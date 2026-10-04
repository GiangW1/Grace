import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from grace_gc.audit import run
from grace_gc.audit.prefix_audit import bundle_from_dict
from grace_gc.data.math_data import MathRecord


def load_script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Engine:
    eos_id = None
    last_rollout = None
    decode = staticmethod(lambda tokens: "Answer: 1")

    def generate_prefix(self, prompts, length, rng, stream):
        return [prompt + [3] * length for prompt in prompts], np.zeros(len(prompts), dtype=bool)

    def continue_selected(self, prefixes, selected, length, rng):
        seeds = [int(rng.integers("continuation", 10, 10000)) for _ in prefixes]
        self.last_rollout = {"continue_finish_reasons": dict.fromkeys(range(len(prefixes)), "stop"),
                             "sampling": {"request_seeds": seeds}}
        return [prefix + [seed] for prefix, seed in zip(prefixes, seeds)]

    def continue_repeated(self, prefix, count, length, rng):
        result = self.continue_selected([prefix] * count, np.ones(count, dtype=bool), length, rng)
        return result, ["stop"] * count, self.last_rollout["sampling"]["request_seeds"]


def collect(monkeypatch, sink=None, resume=None, engine=None):
    monkeypatch.setattr(run, "_policy_grad_vec", lambda e, l, seq, plen, reward, baseline:
                        np.array([seq[-1] / 10000, reward - baseline, 2.]))
    monkeypatch.setattr(run, "_score_grad_vec", lambda *a: np.array([1., 2., 3.]))
    return run._bundles_from_engines(
        [MathRecord("p", "q", "1")], engine or Engine(), SimpleNamespace(dim=3),
        2, 4, [1, 2], 8, 17, lambda rec, n: ([[1]] * n, [rec.problem_id] * n, [rec.answer] * n),
        baseline_fn=lambda pid: .5, store_trajectory_gradients=True, bundle_sink=sink, bundle_resume=resume)


def test_batched_continuations_keep_independent_seeds_and_gradients(monkeypatch):
    batched = collect(monkeypatch)
    sequential = Engine()
    sequential.continue_repeated = None
    reference = collect(monkeypatch, engine=sequential)
    for actual, expected in zip(batched, reference):
        assert actual.continuation_records == expected.continuation_records
        np.testing.assert_array_equal(actual.trajectory_grads, expected.trajectory_grads)
    assert len({row["seed"] for row in batched[0].continuation_records}) == 4


def test_prefix_resume_preserves_rng_and_full_space_moments(monkeypatch, tmp_path):
    script = load_script("audit_separated_thinking")
    reference = collect(monkeypatch)

    def interrupt(bundle, rng):
        script.reduce_bundle(bundle, rng, tmp_path, 17)
        raise InterruptedError

    with pytest.raises(InterruptedError):
        collect(monkeypatch, sink=interrupt)

    def resume(pid, path_id, t, rng):
        path = script.prefix_directory(tmp_path, path_id, t) / "bundle.json"
        if not path.exists():
            return None
        row = json.loads(path.read_text())
        rng.load_state_dict(row["rng_after"])
        return bundle_from_dict(row)

    def sink(bundle, rng):
        script.reduce_bundle(bundle, rng, tmp_path, 17)

    actual = collect(monkeypatch, sink=sink, resume=resume)
    for observed, expected in zip(actual, reference):
        assert observed.continuation_records == expected.continuation_records
        directory = script.prefix_directory(tmp_path, observed.path_id, observed.t)
        row = json.loads((directory / "bundle.json").read_text())
        for half, indices in zip(("a", "b"), row["half_indices"]):
            np.testing.assert_allclose(np.load(directory / f"mean_{half}.npy"),
                                       expected.trajectory_grads[indices].mean(axis=0))
        assert np.mean(row["half_mean_full_norm_sq"]) == pytest.approx(
            np.mean(np.sum(expected.trajectory_grads ** 2, axis=1)))


@pytest.mark.parametrize("p", [.2, .5, .8, 1.])
def test_moment_cost_matches_per_trajectory_estimator(p):
    from grace_gc.audit.variance_cost import variance_times_cost
    script = load_script("thinking_predictor_benefit")
    gradients = np.array([[1., 2.], [3., -1.], [-2., 4.], [2., 1.]])
    predictions = np.array([[.5, 1.], [.5, 1.], [-.2, 1.], [-.2, 1.]])
    norms = np.mean(np.sum(gradients.reshape(2, 2, 2) ** 2, axis=-1), axis=1)
    means = gradients.reshape(2, 2, 2).mean(axis=1)
    residuals = np.mean(np.sum((gradients - predictions).reshape(2, 2, 2) ** 2, axis=-1), axis=1)
    actual = script.moment_product(norms, means @ means.T, residuals,
                                   np.array([1., 2.]), np.array([3., 4.]), p)
    expected = variance_times_cost(gradients, predictions, np.full(4, p),
                                  np.array([1., 1., 2., 2.]), np.array([3., 3., 4., 4.]))
    assert actual["variance_cost_ratio"] == pytest.approx(expected["ratio"])


def test_cpu_analysis_accepts_collected_statistics(monkeypatch, tmp_path):
    script = load_script("audit_separated_thinking")
    monkeypatch.syspath_prepend(str(Path(script.__file__).parent))
    provenance = {"layout_dim": 3, "actor_sha256": "synthetic-test", "checkpoint_sha256": "synthetic-test",
                  "layout_names": ["synthetic-test"], "layout_sha256": "synthetic-test"}
    script.write_json(tmp_path / "worker-0-provenance.json", provenance)
    from grace_gc.core.rng import IsolatedRNG
    for index in range(32):
        directory = script.problem_directory(tmp_path, str(index))
        script.write_json(directory / "summary.json", {"status": "completed"})
        if index < 3:
            bundle = collect(monkeypatch)[0]
            bundle.problem_id, bundle.path_id = str(index), f"{index}:0"
            bundle.features = [1., float(index), 2.]
            bundle.qualification_n, bundle.qualification_mean_reward = 4, .25
            bundle.functional_recoverable, bundle.answer_emitted = False, False
            script.reduce_bundle(bundle, IsolatedRNG.create(17), directory, 17)
    script.merge(SimpleNamespace(run_dir=str(tmp_path), seed=17, allow_legacy=True))
    output = json.loads((tmp_path / "benefit/predictor_benefit_summary.json").read_text())
    assert output["status"] == "completed"
    assert output["n_observed_problems"] == 3
    assert output["strict_pre_answer_undecided_prefixes"] == 3
    for row in output["lopo"]:
        full = next(item for item in row["variance_cost"] if item["p"] == 1.)
        assert full["variance_cost_ratio"] == pytest.approx(1.)


def test_merge_rejects_stale_problem_ids(tmp_path):
    script = load_script("audit_separated_thinking")
    script.write_json(tmp_path / "selection.json", {
        "n_problems": 1,
        "problems": [{"problem_id": "selected"}],
        "reward_protocol_version": 3,
    })
    directory = script.problem_directory(tmp_path, "stale")
    script.write_json(directory / "summary.json", {"status": "completed", "problem_id": "stale"})
    with pytest.raises(ValueError, match="completed problem IDs"):
        script.merge(SimpleNamespace(run_dir=str(tmp_path), seed=17, allow_legacy=False))
