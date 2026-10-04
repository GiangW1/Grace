import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from grace_gc.audit import run
from grace_gc.audit.prefix_audit import bundle_from_dict
from grace_gc.data.math_data import MathRecord


class Engine:
    eos_id = None
    decode = staticmethod(lambda tokens: "Answer: 1")

    def __init__(self):
        self.groups = []

    def generate_prefix(self, prompts, length, rng, stream):
        return [p + [3] * length for p in prompts], np.zeros(len(prompts), dtype=bool)

    def continue_repeated(self, prefix, count, length, rng):
        seeds = [int(rng.integers("continuation", 0, 2**31 - 1)) for _ in range(count)]
        return [prefix + [s] for s in seeds], ["stop"] * count, seeds

    def continue_grouped(self, jobs, rng):
        self.groups.append([len(job["seeds"]) for job in jobs])
        return {job["index"]: ([job["prefix"] + [s] for s in job["seeds"]],
                               ["stop"] * len(job["seeds"]), job["seeds"]) for job in jobs}


def collect(monkeypatch, engine, *, grouped, sink=None, resume=None):
    monkeypatch.setattr(run, "_policy_grad_vec",
                        lambda e, l, seq, plen, reward, baseline:
                        np.array([seq[-1] % 1000, reward - baseline, 2.]))
    monkeypatch.setattr(run, "_score_grad_vec", lambda *a: np.array([1., 2., 3.]))
    return run._bundles_from_engines(
        [MathRecord("p", "q", "1")], engine, SimpleNamespace(dim=3), 2, 4, [1, 2], 8, 17,
        lambda rec, n: ([[1]] * n, [rec.problem_id] * n, [rec.answer] * n),
        baseline_fn=lambda pid: .5, store_trajectory_gradients=True,
        batch_continuations=grouped, bundle_sink=sink, bundle_resume=resume)


def test_grouped_requests_preserve_individual_seeds_gradients_and_resume_states(monkeypatch):
    states = []
    reference = collect(monkeypatch, Engine(), grouped=False,
                        sink=lambda bundle, rng: states.append(rng.state_dict()))
    observed_states, engine = [], Engine()
    observed = collect(monkeypatch, engine, grouped=True,
                       sink=lambda bundle, rng: observed_states.append(rng.state_dict()))
    assert engine.groups == [[4, 4], [4, 4]]
    assert observed_states == states
    for actual, expected in zip(observed, reference):
        assert actual.continuation_records == expected.continuation_records
        np.testing.assert_array_equal(actual.trajectory_grads, expected.trajectory_grads)


@pytest.mark.parametrize("stop_after", [1, 3])
def test_partial_prefix_resume_survives_grouped_generation(monkeypatch, tmp_path, stop_after):
    path = Path(__file__).resolve().parents[1] / "scripts/audit_separated_thinking.py"
    spec = importlib.util.spec_from_file_location("execution_runner", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    reference = collect(monkeypatch, Engine(), grouped=False)
    completed = 0

    def interrupt(bundle, rng):
        nonlocal completed
        script.reduce_bundle(bundle, rng, tmp_path, 17)
        completed += 1
        if completed == stop_after:
            raise InterruptedError

    with pytest.raises(InterruptedError):
        collect(monkeypatch, Engine(), grouped=True, sink=interrupt)

    def resume(pid, path_id, t, rng):
        path = script.prefix_directory(tmp_path, path_id, t) / "bundle.json"
        if not path.exists():
            return None
        row = json.loads(path.read_text())
        rng.load_state_dict(row["rng_after"])
        return bundle_from_dict(row)

    actual = collect(monkeypatch, Engine(), grouped=True, resume=resume,
                     sink=lambda bundle, rng: script.reduce_bundle(bundle, rng, tmp_path, 17))
    for observed, expected in zip(actual, reference):
        assert observed.continuation_records == expected.continuation_records
        directory = script.prefix_directory(tmp_path, observed.path_id, observed.t)
        row = json.loads((directory / "bundle.json").read_text())
        for half, indices in zip(("a", "b"), row["half_indices"]):
            np.testing.assert_allclose(np.load(directory / f"mean_{half}.npy"),
                                       expected.trajectory_grads[indices].mean(axis=0))


def test_gpu_grouping_uses_explicit_seeds_without_advancing_rng(monkeypatch, tmp_path):
    from grace_gc.backends import gpu_engine
    from grace_gc.core.rng import IsolatedRNG

    rng = IsolatedRNG.create(17)
    before = rng.state_dict()
    seen = []

    def phase(llm, prompts, limit, temperature, eos_id, supplied_rng, stream, **kwargs):
        seen.append((prompts, limit, kwargs["request_seeds"]))
        return SimpleNamespace(token_ids=[p + [s] for p, s in zip(prompts, kwargs["request_seeds"])],
                               finish_reasons=["stop", "length", "stop"])

    monkeypatch.setattr(gpu_engine, "generate_phase", phase)
    engines, _ = gpu_engine.make_gpu_engines(
        object(), object(), SimpleNamespace(pad_token_id=0, eos_token_id=9), {}, tmp_path)
    jobs = [{"index": 0, "prefix": [1, 2], "max_new": 8, "seeds": [10, 11]},
            {"index": 3, "prefix": [1, 4], "max_new": 8, "seeds": [12]}]
    result = engines.continue_grouped(jobs, rng)
    assert seen == [([[1, 2], [1, 2], [1, 4]], 8, [10, 11, 12])]
    assert result[0] == ([[1, 2, 10], [1, 2, 11]], ["stop", "length"], [10, 11])
    assert result[3] == ([[1, 4, 12]], ["stop"], [12])
    assert rng.state_dict() == before

