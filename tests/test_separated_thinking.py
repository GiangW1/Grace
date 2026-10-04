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


@pytest.mark.parametrize("with_selection,metric", [
    (False, None), (True, None), (True, np.array([2., .5, 3.]))])
def test_cpu_analysis_preserves_certain_q_despite_uncertain_probes(monkeypatch, tmp_path, with_selection, metric):
    script = load_script("audit_separated_thinking")
    monkeypatch.syspath_prepend(str(Path(script.__file__).parent))
    provenance = {"layout_dim": 3, "actor_sha256": "synthetic-test", "checkpoint_sha256": "synthetic-test",
                  "layout_names": ["synthetic-test"], "layout_sha256": "synthetic-test"}
    if with_selection:
        from grace_gc.versions import sha256_array
        provenance["metric_weights_sha256"] = sha256_array(np.ones(3) if metric is None else metric)
        if metric is not None:
            provenance["metric"] = "test_diagonal"
            script.prepare_metric(tmp_path, metric)
        script.write_json(tmp_path / "selection.json", {
            "n_problems": 32, "problems": [{"problem_id": str(i)} for i in range(32)],
            "reward_protocol_version": 3})
    script.write_json(tmp_path / "worker-0-provenance.json", provenance)
    from grace_gc.core.rng import IsolatedRNG
    for index in range(32):
        directory = script.problem_directory(tmp_path, str(index))
        script.write_json(directory / "summary.json", {
            "status": "completed", "problem_id": str(index), "n_bundles": int(index < 3)})
        if index < 3:
            bundle = collect(monkeypatch)[0]
            bundle.problem_id, bundle.path_id = str(index), f"{index}:0"
            bundle.features = [1., float(index), 2.]
            bundle.qualification_n, bundle.qualification_mean_reward = 4, .25
            bundle.functional_recoverable, bundle.answer_emitted = False, False
            script.reduce_bundle(bundle, IsolatedRNG.create(17), directory, 17, metric=metric,
                                 metric_name="euclidean" if metric is None else "test_diagonal")
    script.merge(SimpleNamespace(run_dir=str(tmp_path), seed=17, allow_legacy=not with_selection))
    output = json.loads((tmp_path / "benefit/predictor_benefit_summary.json").read_text())
    assert output["status"] == "completed"
    assert output["n_observed_problems"] == 3
    assert output["n_original_problems"] == 32
    assert len(output["no_surviving_prefix_problem_ids"]) == 29
    assert output["strict_pre_answer_undecided_prefixes"] == 0
    rows = [json.loads(line) for line in (tmp_path / "replay/prefixes.jsonl").read_text().splitlines()]
    assert all(row["normal_q_mean_reward"] == 1.0 for row in rows)
    assert output["metric_name"] == ("euclidean" if metric is None else "test_diagonal")
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


@pytest.mark.parametrize("n_bundles", [0, 1])
def test_merge_distinguishes_no_surviving_prefix_from_lost_bundles(tmp_path, n_bundles):
    script = load_script("audit_separated_thinking")
    script.write_json(tmp_path / "selection.json", {
        "n_problems": 1, "problems": [{"problem_id": "short"}], "reward_protocol_version": 3})
    script.write_json(script.problem_directory(tmp_path, "short") / "summary.json", {
        "status": "completed", "problem_id": "short", "n_bundles": n_bundles})
    args = SimpleNamespace(run_dir=str(tmp_path), seed=17, allow_legacy=False)
    if n_bundles:
        with pytest.raises(ValueError, match="bundle count differs"):
            script.merge(args)
        assert not (tmp_path / "analysis_status.json").exists()
    else:
        script.merge(args)
        output = json.loads((tmp_path / "benefit/predictor_benefit_summary.json").read_text())
        assert output["status"] == "completed"
        assert output["n_original_problems"] == 1
        assert output["n_observed_problems"] == output["n_prefixes"] == 0
        assert output["no_surviving_prefix_problem_ids"] == ["short"]
        assert output["predictor_benefit"] is None


def test_merge_rejects_bundles_outside_selection(tmp_path):
    script = load_script("audit_separated_thinking")
    script.write_json(tmp_path / "selection.json", {
        "n_problems": 1, "problems": [{"problem_id": "selected"}], "reward_protocol_version": 3})
    script.write_json(script.problem_directory(tmp_path, "selected") / "summary.json", {
        "status": "completed", "problem_id": "selected", "n_bundles": 0})
    script.write_json(script.problem_directory(tmp_path, "stale") / "path-0-t1024/bundle.json", {
        "problem_id": "stale", "reward_protocol_version": 3})
    with pytest.raises(ValueError, match="bundle problem IDs"):
        script.merge(SimpleNamespace(run_dir=str(tmp_path), seed=17, allow_legacy=False))


@pytest.mark.parametrize("reason,count,eos,expected", [
    ("length", 1, None, True), ("stop", 32, None, False),
    (None, 32, None, True), (None, 32, 3, False), (None, 1, None, False)])
def test_probe_uses_real_termination_and_preserves_scoring_evidence(reason, count, eos, expected):
    from grace_gc.audit.qualification import score_probe_sample

    sample = score_probe_sample([9] + [3] * count, 1, " 1\nStill explaining", "1", 32,
                                eos_id=eos, finish_reason=reason)
    assert sample["truncated"] is expected
    assert sample["reward"] == (0.0 if expected else 1.0)
    assert sample["token_ids"] == [3] * count
    assert sample["finish_reason"] == reason


@pytest.mark.parametrize("finish_reason,expected", [("stop", 1.0), ("length", 0.0)])
def test_gpu_audit_probe_passes_actual_finish_reason(monkeypatch, finish_reason, expected):
    from grace_gc.data import tokenize

    monkeypatch.setattr(tokenize, "load_hf_tokenizer", lambda path:
                        SimpleNamespace(encode=lambda text, **kw: [9]))

    class ProbeEngine(Engine):
        decode = staticmethod(lambda tokens: " 1")

        def continue_selected(self, prefixes, selected, length, rng):
            self.last_rollout = {"continue_finish_reasons": {0: finish_reason}}
            return [prefixes[0] + [3]]

    def capture(*args, **kwargs):
        result = kwargs["qualification_fn"]([1, 2], 2, "1", "p", 64, 0)
        assert result["qualification_mean_reward"] == expected
        return []

    monkeypatch.setattr(run, "_bundles_from_engines", capture)
    config = {"model_path": "mock", "method": "full_pg", "enable_thinking": True,
              "audit": {"functional_qualification": {"enabled": True, "n": 1, "max_new_tokens": 32}}}
    run.generate_bundles_gpu([MathRecord("p", "q", "1")], 1, 2, 64, 128, 17, config,
                             engines=ProbeEngine(), layout=SimpleNamespace(dim=3))


def test_sink_keeps_both_euclidean_and_diagonal_second_moments(monkeypatch, tmp_path):
    from grace_gc.core.rng import IsolatedRNG
    from grace_gc.versions import sha256_array

    script = load_script("audit_separated_thinking")
    bundle = collect(monkeypatch)[0]
    gradients = bundle.trajectory_grads.copy()
    metric = np.array([2., .5, 3.])
    script.reduce_bundle(bundle, IsolatedRNG.create(17), tmp_path, 17, metric=metric, metric_name="test_metric")
    row = json.loads((script.prefix_directory(tmp_path, bundle.path_id, bundle.t) / "bundle.json").read_text())
    for half, indices in enumerate(row["half_indices"]):
        assert row["half_full_metric_norm_sq"][half] == pytest.approx(
            np.mean(np.sum(gradients[indices] ** 2 * metric, axis=1)))
        assert row["half_mean_full_norm_sq"][half] == pytest.approx(
            np.mean(np.sum(gradients[indices] ** 2, axis=1)))
    assert row["metric_weights_sha256"] == sha256_array(metric)


def test_metric_resume_rejects_changed_weights_without_overwriting(tmp_path):
    from grace_gc.versions import sha256_array

    script = load_script("audit_separated_thinking")
    metric = np.array([2., .5, 3.])
    script.prepare_metric(tmp_path, metric)
    original = (tmp_path / "metric_weights.npy").read_bytes()
    script.write_json(tmp_path / "worker-0-provenance.json", {"metric_weights_sha256": sha256_array(metric)})
    script.write_json(tmp_path / "problems/p/path-0-t1024/bundle.json", {
        "metric_weights_sha256": sha256_array(metric), "half_full_metric_norm_sq": [1., 2.]})
    script.prepare_metric(tmp_path, metric.copy())
    with pytest.raises(ValueError, match="metric weights differ"):
        script.prepare_metric(tmp_path, np.ones(3))
    assert (tmp_path / "metric_weights.npy").read_bytes() == original


@pytest.mark.parametrize("stale_source", ["prefix", "worker-0", "worker-1"])
def test_merge_rejects_changed_metric_before_writing_replay(monkeypatch, tmp_path, stale_source):
    from grace_gc.core.rng import IsolatedRNG
    from grace_gc.versions import sha256_array

    script = load_script("audit_separated_thinking")
    metric = np.array([2., .5, 3.])
    script.prepare_metric(tmp_path, metric)
    bundle = collect(monkeypatch)[0]
    directory = script.problem_directory(tmp_path, bundle.problem_id)
    script.reduce_bundle(bundle, IsolatedRNG.create(17), directory, 17, metric=metric)
    script.write_json(directory / "summary.json", {
        "status": "completed", "problem_id": bundle.problem_id, "n_bundles": 1})
    script.write_json(tmp_path / "selection.json", {
        "n_problems": 1, "problems": [{"problem_id": bundle.problem_id}], "reward_protocol_version": 3})
    provenance = {"layout_dim": 3, "metric_weights_sha256": sha256_array(metric)}
    script.write_json(tmp_path / "worker-0-provenance.json", provenance)
    if stale_source == "prefix":
        path = script.prefix_directory(directory, bundle.path_id, bundle.t) / "bundle.json"
        row = json.loads(path.read_text())
        row["metric_weights_sha256"] = sha256_array(np.ones(3))
        script.write_json(path, row)
    else:
        script.write_json(tmp_path / f"{stale_source}-provenance.json", {
            **provenance, "metric_weights_sha256": sha256_array(np.ones(3))})
    with pytest.raises(ValueError, match="metric weights differ"):
        script.merge(SimpleNamespace(run_dir=str(tmp_path), seed=17, allow_legacy=False))
    assert not (tmp_path / "replay").exists()


def test_diagonal_metric_requires_weighted_second_moments_even_in_legacy_mode():
    from grace_gc.versions import sha256_array

    script = load_script("audit_separated_thinking")
    metric_hash = sha256_array(np.array([2., .5]))
    with pytest.raises(ValueError, match="second moments"):
        script.validate_metric_record({"metric_weights_sha256": metric_hash}, metric_hash,
                                      False, prefix=True, allow_legacy=True)
    with pytest.raises(ValueError, match="lack metric provenance"):
        script.validate_metric_record({}, metric_hash, False, prefix=True, allow_legacy=True)


def test_benefit_cannot_fall_back_to_euclidean_second_moments(tmp_path):
    from grace_gc.versions import sha256_array

    script = load_script("thinking_predictor_benefit")
    replay = tmp_path / "replay"
    replay.mkdir()
    (replay / "prefixes.jsonl").write_text(json.dumps({"problem_id": "p"}) + "\n")
    np.save(replay / "half_mean_full_grads_a.npy", np.ones((1, 2)))
    np.save(replay / "half_mean_full_grads_b.npy", np.ones((1, 2)))
    np.save(replay / "prefix_score_gradients.npy", np.ones((1, 2, 1)))
    np.save(replay / "half_mean_full_norm_sq_b.npy", np.array([2.]))
    metric = np.array([2., .5])
    np.save(replay / "metric_weights.npy", metric)
    load_script("audit_separated_thinking").write_json(replay / "replay_provenance.json", {
        "metric_weights_sha256": sha256_array(metric)})
    with pytest.raises(ValueError, match="lacks second moments"):
        script.report_benefit(replay, tmp_path / "benefit")


@pytest.mark.parametrize("gpus", [[0, 2], [0, 2, 3]])
def test_supervisor_assigns_all_problems_to_requested_gradient_workers(tmp_path, monkeypatch, gpus):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    script = load_script("supervise_thinking_audit")
    model = tmp_path / "model"
    model.mkdir()
    (model / "thinking_download_verified.txt").write_text(script.REVISION)
    for name in ("ROOT", "MODEL", "CHECKPOINT", "URL"):
        monkeypatch.setattr(script, name, getattr(script, name))
    monkeypatch.setattr(script.signal, "signal", lambda *args: None)
    monkeypatch.setattr(script, "cleanup", lambda *args: None)
    monkeypatch.setattr(script, "wait_for_card", lambda *args: None)
    monkeypatch.setattr(script, "ready", lambda: True)
    monkeypatch.setattr(script, "verify_smoke", lambda: None)
    monkeypatch.setattr(script, "audit_flags", lambda: [])
    phases, servers, workers, analyses = [], [], [], []
    monkeypatch.setattr(script, "status", lambda phase, **extra: phases.append((phase, extra)))

    def launch(command, log, gpu=None):
        if gpu is None:
            analyses.append(list(map(str, command)))
        else:
            servers.append(gpu)
        return SimpleNamespace(poll=lambda: None, wait=lambda: 0)

    def worker(index, gpu, smoke=False, extra_args=()):
        workers.append((index, gpu, smoke, extra_args))

    monkeypatch.setattr(script, "launch", launch)
    monkeypatch.setattr(script, "worker", worker)
    script.main(["--run-dir", str(tmp_path), "--data-dir", str(tmp_path / "inputs"),
                 "--model-path", str(model), "--port", "18099", "--request-concurrency", "16",
                 "--n-continuations", "32",
                 "--gpus", *map(str, gpus)])
    assert servers == [gpus[-1]]
    assert sorted((index, gpu) for index, gpu, smoke, _extra in workers if not smoke) == list(enumerate(gpus[:-1]))
    for _index, _gpu, _smoke, extra in workers:
        assert extra[extra.index("--workers") + 1] == str(len(gpus) - 1)
        assert extra[extra.index("--n-problems") + 1] == "32"
        assert extra[extra.index("--server-url") + 1] == "http://127.0.0.1:18099"
        assert extra[extra.index("--request-concurrency") + 1] == "16"
        assert extra[extra.index("--n-continuations") + 1] == "32"
    scope = json.loads((tmp_path / "scope.json").read_text())
    assert set(scope["gpu_roles"]) == set(map(str, gpus))
    assert scope["n_problems"] == 32
    assert scope["request_concurrency"] == 16
    assert scope["n_continuations"] == 32
    assert len(analyses) == 1 and "--analyze-only" in analyses[0]
    assert phases[-1][0] == "completed"


def test_collection_scope_preserves_sample_count_on_resume(tmp_path):
    script = load_script("audit_separated_thinking")
    expected = script.prepare_collection_scope(tmp_path, 32)
    assert expected["n_continuations"] == 32
    assert script.prepare_collection_scope(tmp_path, 32) == expected
    with pytest.raises(ValueError, match="parameters differ"):
        script.prepare_collection_scope(tmp_path, 64)
    assert json.loads((tmp_path / "collection_scope.json").read_text()) == expected


def test_collection_scope_does_not_relabel_legacy_64_sample_results(tmp_path):
    script = load_script("audit_separated_thinking")
    script.write_json(tmp_path / "worker-0-provenance.json", {"enable_thinking": True})
    with pytest.raises(ValueError, match="continuation count differs"):
        script.prepare_collection_scope(tmp_path, 32)
    assert not (tmp_path / "collection_scope.json").exists()
    assert script.prepare_collection_scope(tmp_path, 64)["n_continuations"] == 64
