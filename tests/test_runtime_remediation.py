import json
import random
from pathlib import Path

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")

from grace_gc.logging_util.ledger import ComputeLedger
from grace_gc.trainer.baseline import HistoricalBaseline
from grace_gc.trainer.checkpoint import load_checkpoint, save_checkpoint
from grace_gc.trainer.loop import build_run_config, run_training
from grace_gc.versions import sha256_file
from scripts.summarize_minimal import summarize


@pytest.fixture
def tiny_config(monkeypatch):
    import grace_gc.trainer.loop as loop
    monkeypatch.setattr(loop, "collect_environment", lambda cfg: {"missing": [], "model": None, "data": None, "file_hashes": {}})
    monkeypatch.setattr("grace_gc.versions.resource_snapshot", lambda: {})
    return build_run_config([], {
        "backend": "cpu_tiny", "method": "full_pg", "seed": 17,
        "n_prompts": 2, "n_start": 4, "decision_tokens": 2,
        "max_new_tokens": 4, "num_steps": 1, "optim": {"lr": 0.02},
        "format_warmup": {"steps": 0}, "save_initial_checkpoint": False,
        "predictor": {"k": 2, "warmup_steps": 0, "audit_s": 1.0, "epochs": 1},
    })


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_interrupted_checkpoint_preserves_previous_archive(tmp_path, monkeypatch):
    path = tmp_path / "checkpoint.npz"
    save_checkpoint(path, {"step": 1})
    old = path.read_bytes()

    def interrupt(stream, payload):
        stream.write(b"partial archive")
        raise OSError("disk interrupted")

    monkeypatch.setattr("grace_gc.trainer.checkpoint._write_archive", interrupt)
    with pytest.raises(OSError, match="interrupted"):
        save_checkpoint(path, {"step": 2})
    assert path.read_bytes() == old
    assert load_checkpoint(path)["step"] == 1
    assert not list(tmp_path.glob("*.tmp"))


def test_one_dump_per_step_and_no_published_step_on_save_failure(tmp_path, tiny_config, monkeypatch):
    import grace_gc.trainer.checkpoint as checkpoint
    original, calls = checkpoint._write_archive, []

    def count(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(checkpoint, "_write_archive", count)
    root = tmp_path / "success"
    run_training(tiny_config, root)
    assert len(calls) == 1
    assert (root / "checkpoint.npz").read_bytes() == (root / "checkpoints/step_1.npz").read_bytes()
    step = rows(root / "steps.jsonl")[0]
    assert step["checkpoint_sha256"] == sha256_file(root / "checkpoint.npz")
    assert step["snapshot_sha"] is not None
    assert step["post_update_snapshot_sha"] is not None
    assert rows(root / "persistence.jsonl")[0]["serialize_write_wall_seconds"] >= 0

    def fail(*args, **kwargs):
        raise OSError("cannot save")

    monkeypatch.setattr(checkpoint, "_write_archive", fail)
    failed = tmp_path / "failed"
    with pytest.raises(OSError, match="cannot save"):
        run_training(tiny_config, failed)
    assert not (failed / "steps.jsonl").exists()
    assert not (failed / "trajectories.jsonl").exists()
    ledger = json.loads((failed / "compute_ledger.json").read_text())
    assert ledger["rows"][-1]["status"] == "failed"
    assert ledger["wall_seconds"] > 0


def test_rewind_forks_without_changing_existing_evidence(tmp_path, tiny_config):
    root = tmp_path / "run"
    run_training({**tiny_config, "num_steps": 3}, root)
    before_log = (root / "steps.jsonl").read_bytes()
    before_checkpoint = sha256_file(root / "checkpoint.npz")
    resumed = run_training({**tiny_config, "resume": str(root / "checkpoints/step_1.npz")}, root)
    fork = Path(resumed["run_dir"])
    assert fork != root
    assert resumed["resume_branch"] is True
    assert [row["step"] for row in rows(fork / "steps.jsonl")] == [2]
    assert (root / "steps.jsonl").read_bytes() == before_log
    assert sha256_file(root / "checkpoint.npz") == before_checkpoint
    assert [row["step"] for row in json.loads((fork / "checkpoints.json").read_text())["steps"]] == [2]


def test_resume_preserves_sessions_effective_optimizer_and_cumulative_cost(tmp_path, tiny_config):
    root = tmp_path / "run"
    first = run_training(tiny_config, root)
    second = run_training({**tiny_config, "optim": {"lr": 0.3}, "seed": 91,
                           "resume": str(root / "checkpoint.npz")}, root)
    assert second["run_dir"] == first["run_dir"]
    assert [row["step"] for row in rows(root / "steps.jsonl")] == [1, 2]
    assert len(list((root / "attempts").glob("*/summary.json"))) == 1
    effective = yaml.safe_load((root / "effective_config.yaml").read_text())
    requested = yaml.safe_load((root / "config.yaml").read_text())
    assert effective["optim"]["lr"] == 0.02
    assert requested["optim"]["lr"] == 0.3
    assert effective["seed"] == 17 and requested["seed"] == 91
    envelope = [row for row in rows(root / "compute_ledger.jsonl") if row["name"] == "train"]
    assert len({row["session_id"] for row in envelope}) == 2
    assert second["cumulative_wall_seconds"] == pytest.approx(sum(row["wall_seconds"] for row in envelope))
    assert second["cumulative_wall_seconds"] > second["session_wall_seconds"]
    info = json.loads((root / "resume.json").read_text())
    assert info["checkpoint_step"] == 1
    assert info["requested_changes"]["seed"] == {"checkpoint": 17, "requested": 91}


def test_ledger_only_excludes_nested_rows_in_their_own_session():
    old = ComputeLedger(1, "a100")
    old.add("train_step", 80.)
    old.add("train", 100., cpu_s=4.)
    resumed = ComputeLedger(1, "a100", rows=old.rows.copy())
    resumed.add("train_step", 20., cpu_s=1.)
    assert resumed.summary()["gpu_reserved_seconds"] == 120.
    assert resumed.summary()["cpu_seconds"] == 5.
    resumed.add("train", 30., cpu_s=2., status="failed")
    assert resumed.summary()["gpu_reserved_seconds"] == 130.
    assert resumed.summary()["cpu_seconds"] == 6.
    unknown = ComputeLedger(1, "a100")
    unknown.add("train", 7.)
    assert unknown.summary()["cpu_seconds"] is None


def test_history_baseline_ht_updates_target_all_starts_not_completions():
    # Enumerate selection of a reward-1 trajectory with p=.25 and a reward-0
    # trajectory with p=1. Expectation must equal the complete-data mean .5.
    estimates = []
    for selected, probability in ((False, .75), (True, .25)):
        baseline = HistoricalBaseline(alpha=1.)
        rewards = [1. if selected else None, 0.]
        baseline.update_ht_mean("p", rewards, [.25, 1.])
        assert rewards == [1. if selected else None, 0.]
        estimates.append(probability * baseline.get("p"))
    assert sum(estimates) == .5
    full = HistoricalBaseline(alpha=.7)
    full.update_ht_mean("p", [1., 0.], [1., 1.])
    assert full.get("p") == .5


def test_global_rng_state_restores_after_actor_and_predictor_initialization(tmp_path, tiny_config):
    from grace_gc.trainer.cpu_tiny import TinyLoRAActor
    from grace_gc.trainer.state_io import restore_train_state, restore_global_rng_state
    root = tmp_path / "run"
    run_training(tiny_config, root)
    payload = load_checkpoint(root / "checkpoint.npz")
    restore_global_rng_state(payload["global_rng"])
    expected = (random.random(), np.random.random(), torch.rand(4))
    actor = TinyLoRAActor()
    optimizer = torch.optim.SGD(actor.trainable_params(), lr=.9)
    restore_train_state(root / "checkpoint.npz", actor, optimizer, 1, 2, "full_pg")
    actual = (random.random(), np.random.random(), torch.rand(4))
    assert expected[0] == actual[0] and expected[1] == actual[1]
    assert torch.equal(expected[2], actual[2])


def test_continuous_and_resumed_grace_match_complete_learned_state(tmp_path, tiny_config):
    cfg = {**tiny_config, "method": "grace", "num_steps": 4}
    run_training(cfg, tmp_path / "continuous")
    root = tmp_path / "split"
    run_training({**cfg, "num_steps": 2}, root)
    run_training({**cfg, "num_steps": 2, "resume": str(root / "checkpoint.npz")}, root)
    a, b = load_checkpoint(tmp_path / "continuous/checkpoint.npz"), load_checkpoint(root / "checkpoint.npz")

    def equal(left, right):
        if isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left:
                equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            assert len(left) == len(right)
            for x, y in zip(left, right):
                equal(x, y)
        elif isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        else:
            assert left == right

    for key in ("actor", "actor_full", "optimizer", "predictor", "basis", "baseline", "reservoir", "rng", "prescan_rng", "history_costs", "global_rng"):
        equal(a[key], b[key])


def test_failed_resume_keeps_prior_cost_and_records_current_envelope(tmp_path, tiny_config, monkeypatch):
    import grace_gc.trainer.loop as loop
    root = tmp_path / "run"
    first = run_training(tiny_config, root)

    def fail(*args, **kwargs):
        raise RuntimeError("resumed operation failed")

    monkeypatch.setattr(loop, "run_tiny_training", fail)
    with pytest.raises(RuntimeError, match="resumed operation failed"):
        run_training({**tiny_config, "resume": str(root / "checkpoint.npz")}, root)
    summary = json.loads((root / "summary.json").read_text())
    assert summary["run_status"] == "failed"
    assert summary["cumulative_wall_seconds"] > first["cumulative_wall_seconds"]
    assert summary["cost_control"]["global_wall_seconds"] > first["cost_control"]["global_wall_seconds"]
    assert rows(root / "cost_control.jsonl")[-1]["event"] == "session_failed"
    assert len(list((root / "attempts").glob("*/summary.json"))) == 1
    assert load_checkpoint(root / "checkpoint.npz")["step"] == 1


def test_rerun_report_uses_actual_stages_and_matching_actor(tmp_path):
    for suffix, hour, avg in (("", "00", .1), ("-new", "01", .9)):
        train = tmp_path / "grace" / ("train" + suffix)
        write_json(train / "summary.json", {"run_status": "complete", "started": f"2026-09-17T{hour}:00:00+00:00",
                                              "finished": f"2026-09-17T{hour}:10:00+00:00", "cumulative_wall_seconds": 700.})
        write_json(tmp_path / "grace" / ("eval-40" + suffix) / "eval_summary.json",
                   {"avg": avg, "checkpoint": str(train / "checkpoints/step_40.npz")})
        write_json(tmp_path / "grace" / ("audit" + suffix) / "audit_summary.json",
                   {"n_bundles": 1, "finished": f"2026-09-17T{hour}:30:00+00:00",
                    "checkpoint": str(train / "checkpoint.npz"), "variance_cost": {"ratio": 9. if suffix else 1.}})
    summarize(tmp_path)
    row = json.loads((tmp_path / "comparison.json").read_text())["methods"][0]
    assert row["final_avg4"] == .9 and row["variance_cost_ratio"] == 9.
    assert row["train_a100_hours"] == 700. / 3600
    # An explicit manifest selects one coherent prior chain, without guessing latest.
    write_json(tmp_path / "stages.json", {"grace": {"train": "grace/train", "eval-40": "grace/eval-40", "audit": "grace/audit"}})
    summarize(tmp_path)
    row = json.loads((tmp_path / "comparison.json").read_text())["methods"][0]
    assert row["final_avg4"] == .1 and row["variance_cost_ratio"] == 1.


def test_checkpoint_cadence_preserves_eval_and_final_snapshots(tmp_path, tiny_config):
    root = tmp_path / "periodic"
    run_training({**tiny_config, "num_steps": 41, "checkpoint_every": 7,
                  "save_initial_checkpoint": True}, root)
    index = json.loads((root / "checkpoints.json").read_text())
    assert [row["step"] for row in index["steps"]] == [0, 7, 14, 20, 21, 28, 35, 40, 41]
    metrics = rows(root / "steps.jsonl")
    assert metrics[0]["checkpoint_sha256"] is None
    assert metrics[0]["last_saved_step"] == 0
    assert metrics[0]["checkpoint_saved"] is False
    assert metrics[-1]["checkpoint_saved"] is True
    assert metrics[-1]["last_saved_step"] == 41
    resumed = run_training({**tiny_config, "num_steps": 2, "checkpoint_every": 7,
                            "resume": str(root / "checkpoint.npz")}, root)
    assert resumed["summary"]["step"] == 43
    assert (root / "checkpoints/step_43.npz").is_file()


def test_training_audit_stays_separate_and_shared_actor_hash_is_evidence(tmp_path):
    train = tmp_path / "grace/train"
    write_json(train / "summary.json", {"run_status": "complete"})
    write_json(train / "initial_actor.json", {"actor_sha256": "same-initial-actor"})
    for stage, ratio in (("audit", 2.), ("audit-training", 3.)):
        write_json(tmp_path / "grace" / stage / "audit_summary.json",
                   {"n_bundles": 1, "finished": "2026-09-18T01:00:00+00:00",
                    "checkpoint": str(train / "checkpoint.npz"), "variance_cost": {"ratio": ratio}})
    summarize(tmp_path)
    result = json.loads((tmp_path / "comparison.json").read_text())
    row = result["methods"][0]
    assert row["variance_cost_ratio"] == 2.
    assert row["training_matched_audit_variance_cost_ratio"] == 3.
    assert row["shared_initial_actor_hash"] == "same-initial-actor"
    assert result["shared_initial_actor_consistency"]["consistent"] is True


def test_report_relocates_absolute_stage_paths_and_exposes_actual_protocol(tmp_path):
    root = tmp_path / "seed-17"
    train = root / "grace/train"
    write_json(train / "summary.json", {"run_status": "complete"})
    (train / "effective_config.yaml").write_text("seed: 23\n", encoding="utf-8")
    protocol = {"problem_ids": ["q1"], "prompt_sha256": "abc", "reward_protocol_version": "strict-v2"}
    write_json(root / "grace/eval-40/eval_summary.json",
               {"avg": .75, "checkpoint": "/old/seed-17/grace/train/checkpoints/step_40.npz"})
    write_json(root / "grace/eval-40/evaluation_manifest.json", protocol)
    write_json(root / "stages.json", {"grace": {
        "train": "/old/seed-17/grace/train", "eval-40": "D:\\old\\seed-17\\grace\\eval-40"}})
    summarize(root)
    row = json.loads((root / "comparison.json").read_text())["methods"][0]
    assert row["actual_seed"] == 23
    assert row["final_avg4"] == .75
    assert row["evaluation_manifest"] == protocol
    assert Path(row["stage_paths"]["train"]) == train


def test_source_snapshot_matches_hash_and_does_not_include_environment_script(tmp_path, monkeypatch):
    import grace_gc.versions as versions
    source, secret = tmp_path / "module.py", tmp_path / "env.sh"
    source.write_bytes(b"value = 17\n")
    secret.write_text("export PRIVATE_VALUE=not-source\n", encoding="utf-8")
    monkeypatch.setattr(versions, "cuda_environment", lambda: {})
    monkeypatch.setattr(versions, "git_info", lambda: {"head": "test", "dirty": False})
    result = versions.collect_versions((), {"grace_gc/module.py": str(source), "scripts/env.sh": str(secret)}, include_source=True)
    assert result["source_files"] == {"grace_gc/module.py": "value = 17\n"}
    assert result["file_hashes"]["grace_gc/module.py"] == versions.sha256_bytes(result["source_files"]["grace_gc/module.py"].encode("utf-8"))


@pytest.mark.parametrize("mismatch,issue", [
    ("temperature", "evaluation_temperature_missing_or_mismatched"),
    ("seed", "actual_seed_missing_or_mismatched"),
    ("initial", "shared_initial_actor_hash_missing_or_mismatched"),
    ("missing", "evaluation_ordered_records_sha256_missing_or_mismatched"),
])
def test_report_withholds_paired_ci_for_same_questions_but_unmatched_protocol(tmp_path, mismatch, issue):
    protocol = {"ordered_records_sha256": "same-questions", "reward_protocol_version": "v2",
                "samples_per_problem": 4, "temperature": 1., "top_p": 1.,
                "max_new_tokens": 2048, "sample_batch_size": 1, "sample_seed_start": 17}
    for method, avg in (("grace", .75), ("grpo", .5)):
        train = tmp_path / method / "train"
        write_json(train / "summary.json", {"run_status": "complete"})
        seed = 23 if method == "grpo" and mismatch == "seed" else 17
        (train / "effective_config.yaml").write_text(f"seed: {seed}\n", encoding="utf-8")
        actor_hash = "different" if method == "grpo" and mismatch == "initial" else "shared"
        write_json(train / "initial_actor.json", {"actor_sha256": actor_hash})
        manifest = dict(protocol)
        if method == "grpo" and mismatch == "temperature":
            manifest["temperature"] = .7
        if method == "grpo" and mismatch == "missing":
            manifest = {}
        write_json(tmp_path / method / "eval-40/eval_summary.json", {
            "avg": avg, "per_problem": [{"problem_id": "q1", "avg": avg}],
            "checkpoint": str(train / "checkpoints/step_40.npz"), "evaluation_manifest": manifest})
    summarize(tmp_path)
    result = json.loads((tmp_path / "comparison.json").read_text())
    assert {row["method"]: row["final_avg4"] for row in result["methods"]} == {"grace": .75, "grpo": .5}
    paired = result["grace_minus_grpo"]
    assert paired["available"] is False and issue in paired["issues"]
    assert "ci95" not in paired and "effect" not in paired
    # Supplying matching evidence makes the same raw observations comparable.
    train = tmp_path / "grpo/train"
    (train / "effective_config.yaml").write_text("seed: 17\n", encoding="utf-8")
    write_json(train / "initial_actor.json", {"actor_sha256": "shared"})
    final_path = tmp_path / "grpo/eval-40/eval_summary.json"
    final = json.loads(final_path.read_text())
    final["evaluation_manifest"] = protocol
    write_json(final_path, final)
    summarize(tmp_path)
    paired = json.loads((tmp_path / "comparison.json").read_text())["grace_minus_grpo"]
    assert paired["available"] is True and paired["effect"] == .25 and paired["issues"] == []


def test_wall_feedback_uses_measured_seconds_and_keeps_next_n_grouped():
    from grace_gc.trainer.cost_control import CostControl
    controller = CostControl({"cost_control": {"enabled": True, "target_step_seconds": 10., "ema_alpha": 1., "max_n": 20}})
    assert controller.observe(8, 20., 1, group=4) == 4
    assert controller.observe(4, 2., 2, group=4) == 20
    assert controller.state["last_n"] == 4
    assert controller.state["ema_seconds_per_start"] == .5
    inferred = CostControl({"cost_control": {"enabled": True}})
    assert inferred.observe(8, 12., 1, group=4) == 8
    assert inferred.state["target_step_seconds"] == 12.
    assert inferred.state["target_source"] == "first_observed_complete_batch"


def test_run_budget_counts_setup_and_persistence_and_stops_at_batch_boundary(tmp_path, tiny_config, monkeypatch):
    import grace_gc.trainer.loop as loop
    clock = [0.]

    class ClockTimer:
        def __init__(self):
            self.start = clock[0]
        def elapsed(self):
            return clock[0] - self.start
        def cpu_elapsed(self):
            return 0.

    monkeypatch.setattr(loop, "Timer", ClockTimer)
    def environment(cfg):
        clock[0] += 5.
        return {"missing": [], "file_hashes": {}}
    monkeypatch.setattr(loop, "collect_environment", environment)
    original_initial, original_step, original_persist, original_final = (
        loop.persist_initial_checkpoint, loop.run_algorithm1_step, loop.persist_training_step, loop.persist_final_checkpoint)
    def initial(*args, **kwargs):
        result = original_initial(*args, **kwargs)
        clock[0] += 3.
        return result
    def step(*args, **kwargs):
        result = original_step(*args, **kwargs)
        clock[0] += 4.
        return result
    def persist(*args, **kwargs):
        result = original_persist(*args, **kwargs)
        clock[0] += 7.
        return result
    def final(run, cfg, state, *args, **kwargs):
        needs_save = cfg.get("_last_saved_step") != state.step
        result = original_final(run, cfg, state, *args, **kwargs)
        if needs_save:
            clock[0] += 2.
        return result
    monkeypatch.setattr(loop, "persist_initial_checkpoint", initial)
    monkeypatch.setattr(loop, "run_algorithm1_step", step)
    monkeypatch.setattr(loop, "persist_training_step", persist)
    monkeypatch.setattr(loop, "persist_final_checkpoint", final)
    root = tmp_path / "budget"
    result = run_training({**tiny_config, "num_steps": 10, "checkpoint_every": 10, "save_initial_checkpoint": True,
                           "run_wall_seconds": 25., "cost_control": {"enabled": True, "target_step_seconds": 5., "ema_alpha": 1.}}, root)
    assert [row["n"] for row in rows(root / "steps.jsonl")] == [4, 2]
    assert result["summary"]["step"] == 2 and result["summary"]["steps"] == 2
    assert result["cost_control"]["stop_reason"] == "run_wall_seconds"
    assert result["cost_control"]["global_wall_seconds"] == 32.
    assert result["cost_control"]["run_overshoot_seconds"] == 7.
    assert result["cost_control"]["cap_before_wall_budget"] is False
    assert load_checkpoint(root / "checkpoint.npz")["step"] == 2
    events = rows(root / "cost_control.jsonl")
    assert next(row for row in events if row["event"] == "setup_complete")["session_wall_seconds"] == 8.
    completed = [row for row in events if row["event"] == "batch_complete"]
    assert [row["last_batch_wall_seconds"] for row in completed] == [11., 13.]


def test_cost_recovery_uses_checkpoint_boundary_decision_but_all_actual_spend(tmp_path):
    from grace_gc.trainer.cost_control import restore_cost_control
    class Clock:
        def elapsed(self):
            return 2.
    root = tmp_path / "run"
    checkpoint = root / "checkpoints/step_2.npz"
    checkpoint.parent.mkdir(parents=True)
    (root / "cost_control.jsonl").write_text("\n".join(json.dumps(row) for row in [
        {"event": "batch_complete", "step": 2, "next_n": 8, "observations": 2, "global_wall_seconds": 30.},
        {"event": "batch_complete", "step": 3, "next_n": 12, "observations": 3, "global_wall_seconds": 40.},
        {"event": "session_failed", "step": 3, "global_wall_seconds": 50.},
    ]) + "\n{incomplete", encoding="utf-8")
    write_json(root / "summary.json", {"cumulative_wall_seconds": 51.})
    controller = restore_cost_control({"resume": str(checkpoint), "cost_control": {"enabled": True},
                                       "run_wall_seconds": 52.}, Clock(), {"step": 2, "cost_control": {"next_n": 4}})
    assert controller.state["next_n"] == 8 and controller.state["observations"] == 2
    assert controller.snapshot()["global_wall_seconds"] == 53.
    assert controller.stop_reason() == "run_wall_seconds"
    assert controller.recovery_source == "checkpoint_step_post_save_cost_log"


def test_zero_step_initialization_and_step_cap_publish_actual_terminal_state(tmp_path, tiny_config):
    root = tmp_path / "initial"
    result = run_training({**tiny_config, "num_steps": 0, "save_initial_checkpoint": False,
                           "run_wall_seconds": 1000.}, root)
    assert load_checkpoint(root / "checkpoint.npz")["step"] == 0
    assert result["summary"]["steps"] == 0
    assert result["cost_control"]["stop_reason"] == "num_steps"
    assert result["cost_control"]["cap_before_wall_budget"] is True
    assert result["cost_control"]["num_steps_cap_reached"] is True


def test_wall_feedback_resumes_post_save_next_n_and_uses_independent_session_budget(tmp_path, tiny_config):
    root = tmp_path / "resume-cost"
    cfg = {**tiny_config, "cost_control": {"enabled": True, "target_step_seconds": 1e-9},
           "session_wall_seconds": 1000.}
    first = run_training(cfg, root)
    assert rows(root / "steps.jsonl")[-1]["n"] == 4
    assert first["cost_control"]["next_n"] == 2
    # The archive was written before observing its own save cost.
    assert load_checkpoint(root / "checkpoint.npz")["cost_control"].get("next_n") is None
    second = run_training({**cfg, "resume": str(root / "checkpoint.npz")}, root)
    assert rows(root / "steps.jsonl")[-1]["n"] == 2
    cost = second["cost_control"]
    assert cost["observations"] == 2
    assert cost["recovery_source"] == "checkpoint_step_post_save_cost_log"
    assert cost["prior_wall_seconds"] >= first["cost_control"]["global_wall_seconds"]
    assert cost["global_wall_seconds"] == cost["prior_wall_seconds"] + cost["session_wall_seconds"]


def test_wall_budget_may_end_in_warmup_without_a_quality_gate(tmp_path, tiny_config):
    result = run_training({**tiny_config, "method": "grace", "run_wall_seconds": 1000.,
                           "predictor": {**tiny_config["predictor"], "warmup_steps": 20}}, tmp_path / "warmup")
    assert result["summary"]["steps"] == 1
    assert result["summary"]["post_warmup_steps"] == 0
    assert result["summary"]["allocation_ready_steps_this_session"] == 0
    assert result["summary"]["allocation_ready_last_batch"] is False


def test_exhausted_resume_into_new_directory_still_publishes_terminal_checkpoint(tmp_path, tiny_config):
    source = tmp_path / "source"
    first = run_training(tiny_config, source)
    destination = tmp_path / "resume-with-no-budget-left"
    result = run_training({**tiny_config, "resume": str(source / "checkpoint.npz"),
                           "run_wall_seconds": 1e-9}, destination)
    assert result["step"] == 1 and result["summary"]["steps"] == 0
    assert result["stop_reason"] == "run_wall_seconds"
    assert load_checkpoint(destination / "checkpoint.npz")["step"] == 1
    assert result["cost_control"]["prior_wall_seconds"] >= first["cost_control"]["global_wall_seconds"]


def test_step_log_distinguishes_token_proxy_from_post_save_cost_schedule(tmp_path, tiny_config):
    controlled = tmp_path / "controlled"
    run_training({**tiny_config, "num_steps": 2,
                  "cost_control": {"enabled": True, "target_step_seconds": 1e-9}}, controlled)
    steps = rows(controlled / "steps.jsonl")
    assert steps[0]["token_proxy_next_n"] == 4
    assert steps[0]["next_n"] is None
    assert steps[0]["next_n_source"] == "post_save_cost_control.jsonl"
    costs = [row for row in rows(controlled / "cost_control.jsonl") if row["event"] == "batch_complete"]
    assert costs[0]["next_n"] == 2 == steps[1]["n"]
    legacy = tmp_path / "legacy"
    run_training(tiny_config, legacy)
    row = rows(legacy / "steps.jsonl")[0]
    assert row["next_n"] == 4
    assert "token_proxy_next_n" not in row
