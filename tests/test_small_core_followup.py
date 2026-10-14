import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from scripts.run_small_core_experiment import main as plan
from scripts.analyze_small_core_prediction import prediction_row, aggregate, analyze
from scripts.summarize_small_core_experiment import paired_delta, summarize_suite, step_health


def arguments(root):
    return ["--model-path", "Qwen3-4B", "--data-path", "train.parquet", "--eval-data-path", "math500.jsonl",
            "--followup", "--run-dir", str(root)]


def test_followup_plan_shared_initial_frozen_audit_and_matched_controls(tmp_path):
    root = tmp_path / "suite"
    assert plan(arguments(root)) == 0
    manifest = json.loads((root / "suite.json").read_text())
    assert manifest["status"] == "planned"
    jobs = manifest["jobs"]
    assert jobs[0]["phase"] == "train" and jobs[0]["arm"] == "core8_full_pg"
    assert len([j for j in jobs if j["phase"] == "initial_eval"]) == 1
    assert len([j for j in jobs if j["phase"] == "train"]) == 3
    assert len([j for j in jobs if j["phase"] == "eval"]) == 3
    assert not any(j["phase"] == "eval_snapshot" for j in jobs)
    assert len([j for j in jobs if j["phase"] == "audit"]) == 1
    assert len([j for j in jobs if j["phase"] == "prediction_report"]) == 1
    assert {j["seed"] for j in jobs} == {17}
    assert {j["arm"] for j in jobs} == {"core8_full_pg", "core8_uniform_ht", "core8_grace"}
    assert len(jobs) == 9  # 3 train + 1 initial + 3 final + 1 audit + 1 CPU report
    for j in jobs:
        if j["phase"] in {"eval", "eval_snapshot", "initial_eval"}:
            assert j["command"][j["command"].index("--seed") + 1] == "17"
        if j["phase"] == "audit":
            assert j["arm"] == "core8_grace"
    cfgs = [yaml.safe_load((root / f"core8_{method}-seed17.yaml").read_text())
            for method in ["full_pg", "uniform_ht", "grace"]]
    assert all({k: v for k, v in cfg.items() if k != "method"} ==
               {k: v for k, v in cfgs[0].items() if k != "method"} for cfg in cfgs)
    cfg = cfgs[-1]
    assert cfg["audit"]["predictor_diagnostic"] is False
    assert cfg["eval"]["n_problems"] == 128 and cfg["eval"]["n"] == cfg["eval"]["k"] == 4
    assert cfg["num_steps"] == cfg["checkpoint_every"] == 50
    assert cfg["predictor"]["warmup_steps"] == 10
    assert [cfg["audit"][k] for k in ["n_problems", "n_prefixes", "n_continuations"]] == [4, 2, 8]
    assert cfg["audit"]["decision_grid"] == [512] and cfg["audit"]["baseline"] is None
    assert cfg["enable_thinking"] is False and cfg["max_new_tokens"] == 8192
    assert cfg["predictor"]["target_space"] == "direct"
    assert cfg["hardware"]["n_gpu"] == 3 and cfg["rollout"]["workers"] == 2
    from grace_gc.backends.rollout_pool import validate_layout
    assert validate_layout(cfg) == 2
    assert cfg["lora"]["rank"] == 8 and cfg["n_start"] == 32 and cfg["n_prompts"] == 8
    assert cfg["cost_control"]["mode"] == "fixed"


def test_followup_explicit_overrides_keep_multiseed_four_card_and_snapshot_support(tmp_path):
    config = tmp_path / "expanded.yaml"
    config.write_text(yaml.safe_dump({"num_steps": 100, "checkpoint_every": 25}))
    root = tmp_path / "suite"
    plan(arguments(root) + ["--config", str(config), "--seeds", "17", "23", "--eval-steps", "25", "50", "75",
                            "--hardware", "configs/hardware/rtx5090_rollout_4.yaml", "--no-audit"])
    jobs = json.loads((root / "suite.json").read_text())["jobs"]
    assert len([j for j in jobs if j["phase"] == "train"]) == 6
    assert len([j for j in jobs if j["phase"] == "eval_snapshot"]) == 18
    assert len([j for j in jobs if j["phase"] == "initial_eval"]) == 1
    cfg = yaml.safe_load((root / "core8_grace-seed23.yaml").read_text())
    assert cfg["hardware"]["n_gpu"] == 4 and cfg["rollout"]["workers"] == 3
    assert cfg["num_steps"] == 100


def test_non_followup_keeps_legacy_seeds_and_single_card_default(tmp_path):
    root = tmp_path / "suite"
    args = arguments(root)
    args.remove("--followup")
    plan(args + ["--arms", "core8_full_pg"])
    jobs = json.loads((root / "suite.json").read_text())["jobs"]
    assert {j["seed"] for j in jobs} == {17, 23, 42}
    assert not any(j["phase"] in {"initial_eval", "eval_snapshot", "audit"} for j in jobs)
    cfg = yaml.safe_load((root / "core8_full_pg-seed17.yaml").read_text())
    assert cfg["hardware"]["n_gpu"] == 1
    assert cfg["num_steps"] == 100


def test_execute_quick_plan_launches_only_initial_and_final_evaluations(tmp_path, monkeypatch):
    import scripts.run_small_core_experiment as runner
    calls = []
    monkeypatch.setattr(runner.subprocess, "run", lambda command, **kwargs:
                        (calls.append(command) or SimpleNamespace(returncode=0)))
    root = tmp_path / "suite"
    plan(arguments(root) + ["--execute"])
    manifest = json.loads((root / "suite.json").read_text())
    assert len(calls) == 9 and manifest["status"] == "complete"
    assert all(j["status"] == "complete" for j in manifest["jobs"])
    evaluations = [c for c in calls if c[1].endswith("/evaluate.py")]
    assert len(evaluations) == 4
    checkpoints = [c[c.index("--checkpoint") + 1] for c in evaluations]
    assert sum(c.endswith("/checkpoints/step_0.npz") for c in checkpoints) == 1
    assert sum(c.endswith("/checkpoint.npz") for c in checkpoints) == 3


@pytest.mark.parametrize("fixed_value,audit_override", [(0., None), (.3, None), (.3, .5), (.3, .3)])
def test_followup_audit_baseline_matches_labels_features_and_manifest(tmp_path, fixed_value, audit_override):
    from grace_gc.audit.run import run_audit, _tiny_actor, _score_grad_vec
    from grace_gc.config import default_config, merge_configs
    from grace_gc.core.layout import collect_lora_layout
    from grace_gc.data.math_data import MathRecord
    from grace_gc.trainer.tiny_engine import make_tiny_engines

    override = {"baseline": {"mode": "fixed", "fixed_value": fixed_value}}
    if audit_override is not None:
        override["audit"] = {"baseline": audit_override}
    config = tmp_path / "override.yaml"
    config.write_text(yaml.safe_dump(override))
    root = tmp_path / "suite"
    plan(arguments(root) + ["--config", str(config), "--arms", "core8_grace", "--seeds", "17"])
    cfg = yaml.safe_load((root / "core8_grace-seed17.yaml").read_text())
    # Exercise the same audit baseline resolution with real CPU autograd, not GPU performance.
    cfg = merge_configs(default_config(), cfg, {"backend": "cpu_tiny", "decision_tokens": 1,
        "max_new_tokens": 3, "hardware": {"n_gpu": 0}, "predictor": {"include_actor_state": False},
        "audit": {"n_problems": 1, "n_prefixes": 1, "n_continuations": 2, "decision_grid": [1],
                  "max_new_tokens": 3, "n_baseline": 0}})
    cfg["lora"] = default_config()["lora"]
    result = run_audit([MathRecord("p", "q1", "1")], cfg, tmp_path / "audit")
    expected = fixed_value if audit_override is None else audit_override
    raw = json.loads((tmp_path / "audit/audit_raw_problems.jsonl").read_text().splitlines()[0])
    assert raw["baseline"] == raw["baseline_feature_value"] == expected
    assert raw["baseline_source"] == ("configured_fixed_baseline" if audit_override is None else
                                       "explicit_audit_baseline")
    assert raw["baseline_samples"] == []  # fixed mode must not pay an independent prescan
    manifest = result["audit_manifest"]
    assert manifest["audit_baseline_override"] == audit_override
    assert manifest["baseline_target_changed_from_configured_fixed"] is (expected != fixed_value)
    stored = json.loads((tmp_path / "audit/audit_bundles.jsonl").read_text().splitlines()[0])
    assert stored["baseline"] == stored["features"][-1] == expected
    actor = _tiny_actor(cfg)
    engines = make_tiny_engines(actor, actor.vocab)
    layout = collect_lora_layout(actor.named_lora_params())
    gradients = np.load(tmp_path / "audit" / stored["trajectory_grads_sidecar"]["path"])
    for gradient, continuation in zip(gradients, stored["continuation_records"]):
        score = _score_grad_vec(engines, layout, continuation["token_ids"], continuation["prompt_len"])
        np.testing.assert_allclose(gradient, (continuation["reward"] - expected) * score, atol=1e-8)


def test_existing_plan_not_overwritten_and_smoke_keeps_short_budget(tmp_path):
    root = tmp_path / "suite"
    args = arguments(root) + ["--smoke", "--seeds", "17", "--no-audit"]
    plan(args)
    original = (root / "suite.json").read_text()
    plan(args)
    assert (root / "suite.json").read_text() == original
    assert len(list(tmp_path.glob("suite*/suite.json"))) == 2
    manifest = json.loads(original)
    assert not any(j["phase"] in {"audit", "eval_snapshot"} for j in manifest["jobs"])
    assert yaml.safe_load((root / "core8_grace-seed17.yaml").read_text())["num_steps"] == 4


@pytest.mark.parametrize("override", [{"format_warmup": {"steps": 1}}, {"init_checkpoint": "trained.npz"}, {"resume": "resume.npz"}])
def test_shared_initial_rejects_nonzero_initialization(tmp_path, override):
    config = tmp_path / "override.yaml"
    config.write_text(yaml.safe_dump(override))
    with pytest.raises(SystemExit):
        plan(arguments(tmp_path / "suite") + ["--config", str(config)])


def test_unsaved_eval_step_is_not_silently_accepted(tmp_path):
    with pytest.raises(SystemExit):
        plan(arguments(tmp_path / "suite") + ["--eval-steps", "13"])


def test_execute_skips_unreached_wall_budget_snapshots_but_runs_final(tmp_path, monkeypatch):
    import scripts.run_small_core_experiment as runner
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(runner.subprocess, "run", run)
    root = tmp_path / "suite"
    plan(arguments(root) + ["--execute", "--no-audit", "--arms", "core8_full_pg", "--seeds", "17",
                            "--config", "configs/experiments/small_core_followup_wall.yaml",
                            "--eval-steps", "50", "100", "150", "--run-wall-seconds", "100"])
    manifest = json.loads((root / "suite.json").read_text())
    assert len(calls) == 3  # train, initial, final; snapshots weren't produced
    assert len([j for j in manifest["jobs"] if j["status"] == "skipped"]) == 3
    assert (root / "comparison_summary.json").is_file()


def bundle(g, f, **overrides):
    g = np.asarray(g, float)
    return {"problem_id": "p", "t": 512, "path_id": "a", "finished": False,
            "true_grad_coords": g.tolist(), "effective_coords": f,
            "basis_gram": {"identity_dim": g.shape[1]}, **overrides}


def test_direct_prediction_noise_correction_and_variance():
    row = prediction_row(bundle([[1, 0], [3, 0]], [2, 0]))
    assert row["zero_prediction_mse"] == 5
    assert row["frozen_prediction_mse"] == 1
    assert row["conditional_noise_trace"] == 2
    assert row["noise_corrected_mean_energy"] == 3
    assert row["noise_corrected_mean_error_sq"] == -1  # do NOT clip uncertainty away
    summary = aggregate([row], .5)
    assert summary["realized_gradient_mse_ratio_vs_zero"] == .2
    assert summary["uniform_ht_added_variance_trace"] == 5
    assert summary["uniform_cv_added_variance_trace"] == 1
    p1 = aggregate([row], 1)
    assert p1["uniform_ht_added_variance_trace"] == p1["uniform_cv_added_variance_trace"] == 0


@pytest.mark.parametrize("p", [.25, .5, 1.])
def test_finished_nonzero_gradient_has_no_added_stopping_variance(p):
    row = prediction_row(bundle([[2, 0]], [0, 0], finished=True))
    summary = aggregate([row], p)
    assert summary["zero_prediction_mse"] == summary["frozen_prediction_mse"] == 4
    assert summary["uniform_ht_added_variance_trace"] == summary["uniform_cv_added_variance_trace"] == 0


def test_missing_predictor_is_not_replaced_by_zero_and_small_samples_are_allowed():
    missing = prediction_row(bundle([[2, 0]], None))
    assert missing["status"] == "unavailable"
    row = prediction_row(bundle([[2, 0]], [0, 0]))
    assert row["conditional_noise_trace"] is None
    assert row["cosine_to_empirical_mean"] is None
    assert aggregate([row], .5)["realized_gradient_mse_ratio_vs_zero"] == 1
    assert aggregate([missing], .5)["frozen_prediction_mse"] is None


def test_direct_report_rejects_pca_and_bad_shapes():
    with pytest.raises(ValueError, match="identity"):
        prediction_row(bundle([[1, 2]], [0, 0], basis_gram=[[1, 0], [0, 1]]))
    with pytest.raises(ValueError, match="dimensions"):
        prediction_row(bundle([[1, 2]], [0]))


def test_report_excludes_finished_prefixes_without_loading_sidecars(tmp_path):
    records = [bundle([[1, 0], [3, 0]], [2, 0]),
               bundle([[2, 0]], [0, 0], finished=True, trajectory_grads_sidecar={"path": "not-needed.npy"})]
    (tmp_path / "audit_bundles.jsonl").write_text("\n".join(map(json.dumps, records)))
    result = analyze(tmp_path)
    assert result["all_prefixes"]["n_prefixes"] == 2
    assert result["eligible_prefixes"]["n_prefixes"] == 1
    assert result["all_prefixes"]["zero_prediction_mse"] == 4.5  # retain finished gradient energy
    assert result["all_prefixes"]["uniform_ht_added_variance_trace"] == 2.5
    assert result["all_prefixes"]["uniform_cv_added_variance_trace"] == .5
    assert result["eligible_prefixes"]["uniform_ht_added_variance_trace"] == 5
    assert result["eligible_prefixes"]["uniform_cv_added_variance_trace"] == 1
    assert result["by_t"][0]["uniform_cv_added_variance_trace"] == 1
    p1 = analyze(tmp_path, p=1.)
    for group in [p1["all_prefixes"], p1["eligible_prefixes"], *p1["by_t"]]:
        assert group["uniform_ht_added_variance_trace"] == group["uniform_cv_added_variance_trace"] == 0


def evaluation(values, step=20, hash_="checkpoint"):
    return {"avg": float(np.mean(values)), "checkpoint_step": step, "checkpoint_sha256": hash_,
            "evaluation_manifest": {"ordered_records_sha256": "same", "sample_seed_start": 17},
            "per_problem": [{"problem_id": str(i), "avg": v} for i, v in enumerate(values)]}


def test_paired_comparison_requires_same_manifest_and_preserves_negative_delta():
    actual, reference = evaluation([0, .5]), evaluation([.5, 1])
    result = paired_delta(actual, reference, bootstrap=100)
    assert result["delta_avg"] == -.5
    assert result["paired_problem_bootstrap_95"] == [-.5, -.5]
    reference["evaluation_manifest"]["sample_seed_start"] = 23
    assert paired_delta(actual, reference)["status"] == "unavailable"
    assert paired_delta(actual, None)["status"] == "unavailable"


def test_summary_checks_checkpoint_clock_hash_and_missing_initial(tmp_path):
    jobs = []
    for arm, values in [("core8_full_pg", [.5, 1]), ("core8_grace", [.5, .5])]:
        train = tmp_path / arm
        train.mkdir()
        (train / "checkpoints.json").write_text(json.dumps({"steps": [{"step": 20, "sha256": "checkpoint",
                                                            "available_command_wall_seconds": 123}]}))
        ev = tmp_path / (arm + "-eval")
        ev.mkdir()
        (ev / "eval_summary.json").write_text(json.dumps(evaluation(values)))
        jobs.extend([{"arm": arm, "seed": 17, "phase": "train", "status": "complete", "run_dir": str(train),
                      "command_wall_seconds": 150},
                     {"arm": arm, "seed": 17, "phase": "eval", "status": "complete", "run_dir": str(ev)}])
    path = tmp_path / "suite.json"
    path.write_text(json.dumps({"status": "complete", "jobs": jobs}))
    result = summarize_suite(path)
    grace = result["curves"][1]
    assert grace["vs_initial"]["status"] == "unavailable"
    assert grace["vs_same_step_full_pg"]["delta_avg"] == -.25
    assert grace["checkpoint_command_seconds"] == 123
    assert grace["training_command_seconds"] == 150
    assert result["final_by_arm"][0]["avg_seed_std"] is None
    (tmp_path / "core8_grace-eval/eval_summary.json").write_text(json.dumps(evaluation([.5, .5], hash_="other")))
    assert summarize_suite(path)["curves"][1]["checkpoint_command_seconds"] is None


def test_health_only_uses_post_warmup_and_requested_steps(tmp_path):
    rows = [{"step": 1, "warmup": True, "n": 32, "n_stopped": 0, "m_shrink": 1},
            {"step": 2, "warmup": False, "n": 32, "n_stopped": 16, "m_shrink": .3,
             "control_variate_used": True, "phases": {"prefix": 2},
             "predictor": {"frozen_new_problems": {"n": 4, "history_complete": True, "residual_to_m0_ratio": .9}}},
            {"step": 3, "warmup": False, "n": 32, "n_stopped": 32, "m_shrink": .9}]
    (tmp_path / "steps.jsonl").write_text("\n".join(map(json.dumps, rows)))
    result = step_health(tmp_path, 2)
    assert result["post_warmup_steps"] == 1
    assert result["stopped_fraction_post_warmup"] == .5
    assert result["m_shrink_mean"] == .3
    assert result["control_variate_steps"] == 1
    assert result["frozen_unseen_problem_diagnostics"] == [
        {"step": 2, "n": 4, "history_complete": True, "residual_to_m0_ratio": .9}]


def test_wall_override_keeps_fixed_n_and_long_step_cap(tmp_path):
    root = tmp_path / "wall"
    plan(arguments(root) + ["--config", "configs/experiments/small_core_followup_wall.yaml",
                            "--arms", "core8_full_pg", "--seeds", "17"])
    cfg = yaml.safe_load((root / "core8_full_pg-seed17.yaml").read_text())
    assert cfg["num_steps"] == 10000 and cfg["run_wall_seconds"] == 7200
    assert cfg["n_start"] == 32 and cfg["cost_control"]["mode"] == "fixed"
