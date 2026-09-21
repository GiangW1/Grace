import json
import shutil

import pytest

from scripts.summarize_cost_quality import collect_curve, select_at_budget, first_observed_target, evaluation_steps


def fixture(tmp_path):
    train = tmp_path / "seed-17/grace/train"
    train.mkdir(parents=True)
    rows, evaluations = [], []
    for step, cost, score in [(0, 5., .50), (10, 25., .55), (20, 45., .90)]:
        rows.append({"step": step, "sha256": f"hash-{step}", "path": f"checkpoints/step_{step}.npz",
                     "available_command_wall_seconds": cost, "available_global_wall_seconds": cost - 2})
        folder = train.parent / f"eval-{step}"
        folder.mkdir()
        summary = {"checkpoint": str(train / rows[-1]["path"]), "checkpoint_step": step,
                   "checkpoint_sha256": rows[-1]["sha256"], "avg": score, "pass_at_k": 1.,
                   "requested_k": 4,
                   "evaluation_manifest": {"ordered_records_sha256": "same-questions", "reward_protocol_version": 2,
                       "samples_per_problem": 4, "temperature": .6, "top_p": .95, "max_new_tokens": 2048,
                       "sample_batch_size": 1, "sample_seed_start": 17}}
        (folder / "eval_summary.json").write_text(json.dumps(summary))
        evaluations.append(folder)
    (train / "checkpoints.json").write_text(json.dumps({"steps": rows}))
    return train, evaluations


def test_budget_uses_latest_available_checkpoint_not_final_or_best_score(tmp_path):
    train, evaluations = fixture(tmp_path)
    points = collect_curve(train, evaluations)
    chosen = select_at_budget(points, 30.)
    assert chosen["step"] == 10 and chosen["avg"] == .55
    assert chosen["wall_seconds"] == 25.
    assert select_at_budget(points, 4.) is None
    assert first_observed_target(points, .8)["wall_seconds"] == 45.


def test_missing_legacy_time_and_mismatched_checkpoint_remain_unusable(tmp_path):
    train, evaluations = fixture(tmp_path)
    index = json.loads((train / "checkpoints.json").read_text())
    index["steps"][0].pop("available_command_wall_seconds")
    index["steps"][1]["sha256"] = "another-checkpoint"
    (train / "checkpoints.json").write_text(json.dumps(index))
    points = collect_curve(train, evaluations)
    assert points[0]["avg"] == .5 and points[0]["wall_seconds"] is None
    assert points[1]["issues"] == ["checkpoint_hash_missing_or_mismatched"]
    assert select_at_budget(points, 30.) is None


def test_protocol_change_is_not_a_single_quality_curve(tmp_path):
    train, evaluations = fixture(tmp_path)
    path = evaluations[-1] / "eval_summary.json"
    summary = json.loads(path.read_text())
    summary["evaluation_manifest"]["ordered_records_sha256"] = "other-questions"
    path.write_text(json.dumps(summary))
    points = collect_curve(train, evaluations)
    assert "evaluation_protocol_changed_within_curve" in points[-1]["issues"]
    assert first_observed_target(points, .8) is None


def test_training_entry_clock_is_explicit_not_a_fallback(tmp_path):
    train, evaluations = fixture(tmp_path)
    points = collect_curve(train, evaluations, clock="training_entry")
    assert points[0]["wall_seconds"] == 3.
    assert points[0]["clock"] == "training_entry"


def test_recipe_hash_ignores_label_but_preserves_real_training_differences():
    import copy
    from scripts.summarize_cost_quality import training_recipe_hash

    config = {"seed": 17, "experiment_variant": "label-a", "vllm": {"seed": 17, "dtype": "bfloat16"},
              "optim": {"lr": .001}, "checkpoint_every": 5}
    original = copy.deepcopy(config)
    same = {**config, "experiment_variant": "label-b"}
    same.pop("seed")
    assert training_recipe_hash(config) == training_recipe_hash(same)
    assert config == original
    for changed in ({"optim": {"lr": .002}}, {"checkpoint_every": 10}, {"vllm": {"dtype": "float32"}}):
        assert training_recipe_hash(config) != training_recipe_hash({**config, **changed})


def test_evaluation_plan_adds_budget_snapshot_without_using_quality(tmp_path):
    train, _ = fixture(tmp_path)
    assert evaluation_steps(train, "0 20 40", budget=30.) == [0, 10, 20, 40]
    assert evaluation_steps(train, "all") == [0, 10, 20]


def test_repeated_checkpoint_evals_cannot_select_lucky_target_crossing(tmp_path):
    train, evaluations = fixture(tmp_path)
    repeat = train.parent / "eval-10-repeat"
    shutil.copytree(evaluations[1], repeat)
    path = repeat / "eval_summary.json"
    row = json.loads(path.read_text())
    row["avg"] = .99
    path.write_text(json.dumps(row))
    points = collect_curve(train, evaluations + [repeat])
    assert all("repeated_checkpoint_evaluation" in p["issues"] for p in points if p["step"] == 10)
    assert first_observed_target(points, .95) is None
    assert select_at_budget(points, 30.)["step"] == 0


def test_real_cpu_checkpoint_to_evaluation_cost_curve(tmp_path, monkeypatch):
    import time
    from grace_gc.config import default_config, merge_configs
    from grace_gc.trainer import loop
    from grace_gc.evaluation import generate
    from grace_gc.data.math_data import MathRecord

    monkeypatch.setattr(loop, "collect_environment", lambda *a, **kw: {})
    monkeypatch.setattr(generate, "collect_environment", lambda *a, **kw: {})
    monkeypatch.setenv("GRACE_COMMAND_START_MONOTONIC", str(time.monotonic()))
    cfg = merge_configs(default_config(), {"backend": "cpu_tiny", "method": "full_pg", "num_steps": 1,
        "n_start": 2, "n_prompts": 1, "decision_tokens": 2, "max_new_tokens": 4, "prompt_max_tokens": 8,
        "format_warmup": {"steps": 0}, "baseline": {"prescan": 1}, "predictor": {"k": 2},
        "eval": {"n": 2, "k": 2, "max_new_tokens": 4}})
    train = tmp_path / "train"
    loop.run_training(cfg, train)
    ev = tmp_path / "eval"
    generate.run_eval([MathRecord("cpu-synthetic", "What is 1+1?", "2")],
                      {**cfg, "checkpoint": str(train / "checkpoint.npz")}, ev)
    point = collect_curve(train, [ev])[0]
    assert point["issues"] == []
    assert point["wall_seconds"] > 0 and point["step"] == 1
    assert point["main_starts"] == 2


def test_seed_pairing_keeps_missing_runs_and_actual_full_job_cost(tmp_path):
    import yaml
    from scripts.summarize_cost_quality import summarize

    train, evaluations = fixture(tmp_path)
    seed = train.parent.parent
    (tmp_path/"experiment.json").write_text(json.dumps({"requested_seeds": [17, 23, 41],
        "requested_methods": ["full_pg", "grace"]}))
    other = seed/"full_pg"
    shutil.copytree(train.parent, other)
    for ev in other.glob("eval-*"):
        path = ev/"eval_summary.json"
        row = json.loads(path.read_text())
        row["checkpoint"] = str(other/"train/checkpoints"/f"step_{row['checkpoint_step']}.npz")
        row["avg"] -= .1
        path.write_text(json.dumps(row))
    manifest = {}
    for method in ("grace", "full_pg"):
        path = seed/method/"train"
        (path/"config.yaml").write_text(yaml.safe_dump({"seed": 17}))
        (path/"initial_actor.json").write_text(json.dumps({"actor_sha256": "shared"}))
        (path/"compute_ledger.json").write_text(json.dumps({"hardware": "synthetic-cpu"}))
        manifest[method] = {"train": f"{method}/train", **{f"eval-{s}": f"{method}/eval-{s}" for s in (0, 10, 20)}}
    (seed/"stages.json").write_text(json.dumps(manifest))
    second = tmp_path/"seed-23"
    second.mkdir()
    (second/"stages.json").write_text(json.dumps({"grace": {"train": "grace/train"}}))
    commands = [{"stage": "seed-17/shared-init", "wall_seconds": 7.},
                {"stage": "seed-17/grace/train", "wall_seconds": 50.}]
    (tmp_path/"command_timing.jsonl").write_text("\n".join(map(json.dumps, commands)))
    result = summarize(tmp_path, [30.])
    pair = result["paired_at_budget"][0]
    assert pair["mean_delta"] == pytest.approx(.1) and pair["n_seeds"] == 1
    assert len(pair["pairs"]) == 3 and all(p["issues"] for p in pair["pairs"][1:])
    assert result["missing_seed_reports"] == ["seed-41"]
    assert result["missing_methods"]["seed-23"] == ["full_pg"]
    method = result["seeds"]["seed-17"]["grace"]
    assert method["actual_command_wall_seconds"] == 50.
    assert method["current_chain_initialization_plus_training_seconds"] == 57.
    assert method["at_budget"]["30.0"]["wall_seconds"] == 25.
    assert result["method_at_budget"][0]["avg"]["seed_sd"] is None


@pytest.mark.parametrize("change", ["questions", "hardware", "recipe", "copied_seed", "none"])
def test_seed_aggregation_does_not_mix_protocols_or_duplicate_seed(tmp_path, change):
    import yaml
    from scripts.summarize_cost_quality import summarize

    test_seed_pairing_keeps_missing_runs_and_actual_full_job_cost(tmp_path)
    seed = tmp_path/"seed-23"
    shutil.copytree(tmp_path/"seed-17", seed, dirs_exist_ok=True)
    for method in ("grace", "full_pg"):
        train = seed/method/"train"
        cfg = {"seed": 17 if change == "copied_seed" else 23}
        if change == "recipe":
            cfg["optim"] = {"lr": .005}
        (train/"config.yaml").write_text(yaml.safe_dump(cfg))
        if change == "hardware":
            (train/"compute_ledger.json").write_text(json.dumps({"hardware": "another-device"}))
        for ev in (seed/method).glob("eval-*"):
            path = ev/"eval_summary.json"
            row = json.loads(path.read_text())
            row["checkpoint"] = str(train/"checkpoints"/f"step_{row['checkpoint_step']}.npz")
            row["evaluation_manifest"]["sample_seed_start"] = 23
            if change == "questions":
                row["evaluation_manifest"]["ordered_records_sha256"] = "other-problems"
            path.write_text(json.dumps(row))
    result = summarize(tmp_path, [30.])
    pair = result["paired_at_budget"][0]
    methods = result["method_at_budget"]
    if change in {"questions", "hardware", "recipe"}:
        assert pair["n_eligible_pairs"] == 2 and pair["mean_delta"] is None
        assert pair["aggregation_issues"] == ["cross_seed_protocol_mismatch"]
        assert all(m["avg"]["mean"] is None and m["aggregation_issues"] for m in methods)
    else:
        expected = 1 if change == "copied_seed" else 2
        assert pair["n_seeds"] == expected
        assert all(m["avg"]["n_seeds"] == expected for m in methods)
