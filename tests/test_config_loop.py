import json
from pathlib import Path

import pytest

from grace_gc.config import default_config, merge_configs, require_training_leaves_warmup, validate_config
from grace_gc.logging_util.ledger import ComputeLedger
from grace_gc.trainer.checkpoint import load_checkpoint, required_keys, save_checkpoint
from grace_gc.trainer.loop import build_run_config, run_training
from grace_gc.versions import collect_environment, collect_versions


def test_hf_model_id_is_allowed():
    cfg = default_config()
    cfg["model_path"] = "Qwen/Qwen3-4B-Base"
    assert validate_config(cfg)["model_path"] == "Qwen/Qwen3-4B-Base"
    assert cfg["predictor"]["constant_cost"] is True


def test_validate_rejects_decision_beyond_budget():
    cfg = default_config()
    cfg["decision_tokens"] = 8
    cfg["max_new_tokens"] = 4
    cfg["predictor"]["warmup_steps"] = 0
    with pytest.raises(ValueError, match="decision_tokens"):
        validate_config(cfg)


def test_train_rejects_all_warmup_grace():
    cfg = default_config()
    cfg["num_steps"] = 1
    cfg["predictor"]["warmup_steps"] = 20
    with pytest.raises(ValueError, match="warmup"):
        require_training_leaves_warmup(cfg)


def test_validate_rejects_bad_pmin():
    cfg = default_config()
    cfg["allocation"]["p_min"] = 0.0
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_merge_and_missing_optional_versions(tmp_path: Path):
    merged = merge_configs(default_config(), {"method": "uniform_ht"})
    assert merged["method"] == "uniform_ht"
    info = collect_versions()
    assert "missing" in info
    assert "hostname" in info
    data = tmp_path / "math.jsonl"
    data.write_text('{"problem_id":"1","prompt":"1","answer":"2"}\n', encoding="utf-8")
    env = collect_environment({"data_path": str(data), "model_path": "Qwen/Qwen3-4B-Base"})
    assert env["data"]["sha256"]
    assert env["model"]["note"] == "not_a_local_path"


def test_checkpoint_roundtrip(tmp_path: Path):
    path = tmp_path / "ckpt.npz"
    payload = {k: 1 for k in required_keys()}
    save_checkpoint(path, payload)
    loaded = load_checkpoint(path)
    assert set(loaded) >= set(required_keys())


def test_cpu_train_script_path(tmp_path: Path):
    cfg = build_run_config(
        ["configs/experiments/minimal.yaml"],
        {
            "seed": 17,
            "backend": "cpu_tiny",
            "n_start": 4,
            "n_prompts": 2,
            "decision_tokens": 3,
            "max_new_tokens": 4,
            "num_steps": 2,
        },
    )
    out = run_training(cfg, tmp_path / "run")
    assert out["run_status"] == "complete"
    root = tmp_path / "run"
    assert (root / "config.yaml").is_file()
    assert (root / "run.log").is_file()
    assert (root / "environment.json").is_file()
    assert (root / "steps.jsonl").is_file()
    assert (root / "compute_ledger.jsonl").is_file()
    assert (root / "checkpoints.json").is_file()
    assert (root / "health.json").is_file()
    assert (root / "run_meta.json").is_file()
    assert (root / "data_splits.json").is_file()
    health = json.loads((root / "health.json").read_text(encoding="utf-8"))
    meta = json.loads((root / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["kind"] == "train"
    assert "T" in meta["started"]
    assert out["started"]
    assert out["finished"]
    assert out["run_dir"]
    assert health["written_at"]
    assert "all_reward_zero" in health
    env = json.loads((root / "environment.json").read_text(encoding="utf-8"))
    assert "run_knobs" in env
    assert (root / "checkpoints" / "step_1.npz").is_file()
    assert (root / "checkpoints" / "step_2.npz").is_file()
    traj = [json.loads(line) for line in (root / "trajectories.jsonl").read_text(encoding="utf-8").splitlines() if line]
    steps = [json.loads(line) for line in (root / "steps.jsonl").read_text(encoding="utf-8").splitlines() if line]
    assert len(steps) == 2
    assert {row["step"] for row in steps} == {1, 2}
    assert traj
    row = traj[0]
    for key in (
        "p_continue",
        "z_continue",
        "r_hat_full",
        "c_hat_remaining",
        "prediction_coords",
        "finish_reason",
        "prefix_tokens",
        "response_tokens",
        "prompt_token_ids",
        "global_starts_denominator",
        "rng_counters",
        "prompt_truncated",
    ):
        assert key in row
    assert "lora_A_norm_sq" in health
    assert "mean_baseline_b" in health
    assert "mean_response_tokens" in health
    assert "mean_suffix_tokens" in health
    assert "n_prefix_finished" in health
    assert "n_eligible" in health
    assert "n_continued" in health
    assert "n_short_response" in health
    assert "n_baseline_zero" in health
    assert "baseline_collapsed_with_zero_reward" in health
    assert "lora_A_grad_zero_expected" in health
    assert "n_lora_A_grad_missing" in health
    assert "steps_after_warmup" in health
    assert "allocating_with_untrained_predictor" in health
    assert "resources" in health
    assert "pid" in health["resources"]
    assert "hostname" in env
    assert "process" in env
    assert env["started"] == meta["started"]
    assert "thinking_closed" in row
    ledger = json.loads((root / "compute_ledger.json").read_text(encoding="utf-8"))
    assert any(item["name"] == "train_step" for item in ledger["rows"])
    assert any(str(item["name"]).startswith("phase_") for item in ledger["rows"])


def test_second_train_keeps_first_run_dir(tmp_path: Path):
    from grace_gc.logging_util.run_dir import default_run_dir, resolve_run_dir

    cfg = build_run_config(
        ["configs/experiments/minimal.yaml"],
        {
            "seed": 17,
            "backend": "cpu_tiny",
            "n_start": 4,
            "n_prompts": 2,
            "decision_tokens": 3,
            "max_new_tokens": 4,
            "num_steps": 2,
        },
    )
    first = run_training(cfg, tmp_path / "run")
    second = run_training(cfg, tmp_path / "run")
    assert Path(first["run_dir"]).resolve() == (tmp_path / "run").resolve()
    assert Path(second["run_dir"]).name.startswith("run-")
    assert Path(second["run_dir"]).resolve() != Path(first["run_dir"]).resolve()
    assert (tmp_path / "run" / "summary.json").is_file()
    assert Path(second["run_dir"]).joinpath("summary.json").is_file()
    empty = tmp_path / "fresh"
    assert resolve_run_dir(empty) == empty
    name = default_run_dir("train").name
    assert name.startswith("train-")
    assert len(name) > len("train-")


def test_describe_path_records_download_meta(tmp_path: Path):
    from grace_gc.versions import describe_path, resource_snapshot

    model_dir = tmp_path / "Qwen3-4B-Base"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type":"qwen3","vocab_size":16}', encoding="utf-8")
    (model_dir / "download_meta.json").write_text(
        '{"repo_id":"Qwen/Qwen3-4B-Base","sha":"abc123"}',
        encoding="utf-8",
    )
    info = describe_path(str(model_dir))
    assert info["download_meta"]["sha"] == "abc123"
    assert info["config"]["model_type"] == "qwen3"
    snap = resource_snapshot()
    assert snap["pid"]


def test_logprob_probe_unavailable_without_engines():
    from grace_gc.backends.logprob_probe import compare_hf_vllm_logprob, probe_first_completed

    out = compare_hf_vllm_logprob(None, None, [1, 2, 3, 4], 1, 0)
    assert out["status"] == "unavailable"
    assert probe_first_completed(None, None, [], pad_id=0)["status"] == "no_completed_sequence"


def test_ledger_separates_hardware():
    a100 = ComputeLedger(n_gpu=4, hardware="a100")
    a100.add("train", 10.0, cpu_s=2.0)
    rtx = ComputeLedger(n_gpu=8, hardware="rtx5090")
    rtx.add("train", 10.0)
    assert a100.summary()["hardware"] != rtx.summary()["hardware"]
    assert a100.summary()["gpu_reserved_seconds"] == 40.0
