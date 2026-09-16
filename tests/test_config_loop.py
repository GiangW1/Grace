from pathlib import Path

import pytest

from grace_gc.config import default_config, merge_configs, require_training_leaves_warmup, validate_config
from grace_gc.logging_util.ledger import ComputeLedger
from grace_gc.trainer.checkpoint import load_checkpoint, required_keys, save_checkpoint
from grace_gc.trainer.loop import build_run_config, run_training
from grace_gc.versions import collect_versions


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


def test_merge_and_missing_optional_versions():
    merged = merge_configs(default_config(), {"method": "uniform_ht"})
    assert merged["method"] == "uniform_ht"
    info = collect_versions()
    assert "missing" in info


def test_checkpoint_roundtrip(tmp_path: Path):
    path = tmp_path / "ckpt.npz"
    payload = {k: 1 for k in required_keys()}
    save_checkpoint(path, payload)
    loaded = load_checkpoint(path)
    assert set(loaded) >= set(required_keys())


def test_cpu_train_script_path(tmp_path: Path):
    cfg = build_run_config(
        ["configs/experiments/minimal.yaml"],
        {"seed": 17, "backend": "cpu_tiny", "n_start": 4, "decision_tokens": 3, "max_new_tokens": 4},
    )
    out = run_training(cfg, tmp_path / "run")
    assert out["run_status"] == "complete"
    assert (tmp_path / "run" / "config.yaml").is_file()
    assert (tmp_path / "run" / "trajectories.jsonl").is_file()
    assert (tmp_path / "run" / "compute_ledger.json").is_file()


def test_ledger_separates_hardware():
    a100 = ComputeLedger(n_gpu=4, hardware="a100")
    a100.add("train", 10.0, cpu_s=2.0)
    rtx = ComputeLedger(n_gpu=8, hardware="rtx5090")
    rtx.add("train", 10.0)
    assert a100.summary()["hardware"] != rtx.summary()["hardware"]
    assert a100.summary()["gpu_reserved_seconds"] == 40.0
