from pathlib import Path

import pytest


def test_format_warmup_tiny_moves_lora_b(tmp_path: Path):
    pytest.importorskip("torch")
    import torch

    from grace_gc.data.math_data import MathRecord
    from grace_gc.trainer.cpu_tiny import TinyLoRAActor
    from grace_gc.trainer.format_warmup import run_format_warmup_tiny
    from grace_gc.trainer.loop import build_run_config, run_training

    actor = TinyLoRAActor()
    before = [p.detach().clone() for name, p in actor.named_lora_params() if "lora_B" in name]
    recs = [MathRecord("1", "1+1", "2"), MathRecord("2", "2+2", "4")]
    info = run_format_warmup_tiny(
        actor,
        recs,
        actor.vocab,
        {"format_warmup": {"steps": 4, "batch_size": 2, "lr": 0.2}, "prompt_max_tokens": 16},
    )
    assert info["steps"] == 4
    assert info["skipped"] is False
    after = [p.detach() for name, p in actor.named_lora_params() if "lora_B" in name]
    moved = any(not torch.equal(a, b) for a, b in zip(before, after))
    assert moved

    cfg = build_run_config(
        ["configs/experiments/minimal.yaml"],
        {
            "seed": 3,
            "backend": "cpu_tiny",
            "n_start": 4,
            "n_prompts": 2,
            "decision_tokens": 2,
            "max_new_tokens": 3,
            "num_steps": 1,
            "format_warmup": {"steps": 2, "batch_size": 2, "lr": 0.1},
        },
    )
    out = run_training(cfg, tmp_path / "fw")
    assert out["run_status"] == "complete"
    assert (tmp_path / "fw" / "format_warmup.json").is_file()
