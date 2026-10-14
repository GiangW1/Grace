import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from grace_gc.backends.memory_profile import ActorMemoryProfile, full_length_pg_probe
from grace_gc.config import load_config, merge_configs
from grace_gc.logging_util.run_dir import RunDirectory


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("n_gpu", [3, 4])
def test_5090_capacity_config_preserves_full_response_horizon(n_gpu):
    from grace_gc.backends.rollout_pool import validate_layout
    from grace_gc.backends.verl_trainer import vllm_needed_max_model_len
    cfg = merge_configs(*(load_config(ROOT / p) for p in [
        f"configs/hardware/rtx5090_rollout_{n_gpu}.yaml",
        "configs/experiments/small_core_comparison.yaml",
        "configs/experiments/small_core_r8.yaml",
        "configs/experiments/small_core_memory_smoke.yaml",
    ]))
    assert validate_layout(cfg) == n_gpu - 1
    assert cfg["lora"]["rank"] == 8 and cfg["lora"]["gradient_checkpointing"]
    assert cfg["method"] == "full_pg" and cfg["max_new_tokens"] == 8192
    assert cfg["enable_thinking"] is False
    assert cfg["num_steps"] == cfg["n_start"] == 4
    assert vllm_needed_max_model_len(cfg) == 9280
    assert cfg["vllm"]["enable_chunked_prefill"]
    audit_cfg = merge_configs(cfg, load_config(ROOT / "configs/experiments/small_core_audit.yaml"),
                             load_config(ROOT / "configs/experiments/small_core_audit_smoke.yaml"))
    assert audit_cfg["enable_thinking"] is False
    assert audit_cfg["audit"]["max_new_tokens"] == 8192
    assert validate_layout(audit_cfg) == n_gpu - 1


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("external_cache", [False, True])
@pytest.mark.parametrize("n_gpu", [3, 4])
def test_gpu_audit_uses_separate_pool_and_closes_owned_resources(monkeypatch, fail, external_cache, n_gpu):
    from grace_gc.audit import run
    from grace_gc.backends import gpu_engine, hf_actor, rollout_pool, verl_trainer
    from grace_gc.data import reward, tokenize
    from grace_gc.trainer import state_io

    events = []
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: events.append(("actor", device)))
    monkeypatch.setattr(reward, "require_math_verify", lambda: None)
    monkeypatch.setattr(tokenize, "load_hf_tokenizer", lambda _: object())
    monkeypatch.setattr(verl_trainer, "load_lora_actor", lambda *args: object())
    monkeypatch.setattr(hf_actor, "named_lora_params", lambda _: [("q_proj.lora_core", torch.zeros(2, 2))])
    monkeypatch.setattr(state_io, "check_snapshot_identity", lambda *args: None)
    monkeypatch.setattr(run, "load_numpy_module_state", lambda *args: None)
    monkeypatch.setattr(verl_trainer, "build_vllm_engine", lambda *args: pytest.fail("colocated engine"))
    pool = SimpleNamespace(metadata={"actor_device": "0", "rollout_devices": [str(i) for i in range(1, n_gpu)]},
                           close=lambda: events.append("closed"))
    def launch(model, cfg, rank, gpu_count):
        assert cfg["gpu_memory_utilization"] == .75  # Not capped for colocation.
        assert cfg["max_model_len"] == 9280
        assert rank == 8 and gpu_count == n_gpu
        return pool
    monkeypatch.setattr(rollout_pool.RolloutPool, "launch", launch)
    def build_engines(actor, llm, *args):
        assert llm is pool
        def sync():
            events.append("sync")
            if fail:
                raise RuntimeError("sync failed")
        return object(), {"sync": sync}
    monkeypatch.setattr(gpu_engine, "make_gpu_engines", build_engines)
    cfg = {"model_path": "org/model", "checkpoint": "unused", "hardware": {"n_gpu": n_gpu},
           "rollout": {"workers": n_gpu - 1}, "lora": {"rank": 8}, "audit": {"max_new_tokens": 8192},
           "vllm": {"gpu_memory_utilization": .75}}
    cache = {} if external_cache else None
    def call():
        return run.generate_bundles_gpu([], 1, 2, 512, 8192, 17, cfg, payload={}, cache=cache)
    if fail:
        with pytest.raises(RuntimeError, match="sync failed"):
            call()
    else:
        assert call() == []
    if external_cache:
        assert events == [("actor", 0), "sync"]
        run.close_gpu_audit_cache(cache)
        run.close_gpu_audit_cache(cache)  # Idempotent cleanup.
    assert events == [("actor", 0), "sync", "closed"]


def test_full_length_probe_uses_real_pg_path_without_state_changes(monkeypatch):
    from grace_gc.backends import hf_actor
    from tests.test_small_core import small_model
    actor = small_model()
    params = list(actor.parameters())
    before = [p.detach().clone() for p in params]
    for p in params:
        if p.requires_grad:
            p.grad = torch.ones_like(p)
    saved_grads = [p.grad for p in params]
    rng = torch.get_rng_state().clone()
    seen = []
    def logprob(model, ids, prompt_len, pad, eos_id):
        seen.append((len(ids), prompt_len, eos_id))
        assert all(t not in [0, 99] for t in ids)
        return model.q_proj(torch.randn(1, 7)).sum() + model.v_proj(torch.randn(1, 7)).sum()
    monkeypatch.setattr(hf_actor, "logprob_one", logprob)
    tokenizer = SimpleNamespace(pad_token_id=0, all_special_ids=[0, 99],
                                encode=lambda *a, **k: [0, 1, 2, 99])
    result = full_length_pg_probe(actor, tokenizer, {"method": "full_pg", "prompt_max_tokens": 1024,
                                                    "max_new_tokens": 8192})
    assert seen == [(9216, 1024, None)]
    assert result["all_finite"] and result["synthetic"] and not result["optimizer_updated"]
    assert torch.equal(rng, torch.get_rng_state())
    for p, original, grad in zip(params, before, saved_grads):
        torch.testing.assert_close(p, original, rtol=0, atol=0)
        assert p.grad is grad


@pytest.mark.parametrize("oom", [False, True])
def test_memory_profile_persists_peaks_lengths_and_oom(monkeypatch, tmp_path, oom):
    import grace_gc.backends.memory_profile as module
    monkeypatch.setattr(module.shutil, "which", lambda _: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "synthetic GPU")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: SimpleNamespace(total_memory=32 * 2**30))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda _: None)
    for fn in ("max_memory_allocated", "max_memory_reserved", "memory_allocated", "memory_reserved"):
        monkeypatch.setattr(torch.cuda, fn, lambda _: 12 * 2**30)
    profile = ActorMemoryProfile(RunDirectory(tmp_path))
    def action():
        with profile.phase("algorithm_step", step=1):
            profile.observe_sequence([1] * 9216, 1024)
            if oom:
                raise torch.cuda.OutOfMemoryError("synthetic failure")
    if oom:
        with pytest.raises(torch.cuda.OutOfMemoryError):
            action()
    else:
        action()
    profile.close("failed" if oom else "completed")
    report = json.loads((tmp_path / "memory_profile.json").read_text())
    row = report["phases"][1]
    assert row["status"] == ("out_of_memory" if oom else "completed")
    assert row["peak_allocated_gib"] == 12
    assert row["max_sequence_tokens"] == 9216 and row["max_response_tokens"] == 8192
    assert report["board_monitor"] == "unavailable"
