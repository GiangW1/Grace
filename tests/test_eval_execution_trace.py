"""Execution evidence helps diagnose differences; it promises no CUDA determinism."""
from types import SimpleNamespace

from grace_gc.data.math_data import MathRecord


def test_vllm_trace_records_seeds_ids_chunks_and_serial_fallback(monkeypatch):
    from grace_gc.backends import vllm_two_phase as vllm
    from grace_gc.data import tokenize
    from grace_gc.evaluation.generate import generate_answers_vllm
    class Sampling:
        def __init__(self, **kwargs): self.kwargs = kwargs
    class LLM:
        def generate(self, prompts, sampling_params, **kwargs):
            if isinstance(sampling_params, list):
                raise TypeError("test old engine serial fallback")
            seed = sampling_params.kwargs["seed"]
            return [SimpleNamespace(request_id=f"request-{seed}", prompt_token_ids=prompts[0],
                    outputs=[SimpleNamespace(token_ids=[seed]*4, finish_reason="length", stop_reason=None)])]
    monkeypatch.setattr(vllm, "_require_vllm", lambda: (LLM, Sampling))
    monkeypatch.setattr(vllm, "_vllm_prompts", lambda ids: ids)
    monkeypatch.setattr(tokenize, "encode_records_hf", lambda recs, tok, mx, **kw:
                        ([[1, 2]]*len(recs), ["p"]*len(recs), ["1"]*len(recs)))
    monkeypatch.setattr(tokenize, "decode_hf", lambda tok, ids: str(ids[0]))
    monkeypatch.setattr(tokenize, "collect_stop_token_ids", lambda tok: [99])
    trace = []
    items = generate_answers_vllm([MathRecord("p", "q", "1")], None, LLM(), 4, 4, .6, .95, 23,
                                  sample_batch_size=4, request_recorder=trace.append)
    assert items[0].sample_seeds == [23, 24, 25, 26]
    assert [r["request_seed"] for r in trace] == items[0].sample_seeds
    assert [r["request_id"] for r in trace] == [f"request-{s}" for s in range(23, 27)]
    assert all(r["execution_batch_size"] == 1 and r["requested_batch_size"] == 4 for r in trace)
    assert all(r["serial_fallback"] and r["prompt_token_count"] == 2 for r in trace)
    assert len({r["prompt_token_sha256"] for r in trace}) == 1
    assert len({r["response_token_sha256"] for r in trace}) == 4


def test_eval_engine_seed_inherits_run_seed_and_honors_explicit_override():
    from grace_gc.evaluation.generate import eval_engine_config
    cfg = {"seed": 41, "prompt_max_tokens": 1024, "vllm": {"lora_dtype": "bfloat16"}}
    result = eval_engine_config(cfg, 4096)
    assert result["seed"] == 41
    assert result["max_model_len"] >= 5120
    assert "seed" not in cfg["vllm"]
    cfg["vllm"]["seed"] = 7
    assert eval_engine_config(cfg, 4096)["seed"] == 7


def test_gpu_audit_engine_seed_inherits_run_seed(monkeypatch):
    import pytest
    from grace_gc.audit import run
    from grace_gc.backends import hf_actor, verl_trainer
    from grace_gc.data import reward, tokenize
    from grace_gc.trainer import state_io

    monkeypatch.setattr(reward, "require_math_verify", lambda: None)
    monkeypatch.setattr(tokenize, "load_hf_tokenizer", lambda _: object())
    monkeypatch.setattr(verl_trainer, "load_lora_actor", lambda *args: object())
    monkeypatch.setattr(hf_actor, "named_lora_params", lambda _: [])
    monkeypatch.setattr(state_io, "check_snapshot_identity", lambda *args: None)
    monkeypatch.setattr(run, "load_numpy_module_state", lambda *args: None)
    received = []
    class EngineReached(Exception):
        pass
    def build(model, cfg, rank):
        received.append(cfg)
        raise EngineReached
    monkeypatch.setattr(verl_trainer, "build_vllm_engine", build)
    cfg = {"seed": 41, "model_path": "org/model", "checkpoint": "unused", "vllm": {}}
    for override in (None, 7):
        if override is not None:
            cfg["vllm"]["seed"] = override
        with pytest.raises(EngineReached):
            run.generate_bundles_gpu([], 1, 2, 512, 2048, 41, cfg, payload={})
        assert received[-1]["seed"] == (41 if override is None else override)
    assert received[0] is not cfg["vllm"]
