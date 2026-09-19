"""Generation observations include failed attempts without changing sampling."""
from types import SimpleNamespace
import json

import numpy as np
import pytest

from grace_gc.backends import gpu_engine, vllm_two_phase
from grace_gc.core.rng import IsolatedRNG


@pytest.fixture
def fake_vllm(monkeypatch):
    class SamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(vllm_two_phase, "_require_vllm", lambda: (object, SamplingParams))
    monkeypatch.setattr(vllm_two_phase, "_vllm_prompts", lambda ids: ids)
    clock = iter(range(100))
    monkeypatch.setattr(vllm_two_phase, "perf_counter", lambda: float(next(clock)), raising=False)


def output(prompt, cached=None):
    result = SimpleNamespace(prompt_token_ids=prompt, outputs=[SimpleNamespace(
        token_ids=[7], finish_reason="length", stop_reason=None,
        logprobs=[{7: SimpleNamespace(logprob=-0.25)}],
    )])
    if cached is not None:
        result.num_cached_tokens = cached
    return result


def generate(llm, execution=None):
    kwargs = {} if execution is None else {"execution": execution}
    return vllm_two_phase.generate_phase(
        llm, [[1, 2], [3, 4]], 1, 1., 99, IsolatedRNG.create(17), "token",
        **kwargs,
    )


@pytest.mark.parametrize("message", [
    "CUDA kernel received an invalid argument",
    "sampling_params list length must match prompts",
    "sampling_params unsupported processor type",
])
def test_unrelated_type_error_is_not_retried(fake_vllm, message):
    calls = []
    error = TypeError(message)

    class LLM:
        def generate(self, prompts, sampling_params, **kwargs):
            calls.append(prompts)
            if isinstance(sampling_params, list):
                raise error
            return [output(prompts[0])]

    with pytest.raises(TypeError) as caught:
        generate(LLM())
    assert caught.value is error
    assert len(calls) == 1


@pytest.mark.parametrize("reason", [
    "sampling_params must be a SamplingParams instance, not list",
    "Argument 'sampling_params' has incorrect type (expected SamplingParams, got list)",
])
def test_explicit_sampling_list_rejection_keeps_seeds_and_failed_cost(fake_vllm, reason):
    calls, execution = [], {}

    class LLM:
        def generate(self, prompts, sampling_params, **kwargs):
            calls.append((prompts, sampling_params))
            if isinstance(sampling_params, list):
                raise TypeError(reason)
            return [output(prompts[0], cached=1)]

    phase = generate(LLM(), execution)
    assert phase.num_cached_tokens == [1, 1]
    assert execution["num_generate_calls"] == 3
    assert execution["successful_batch_sizes"] == [1, 1]
    assert execution["failed_generate_batches"] == 1
    assert execution["serial_fallback_reason"] == reason
    assert execution["wall_seconds"] == 3.
    assert [params.kwargs["seed"] for _, params in calls[1:]] == phase.sampling["request_seeds"]
    assert [params.kwargs["seed"] for params in calls[0][1]] == phase.sampling["request_seeds"]


def test_batched_generation_reports_cache_missingness_and_actual_calls(fake_vllm):
    class LLM:
        def generate(self, prompts, sampling_params, **kwargs):
            return [output(prompts[0], cached=0), output(prompts[1])]

    phase = generate(LLM())
    assert phase.num_cached_tokens == [0, None]
    assert phase.execution["requested_batch_size"] == 2
    assert phase.execution["successful_batch_sizes"] == [2]
    assert phase.execution["num_generate_calls"] == 1
    assert phase.execution["wall_seconds"] == 1.
    assert phase.token_logprobs == [[-0.25], [-0.25]]


def test_serial_failure_keeps_all_prior_attempt_costs(fake_vllm):
    execution = {}
    error = RuntimeError("worker stopped")

    class LLM:
        def generate(self, prompts, sampling_params, **kwargs):
            if isinstance(sampling_params, list):
                raise TypeError("sampling_params must be a SamplingParams instance, not list")
            if prompts[0] == [3, 4]:
                raise error
            return [output(prompts[0])]

    with pytest.raises(RuntimeError) as caught:
        generate(LLM(), execution)
    assert caught.value is error
    assert execution["successful_generate_batches"] == 1
    assert execution["failed_generate_batches"] == 2
    assert execution["wall_seconds"] == 3.


def test_gpu_rollout_metadata_preserves_global_indices_and_tuple_return(fake_vllm, tmp_path):
    class LLM:
        def generate(self, prompts, sampling_params, **kwargs):
            return [output(prompt, cached=len(prompt)-1) for prompt in prompts]

    tok = SimpleNamespace(pad_token_id=0, eos_token_id=99)
    engines, _extra = gpu_engine.make_gpu_engines(object(), LLM(), tok, {}, tmp_path)
    prefixes, finished = engines.generate_prefix([[1, 2], [3, 4]], 1, IsolatedRNG.create(17), "token")
    assert finished.tolist() == [False, False]
    engines.continue_selected(prefixes, np.array([False, True]), 1, IsolatedRNG.create(18))
    engines.continue_selected(prefixes, np.array([True, False]), 1, IsolatedRNG.create(19))
    roll = engines.last_rollout
    assert roll["prefix_num_cached_tokens"] == [1, 1]
    assert roll["continue_num_cached_tokens"] == {1: 2, 0: 2}
    assert roll["prefix_execution"]["successful_batch_sizes"] == [2]
    assert [item["request_indices"] for item in roll["continue_execution"]] == [[1], [0]]


def test_gpu_failed_generate_preserves_execution_metadata(fake_vllm, tmp_path):
    class LLM:
        def generate(self, *args, **kwargs):
            raise TypeError("bad CUDA kernel argument")

    tok = SimpleNamespace(pad_token_id=0, eos_token_id=99)
    engines, _extra = gpu_engine.make_gpu_engines(object(), LLM(), tok, {}, tmp_path)
    with pytest.raises(TypeError, match="CUDA"):
        engines.generate_prefix([[1, 2]], 1, IsolatedRNG.create(17), "token")
    assert engines.last_rollout["prefix_execution"]["failed_generate_batches"] == 1
    assert engines.last_rollout["prefix_execution"]["wall_seconds"] == 1.


def test_failed_continuation_keeps_prefix_and_failed_attempt(fake_vllm, tmp_path):
    class LLM:
        def generate(self, prompts, sampling_params, **kwargs):
            if len(prompts[0]) > 2:
                raise RuntimeError("suffix worker failed")
            return [output(prompt, cached=1) for prompt in prompts]

    tok = SimpleNamespace(pad_token_id=0, eos_token_id=99)
    engines, _extra = gpu_engine.make_gpu_engines(object(), LLM(), tok, {}, tmp_path)
    prefixes, _finished = engines.generate_prefix([[1, 2]], 1, IsolatedRNG.create(17), "token")
    with pytest.raises(RuntimeError, match="suffix"):
        engines.continue_selected(prefixes, np.array([True]), 1, IsolatedRNG.create(18))
    assert engines.last_rollout["prefix_num_cached_tokens"] == [1]
    detail = engines.last_rollout["continue_execution"][0]
    assert detail["request_indices"] == [0]
    assert detail["failed_generate_batches"] == 1
    assert detail["wall_seconds"] == 1.


def test_output_validation_failure_keeps_completed_generate_cost(fake_vllm):
    class LLM:
        def generate(self, prompts, sampling_params, **kwargs):
            return []

    execution = {}
    with pytest.raises(ValueError, match="0 outputs for 2 prompts"):
        generate(LLM(), execution)
    assert execution["successful_generate_batches"] == 1
    assert execution["successful_batch_sizes"] == [2]
    assert execution["wall_seconds"] == 1.


def test_empty_continuation_records_zero_engine_calls(fake_vllm):
    execution = {}
    result = vllm_two_phase.continue_selected(
        object(), [[1, 99], [2, 7]], np.array([True, False]), 4, 1., 99,
        IsolatedRNG.create(17), execution=execution,
    )
    assert result == [[1, 99], None]
    assert execution["num_generate_calls"] == 0
    assert execution["wall_seconds"] == 0.


@pytest.mark.parametrize("fail_load", [False, True])
def test_sync_metadata_includes_attempted_stage_and_preserves_order(monkeypatch, tmp_path, fail_load):
    calls = []
    clock = iter(range(100))
    monkeypatch.setattr(gpu_engine, "perf_counter", lambda: float(next(clock)), raising=False)
    monkeypatch.setattr(gpu_engine, "save_lora_adapter", lambda *args: calls.append("save"))
    monkeypatch.setattr(gpu_engine, "make_lora_request", lambda *args: "adapter")

    def load(*args, **kwargs):
        calls.append("load")
        if fail_load:
            raise RuntimeError("load failed")

    monkeypatch.setattr(gpu_engine, "apply_lora_request", load)
    monkeypatch.setattr(gpu_engine, "reset_vllm_prefix_cache", lambda *args: calls.append("reset"))
    extra = {"lora_id": 1, "llm": object()}
    if fail_load:
        with pytest.raises(RuntimeError, match="load failed"):
            gpu_engine._sync(object(), tmp_path, extra)
    else:
        assert gpu_engine._sync(object(), tmp_path, extra) == "adapter"
    assert calls == (["save", "load"] if fail_load else ["save", "load", "reset"])
    detail = extra["last_sync"]
    assert detail["timings"]["adapter_save"] == 1.
    assert detail["timings"]["adapter_load"] == 1.
    assert detail["status"] == ("failed" if fail_load else "complete")
    assert ("prefix_cache_reset" in detail["timings"]) is not fail_load


@pytest.fixture
def simulated_gpu_train(monkeypatch, tmp_path):
    import torch
    from grace_gc.backends import verl_trainer as trainer
    from grace_gc.data import reward
    from grace_gc.logging_util import forensics
    from grace_gc.logging_util.run_dir import RunDirectory
    from grace_gc.trainer import initialization, state_io

    monkeypatch.setattr(trainer, "require_gpu_stack", lambda: {})
    monkeypatch.setattr(trainer, "load_hf_tokenizer", lambda _: object())
    monkeypatch.setattr(trainer, "tokenizer_inventory", lambda _: {})
    monkeypatch.setattr(trainer, "seed_all", lambda _: None)
    monkeypatch.setattr(reward, "require_math_verify", lambda: None)
    monkeypatch.setattr(trainer, "load_lora_actor", lambda *args: object())
    monkeypatch.setattr(trainer, "maybe_wrap_fsdp", lambda actor, _: actor)
    monkeypatch.setattr(initialization, "initialize_actor", lambda *args: None)
    monkeypatch.setattr(trainer, "build_vllm_engine", lambda *args: object())
    monkeypatch.setattr(trainer, "named_lora_params", lambda _: [
        ("q_proj.lora_A.weight", torch.nn.Parameter(torch.zeros(1, 1))),
    ])
    monkeypatch.setattr(trainer, "actor_numerics", lambda _: {})
    monkeypatch.setattr(trainer, "persist_initial_checkpoint", lambda *args, **kwargs: None)
    monkeypatch.setattr(state_io, "check_snapshot_identity", lambda *args: None)
    monkeypatch.setattr(forensics, "record_effective_config", lambda *args: None)
    monkeypatch.setattr(trainer, "restore_train_state", lambda *args, **kwargs: SimpleNamespace(step=4))
    return trainer, RunDirectory(tmp_path), {
        "model_path": "test/model", "method": "full_pg", "prompt_token_ids": [[1, 2]],
        "golds": ["1"], "num_steps": 1, "n_start": 1, "n_prompts": 1,
        "cost_control": {"mode": "fixed"},
    }


@pytest.mark.parametrize("failure", ["initial_sync", "resume_sync", "prefix", "continue", "algorithm", "step_sync"])
def test_gpu_train_persists_failure_observations(simulated_gpu_train, monkeypatch, failure):
    trainer, run, cfg = simulated_gpu_train
    error = RuntimeError("original failure")
    failed_call = {"status": "failed", "batch_size": 1, "wall_seconds": 2.5,
                   "error_type": "RuntimeError", "error": str(error)}
    engines = SimpleNamespace(last_rollout=None)
    extra = {"pad_id": 0, "eos_id": 99}
    sync_calls = []

    def sync():
        sync_calls.append(1)
        fail = (failure == "initial_sync" or
                (failure in {"resume_sync", "step_sync"} and len(sync_calls) == 2))
        extra["last_sync"] = {"status": "failed" if fail else "complete",
                              "timings": {"adapter_load": 1.5}}
        if fail:
            extra["last_sync"]["failed_stage"] = "adapter_load"
            raise error

    def step(_engines, state, *args, **kwargs):
        engines.last_rollout = {
            "prefix_num_cached_tokens": [0], "prefix_token_ids": [[123456789]],
            "prefix_execution": {"generate_calls": [failed_call] if failure == "prefix" else
                                 [failed_call, {"status": "returned", "wall_seconds": 1.}]
                                 if failure == "algorithm" else []},
            "continue_execution": [{"request_indices": [0], "generate_calls": [failed_call]}]
                if failure == "continue" else [],
        }
        if failure in {"prefix", "continue", "algorithm"}:
            raise error
        state.step += 1
        return {"timings": {}}

    extra["sync"] = sync
    monkeypatch.setattr(trainer, "make_gpu_engines", lambda *args: (engines, extra))
    monkeypatch.setattr(trainer, "run_algorithm1_step", step)
    if failure == "resume_sync":
        cfg.update(resume="unused", _resume_payload={"step": 4})
    with pytest.raises(RuntimeError) as caught:
        trainer.train(cfg, run)
    assert caught.value is error
    saved = json.loads((run.root / "failed_execution.json").read_text(encoding="utf-8"))
    assert saved["phase"] == ("algorithm_step" if failure in {"prefix", "continue", "algorithm"} else failure)
    assert saved["step"] == (4 if failure == "resume_sync" else 0 if failure == "initial_sync" else 1)
    assert saved["error_type"] == "RuntimeError"
    assert saved["last_sync"]["timings"]["adapter_load"] == 1.5
    assert "123456789" not in json.dumps(saved)
    if failure in {"prefix", "continue"}:
        assert saved["failed_attempt"]["phase"] == failure
        assert saved["failed_attempt"]["wall_seconds"] == 2.5
    else:
        assert saved["failed_attempt"] is None


def test_failure_record_write_error_does_not_replace_original(simulated_gpu_train, monkeypatch):
    trainer, run, cfg = simulated_gpu_train
    error = RuntimeError("original generation failure")
    engines = SimpleNamespace(last_rollout={})

    def sync():
        raise error

    original_write = run.write_json

    def write(name, payload):
        if name == "failed_execution.json":
            raise OSError("disk full")
        return original_write(name, payload)

    monkeypatch.setattr(run, "write_json", write)
    monkeypatch.setattr(trainer, "make_gpu_engines", lambda *args: (engines, {"sync": sync}))
    with pytest.raises(RuntimeError) as caught:
        trainer.train(cfg, run)
    assert caught.value is error
    assert any("failed_execution.json" in note and "disk full" in note
               for note in getattr(error, "__notes__", []))


def test_successful_gpu_sync_details_reach_steps_jsonl(simulated_gpu_train, monkeypatch):
    from grace_gc.logging_util import forensics

    trainer, run, cfg = simulated_gpu_train
    engines = SimpleNamespace(last_rollout={})
    extra = {}
    sync_steps = []

    def sync():
        sync_steps.append(len(sync_steps))
        extra["last_sync"] = {"status": "complete", "timings": {"adapter_load": float(len(sync_steps))}}

    def step(_engines, state, *args, **kwargs):
        state.step += 1
        return {"timings": {}, "n": 1, "n_completed": 1, "n_audited": 0, "loss": 0.}

    extra["sync"] = sync
    monkeypatch.setattr(trainer, "make_gpu_engines", lambda *args: (engines, extra))
    monkeypatch.setattr(trainer, "run_algorithm1_step", step)
    monkeypatch.setattr(forensics, "_publish_snapshot", lambda *args, **kwargs: ({}, "test-checkpoint"))
    monkeypatch.setattr(trainer, "persist_final_checkpoint", lambda *args, **kwargs: None)
    result = trainer.train(cfg, run)
    rows = [json.loads(line) for line in (run.root / "steps.jsonl").read_text(encoding="utf-8").splitlines()]
    assert result["steps"] == 1
    assert len(rows) == 1
    assert rows[0]["sync_details"] == {"status": "complete", "timings": {"adapter_load": 2.}}
    assert not (run.root / "failed_execution.json").exists()
