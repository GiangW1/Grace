"""U6/U7 GPU entries. They skip on this machine when CUDA/verl are absent."""

from pathlib import Path

import pytest

from grace_gc.audit.run import generate_bundles_gpu, run_audit
from grace_gc.backends import require_gpu_stack
from grace_gc.backends.distributed import apply_update_order
from grace_gc.backends.fsdp_actor import maybe_wrap_fsdp
from grace_gc.data.math_data import MathRecord
import numpy as np


def test_u6_gpu_fp64_entry_is_defined():
    try:
        versions = require_gpu_stack()
    except ImportError:
        pytest.skip("GPU stack not installed; U6 must be run on the server")
    import torch

    from grace_gc.core.estimator import ht_estimate

    g = np.array([[1.0, -2.0, 0.5]], dtype=np.float64)
    m = np.array([[0.2, 0.1, -0.3]], dtype=np.float64)
    p = np.array([0.4], dtype=np.float64)
    z = np.array([1.0], dtype=np.float64)
    ref = ht_estimate(g, m, p, z)
    device = torch.device("cuda")
    pt = torch.as_tensor(p, device=device, dtype=torch.float64)
    zt = torch.as_tensor(z, device=device, dtype=torch.float64)
    gt = torch.as_tensor(g, device=device, dtype=torch.float64)
    mt = torch.as_tensor(m, device=device, dtype=torch.float64)
    hat = mt + (zt / pt)[:, None] * (gt - mt)
    np.testing.assert_allclose(hat.cpu().numpy(), ref, atol=1e-12)
    assert "torch" in versions


def test_u7_amp_reduce_clip_order_cpu_reference():
    u = np.eye(3, 1)
    f = np.zeros((2, 1))
    z = np.array([1.0, 1.0])
    p = np.array([1.0, 1.0])
    out = apply_update_order(np.array([3.0, 0.0, 0.0]), u, f, z, p, global_n=2, clip=1.0, amp_scale=1.0)
    assert out.clip_triggered is True
    assert out.global_n == 2
    np.testing.assert_allclose(np.linalg.norm(out.grad), 1.0, atol=1e-12)


def test_vllm_stop_without_eos_token_is_finished():
    from grace_gc.backends.vllm_two_phase import trim_generated_tokens

    cut, ended = trim_generated_tokens([10, 11, 2, 99], 2, None)
    assert cut == [10, 11, 2]
    assert ended
    kept, open_end = trim_generated_tokens([10, 11, 12], 2, "length")
    assert kept == [10, 11, 12]
    assert not open_end
    omitted, stopped = trim_generated_tokens([10, 11, 12], 2, "stop")
    assert omitted == [10, 11, 12]
    assert stopped
    restored, restored_stop = trim_generated_tokens([10, 11, 12], 2, "stop", 2)
    assert restored == [10, 11, 12, 2]
    assert restored_stop
    empty, empty_stop = trim_generated_tokens([], 2, "stop")
    assert empty == []
    assert empty_stop
    empty_id, empty_id_stop = trim_generated_tokens([], 2, "stop", 2)
    assert empty_id == [2]
    assert empty_id_stop
    qwen_stops = [151643, 151645]
    no_guess, no_guess_stop = trim_generated_tokens([10, 11], qwen_stops, "stop")
    assert no_guess == [10, 11]
    assert no_guess_stop
    named, named_stop = trim_generated_tokens([10, 11], qwen_stops, "stop", 151645)
    assert named == [10, 11, 151645]
    assert named_stop
    im_end, im_stopped = trim_generated_tokens([10, 11, 151645, 7], qwen_stops, None)
    assert im_end == [10, 11, 151645]
    assert im_stopped
    class _Stop:
        name = "STOP"
        value = 0

    enum_cut, enum_stop = trim_generated_tokens([10, 11], 2, _Stop())
    assert enum_cut == [10, 11]
    assert enum_stop
    class _Length:
        name = "LENGTH"

    length_cut, length_stop = trim_generated_tokens([10, 11, 12], 2, _Length())
    assert length_cut == [10, 11, 12]
    assert not length_stop


def test_generate_phase_rejects_abort_finish(monkeypatch):
    from types import SimpleNamespace

    import grace_gc.backends.vllm_two_phase as vllm_mod
    from grace_gc.core.rng import IsolatedRNG

    class _SP:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _LLM:
        def generate(self, prompts, sampling_params=None, **kwargs):
            return [
                SimpleNamespace(
                    prompt_token_ids=[1, 2],
                    outputs=[SimpleNamespace(token_ids=[9], finish_reason="abort")],
                )
            ]

    monkeypatch.setattr(vllm_mod, "_require_vllm", lambda: (_LLM, _SP))
    monkeypatch.setattr(vllm_mod, "_vllm_prompts", lambda ids: ids)
    with pytest.raises(ValueError, match="finish_reason"):
        vllm_mod.generate_phase(_LLM(), [[1, 2]], 4, 1.0, 2, IsolatedRNG.create(0), "token")


def test_length_finish_short_of_budget_is_truncated():
    from grace_gc.backends.vllm_two_phase import generated_was_truncated

    assert generated_was_truncated(4095, 4096, False, "length") is True
    assert generated_was_truncated(4096, 4096, False, "length") is True
    assert generated_was_truncated(200, 4096, True, "stop") is False
    assert generated_was_truncated(200, 4096, False, "stop") is False
    class _Length:
        name = "LENGTH"

    assert generated_was_truncated(100, 4096, False, _Length()) is True


def test_qwen_chat_stop_ids_include_im_end_and_endoftext():
    from grace_gc.data.tokenize import collect_stop_token_ids
    from grace_gc.trainer.algorithm import _length_truncated

    vocab = {"<|endoftext|>": 151643, "<|im_end|>": 151645, "unk": 0}

    class _Tok:
        eos_token_id = 151645
        unk_token_id = 0

        def convert_tokens_to_ids(self, name):
            return vocab.get(name, 0)

        def convert_ids_to_tokens(self, tid):
            inv = {v: k for k, v in vocab.items()}
            return inv.get(int(tid), "unk")

    ids = collect_stop_token_ids(_Tok())
    assert 151645 in ids
    assert 151643 in ids
    assert _length_truncated([1, 2, 3, 151643], 1, 3, False, ids) is False
    assert _length_truncated([1, 2, 3, 4], 1, 3, False, ids) is True


def test_build_sampling_params_passes_stop_token_ids(monkeypatch):
    import grace_gc.backends.vllm_two_phase as vllm_mod

    seen = {}

    class _SP:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr(vllm_mod, "_require_vllm", lambda: (None, _SP))
    params = vllm_mod.build_sampling_params(8, 1.0, 3, eos_id=2, top_p=0.95)
    assert isinstance(params, _SP)
    assert seen["stop_token_ids"] == [2]
    assert seen["stop"] == []
    assert seen["seed"] == 3
    assert seen["top_p"] == 0.95
    assert seen["top_k"] == -1
    assert seen["repetition_penalty"] == 1.0
    assert seen["min_p"] == 0.0
    assert "ignore_eos" not in seen


def test_build_sampling_params_rejects_missing_stop_ids(monkeypatch):
    import grace_gc.backends.vllm_two_phase as vllm_mod

    class _SP:
        def __init__(self, **kwargs):
            if "stop_token_ids" in kwargs:
                raise TypeError("stop_token_ids")
            self.kwargs = kwargs

    monkeypatch.setattr(vllm_mod, "_require_vllm", lambda: (None, _SP))
    with pytest.raises(ValueError, match="required sampling field"):
        vllm_mod.build_sampling_params(4, 1.0, 1, eos_id=7)


def test_build_sampling_params_drops_unsupported_min_p(monkeypatch):
    import grace_gc.backends.vllm_two_phase as vllm_mod

    class _SP:
        def __init__(self, **kwargs):
            if "min_p" in kwargs:
                raise TypeError("min_p")
            self.kwargs = kwargs

    monkeypatch.setattr(vllm_mod, "_require_vllm", lambda: (None, _SP))
    params = vllm_mod.build_sampling_params(4, 1.0, 1, eos_id=7)
    assert "min_p" not in params.kwargs
    assert params.kwargs["stop_token_ids"] == [7]
    assert params.kwargs["stop"] == []
    assert params.kwargs["top_k"] == -1
    assert params.kwargs["seed"] == 1


def test_build_sampling_params_rejects_unseeded_fallback(monkeypatch):
    import grace_gc.backends.vllm_two_phase as vllm_mod

    class _SP:
        def __init__(self, **kwargs):
            raise TypeError("no seed")

    monkeypatch.setattr(vllm_mod, "_require_vllm", lambda: (None, _SP))
    with pytest.raises(ValueError, match="seed"):
        vllm_mod.build_sampling_params(4, 1.0, 1, eos_id=7)


def test_generate_phase_rejects_reordered_prompts(monkeypatch):
    from types import SimpleNamespace

    import grace_gc.backends.vllm_two_phase as vllm_mod
    from grace_gc.core.rng import IsolatedRNG

    class _SP:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _LLM:
        def generate(self, prompts, sampling_params=None, **kwargs):
            return [
                SimpleNamespace(
                    prompt_token_ids=[3, 4],
                    outputs=[SimpleNamespace(token_ids=[9], finish_reason="length")],
                ),
                SimpleNamespace(
                    prompt_token_ids=[1, 2],
                    outputs=[SimpleNamespace(token_ids=[8], finish_reason="length")],
                ),
            ]

    monkeypatch.setattr(vllm_mod, "_require_vllm", lambda: (_LLM, _SP))
    monkeypatch.setattr(vllm_mod, "_vllm_prompts", lambda ids: ids)
    with pytest.raises(ValueError, match="request order"):
        vllm_mod.generate_phase(_LLM(), [[1, 2], [3, 4]], 4, 1.0, 2, IsolatedRNG.create(0), "token")


def test_generate_phase_rejects_empty_completions(monkeypatch):
    from types import SimpleNamespace

    import grace_gc.backends.vllm_two_phase as vllm_mod
    from grace_gc.core.rng import IsolatedRNG

    class _SP:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _LLM:
        def generate(self, prompts, sampling_params=None, **kwargs):
            return [
                SimpleNamespace(prompt_token_ids=[1, 2], outputs=[]),
                SimpleNamespace(
                    prompt_token_ids=[3, 4],
                    outputs=[SimpleNamespace(token_ids=[8], finish_reason="length")],
                ),
            ]

    monkeypatch.setattr(vllm_mod, "_require_vllm", lambda: (_LLM, _SP))
    monkeypatch.setattr(vllm_mod, "_vllm_prompts", lambda ids: ids)
    with pytest.raises(ValueError, match="no completions"):
        vllm_mod.generate_phase(_LLM(), [[1, 2], [3, 4]], 4, 1.0, 2, IsolatedRNG.create(0), "token")


def test_generate_phase_rejects_output_count_mismatch(monkeypatch):
    from types import SimpleNamespace

    import grace_gc.backends.vllm_two_phase as vllm_mod
    from grace_gc.core.rng import IsolatedRNG

    class _SP:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _LLM:
        def generate(self, prompts, sampling_params=None, **kwargs):
            return [SimpleNamespace(outputs=[SimpleNamespace(token_ids=[9], finish_reason="length")])]

    monkeypatch.setattr(vllm_mod, "_require_vllm", lambda: (_LLM, _SP))
    monkeypatch.setattr(vllm_mod, "_vllm_prompts", lambda ids: ids)
    with pytest.raises(ValueError, match="vLLM returned"):
        vllm_mod.generate_phase(_LLM(), [[1, 2], [3, 4]], 4, 1.0, 2, IsolatedRNG.create(0), "token")


def test_generate_phase_rejects_missing_prompt_ids(monkeypatch):
    from types import SimpleNamespace

    import grace_gc.backends.vllm_two_phase as vllm_mod
    from grace_gc.core.rng import IsolatedRNG

    class _SP:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _LLM:
        def generate(self, prompts, sampling_params=None, **kwargs):
            return [
                SimpleNamespace(outputs=[SimpleNamespace(token_ids=[9], finish_reason="length")]),
            ]

    monkeypatch.setattr(vllm_mod, "_require_vllm", lambda: (_LLM, _SP))
    monkeypatch.setattr(vllm_mod, "_vllm_prompts", lambda ids: ids)
    with pytest.raises(ValueError, match="prompt_token_ids"):
        vllm_mod.generate_phase(_LLM(), [[1, 2]], 4, 1.0, 2, IsolatedRNG.create(0), "token")


def test_generate_phase_keeps_identical_answers_with_independent_seeds(monkeypatch):
    from types import SimpleNamespace

    import grace_gc.backends.vllm_two_phase as vllm_mod
    from grace_gc.core.rng import IsolatedRNG

    class _SP:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _LLM:
        def generate(self, prompts, sampling_params=None, **kwargs):
            assert len({sp.kwargs["seed"] for sp in sampling_params}) == 2
            same = list(range(10, 18))
            return [
                SimpleNamespace(
                    prompt_token_ids=[1, 2],
                    outputs=[SimpleNamespace(token_ids=same, finish_reason="length")],
                ),
                SimpleNamespace(
                    prompt_token_ids=[1, 2],
                    outputs=[SimpleNamespace(token_ids=same, finish_reason="length")],
                ),
            ]

    monkeypatch.setattr(vllm_mod, "_require_vllm", lambda: (_LLM, _SP))
    monkeypatch.setattr(vllm_mod, "_vllm_prompts", lambda ids: ids)
    result = vllm_mod.generate_phase(_LLM(), [[1, 2], [1, 2]], 8, 1.0, 2, IsolatedRNG.create(0), "token")
    assert result.token_ids[0] == result.token_ids[1]
    assert len(set(result.sampling["request_seeds"])) == 2


def test_generate_phase_accepts_distinct_same_prompt_rollouts(monkeypatch):
    from types import SimpleNamespace

    import grace_gc.backends.vllm_two_phase as vllm_mod
    from grace_gc.core.rng import IsolatedRNG

    class _SP:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _LLM:
        def generate(self, prompts, sampling_params=None, **kwargs):
            return [
                SimpleNamespace(
                    prompt_token_ids=[1, 2],
                    outputs=[SimpleNamespace(token_ids=list(range(10, 18)), finish_reason="length")],
                ),
                SimpleNamespace(
                    prompt_token_ids=[1, 2],
                    outputs=[SimpleNamespace(token_ids=list(range(20, 28)), finish_reason="length")],
                ),
            ]

    monkeypatch.setattr(vllm_mod, "_require_vllm", lambda: (_LLM, _SP))
    monkeypatch.setattr(vllm_mod, "_vllm_prompts", lambda ids: ids)
    out = vllm_mod.generate_phase(_LLM(), [[1, 2], [1, 2]], 8, 1.0, 2, IsolatedRNG.create(0), "token")
    assert out.token_ids[0] != out.token_ids[1]


def test_require_gpu_stack_does_not_require_unused_verl():
    import inspect

    from grace_gc.backends import require_gpu_stack

    src = inspect.getsource(require_gpu_stack)
    assert 'missing.append("verl")' not in src
    assert 'missing.append("vllm")' in src


def test_lora_sync_resets_prefix_cache_and_uses_new_dir(tmp_path: Path, monkeypatch):
    import grace_gc.backends.gpu_engine as gpu_mod

    saved = []

    class _Actor:
        def save_pretrained(self, path):
            saved.append(str(path))
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "adapter_config.json").write_text("{}", encoding="utf-8")
            (Path(path) / "adapter_model.safetensors").write_bytes(b"lora")

    class _LLM:
        def __init__(self):
            self.resets = 0
            self.added = []
            self.removed = []

        def add_lora(self, request):
            self.added.append(request)
            return True

        def remove_lora(self, ident):
            self.removed.append(int(ident))
            return True

        def reset_prefix_cache(self):
            self.resets += 1
            return True

    monkeypatch.setattr(gpu_mod, "make_lora_request", lambda path, ident: ("req", str(path), int(ident)))
    llm = _LLM()
    extra = {"lora_id": 1, "llm": llm}
    req = gpu_mod._sync(_Actor(), tmp_path / "lora", extra)
    assert extra["lora_id"] == 2
    assert extra["adapter_path"] == tmp_path / "lora" / "step-2"
    assert saved == [str(tmp_path / "lora" / "step-2")]
    assert llm.added == [req]
    assert llm.removed == []
    assert llm.resets == 1
    assert req == ("req", str(tmp_path / "lora" / "step-2"), 2)
    gpu_mod._sync(_Actor(), tmp_path / "lora", extra)
    assert extra["lora_id"] == 3
    assert llm.removed == [2]


def test_apply_lora_request_requires_vllm_api():
    from grace_gc.backends.weight_sync import apply_lora_request

    class _Engine:
        def add_lora(self, request):
            self.req = request
            return True

    class _LLM:
        llm_engine = _Engine()

    apply_lora_request(_LLM(), "req")
    class _AddOnly:
        def add_lora(self, request):
            return True

    with pytest.raises(TypeError, match="remove_lora"):
        apply_lora_request(_AddOnly(), "req", remove_id=1)
    with pytest.raises(TypeError, match="add_lora"):
        apply_lora_request(object(), "req")
    with pytest.raises(TypeError, match="required"):
        apply_lora_request(None, "req")
    with pytest.raises(TypeError, match="LoRARequest"):
        apply_lora_request(object(), None)


def test_save_lora_adapter_requires_adapter_config(tmp_path: Path):
    from grace_gc.backends.weight_sync import save_lora_adapter

    class _Empty:
        def save_pretrained(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)

    with pytest.raises(ValueError, match="adapter_config"):
        save_lora_adapter(_Empty(), tmp_path / "empty")


def test_save_lora_adapter_requires_weights(tmp_path: Path):
    from grace_gc.backends.weight_sync import save_lora_adapter

    class _CfgOnly:
        def save_pretrained(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "adapter_config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="adapter weights"):
        save_lora_adapter(_CfgOnly(), tmp_path / "cfg")


def test_apply_lora_rejects_awaitable():
    from grace_gc.backends.weight_sync import apply_lora_request, reset_vllm_prefix_cache

    async def _add(_request):
        return True

    async def _reset():
        return True

    class _LLM:
        add_lora = staticmethod(_add)
        reset_prefix_cache = staticmethod(_reset)

    with pytest.raises(TypeError, match="awaitable"):
        apply_lora_request(_LLM(), "req")
    with pytest.raises(TypeError, match="awaitable"):
        reset_vllm_prefix_cache(_LLM())


def test_prefix_cache_reset_requires_vllm_api():
    from grace_gc.backends.weight_sync import reset_vllm_prefix_cache

    class _Engine:
        def reset_prefix_cache(self):
            return True

    class _LLM:
        llm_engine = _Engine()

    reset_vllm_prefix_cache(_LLM())
    with pytest.raises(TypeError, match="reset_prefix_cache"):
        reset_vllm_prefix_cache(object())
    with pytest.raises(TypeError, match="required"):
        reset_vllm_prefix_cache(None)


def test_lora_request_name_includes_id(monkeypatch):
    import sys
    import types

    import grace_gc.backends.weight_sync as ws

    seen = {}

    def _fake_request(*args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        return "ok"

    for name in ("vllm", "vllm.lora", "vllm.lora.request"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["vllm.lora.request"].LoRARequest = _fake_request
    assert ws.make_lora_request("/tmp/a", 3) == "ok"
    if seen["kwargs"]:
        assert seen["kwargs"]["lora_name"] == "grace-gc-3"
        assert seen["kwargs"]["lora_int_id"] == 3
        assert Path(seen["kwargs"]["lora_path"]).is_absolute()
    else:
        assert seen["args"][0] == "grace-gc-3"
        assert seen["args"][1] == 3
        assert Path(seen["args"][2]).is_absolute()


def test_gpu_loads_qwen_with_trust_remote_code():
    import inspect

    from grace_gc.backends import verl_trainer

    assert "trust_remote_code=True" in inspect.getsource(verl_trainer.load_lora_actor)
    src = inspect.getsource(verl_trainer.build_vllm_engine)
    assert "trust_remote_code" in src
    assert 'generation_config": "vllm"' in src
    assert 'pop("generation_config"' not in src


def test_build_vllm_engine_keeps_generation_config(monkeypatch):
    import grace_gc.backends.verl_trainer as vt
    import grace_gc.backends.vllm_two_phase as vllm_mod

    class _LLM:
        def __init__(self, **kwargs):
            if "generation_config" not in kwargs:
                raise AssertionError("generation_config must not be dropped")
            if "max_loras" in kwargs:
                raise TypeError("max_loras")
            self.kwargs = kwargs

    monkeypatch.setattr(vllm_mod, "_require_vllm", lambda: (_LLM, None))
    llm = vt.build_vllm_engine("org/model", {}, 16)
    assert llm.kwargs["generation_config"] == "vllm"
    assert "max_loras" not in llm.kwargs
    assert vt.build_vllm_engine.last["accepted"]["generation_config"] == "vllm"
    assert "max_loras" in vt.build_vllm_engine.last["dropped"]


def test_build_vllm_engine_rejects_missing_generation_config(monkeypatch):
    import grace_gc.backends.verl_trainer as vt
    import grace_gc.backends.vllm_two_phase as vllm_mod

    class _LLM:
        def __init__(self, **kwargs):
            raise TypeError("generation_config")

    monkeypatch.setattr(vllm_mod, "_require_vllm", lambda: (_LLM, None))
    with pytest.raises(ValueError, match="generation_config"):
        vt.build_vllm_engine("org/model", {}, 16)


def test_build_vllm_engine_leaves_room_for_hf_actor(monkeypatch):
    import grace_gc.backends.verl_trainer as vt
    import grace_gc.backends.vllm_two_phase as vllm_mod

    class _LLM:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(vllm_mod, "_require_vllm", lambda: (_LLM, None))
    llm = vt.build_vllm_engine("org/model", {}, 16)
    assert llm.kwargs["gpu_memory_utilization"] == 0.5
    assert llm.kwargs["max_model_len"] == 5120
    custom = vt.build_vllm_engine("org/model", {"gpu_memory_utilization": 0.4, "max_model_len": 8192}, 16)
    assert custom.kwargs["gpu_memory_utilization"] == 0.4
    assert custom.kwargs["max_model_len"] == 8192


def test_vllm_registers_worker_shutdown(monkeypatch):
    from types import SimpleNamespace

    import grace_gc.backends.verl_trainer as vt
    import grace_gc.backends.vllm_two_phase as vllm_mod

    callbacks = []
    events = []
    core = SimpleNamespace(shutdown=lambda: events.append("shutdown"))
    llm = SimpleNamespace(
        llm_engine=SimpleNamespace(engine_core=core),
        collective_rpc=lambda callback: events.append(callback),
    )
    monkeypatch.setattr(vllm_mod, "_require_vllm", lambda: (lambda **kwargs: llm, None))
    monkeypatch.setattr(vt.atexit, "register", lambda fn, *args: callbacks.append((fn, args)))

    assert vt.build_vllm_engine("org/model", {}, 16) is llm
    assert not events
    assert len(callbacks) == 1
    callback, args = callbacks[0]
    callback(*args)
    assert events == ["grace_destroy_process_groups", "shutdown"]


def test_vllm_shutdown_still_stops_core_if_rpc_fails():
    from types import SimpleNamespace

    from grace_gc.backends.verl_trainer import _shutdown_vllm_engine

    closed = []

    def broken_rpc(callback):
        raise RuntimeError("worker failed")

    llm = SimpleNamespace(
        llm_engine=SimpleNamespace(engine_core=SimpleNamespace(shutdown=lambda: closed.append(True))),
        collective_rpc=broken_rpc,
    )
    with pytest.raises(RuntimeError, match="worker failed"):
        _shutdown_vllm_engine(llm)
    assert closed == [True]


def test_eval_vllm_keeps_identical_answers_with_independent_seeds(monkeypatch):
    from types import SimpleNamespace

    import grace_gc.backends.vllm_two_phase as vllm_mod
    import grace_gc.data.tokenize as tok
    from grace_gc.evaluation import generate

    class _SP:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    seeds = []

    class _LLM:
        def generate(self, prompts, sampling_params=None, **kwargs):
            seeds.append(sampling_params.kwargs["seed"])
            return [
                SimpleNamespace(
                    prompt_token_ids=[1, 2],
                    outputs=[SimpleNamespace(token_ids=[9, 9, 9], finish_reason="length")],
                )
            ]

    monkeypatch.setattr(vllm_mod, "_require_vllm", lambda: (_LLM, _SP))
    monkeypatch.setattr(vllm_mod, "_vllm_prompts", lambda ids: ids)
    monkeypatch.setattr(
        tok,
        "encode_records_hf",
        lambda recs, tokenizer, max_prompt, **kwargs: ([[1, 2]] * len(recs), ["p"] * len(recs), ["2"] * len(recs)),
    )
    monkeypatch.setattr(tok, "decode_hf", lambda tokenizer, gen: "same")
    monkeypatch.setattr(tok, "collect_stop_token_ids", lambda tokenizer: [2])
    recs = [MathRecord(problem_id="a", prompt="1+1", answer="2")]
    items = generate.generate_answers_vllm(recs, object(), _LLM(), 4, 8, 0.6, 0.95, 1)
    assert items[0].answers == ["same"] * 4
    assert seeds == [1, 2, 3, 4]
    assert items[0].sample_seeds == seeds


def test_eval_vllm_accepts_distinct_avg_at_4(monkeypatch):
    from types import SimpleNamespace

    import grace_gc.backends.vllm_two_phase as vllm_mod
    import grace_gc.data.tokenize as tok
    from grace_gc.evaluation import generate

    class _SP:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _LLM:
        def __init__(self):
            self.n = 0

        def generate(self, prompts, sampling_params=None, **kwargs):
            self.n += 1
            return [
                SimpleNamespace(
                    prompt_token_ids=[1, 2],
                    outputs=[SimpleNamespace(token_ids=[10 + self.n, 11, 12], finish_reason="length")],
                )
            ]

    monkeypatch.setattr(vllm_mod, "_require_vllm", lambda: (_LLM, _SP))
    monkeypatch.setattr(vllm_mod, "_vllm_prompts", lambda ids: ids)
    monkeypatch.setattr(
        tok,
        "encode_records_hf",
        lambda recs, tokenizer, max_prompt, **kwargs: ([[1, 2]] * len(recs), ["p"] * len(recs), ["2"] * len(recs)),
    )
    monkeypatch.setattr(tok, "decode_hf", lambda tokenizer, gen: f"ans{gen[0]}")
    monkeypatch.setattr(tok, "collect_stop_token_ids", lambda tokenizer: [2])
    recs = [MathRecord(problem_id="a", prompt="1+1", answer="2")]
    items = generate.generate_answers_vllm(recs, object(), _LLM(), 4, 8, 0.6, 0.95, 1)
    assert len(items[0].answers) == 4
    assert len(set(items[0].answers)) == 4


def test_eval_writes_lora_before_starting_vllm():
    import inspect

    from grace_gc.evaluation import generate

    src = inspect.getsource(generate._generate_eval_items)
    assert src.index("save_lora_adapter") < src.index("llm = build_vllm_engine")
    assert src.index("del actor") < src.index("llm = build_vllm_engine")
    assert src.index("apply_lora_request") < src.index("generate_answers_vllm")
    assert "max_model_len" in src
    assert "max(have, 5120, need)" in src
    assert src.index("prompt_max_tokens") < src.index("llm = build_vllm_engine")


def test_logprob_forward_skips_hidden_states():
    import inspect

    from grace_gc.backends import hf_actor

    src = inspect.getsource(hf_actor.logprob_one)
    assert "output_hidden_states=False" in src
    feat = inspect.getsource(hf_actor.prefix_feature_bundle)
    assert "output_hidden_states=True" in feat


def test_gpu_train_seeds_before_lora_init():
    import inspect

    from grace_gc.backends import verl_trainer

    src = inspect.getsource(verl_trainer.train)
    assert src.index("seed_all") < src.index("load_lora_actor")
    assert src.index("apply_method_defaults") < src.index("resolve_start_counts")


def test_gpu_train_rejects_n_gpu_gt_1(tmp_path: Path):
    from grace_gc.backends.verl_trainer import train
    from grace_gc.logging_util.run_dir import RunDirectory

    with pytest.raises(RuntimeError, match="n_gpu"):
        train({"hardware": {"n_gpu": 4}, "model_path": "org/model"}, RunDirectory(tmp_path / "g"))


def test_fsdp_multigpu_hard_fails_without_process_group():
    module = object()
    assert maybe_wrap_fsdp(module, 1) is module
    with pytest.raises((ImportError, RuntimeError)):
        maybe_wrap_fsdp(module, 2)


def test_gpu_audit_entry_is_wired(tmp_path: Path):
    recs = [MathRecord(problem_id="a", prompt="1+1", answer="2")]
    assert callable(generate_bundles_gpu)
    with pytest.raises(ValueError, match="model_path"):
        run_audit(recs, {"backend": "gpu_verl"}, tmp_path / "audit-missing")
    with pytest.raises(FileNotFoundError, match="model path"):
        run_audit(
            recs,
            {"backend": "gpu_verl", "model_path": str(tmp_path / "no-such-model")},
            tmp_path / "audit-missing-file",
        )
    with pytest.raises(ValueError, match="checkpoint"):
        run_audit(
            recs,
            {"backend": "gpu_verl", "model_path": "org/model"},
            tmp_path / "audit-no-ckpt",
        )


def test_gpu_eval_requires_checkpoint(tmp_path: Path):
    from grace_gc.evaluation.generate import run_eval

    with pytest.raises(ValueError, match="checkpoint"):
        run_eval([], {"backend": "gpu_verl", "model_path": "org/model"}, tmp_path / "eval-no-ckpt")
