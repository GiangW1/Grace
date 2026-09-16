"""Copy updated LoRA weights onto the vLLM snapshot after each actor step."""

from __future__ import annotations

import inspect
from pathlib import Path

_ADAPTER_WEIGHTS = (
    "adapter_model.safetensors",
    "adapter_model.bin",
    "adapter_model.pt",
    "pytorch_lora_weights.safetensors",
    "pytorch_lora_weights.bin",
)


def _require_finished(ok, what: str) -> None:
    if inspect.isawaitable(ok):
        close = getattr(ok, "close", None)
        if callable(close):
            close()
        raise TypeError(
            f"vLLM {what} returned an awaitable; rollout would not wait for the snapshot"
        )


def save_lora_adapter(actor, path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if not hasattr(actor, "save_pretrained"):
        raise TypeError("actor must be a PEFT model with save_pretrained")
    actor.save_pretrained(path)
    if not (path / "adapter_config.json").is_file():
        raise ValueError(f"PEFT save_pretrained wrote no adapter_config.json under {path}")
    if not any((path / name).is_file() for name in _ADAPTER_WEIGHTS):
        raise ValueError(f"PEFT save_pretrained wrote no adapter weights under {path}")
    return path


def make_lora_request(adapter_dir: str | Path, lora_id: int):
    try:
        from vllm.lora.request import LoRARequest
    except Exception as exc:
        raise ImportError("vLLM LoRARequest is required to refresh the rollout snapshot") from exc
    path = str(Path(adapter_dir).expanduser().resolve())
    ident = int(lora_id)
    name = f"grace-gc-{ident}"
    try:
        return LoRARequest(lora_name=name, lora_int_id=ident, lora_path=path)
    except TypeError:
        return LoRARequest(name, ident, path)


def apply_lora_request(llm, request, remove_id: int | None = None) -> None:
    """Register the adapter. generate(lora_request=...) is not enough on every vLLM."""
    if llm is None:
        raise TypeError("vLLM engine is required to load the LoRA snapshot")
    if request is None:
        raise TypeError("LoRARequest is required; generate would use the base model")
    add = getattr(llm, "add_lora", None)
    engine = getattr(llm, "llm_engine", None) or getattr(llm, "engine", None)
    if add is None and engine is not None:
        add = getattr(engine, "add_lora", None)
    if add is None:
        raise TypeError("vLLM has no add_lora; rollout would use the base model")
    if remove_id is not None:
        rem = getattr(llm, "remove_lora", None)
        if rem is None and engine is not None:
            rem = getattr(engine, "remove_lora", None)
        if rem is None:
            raise TypeError(
                "vLLM has no remove_lora; max_loras would evict the live snapshot"
            )
        ok_rem = rem(int(remove_id))
        _require_finished(ok_rem, "remove_lora")
        if ok_rem is False:
            raise RuntimeError(
                "vLLM remove_lora failed; the next add_lora may evict the live adapter"
            )
    ok = add(request)
    _require_finished(ok, "add_lora")
    if ok is False:
        raise RuntimeError("vLLM add_lora failed; rollout would not match the actor")


def reset_vllm_prefix_cache(llm) -> None:
    """KV from the previous LoRA snapshot must not be reused after a sync."""
    if llm is None:
        raise TypeError("vLLM engine is required to reset the prefix cache after LoRA sync")
    reset = getattr(llm, "reset_prefix_cache", None)
    engine = getattr(llm, "llm_engine", None) or getattr(llm, "engine", None)
    if reset is None and engine is not None:
        reset = getattr(engine, "reset_prefix_cache", None)
    if reset is None:
        raise TypeError("vLLM has no reset_prefix_cache; prefix KV would stay on the old LoRA")
    ok = reset()
    _require_finished(ok, "reset_prefix_cache")
    if ok is False:
        raise RuntimeError("vLLM prefix cache reset failed after LoRA sync")
