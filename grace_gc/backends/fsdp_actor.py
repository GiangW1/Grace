"""FSDP actor update order. Imports torch.distributed only when called."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from grace_gc.backends.distributed import apply_update_order
from grace_gc.core.layout import ParamLayout, pack_grads, unpack_to_dict


@dataclass
class ActorUpdateResult:
    clip_triggered: bool
    global_n: int
    grad_norm: float


def _require_torch():
    try:
        import torch
    except Exception as exc:
        raise ImportError("PyTorch is required for the FSDP actor path") from exc
    return torch


def wrap_fsdp(module, **kwargs):
    torch = _require_torch()
    if not torch.cuda.is_available():
        raise ImportError("CUDA is required to wrap the FSDP actor")
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        raise RuntimeError(
            "FSDP wrap requires torch.distributed.init_process_group. "
            "Launch with torchrun or verl; do not fall back to a single process."
        )
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    return FSDP(module, **kwargs)


def maybe_wrap_fsdp(module, n_gpu: int, **kwargs):
    if int(n_gpu) <= 1:
        return module
    return wrap_fsdp(module, **kwargs)


def token_sum_logprob(logits, token_ids, prompt_lens, valid_lens=None, eos_id=None):
    torch = _require_torch()
    logp = torch.log_softmax(logits[:, :-1], dim=-1)
    target = token_ids[:, 1:]
    token_lp = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    mask = torch.zeros_like(token_lp)
    width = token_lp.shape[1]
    for i, plen in enumerate(prompt_lens):
        end = int(valid_lens[i]) - 1 if valid_lens is not None else width
        end = min(max(end, 0), width)
        if eos_id is not None:
            from grace_gc.data.tokenize import as_stop_ids

            stops = set(as_stop_ids(eos_id))
            seq = token_ids[i]
            last = int(valid_lens[i]) if valid_lens is not None else int(seq.shape[0])
            for j in range(int(plen), last):
                if int(seq[j].item()) in stops:
                    end = min(end, j)
                    break
        start = max(int(plen) - 1, 0)
        if start < end:
            mask[i, start:end] = 1.0
    return (token_lp * mask).sum(dim=-1)


def apply_fsdp_step(
    named_grads,
    layout: ParamLayout,
    u: np.ndarray,
    f: np.ndarray,
    z: np.ndarray,
    p: np.ndarray,
    global_n: int,
    optimizer,
    clip: float = 1.0,
    amp_scale: float = 1.0,
    extra_shards: list[np.ndarray] | None = None,
) -> ActorUpdateResult:
    packed = pack_grads(named_grads, layout)
    result = apply_update_order(
        packed,
        u,
        f,
        z,
        p,
        global_n,
        shard_grads=extra_shards,
        clip=clip,
        amp_scale=amp_scale,
    )
    unpacked = unpack_to_dict(result.grad, layout)
    torch = _require_torch()
    for name, param in named_grads:
        if name not in unpacked or not hasattr(param, "view_as"):
            continue
        chunk = torch.as_tensor(unpacked[name], dtype=param.dtype, device=param.device)
        param.grad = chunk.view_as(param)
    if optimizer is not None:
        optimizer.step()
    return ActorUpdateResult(
        clip_triggered=result.clip_triggered,
        global_n=result.global_n,
        grad_norm=result.clip_norm,
    )
