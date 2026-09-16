"""Shared real-stream backward, audit, prediction correction, clip, step."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from grace_gc.core.layout import ParamLayout, pack_grads, unpack_to_dict
from grace_gc.core.losses import prediction_grad_correction, real_stream_loss_scale


def _torch():
    import torch

    return torch


@dataclass
class UpdateResult:
    loss: float
    n_audited: int
    clip_triggered: bool
    packed_grad: np.ndarray


def real_stream_backward(logprob_sums, advantages: np.ndarray, p: np.ndarray, z: np.ndarray, n: int):
    torch = _torch()
    scales = real_stream_loss_scale(advantages, p, z, n)
    scale_t = torch.as_tensor(scales, dtype=logprob_sums.dtype, device=logprob_sums.device)
    loss = (logprob_sums * scale_t).sum()
    if loss.requires_grad:
        loss.backward()
    return loss


def real_stream_backward_each(logprob_fn, advantages: np.ndarray, p: np.ndarray, z: np.ndarray, n: int, chosen):
    """Same scale as `real_stream_backward`, one completed start at a time.

    A single 4B forward over N=512 padded sequences materializes vocab logits
    that do not fit next to vLLM on one GPU. Per-start backward accumulates
    the same `.grad`.
    """
    torch = _torch()
    scales = real_stream_loss_scale(advantages, p, z, n)
    chosen = np.asarray(chosen, dtype=bool).reshape(-1)
    if chosen.shape[0] != n:
        raise ValueError("chosen mask dimension does not match N")
    total = None
    for i, keep in enumerate(chosen):
        if not keep:
            continue
        lp = logprob_fn(i)
        if lp is None:
            continue
        term = lp * torch.as_tensor(float(scales[i]), dtype=lp.dtype, device=lp.device)
        if term.requires_grad:
            term.backward()
        total = term.detach() if total is None else total + term.detach()
    if total is None:
        return torch.zeros(())
    return total


def snapshot_grads(params) -> list:
    out = []
    for param in params:
        out.append(None if param.grad is None else param.grad.detach().clone())
    return out


def restore_grads(params, saved) -> None:
    for param, grad in zip(params, saved):
        param.grad = grad


def audit_one(scalar, params, names, layout: ParamLayout) -> np.ndarray:
    torch = _torch()
    grads = torch.autograd.grad(scalar, params, retain_graph=True, allow_unused=True)
    named = []
    for name, param, grad in zip(names, params, grads):
        if grad is None:
            named.append((name, np.zeros(tuple(param.shape), dtype=np.float64)))
        else:
            named.append((name, grad.detach().cpu().numpy()))
    return pack_grads(named, layout)


def apply_correction_clip_step(
    named_params,
    layout: ParamLayout,
    u: np.ndarray,
    f: np.ndarray,
    z: np.ndarray,
    p: np.ndarray,
    n: int,
    optimizer,
    clip: float = 1.0,
    use_correction: bool = True,
) -> tuple[np.ndarray, bool]:
    torch = _torch()
    packed = pack_grads(
        [
            (
                name,
                param.grad.detach().cpu().numpy()
                if param.grad is not None
                else np.zeros(tuple(param.shape), dtype=np.float64),
            )
            for name, param in named_params
        ],
        layout,
    )
    if use_correction:
        packed = packed + prediction_grad_correction(u, f, z, p, n)
    norm = float(np.linalg.norm(packed))
    triggered = clip > 0.0 and norm > clip
    if triggered:
        packed = packed * (clip / norm)
    unpacked = unpack_to_dict(packed, layout)
    written = 0
    for name, param in named_params:
        if name not in unpacked:
            continue
        chunk = torch.as_tensor(unpacked[name], dtype=param.dtype, device=param.device)
        param.grad = chunk.view_as(param)
        written += 1
    if written != len(layout.entries):
        raise ValueError("correction writeback missed layout entries")
    optimizer.step()
    return packed, triggered
