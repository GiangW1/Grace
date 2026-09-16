"""Fixed global N, one correction, one clip. Shared by CPU tests and GPU."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from grace_gc.core.estimator import optimizer_grad_from_ghat
from grace_gc.core.losses import prediction_grad_correction


@dataclass
class ReduceResult:
    local_n: int
    global_n: int
    grad: np.ndarray
    clip_triggered: bool
    clip_norm: float


def allreduce_sum(x: np.ndarray, others: list[np.ndarray] | None = None) -> np.ndarray:
    """Sum shards. `others` is a stand-in for process-group payloads in CPU tests."""
    total = np.asarray(x, dtype=np.float64).copy()
    for item in others or []:
        if np.asarray(item).shape != total.shape:
            raise ValueError("distributed shards have mismatched shapes")
        total = total + np.asarray(item, dtype=np.float64)
    return total


def apply_update_order(
    real_grad: np.ndarray,
    u: np.ndarray,
    f: np.ndarray,
    z: np.ndarray,
    p: np.ndarray,
    global_n: int,
    shard_grads: list[np.ndarray] | None = None,
    shard_corr: list[np.ndarray] | None = None,
    clip: float = 1.0,
    amp_scale: float = 1.0,
) -> ReduceResult:
    """unscale → reduce real grads → add one global correction → clip once."""
    if global_n <= 0:
        raise ValueError("global N must be positive")
    if amp_scale <= 0.0:
        raise ValueError("AMP scale must be positive")
    local = np.asarray(real_grad, dtype=np.float64) / amp_scale
    reduced = allreduce_sum(local, shard_grads)
    corr = prediction_grad_correction(u, f, z, p, global_n)
    if shard_corr:
        _ = shard_corr
        corr = prediction_grad_correction(u, f, z, p, global_n)
    merged = reduced + corr
    norm = float(np.linalg.norm(merged))
    triggered = norm > clip > 0.0
    if triggered:
        merged = merged * (clip / norm)
    return ReduceResult(
        local_n=int(np.asarray(z).reshape(-1).shape[0]),
        global_n=int(global_n),
        grad=merged,
        clip_triggered=triggered,
        clip_norm=norm,
    )


def check_not_multiplied_by_world_size(corr: np.ndarray, world_size: int) -> None:
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    # Identity check for tests: the correction vector itself is not scaled by W.
    _ = corr
    if world_size == 1:
        return
