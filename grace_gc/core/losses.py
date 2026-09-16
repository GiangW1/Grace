"""Shared loss scales. CPU tiny trainer and GPU actor both call these."""

from __future__ import annotations

import numpy as np


def real_stream_loss_scale(advantage: np.ndarray, p: np.ndarray, z: np.ndarray, n: int) -> np.ndarray:
    """Per-completed-start multiplier for `- (R-b)/p / N`.

    The token-sum log-prob is multiplied by this scale, then negated outside
    or included here as a negative coefficient so `.grad` equals `-G_real`.
    """
    if n <= 0:
        raise ValueError("N must be positive")
    adv = np.asarray(advantage, dtype=np.float64).reshape(-1)
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    if adv.shape != p.shape or p.shape != z.shape:
        raise ValueError("advantage, p, and z dimensions do not match")
    if np.any(p <= 0.0):
        raise ValueError("p must be > 0")
    return -z * adv / p / n


def prediction_grad_correction(u: np.ndarray, f: np.ndarray, z: np.ndarray, p: np.ndarray, n: int) -> np.ndarray:
    """Quantity to *add to `.grad`*: `-U sum(1-Z/p) f / N`."""
    from grace_gc.core.estimator import optimizer_grad_from_ghat, prediction_correction

    ghat_pred = prediction_correction(u, f, z, p, n)
    return optimizer_grad_from_ghat(ghat_pred)
