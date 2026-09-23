"""Full-space Horvitz–Thompson gradient completion.

G is the ascent direction. A standard optimizer `.grad` must be `-Ghat`.
"""

from __future__ import annotations

import numpy as np


def _as_2d(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        return arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"expected 1d or 2d array, got shape {arr.shape}")
    return arr


def ht_estimate(g: np.ndarray, m: np.ndarray, p: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Single-start or batched `Ghat = m + Z/p (G - m)`."""
    g = _as_2d(g)
    m = np.broadcast_to(_as_2d(m), g.shape).copy()
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    if g.shape[0] != p.shape[0] or p.shape[0] != z.shape[0]:
        raise ValueError("g, m, p, z batch dimensions do not match")
    if np.any(p <= 0.0):
        raise ValueError("p must be > 0")
    return m + (z / p)[:, None] * (g - m)


def batch_ht_mean(
    g: np.ndarray,
    m: np.ndarray,
    p: np.ndarray,
    z: np.ndarray,
    n: int | None = None,
) -> np.ndarray:
    """Mean of HT estimates over a predetermined start count N."""
    ghat = ht_estimate(g, m, p, z)
    denom = int(ghat.shape[0] if n is None else n)
    if denom <= 0:
        raise ValueError("N must be positive")
    return ghat.sum(axis=0) / denom


def two_checkpoint_ht_estimate(
    g: np.ndarray,
    m_first: np.ndarray,
    m_second: np.ndarray,
    p_first: np.ndarray,
    p_second: np.ndarray,
    z_first: np.ndarray,
    z_second: np.ndarray,
) -> np.ndarray:
    """Nested CV/HT estimate for decisions at two prefix checkpoints.

    ``m_second`` is used only after surviving the first decision.  The second
    draw can depend on the longer prefix; it must be made before the unseen
    suffix.  Natural finishes use p=z=1 at the checkpoint they skip.
    """
    g = _as_2d(g)
    first = np.broadcast_to(_as_2d(m_first), g.shape)
    second = np.broadcast_to(_as_2d(m_second), g.shape)
    p1 = np.asarray(p_first, dtype=np.float64).reshape(-1)
    p2 = np.asarray(p_second, dtype=np.float64).reshape(-1)
    z1 = np.asarray(z_first, dtype=np.float64).reshape(-1)
    z2 = np.asarray(z_second, dtype=np.float64).reshape(-1)
    if any(x.shape[0] != g.shape[0] for x in (p1, p2, z1, z2)):
        raise ValueError("two-checkpoint batch dimensions do not match")
    if (np.any(~np.isfinite(p1)) or np.any(~np.isfinite(p2))
            or np.any((p1 <= 0) | (p1 > 1)) or np.any((p2 <= 0) | (p2 > 1))):
        raise ValueError("checkpoint probabilities must be in (0, 1]")
    if (np.any(~np.isin(z1, (0.0, 1.0))) or np.any(~np.isin(z2, (0.0, 1.0)))
            or np.any(z2 > z1)):
        raise ValueError("second selection must be nested inside the first")
    return first + (z1 / p1)[:, None] * (second - first) + (
        (z2 / (p1 * p2))[:, None] * (g - second)
    )


def prediction_correction(u: np.ndarray, f: np.ndarray, z: np.ndarray, p: np.ndarray, n: int) -> np.ndarray:
    """`(U/N) sum_i (1 - Z_i/p_i) f_i`. Includes completed starts."""
    u = np.asarray(u, dtype=np.float64)
    f = _as_2d(f)
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    if n <= 0:
        raise ValueError("N must be positive")
    if u.ndim != 2:
        raise ValueError(f"U must be (d, k), got {u.shape}")
    if f.shape[1] != u.shape[1]:
        raise ValueError(f"f dim {f.shape[1]} != U k {u.shape[1]}")
    if f.shape[0] != z.shape[0] or z.shape[0] != p.shape[0]:
        raise ValueError("f, z, p batch dimensions do not match")
    if np.any(p <= 0.0):
        raise ValueError("p must be > 0")
    weights = 1.0 - z / p
    acc = weights @ f
    return (u @ acc) / n


def dual_stream_mean(
    g_completed: np.ndarray,
    p_completed: np.ndarray,
    u: np.ndarray,
    f_all: np.ndarray,
    z_all: np.ndarray,
    p_all: np.ndarray,
    n: int,
) -> np.ndarray:
    """`(1/N) sum_{Z=1} G/p + (U/N) sum (1-Z/p) f`.

    ``g_completed`` / ``p_completed`` contain only starts with Z=1.
    Stopped starts contribute only through the prediction term.
    """
    if n <= 0:
        raise ValueError("N must be positive")
    real = np.zeros(u.shape[0], dtype=np.float64)
    if g_completed is not None and np.size(g_completed):
        g_c = _as_2d(g_completed)
        p_c = np.asarray(p_completed, dtype=np.float64).reshape(-1)
        if g_c.shape[0] != p_c.shape[0]:
            raise ValueError("completed g/p dimensions do not match")
        if np.any(p_c <= 0.0):
            raise ValueError("p must be > 0")
        real = (g_c / p_c[:, None]).sum(axis=0) / n
    pred = prediction_correction(u, f_all, z_all, p_all, n)
    return real + pred


def optimizer_grad_from_ghat(ghat: np.ndarray) -> np.ndarray:
    """Standard descent `.grad` is the negative ascent estimate."""
    return -np.asarray(ghat, dtype=np.float64)


def ht_variance_trace(g: np.ndarray, m: np.ndarray, p: np.ndarray) -> tuple[float, float, float]:
    """Prop. 2 check: E[||Ghat-EG||^2] = tr Cov(G) + E[(1/p-1) ||G-m||^2].

    Returns (full_var, extra, total) using the empirical distribution of rows.
    """
    g = _as_2d(g)
    m = np.broadcast_to(_as_2d(m), g.shape)
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    if np.any(p <= 0.0) or np.any(p > 1.0):
        raise ValueError("p must be in (0, 1]")
    centered = g - g.mean(axis=0, keepdims=True)
    full_var = float(np.sum(centered * centered) / g.shape[0])
    residual = np.sum((g - m) ** 2, axis=1)
    extra = float(np.mean((1.0 / p - 1.0) * residual))
    return full_var, extra, full_var + extra
