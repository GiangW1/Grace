"""Low-rank basis from recent residual gradients."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_SINGULAR_REL = 1e-6


@dataclass
class Basis:
    u: np.ndarray
    basis_id: int
    k: int
    rank: int = 0

    def project(self, g: np.ndarray) -> np.ndarray:
        g = np.asarray(g, dtype=np.float64)
        if g.ndim == 1:
            return self.u.T @ g
        return g @ self.u


def effective_rank(singular_values: np.ndarray, k: int) -> int:
    s = np.asarray(singular_values, dtype=np.float64).reshape(-1)
    if s.size == 0:
        return 0
    peak = float(np.max(s))
    if not np.isfinite(peak) or peak <= 0.0:
        return 0
    return int(min(int(k), int(np.sum(s >= peak * _SINGULAR_REL))))


def pad_basis_cols(u: np.ndarray, k: int) -> np.ndarray:
    u = np.asarray(u, dtype=np.float64)
    if u.ndim != 2:
        raise ValueError("U must be 2d")
    k = int(k)
    if k <= 0:
        raise ValueError("k must be positive")
    if u.shape[1] >= k:
        return u[:, :k]
    pad = np.zeros((u.shape[0], k - u.shape[1]), dtype=np.float64)
    return np.concatenate([u, pad], axis=1)


def randomized_svd(matrix: np.ndarray, k: int, oversample: int = 16, seed: int = 0) -> tuple[np.ndarray, int]:
    """Return padded top-k left singular vectors (d, k) and the effective rank."""
    x = np.asarray(matrix, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("matrix must be 2d")
    n, d = x.shape
    k_req = int(k)
    k = int(min(k_req, n, d))
    if k_req <= 0 or k <= 0:
        raise ValueError("k must be positive")
    rng = np.random.default_rng(seed)
    p = min(d, k + oversample)
    omega = rng.normal(size=(d, p))
    y = x @ omega
    q, _ = np.linalg.qr(y, mode="reduced")
    b = q.T @ x
    _u_small, s, vt = np.linalg.svd(b, full_matrices=False)
    rank = effective_rank(s, k)
    if rank <= 0:
        return np.zeros((d, k_req), dtype=np.float64), 0
    u = vt[:rank].T
    return pad_basis_cols(u, k_req), rank


def should_refresh_basis(
    n_grads: int,
    k: int,
    step: int,
    basis_id: int,
    refresh_every: int,
) -> bool:
    """Form U as soon as k grads exist; then every refresh_every actor steps.

    Identity init is a placeholder, not a residual-risk basis. Residual-risk
    Neyman allocation must wait for a real U and heads trained on that U.
    """
    if int(n_grads) < int(k) or int(k) <= 0:
        return False
    if int(basis_id) == 0:
        return True
    every = int(refresh_every)
    return every > 0 and (int(step) + 1) % every == 0


def _stable_center(rows: np.ndarray) -> np.ndarray:
    """Subtract the first row, then mean-center. Avoids float mean residual on copies."""
    x = np.asarray(rows, dtype=np.float64)
    if x.ndim == 1:
        return np.zeros_like(x)
    diffs = x - x[0]
    return diffs - diffs.mean(axis=0, keepdims=True)


def refresh_basis(
    grads: np.ndarray,
    k: int,
    basis_id: int,
    oversample: int = 16,
    seed: int = 0,
    problem_ids: list[str] | None = None,
) -> Basis | None:
    g = np.asarray(grads, dtype=np.float64)
    if g.ndim != 2:
        raise ValueError("grads must be (n, d)")
    if problem_ids is None:
        centered = _stable_center(g)
    else:
        pids = np.asarray(problem_ids)
        if pids.shape[0] != g.shape[0]:
            raise ValueError("problem_ids length does not match grads")
        centered = np.zeros_like(g)
        for pid in np.unique(pids):
            mask = pids == pid
            centered[mask] = _stable_center(g[mask])
    u, rank = randomized_svd(centered, k=k, oversample=oversample, seed=seed)
    if rank <= 0:
        return None
    return Basis(u=u, basis_id=int(basis_id), k=int(u.shape[1]), rank=int(rank))


def align_basis(new_u: np.ndarray, old_u: np.ndarray) -> np.ndarray:
    """Put new columns into old slots by |inner product| and matching sign.

    Same subspace, opposite sign or swapped columns would otherwise look like
    a new coordinate task to the heads. Zero columns stay unused.
    """
    new_u = np.asarray(new_u, dtype=np.float64)
    old_u = np.asarray(old_u, dtype=np.float64)
    if new_u.shape != old_u.shape:
        raise ValueError(f"align_basis shapes {new_u.shape} != {old_u.shape}")
    k = new_u.shape[1]
    new_live = np.linalg.norm(new_u, axis=0) > 1e-12
    old_live = np.linalg.norm(old_u, axis=0) > 1e-12
    dots = old_u.T @ new_u
    scores = np.abs(dots)
    scores[~old_live, :] = -1.0
    scores[:, ~new_live] = -1.0
    out = np.zeros_like(new_u)
    used_old = np.zeros(k, dtype=bool)
    used_new = np.zeros(k, dtype=bool)
    for _ in range(int(min(np.sum(old_live), np.sum(new_live)))):
        i, j = np.unravel_index(int(np.argmax(scores)), scores.shape)
        if scores[i, j] < 0.0:
            break
        sign = 1.0 if dots[i, j] >= 0.0 else -1.0
        out[:, i] = sign * new_u[:, j]
        used_old[i] = True
        used_new[j] = True
        scores[i, :] = -1.0
        scores[:, j] = -1.0
    leftover_new = np.where(new_live & ~used_new)[0]
    leftover_old = np.where(~used_old)[0]
    for slot, j in zip(leftover_old, leftover_new):
        out[:, slot] = new_u[:, j]
    return out


def reproject(
    old_coords: np.ndarray,
    old_u: np.ndarray,
    new_u: np.ndarray,
    g: np.ndarray | None = None,
) -> np.ndarray:
    """Map labels onto a new basis from full-space G when available."""
    new_u = np.asarray(new_u, dtype=np.float64)
    if g is not None:
        g = np.asarray(g, dtype=np.float64)
        if g.ndim == 1:
            return new_u.T @ g
        return g @ new_u
    approx = np.asarray(old_u, dtype=np.float64) @ np.asarray(old_coords, dtype=np.float64).T
    return (new_u.T @ approx).T
