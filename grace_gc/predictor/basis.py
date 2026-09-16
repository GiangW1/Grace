"""Low-rank basis from recent residual gradients."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Basis:
    u: np.ndarray
    basis_id: int
    k: int

    def project(self, g: np.ndarray) -> np.ndarray:
        g = np.asarray(g, dtype=np.float64)
        if g.ndim == 1:
            return self.u.T @ g
        return g @ self.u


def randomized_svd(matrix: np.ndarray, k: int, oversample: int = 16, seed: int = 0) -> np.ndarray:
    """Return top-k left singular vectors as columns of U (d, k)."""
    x = np.asarray(matrix, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("matrix must be 2d")
    n, d = x.shape
    k = int(min(k, n, d))
    if k <= 0:
        raise ValueError("k must be positive")
    rng = np.random.default_rng(seed)
    p = min(d, k + oversample)
    omega = rng.normal(size=(d, p))
    y = x @ omega
    q, _ = np.linalg.qr(y, mode="reduced")
    b = q.T @ x
    _u_small, _s, vt = np.linalg.svd(b, full_matrices=False)
    u = vt[:k].T
    return u.astype(np.float64)


def refresh_basis(
    grads: np.ndarray,
    k: int,
    basis_id: int,
    oversample: int = 16,
    seed: int = 0,
    problem_ids: list[str] | None = None,
) -> Basis:
    g = np.asarray(grads, dtype=np.float64)
    if g.ndim != 2:
        raise ValueError("grads must be (n, d)")
    if problem_ids is None:
        centered = g - g.mean(axis=0, keepdims=True)
    else:
        pids = np.asarray(problem_ids)
        if pids.shape[0] != g.shape[0]:
            raise ValueError("problem_ids length does not match grads")
        centered = np.zeros_like(g)
        for pid in np.unique(pids):
            mask = pids == pid
            centered[mask] = g[mask] - g[mask].mean(axis=0, keepdims=True)
    u = randomized_svd(centered, k=k, oversample=oversample, seed=seed)
    return Basis(u=u, basis_id=int(basis_id), k=int(u.shape[1]))


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
