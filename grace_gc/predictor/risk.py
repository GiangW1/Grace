"""Full-space residual labels and risk-head NLL."""

from __future__ import annotations

import numpy as np


def full_space_residual(g: np.ndarray, f: np.ndarray, u: np.ndarray, block_size: int = 65536) -> np.ndarray:
    """e = ||G - U f||^2. Equivalent to the Gram expansion when all terms are finite."""
    g = np.asarray(g, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    if g.ndim == 1:
        g = g[None, :]
        f = f[None, :]
        squeeze = True
    else:
        squeeze = False
    if g.shape[1] != u.shape[0] or f.shape[1] != u.shape[1]:
        raise ValueError("G/f/U dimensions do not match")
    if g.shape[0] != f.shape[0] or int(block_size) <= 0:
        raise ValueError("residual batch size or block size is invalid")
    e = np.zeros(g.shape[0], dtype=np.float64)
    # Direct differences remain accurate even if G and Uf nearly cancel.
    for start in range(0, g.shape[1], int(block_size)):
        stop = start + int(block_size)
        residual = g[:, start:stop] - f @ u[start:stop].T
        e += np.einsum("ij,ij->i", residual, residual)
    if not np.all(np.isfinite(e)):
        raise ValueError("full-space residual is not finite")
    if squeeze:
        return e.reshape(())
    return e


def risk_nll(e: np.ndarray, r_hat: np.ndarray) -> np.ndarray:
    """e / r_hat + log r_hat. Conditional optimum is E[e|h]."""
    e = np.asarray(e, dtype=np.float64)
    r = np.asarray(r_hat, dtype=np.float64)
    if np.any(r <= 0.0):
        raise ValueError("r_hat must be > 0")
    return e / r + np.log(r)
