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
    warmup_steps: int | None = None,
    refresh_after_warmup: bool = False,
) -> bool:
    """Form U as soon as k grads exist; then every refresh_every actor steps.

    Identity init is a placeholder, not a residual-risk basis. Residual-risk
    Neyman allocation must wait for a real U and heads trained on that U.
    """
    if int(n_grads) < int(k) or int(k) <= 0:
        return False
    if int(basis_id) == 0:
        return True
    if refresh_after_warmup and warmup_steps is not None and int(step) + 1 == int(warmup_steps):
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
    weights: np.ndarray | None = None,
    solver: str = "auto",
    center: str = "problem",
) -> Basis | None:
    g = np.asarray(grads, dtype=np.float64)
    if g.ndim != 2:
        raise ValueError("grads must be (n, d)")
    if center not in {"problem", "global", "none"}:
        raise ValueError("basis center must be problem, global or none")
    w = np.ones(g.shape[0]) if weights is None else np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.shape[0] != g.shape[0] or np.any(w < 0) or not np.all(np.isfinite(w)):
        raise ValueError("basis weights must be finite, nonnegative and match grads")
    if not g.shape[0] or not np.any(w > 0):
        return None
    if center == "none":
        centered = g.copy()
    elif problem_ids is None or center == "global":
        diffs = g - g[0]
        centered = diffs - np.average(diffs, axis=0, weights=w)
    else:
        pids = np.asarray(problem_ids)
        if pids.shape[0] != g.shape[0]:
            raise ValueError("problem_ids length does not match grads")
        centered = np.zeros_like(g)
        for pid in np.unique(pids):
            mask = pids == pid
            if np.any(w[mask] > 0):
                diffs = g[mask] - g[mask][0]
                centered[mask] = diffs - np.average(diffs, axis=0, weights=w[mask])
    centered *= np.sqrt(w)[:, None]
    if solver == "randomized":
        u, rank = randomized_svd(centered, k=k, oversample=oversample, seed=seed)
    elif solver in {"auto", "gram", "svd"}:
        u, rank = exact_basis(centered, k, use_gram=solver == "gram" or (solver == "auto" and g.shape[0] < g.shape[1]))
    else:
        raise ValueError("basis solver must be auto, gram, svd or randomized")
    if rank <= 0:
        return None
    return Basis(u=u, basis_id=int(basis_id), k=int(u.shape[1]), rank=int(rank))


def exact_basis(matrix: np.ndarray, k: int, use_gram: bool = True) -> tuple[np.ndarray, int]:
    """Top right singular vectors; small-n Gram avoids a D×(k+oversample) array."""
    x = np.asarray(matrix, dtype=np.float64)
    if x.ndim != 2 or int(k) <= 0:
        raise ValueError("matrix must be 2d and k must be positive")
    if not np.all(np.isfinite(x)):
        raise ValueError("basis gradients must be finite")
    if use_gram:
        eigenvalues, left = np.linalg.eigh(x @ x.T)
        order = np.argsort(eigenvalues)[::-1]
        singular = np.sqrt(np.maximum(eigenvalues[order], 0.0))
        rank = effective_rank(singular, min(int(k), *x.shape))
        if rank:
            u = x.T @ (left[:, order[:rank]] / singular[:rank])
            # Gram eigenvectors lose a little orthogonality near rank deficiency.
            u, _ = np.linalg.qr(u, mode="reduced")
        else:
            u = np.zeros((x.shape[1], 0))
    else:
        _left, singular, vt = np.linalg.svd(x, full_matrices=False)
        rank = effective_rank(singular, int(k))
        u = vt[:rank].T
    return pad_basis_cols(u, int(k)), rank


def crossfit_ridge_operator(features, problem_ids, weights=None, ridge_l2=1.0):
    """A @ G predicts G while excluding every label from each row's problem.

    Each fold fits weighted-mean ridge with unpenalized intercept, feature scales
    fit on that fold's training problems, and a small n_train dual system. A is
    independent of gradient labels; entire same-problem blocks are exactly zero.
    """
    x = np.asarray(features, dtype=np.float64)
    pids = np.asarray(problem_ids)
    if x.ndim != 2 or pids.shape != (len(x),):
        raise ValueError("crossfit features and problem ids must match")
    w = np.ones(len(x)) if weights is None else np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.shape != (len(x),) or np.any(w < 0) or not np.all(np.isfinite(w)) or not np.all(np.isfinite(x)):
        raise ValueError("crossfit features/weights must be finite with nonnegative weights")
    if not np.isfinite(ridge_l2) or ridge_l2 < 0:
        raise ValueError("crossfit ridge_l2 must be finite and nonnegative")
    operator = np.zeros((len(x), len(x)), dtype=np.float64)
    available = np.zeros(len(x), dtype=bool)
    for pid in np.unique(pids):
        hold = np.flatnonzero((pids == pid) & (w > 0))
        train = np.flatnonzero((pids != pid) & (w > 0))
        if not len(train) or not len(hold):
            continue
        tw = w[train] / w[train].sum()
        mean = np.average(x[train], axis=0, weights=tw)
        centered = x[train] - mean
        std = np.sqrt(np.average(centered ** 2, axis=0, weights=tw))
        std = np.where(std > 1e-6, std, 1.)
        xc = centered / std * np.sqrt(tw)[:, None]
        xh = (x[hold] - mean) / std
        gram = xc @ xc.T + float(ridge_l2) * np.eye(len(train))
        rhs = np.diag(np.sqrt(tw))
        solved = (np.linalg.solve(gram, rhs) if ridge_l2 > 0
                  else np.linalg.lstsq(gram, rhs, rcond=None)[0])
        smoother = (xh @ xc.T) @ solved
        # Apply the intercept to the original G, not a target-centered copy.
        operator[np.ix_(hold, train)] = smoother + (1. - smoother.sum(axis=1))[:, None] * tw
        available[hold] = True
    return operator, available


def refresh_predictable_basis(grads, features, problem_ids, k, basis_id,
                              weights=None, ridge_l2=1.0,
                              objective="prediction_energy") -> tuple[Basis | None, dict]:
    """Low-rank conditional-mean variant; caller supplies ONLY fit-side labels.

    M = A G is leave-one-problem-out ridge prediction, not in-sample fitted G.
    Maximize sum_i w_i ||U.T M_i||² subject to U.T U=I; no problem-mean removal.
    Explicit cross_moment instead uses positive eigenvectors of
    C = (G.T W M + M.T W G)/2, checking held-out target alignment rather than
    predicted energy alone. Its finite-sample positive spectrum can still be
    noise; it is not an independent evaluation of the selected basis.
    This is a predictive-signal surrogate, not proof of predictable energy on
    future policies. Hold/calibration labels must never enter this function.

    Stream M in dimension blocks: O(n²D + nDk) work outside ridge folds and
    O(n² + n*block + Dk) extra storage, without storing another full n-by-D M.
    """
    g = np.asarray(grads, dtype=np.float64)
    if g.ndim != 2 or int(k) <= 0:
        raise ValueError("grads must be (n,d) and k must be positive")
    if not all(np.all(np.isfinite(row)) for row in g):
        raise ValueError("basis gradients must be finite")
    if len(features) != len(g) or len(problem_ids) != len(g):
        raise ValueError("predictable basis rows do not match")
    if objective not in {"prediction_energy", "cross_moment"}:
        raise ValueError("predictable basis objective must be prediction_energy or cross_moment")
    w = np.ones(len(g)) if weights is None else np.asarray(weights, dtype=np.float64).reshape(-1)
    operator, available = crossfit_ridge_operator(features, problem_ids, w, ridge_l2)
    metrics = {"variant": "predictable_crossfit", "crossfit_problems": len(set(problem_ids)),
               "prediction_rows": int(available.sum()), "rank": 0,
               "objective": "uncentered weighted energy of leave-one-problem-out ridge predictions",
               "ridge_l2": float(ridge_l2), "gradient_labels": "historical, not recomputed under current policy"}
    if objective == "cross_moment":
        metrics.update(variant="predictable_crossfit_signal",
                       objective="positive spectrum of symmetric weighted held-out G/prediction cross moment")
    if not np.any(available):
        metrics["reason"] = "no_cross_problem_training_rows"
        return None, metrics
    if objective == "cross_moment":
        return _cross_moment_basis(g, operator, w * available, k, basis_id, metrics)
    weighted = operator * np.sqrt(w * available)[:, None]
    gram = np.zeros((len(g), len(g)), dtype=np.float64)
    block = 65536
    for start in range(0, g.shape[1], block):
        prediction = weighted @ g[:, start:start + block]
        gram += prediction @ prediction.T
    eigenvalues, left = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    singular = np.sqrt(np.maximum(eigenvalues[order], 0.))
    rank = effective_rank(singular, min(int(k), *g.shape))
    metrics["rank"] = rank
    metrics["predicted_energy"] = float(np.trace(gram))
    if not rank:
        metrics["reason"] = "zero_predicted_rank"
        return None, metrics
    projected = left[:, order[:rank]] / singular[:rank]
    u = np.empty((g.shape[1], rank), dtype=np.float64)
    for start in range(0, g.shape[1], block):
        prediction = weighted @ g[:, start:start + block]
        u[start:start + block] = prediction.T @ projected
    u, _ = np.linalg.qr(u, mode="reduced")
    metrics["reason"] = "formed"
    return Basis(pad_basis_cols(u, int(k)), int(basis_id), int(k), rank), metrics


def _cross_moment_basis(g, operator, weights, k, basis_id, metrics):
    """Exact small-n signed eigensystem of G.T B G; never construct D×D C.

    If G G.T = V diag(s²) V.T, E=G.T V/s is an orthonormal row-space basis
    and E.T C E = diag(s) V.T B V diag(s), B=(W A+A.T W)/2.
    Gradients and cross moments are uncentered. The relative spectral cutoff
    is numerical rank only, using the existing squared singular tolerance.
    """
    active = weights > 0
    gram = np.zeros((len(g), len(g)), dtype=np.float64)
    block = 65536
    for start in range(0, g.shape[1], block):
        # Excluded/expired rows must not affect even the numerical row space.
        rows = g[:, start:start + block] * active[:, None]
        gram += rows @ rows.T
    weighted = weights[:, None] * operator
    b = (weighted + weighted.T) / 2.
    target_energy = float(np.dot(weights, np.diag(gram)))
    predicted_energy = float(np.sum((operator @ gram) * weighted))
    cross_trace = float(np.sum(b * gram))
    metrics.update(target_energy=target_energy, predicted_energy=predicted_energy,
                   cross_moment_trace=cross_trace,
                   oof_residual_energy=max(0., target_energy + predicted_energy - 2. * cross_trace),
                   positive_eigenvalue_n=0, negative_eigenvalue_n=0,
                   cross_moment_eigenvalues=[])
    values, left = np.linalg.eigh(gram)
    live = values > max(0., float(values.max())) * _SINGULAR_REL ** 2
    if not np.any(live):
        metrics["reason"] = "zero_gradient_rank"
        return None, metrics
    root = np.sqrt(values[live])
    left = left[:, live]
    small = root[:, None] * (left.T @ b @ left) * root[None, :]
    signal, vectors = np.linalg.eigh((small + small.T) / 2.)
    tol = float(np.max(np.abs(signal))) * _SINGULAR_REL ** 2
    positive = np.flatnonzero(signal > tol)[::-1]
    metrics.update(cross_moment_eigenvalues=signal[::-1].tolist(),
                   positive_eigenvalue_n=len(positive),
                   negative_eigenvalue_n=int(np.count_nonzero(signal < -tol)))
    rank = min(int(k), len(positive))
    metrics["rank"] = rank
    if not rank:
        metrics["reason"] = "no_positive_cross_moment_rank"
        return None, metrics
    projected = (left / root) @ vectors[:, positive[:rank]]
    projected[~active] = 0.
    u = np.empty((g.shape[1], rank), dtype=np.float64)
    for start in range(0, g.shape[1], block):
        u[start:start + block] = g[:, start:start + block].T @ projected
    u, _ = np.linalg.qr(u, mode="reduced")
    metrics["reason"] = "formed"
    return Basis(pad_basis_cols(u, int(k)), int(basis_id), int(k), rank), metrics


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
