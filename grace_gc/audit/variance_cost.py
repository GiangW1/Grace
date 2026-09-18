"""Numerical variance × cost at the actual p. Closed forms are display-only."""

from __future__ import annotations

import numpy as np

from grace_gc.core.estimator import ht_estimate


def variance_times_cost(
    g: np.ndarray,
    m: np.ndarray,
    p: np.ndarray,
    prefix_cost: float,
    suffix_cost: np.ndarray,
) -> dict:
    g = np.asarray(g, dtype=np.float64)
    m = np.asarray(m, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    suffix_cost = np.asarray(suffix_cost, dtype=np.float64).reshape(-1)
    prefix = np.asarray(prefix_cost, dtype=np.float64).reshape(-1)
    if prefix.size == 1:
        prefix = np.full(g.shape[0], float(prefix[0]))
    if g.shape[0] != p.shape[0]:
        raise ValueError("G and p dimensions do not match")
    if prefix.shape[0] != g.shape[0] or suffix_cost.shape[0] != g.shape[0]:
        raise ValueError("prefix/suffix cost dimensions do not match G")
    mean_g = g.mean(axis=0)
    full_var = float(np.mean(np.sum((g - mean_g) ** 2, axis=1)))
    full_cost = float(np.mean(prefix + suffix_cost))
    extras = []
    for i in range(g.shape[0]):
        z1 = ht_estimate(g[i], m[i], p[i], 1.0)
        z0 = ht_estimate(g[i], m[i], p[i], 0.0)
        # Monte Carlo over Z using the actual p; G is the observed complete label.
        mean = p[i] * z1 + (1.0 - p[i]) * z0
        var = p[i] * np.sum((z1 - mean) ** 2) + (1.0 - p[i]) * np.sum((z0 - mean) ** 2)
        extras.append(var)
    extra_var = float(np.mean(extras))
    new_var = full_var + extra_var
    new_cost = float(np.mean(prefix + p * suffix_cost))
    return {
        "full_var": full_var,
        "extra_var": extra_var,
        "full_cost": full_cost,
        "full_product": full_var * full_cost,
        "actual_p_var": new_var,
        "actual_p_cost": new_cost,
        "actual_p_product": new_var * new_cost,
        "ratio": None if full_var * full_cost == 0 else (new_var * new_cost) / (full_var * full_cost),
    }


def fit_eval_split(
    n: int,
    rng: np.random.Generator,
    keys: list | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Split bundles. Same path (or problem) stays on one side."""
    if n < 2:
        idx = np.arange(n)
        return idx, idx
    if keys is None:
        perm = rng.permutation(n)
        mid = n // 2
        return perm[:mid], perm[mid:]
    if len(keys) != n:
        raise ValueError("fit/eval keys must match bundle count")
    unique: list = []
    seen: set = set()
    for key in keys:
        if key not in seen:
            seen.add(key)
            unique.append(key)
    if len(unique) < 2:
        idx = np.arange(n)
        return idx, idx
    order = [unique[i] for i in rng.permutation(len(unique))]
    mid = max(1, len(order) // 2)
    if mid == len(order):
        mid = len(order) - 1
    fit_keys = set(order[:mid])
    eval_keys = set(order[mid:])
    fit = np.array([i for i, key in enumerate(keys) if key in fit_keys], dtype=np.int64)
    ev = np.array([i for i, key in enumerate(keys) if key in eval_keys], dtype=np.int64)
    if fit.size == 0 or ev.size == 0:
        idx = np.arange(n)
        return idx, idx
    return fit, ev


def match_observed_cost(risk, cost, observed_cost, target, p_min, weights, finished):
    """Offline allocation diagnostic; match an observed suffix budget by bisection."""
    risk = np.maximum(np.asarray(risk, dtype=float), 1e-12)
    risk = risk / max(float(risk.mean()), 1e-12)
    cost = np.maximum(np.asarray(cost, dtype=float), 1e-12)
    observed_cost = np.asarray(observed_cost, dtype=float)
    weights = np.asarray(weights, dtype=float)
    finished = np.asarray(finished, dtype=bool)
    lo, hi = 1e-30, 1e30
    for _ in range(80):
        mid = np.sqrt(lo * hi)
        p = np.where(finished, 1., np.clip(np.sqrt(risk / (mid * cost)), p_min, 1.))
        if float(np.sum(weights * p * observed_cost)) > target:
            lo = mid
        else:
            hi = mid
    return np.where(finished, 1., np.clip(np.sqrt(risk / (np.sqrt(lo * hi) * cost)), p_min, 1.))
