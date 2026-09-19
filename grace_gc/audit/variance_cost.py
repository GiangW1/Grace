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


def cluster_bootstrap_variance_cost(g, m, p, prefix_cost, suffix_cost, cluster_ids,
                                    *, samples=1000, seed=17) -> dict:
    """Conditional percentile interval, resampling whole problems at fixed m/p.

    Small per-problem sufficient statistics preserve every nested path, t and
    continuation. No refitting/reallocation occurs inside this supplement.
    """
    g, m = np.asarray(g, dtype=float), np.asarray(m, dtype=float)
    p, suffix = np.asarray(p, dtype=float).reshape(-1), np.asarray(suffix_cost, dtype=float).reshape(-1)
    n = len(g)
    prefix = np.broadcast_to(np.asarray(prefix_cost, dtype=float), (n,))
    ids = list(dict.fromkeys(cluster_ids))
    if g.ndim != 2 or m.shape != g.shape or len(p) != n or len(suffix) != n or len(cluster_ids) != n:
        raise ValueError("variance-cost bootstrap row dimensions do not match")
    if int(samples) != samples or samples < 0:
        raise ValueError("bootstrap samples must be a nonnegative integer")
    result = {"point_ratio": None, "ratio_ci95": None, "n_rows": n, "n_clusters": len(ids),
              "cluster_unit": "problem_id", "samples_requested": int(samples), "seed": int(seed),
              "replicates_drawn": 0, "valid_replicates": 0, "undefined_replicates": 0,
              "cluster_sufficient_statistics": [], "unavailable_reason": None,
              "scope": "95% percentile problem-cluster bootstrap of the pooled report-row variance-times-response-token proxy. Assumes sampled problems are exchangeable; all paths/t/continuations of a problem move together. Conditional on frozen actor, predictor, p, fit/report split and observed suffixes; no refitting or reallocation. Not training-seed uncertainty, a fixed-prompt training-batch variance, full-pipeline uncertainty, or GPU savings. Undefined zero-variance/cost draws are counted; finite-draw percentiles are conditional on a defined ratio and can be unstable for few problems."}
    if not n or not all(np.all(np.isfinite(x)) for x in (g, m, p, prefix, suffix)):
        result["unavailable_reason"] = "empty_or_nonfinite_report_rows"
        return result
    if np.any(p <= 0) or np.any(p > 1):
        raise ValueError("bootstrap continuation probabilities must be in (0,1]")
    origin = g[0]
    clusters = []
    for pid in ids:
        ix = np.flatnonzero(np.asarray(cluster_ids) == pid)
        differences, residual = g[ix] - g[ix[0]], g[ix] - m[ix]
        local_mean = differences.mean(axis=0)
        centered = differences - local_mean
        clusters.append({"problem_id": pid, "n_rows": len(ix),
                         "mean_centered_g": (g[ix[0]] - origin + local_mean).tolist(),
                         "within_gradient_m2": float(np.einsum("ij,ij->", centered, centered)),
                         "sum_extra_variance": float(np.dot(1. / p[ix] - 1., np.einsum("ij,ij->i", residual, residual))),
                         "sum_full_cost": float(np.sum(prefix[ix] + suffix[ix])),
                         "sum_actual_cost": float(np.sum(prefix[ix] + p[ix] * suffix[ix]))})
    result["cluster_sufficient_statistics"] = clusters
    counts = np.array([row["n_rows"] for row in clusters])
    means = np.array([row["mean_centered_g"] for row in clusters])
    within, extras, full_costs, actual_costs = (np.array([row[key] for row in clusters]) for key in
        ("within_gradient_m2", "sum_extra_variance", "sum_full_cost", "sum_actual_cost"))
    distances = np.zeros((len(ids), len(ids)))
    for i in range(len(ids)):
        difference = means[i + 1:] - means[i]
        distances[i, i + 1:] = np.einsum("ij,ij->i", difference, difference)
    distances += distances.T

    def ratios(multiplicity):
        total = multiplicity @ counts
        weights = multiplicity * counts
        # Within/between decomposition avoids subtracting two large moments,
        # and keeps identical-gradient cluster draws exactly zero-variance.
        variance = ((multiplicity @ within) / total +
                    np.einsum("bi,ij,bj->b", weights, distances, weights) / (2. * total ** 2))
        denominator = variance * (multiplicity @ full_costs) / total
        numerator = (variance + (multiplicity @ extras) / total) * (multiplicity @ actual_costs) / total
        return np.divide(numerator, denominator, out=np.full(len(total), np.nan), where=denominator > 0)

    point = ratios(np.ones((1, len(ids))))[0]
    result["point_ratio"] = float(point) if np.isfinite(point) else None
    if len(ids) < 2 or samples == 0:
        result["unavailable_reason"] = "fewer_than_two_problem_clusters" if len(ids) < 2 else "resampling_disabled"
        return result
    draws = np.random.default_rng(seed).integers(0, len(ids), size=(int(samples), len(ids)))
    multiplicity = np.array([np.bincount(draw, minlength=len(ids)) for draw in draws])
    values = ratios(multiplicity)
    valid = values[np.isfinite(values)]
    result.update(replicates_drawn=int(samples), valid_replicates=len(valid),
                  undefined_replicates=int(samples) - len(valid))
    if len(valid):
        result["ratio_ci95"] = np.quantile(valid, [.025, .975]).tolist()
    else:
        result["unavailable_reason"] = "no_defined_bootstrap_ratios"
    return result


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
