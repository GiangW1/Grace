"""Exact suffix posterior and fixed-start aggregation (NumPy reference)."""

from __future__ import annotations

import numpy as np


def suffix_posterior(log_likelihoods):
    """Uniform-donor posterior, using summed, untempered sequence log-probs."""
    values = np.asarray(log_likelihoods, dtype=np.float64)
    if values.ndim != 1 or not len(values) or np.isnan(values).any() or np.isposinf(values).any():
        raise ValueError("suffix log likelihoods must be a nonempty vector without NaN/+inf")
    maximum = values.max()
    if not np.isfinite(maximum):
        raise ValueError("sampled suffix has zero probability under every prefix")
    weights = np.exp(values - maximum)
    return weights / weights.sum()


def prefix_coefficients(alpha, advantages, donor: int, strength: float = 1.0):
    """GD + strength*C; strength must be fixed before the current suffix draw."""
    alpha, advantages = np.asarray(alpha, dtype=float), np.asarray(advantages, dtype=float)
    if (alpha.ndim != 1 or alpha.shape != advantages.shape or not len(alpha)
            or not np.isfinite(alpha).all() or not np.isfinite(advantages).all()
            or (alpha < 0).any() or not np.isclose(alpha.sum(), 1.0)
            or not 0 <= donor < len(alpha) or not np.isfinite(strength)):
        raise ValueError("invalid posterior, advantages, donor or strength")
    coefficients = strength * alpha * advantages
    coefficients[donor] += (1.0 - strength) * advantages[donor]
    return coefficients


def transport_gradient(prefix_grads, donor_full_grad, alpha, advantages, donor, strength=1.0):
    """All vectors are ascent directions; donor_full_grad already includes A_D."""
    prefix_grads = np.asarray(prefix_grads, dtype=np.float64)
    donor_full_grad = np.asarray(donor_full_grad, dtype=np.float64)
    coefficients = prefix_coefficients(alpha, advantages, donor, strength)
    if prefix_grads.shape != (len(coefficients), *donor_full_grad.shape) or donor_full_grad.ndim != 1:
        raise ValueError("prefix and donor gradient layouts disagree")
    coefficients[donor] -= float(advantages[donor])
    result = donor_full_grad + coefficients @ prefix_grads
    if not np.isfinite(result).all():
        raise ValueError("nonfinite transport gradient")
    return result


def aggregate_groups(groups, early_grads, n_start: int):
    """groups=(original live group size, group mean); early grads are singletons."""
    groups, early_grads = list(groups), list(early_grads)
    count = sum(int(size) for size, _ in groups) + len(early_grads)
    if n_start < 1 or count != n_start or any(size < 1 for size, _ in groups):
        raise ValueError("group sizes plus natural completions must equal fixed n_start")
    vectors = [np.asarray(g, dtype=np.float64) * size for size, g in groups]
    vectors += [np.asarray(g, dtype=np.float64) for g in early_grads]
    if any(v.ndim != 1 or v.shape != vectors[0].shape or not np.isfinite(v).all() for v in vectors):
        raise ValueError("aggregate gradient layout or values are invalid")
    return sum(vectors) / n_start
