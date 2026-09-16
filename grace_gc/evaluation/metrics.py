"""Eval metrics. Small n and wide intervals are valid outputs."""

from __future__ import annotations

import math

import numpy as np


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k: 1 - C(n-c, k) / C(n, k)."""
    if n <= 0 or k <= 0:
        raise ValueError("n and k must be positive")
    if c < 0 or c > n or k > n:
        raise ValueError("invalid c or k for given n")
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def avg_at_k(rewards: list[float]) -> float:
    if not rewards:
        return float("nan")
    return float(np.mean(rewards))


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n <= 0:
        return float("nan"), float("nan"), float("nan")
    phat = successes / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    return phat, max(0.0, center - half), min(1.0, center + half)


def time_to_target(curve: list[tuple[float, float]], target: float, consecutive: int = 2) -> float | None:
    """Return compute at which `consecutive` points stay at/above target."""
    hit = 0
    for compute, value in curve:
        if value >= target:
            hit += 1
            if hit >= consecutive:
                return float(compute)
        else:
            hit = 0
    return None


def hvd(hard_first_success: dict[str, float | None]) -> dict[str, float | None]:
    """Hard-problem first verified success time. Missing stays None."""
    values = [v for v in hard_first_success.values() if v is not None]
    return {
        "n_hard": len(hard_first_success),
        "n_solved": len(values),
        "mean_success_compute": None if not values else float(np.mean(values)),
        "per_problem": dict(hard_first_success),
    }
