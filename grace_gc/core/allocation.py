"""Neyman continuation allocation with p_min clip."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AllocationResult:
    p: np.ndarray
    lam: float
    budget_target: float
    budget_expected: float
    budget_deviation: float
    n_eligible: int


def _validate_probs(p_min: float, beta: float) -> None:
    if p_min <= 0.0 or p_min > 1.0:
        raise ValueError(f"p_min must be in (0, 1], got {p_min}")
    if beta <= 0.0 or beta > 1.0:
        raise ValueError(f"beta must be in (0, 1], got {beta}")


def _clipped_p(risk: np.ndarray, cost: np.ndarray, lam: float, p_min: float) -> np.ndarray:
    # Zero incremental cost: continue for free if there is residual risk.
    p = np.empty_like(risk)
    zero_c = cost <= 0.0
    pos = ~zero_c
    p[zero_c] = np.where(risk[zero_c] > 0.0, 1.0, p_min)
    if np.any(pos):
        raw = np.sqrt(np.maximum(risk[pos], 0.0) / (lam * cost[pos]))
        p[pos] = np.clip(raw, p_min, 1.0)
    return p


def allocate_continuation(
    risk: np.ndarray,
    cost: np.ndarray,
    beta: float,
    p_min: float,
    finished: np.ndarray | None = None,
    iters: int = 20,
    uniform_shrink: float = 0.0,
) -> AllocationResult:
    """Solve λ so that E[p c] / E[c] ≈ β on unfinished prefixes.

    Natural finishes are excluded and forced to p=1. Empty eligible sets
    return p=1 everywhere. Optional shrinkage mixes with a uniform probability
    at the same achieved expected cost, preserving solver budget deviations.
    Lambda describes the original Neyman allocation before this mixture.
    """
    _validate_probs(p_min, beta)
    uniform_shrink = float(uniform_shrink)
    if not np.isfinite(uniform_shrink) or not 0.0 <= uniform_shrink <= 1.0:
        raise ValueError("uniform_shrink must be finite and in [0, 1]")
    risk = np.asarray(risk, dtype=np.float64).reshape(-1)
    cost = np.asarray(cost, dtype=np.float64).reshape(-1)
    if risk.shape[0] != cost.shape[0]:
        raise ValueError("risk and cost dimensions do not match")
    if np.any(~np.isfinite(risk)) or np.any(~np.isfinite(cost)):
        raise ValueError("risk/cost must be finite")
    n = risk.shape[0]
    if finished is None:
        finished = np.zeros(n, dtype=bool)
    else:
        finished = np.asarray(finished, dtype=bool).reshape(-1)
        if finished.shape[0] != n:
            raise ValueError("finished mask dimension does not match")

    p = np.ones(n, dtype=np.float64)
    eligible = ~finished
    if not np.any(eligible):
        return AllocationResult(
            p=p,
            lam=float("inf"),
            budget_target=0.0,
            budget_expected=0.0,
            budget_deviation=0.0,
            n_eligible=0,
        )

    r_raw = np.maximum(risk[eligible], 0.0)
    c = cost[eligible]
    # Solve on r / mean(r) so a global risk scale does not move p.
    scale = float(np.mean(r_raw))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    r = r_raw / scale
    c_sum = float(np.sum(np.maximum(c, 0.0)))
    target = beta * c_sum

    def expected_cost(lam: float) -> float:
        return float(np.sum(_clipped_p(r, c, lam, p_min) * np.maximum(c, 0.0)))

    # λ → 0 drives p → 1; λ → ∞ drives p → p_min (or 1 if c=0 and r>0).
    lo, hi = 1e-18, 1e18
    if expected_cost(lo) < target:
        lam = lo
    elif expected_cost(hi) > target:
        lam = hi
    else:
        lam = 1.0
        for _ in range(iters):
            mid = np.sqrt(lo * hi)
            if expected_cost(mid) > target:
                lo = mid
            else:
                hi = mid
            lam = np.sqrt(lo * hi)

    p_elig = _clipped_p(r, c, lam, p_min)
    if uniform_shrink:
        # Matching the achieved cost also handles clipping, all-zero risks and
        # finite bisection error without silently changing the budget.
        uniform = float(np.dot(p_elig, np.maximum(c, 0.0)) / c_sum) if c_sum > 0.0 else 1.0
        uniform = float(np.clip(uniform, p_min, 1.0))
        p_elig = (1.0 - uniform_shrink) * p_elig + uniform_shrink * uniform
    p[eligible] = p_elig
    expected = float(np.sum(p_elig * np.maximum(c, 0.0)))
    deviation = expected - target
    return AllocationResult(
        p=p,
        lam=float(lam * scale),
        budget_target=target,
        budget_expected=expected,
        budget_deviation=deviation,
        n_eligible=int(np.sum(eligible)),
    )


def sample_z(p: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Continuation Z requires p in (0, 1]. Audit draws may use IsolatedRNG.bernoulli with 0."""
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    if np.any(p <= 0.0) or np.any(p > 1.0):
        raise ValueError("p must be in (0, 1]")
    return (rng.random(p.shape[0]) < p).astype(np.float64)
