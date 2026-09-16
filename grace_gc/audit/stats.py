"""Prefix-audit statistics. Filters are analysis options, not run gates."""

from __future__ import annotations

import numpy as np


def rho_a(var_r_prefix: np.ndarray, var_r_prompt: float) -> float:
    if var_r_prompt <= 0.0:
        return float("nan")
    return float(np.mean(var_r_prefix) / var_r_prompt)


def rho_l(cond_var_g: np.ndarray, prompt_var_g: float) -> float:
    if prompt_var_g <= 0.0:
        return float("nan")
    return float(np.mean(cond_var_g) / prompt_var_g)


def gate_mask(var_r: np.ndarray, mean_energy: np.ndarray, undecided: float, kappa: float) -> np.ndarray:
    return (var_r > undecided) & (mean_energy >= kappa)


def wilson_q_inside(rewards, lo: float = 0.05, hi: float = 0.95) -> bool:
    """Paper §4.2: q(h) Wilson interval lies in (lo, hi)."""
    from grace_gc.evaluation.metrics import wilson_interval

    r = np.asarray(rewards, dtype=np.float64).reshape(-1)
    n = int(r.size)
    if n <= 0:
        return False
    _p, wlo, whi = wilson_interval(int(np.sum(r >= 0.5)), n)
    return bool(np.isfinite(wlo) and np.isfinite(whi) and wlo > lo and whi < hi)


def headline_mask(
    var_r: np.ndarray,
    mean_energy: np.ndarray,
    undecided: float,
    kappa: float,
    answer_emitted,
    rewards,
    wilson_lo: float = 0.05,
    wilson_hi: float = 0.95,
    require_pre_emit: bool = True,
) -> np.ndarray:
    """Fig. 2 / §4.2 population: undecided, nonzero signal, pre-emit, Wilson-open q."""
    mask = gate_mask(var_r, mean_energy, undecided, kappa)
    emitted = np.asarray(answer_emitted, dtype=bool).reshape(-1)
    if emitted.shape[0] != mask.shape[0]:
        raise ValueError("answer_emitted length does not match prefixes")
    if require_pre_emit:
        mask = mask & ~emitted
    if wilson_lo > 0.0 or wilson_hi < 1.0:
        inside = np.array([wilson_q_inside(r, wilson_lo, wilson_hi) for r in rewards], dtype=bool)
        if inside.shape[0] != mask.shape[0]:
            raise ValueError("rewards length does not match prefixes")
        mask = mask & inside
    return mask


def t_learn(rho_curve: np.ndarray, times: np.ndarray, eps: float) -> float | None:
    for t, val in zip(times, rho_curve):
        if np.isfinite(val) and val <= eps:
            return float(t)
    return None


def rho_ucb(vals, z: float = 1.96) -> float:
    """Upper bound of the problem-level mean. One problem keeps the point estimate."""
    vals = np.asarray(vals, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan")
    mean = float(np.mean(vals))
    if vals.size < 2:
        return mean
    return mean + float(z) * float(np.std(vals, ddof=1) / np.sqrt(vals.size))


def t_answer(var_curve: np.ndarray, times: np.ndarray, delta: float) -> float | None:
    """t_A = min{t: Var(R|h_t) ≤ δ}. The curve is variance, not ρ_A."""
    for t, val in zip(times, var_curve):
        if np.isfinite(val) and val <= delta:
            return float(t)
    return None


def elf(t_l: np.ndarray, t_a: np.ndarray) -> float:
    """P(t_L < t_A). Missing times are +inf so censored paths stay in the denominator."""
    t_l = np.asarray(t_l, dtype=np.float64)
    t_a = np.asarray(t_a, dtype=np.float64)
    if t_l.size == 0:
        return float("nan")
    if not np.any(np.isfinite(t_l) | np.isfinite(t_a)):
        return float("nan")
    tl = np.where(np.isfinite(t_l), t_l, np.inf)
    ta = np.where(np.isfinite(t_a), t_a, np.inf)
    return float(np.mean(tl < ta))


def lag_index(
    rho_a_curve: np.ndarray,
    rho_l_curve: np.ndarray,
    times: np.ndarray,
    length: float | None = None,
) -> float:
    """Integrate ρ_A-ρ_L against t/L, not a stretched local grid."""
    a = np.asarray(rho_a_curve, dtype=np.float64)
    b = np.asarray(rho_l_curve, dtype=np.float64)
    t = np.asarray(times, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(t)
    if int(ok.sum()) < 2:
        return float("nan")
    a, b, t = a[ok], b[ok], t[ok]
    horizon = float(length) if length is not None and float(length) > 0 else float(np.max(t))
    if horizon <= 0.0:
        return float("nan")
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:
        integrate = np.trapz
    return float(integrate(a - b, t / horizon))


def plc(tokens_after_tl: float, backward_after_tl: float, total_cost: float) -> float:
    if total_cost <= 0.0:
        return float("nan")
    return float((tokens_after_tl + backward_after_tl) / total_cost)


def summarize_with_filters(values: np.ndarray, mask: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    return {
        "all": {
            "n": int(values.size),
            "mean": None if values.size == 0 or not np.any(np.isfinite(values)) else float(np.nanmean(values)),
        },
        "selected": {
            "n": int(mask.sum()),
            "mean": None
            if not np.any(mask) or not np.any(np.isfinite(values[mask]))
            else float(np.nanmean(values[mask])),
        },
    }
