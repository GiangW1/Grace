"""Inverse-propensity weights and problem-level splits."""

from __future__ import annotations

import numpy as np


def ipw_weights(p: np.ndarray, s: np.ndarray | float) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    s_arr = np.asarray(s, dtype=np.float64)
    if s_arr.ndim == 0:
        s_arr = np.full_like(p, float(s_arr))
    else:
        s_arr = s_arr.reshape(-1)
        if s_arr.shape[0] != p.shape[0]:
            raise ValueError("p and s dimensions do not match")
    if np.any(p <= 0.0) or np.any(s_arr <= 0.0):
        raise ValueError("p and s must be > 0")
    return 1.0 / (p * s_arr)


def assign_new_problems(
    fit_ids: set[str],
    hold_ids: set[str],
    problem_ids: list[str],
    rng: np.random.Generator,
) -> None:
    """Assign each newly seen problem once. Later arrivals are not dumped into hold."""
    seen: set[str] = set()
    for pid in problem_ids:
        if pid in seen:
            continue
        seen.add(pid)
        if pid in fit_ids or pid in hold_ids:
            continue
        if not fit_ids:
            fit_ids.add(pid)
        elif not hold_ids:
            hold_ids.add(pid)
        elif float(rng.random()) < 0.5:
            fit_ids.add(pid)
        else:
            hold_ids.add(pid)


def split_by_problem(problem_ids: list[str], rng: np.random.Generator, frac: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    ids = np.array(problem_ids)
    unique = np.unique(ids)
    rng.shuffle(unique)
    n_fit = max(1, int(round(len(unique) * frac))) if len(unique) > 1 else 1
    fit_set = set(unique[:n_fit])
    fit = np.array([pid in fit_set for pid in ids])
    if len(unique) > 1 and not np.any(~fit):
        fit[ids == unique[-1]] = False
    return fit, ~fit
