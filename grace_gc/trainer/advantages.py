"""Advantages for the mechanism track (history EMA) and GRPO (group mean)."""

from __future__ import annotations

import numpy as np

from grace_gc.trainer.baseline import HistoricalBaseline


def history_advantages(
    rewards: list[float | None],
    problem_ids: list[str],
    baseline: HistoricalBaseline,
) -> np.ndarray:
    adv = np.zeros(len(rewards), dtype=np.float64)
    for i, (reward, pid) in enumerate(zip(rewards, problem_ids)):
        if reward is None:
            continue
        adv[i] = float(reward) - baseline.get(pid)
    return adv


def grpo_advantages(rewards: list[float | None], problem_ids: list[str]) -> np.ndarray:
    """Dr. GRPO: group-mean advantage, no std normalization.

    Stopped starts (reward=None) are excluded from the group mean.
    """
    groups: dict[str, list[tuple[int, float]]] = {}
    for i, (reward, pid) in enumerate(zip(rewards, problem_ids)):
        if reward is None:
            continue
        groups.setdefault(pid, []).append((i, float(reward)))
    adv = np.zeros(len(rewards), dtype=np.float64)
    for items in groups.values():
        mean = float(np.mean([r for _, r in items]))
        for i, reward in items:
            adv[i] = reward - mean
    return adv


def advantages_for_method(
    method_objective: str,
    rewards: list[float | None],
    problem_ids: list[str],
    baseline: HistoricalBaseline,
) -> np.ndarray:
    if method_objective == "grpo":
        return grpo_advantages(rewards, problem_ids)
    return history_advantages(rewards, problem_ids, baseline)
