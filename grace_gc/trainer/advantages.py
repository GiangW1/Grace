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


def loo_advantages(
    rewards: list[float | None],
    problem_ids: list[str],
    baseline: HistoricalBaseline,
) -> np.ndarray:
    """R_i minus the mean observed reward of the other starts of the same problem.

    b_i depends only on other starts (their Z and R), never on trajectory i,
    so E[(R_i - b_i) grad log pi_i] is still the raw policy gradient, HT
    weighting included. Without another observed start, history is the fallback.
    """
    groups: dict[str, list[float]] = {}
    for reward, pid in zip(rewards, problem_ids):
        if reward is not None:
            groups.setdefault(pid, []).append(float(reward))
    adv = np.zeros(len(rewards), dtype=np.float64)
    for i, (reward, pid) in enumerate(zip(rewards, problem_ids)):
        if reward is None:
            continue
        others = len(groups[pid]) - 1
        b = (sum(groups[pid]) - float(reward)) / others if others else baseline.get(pid)
        adv[i] = float(reward) - b
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
    if baseline.mode == "loo":
        return loo_advantages(rewards, problem_ids, baseline)
    return history_advantages(rewards, problem_ids, baseline)


def informative_groups(rewards: list[float | None], problem_ids: list[str], max_starts: int):
    """DAPO dynamic sampling: keep groups whose observed rewards differ, up to max_starts.

    Zero-variance groups have zero group-mean advantage, so they only cost
    generation. Groups are kept in draw order until the next would exceed the
    fixed start budget; the loss denominator stays that budget.
    """
    order, members = [], {}
    for i, pid in enumerate(problem_ids):
        if pid not in members:
            order.append(pid)
            members[pid] = []
        members[pid].append(i)
    keep = np.zeros(len(rewards), dtype=bool)
    informative = kept = kept_starts = 0
    for pid in order:
        observed = {float(rewards[i]) for i in members[pid] if rewards[i] is not None}
        if len(observed) < 2:
            continue
        informative += 1
        if kept_starts + len(members[pid]) > int(max_starts):
            continue
        keep[members[pid]] = True
        kept += 1
        kept_starts += len(members[pid])
    return keep, {"n_groups": len(order), "n_informative": informative, "n_kept": kept,
                  "n_kept_starts": kept_starts, "max_starts": int(max_starts)}
