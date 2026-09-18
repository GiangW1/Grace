"""Independent full-answer evaluation. Training stoppers are removed."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from grace_gc.data.reward import extract_answer, rule_reward
from grace_gc.evaluation.metrics import avg_at_k, pass_at_k, wilson_interval


@dataclass
class EvalItem:
    problem_id: str
    gold: str
    answers: list[str]
    truncated: list[bool]
    extracted: list[str | None] | None = None
    response_tokens: list[int] | None = None
    finish_reasons: list[str] | None = None
    token_ids: list[list[int]] | None = None
    sample_seeds: list[int] | None = None
    prompt_truncated: list[bool] | None = None
    vllm_finish_reasons: list[str | None] | None = None


def evaluate_items(items: list[EvalItem], k: int = 1) -> dict:
    per = []
    parse_ok = 0
    trunc = 0
    total = 0
    for item in items:
        if len(item.answers) != len(item.truncated):
            raise ValueError("answers and truncated lengths do not match")
        rewards = [rule_reward(ans, item.gold, truncated=t) or 0.0 for ans, t in zip(item.answers, item.truncated)]
        n = len(rewards)
        c = int(sum(r >= 1.0 for r in rewards))
        total += n
        trunc += int(sum(item.truncated))
        parse_ok += int(sum(extract_answer(ans) is not None for ans in item.answers))
        if n == 0:
            metric = None
        elif n < k:
            metric = None
        else:
            metric = pass_at_k(n, c, k)
        per.append(
            {
                "problem_id": item.problem_id,
                "n": n,
                "c": c,
                "avg": avg_at_k(rewards),
                "pass_at_k": metric,
                "requested_k": k,
            }
        )
    successes = int(sum(row["c"] > 0 for row in per))
    rate, lo, hi = wilson_interval(successes, max(len(per), 1))
    avgs = [row["avg"] for row in per]
    passes = [row["pass_at_k"] for row in per if row["pass_at_k"] is not None]
    resp_lens = [int(n) for item in items for n in (item.response_tokens or [])]
    avg_ci = {"low": None, "high": None, "n_problems": len(avgs),
              "method": "percentile problem bootstrap", "resampling_unit": "problem",
              "note": "conditional on this checkpoint and sampled answers; not variation across training seeds; unavailable for fewer than two problems"}
    if len(avgs) >= 2:
        draws = np.random.default_rng(0).choice(np.asarray(avgs), size=(2000, len(avgs))).mean(axis=1)
        avg_ci.update(low=float(np.quantile(draws, .025)), high=float(np.quantile(draws, .975)))
    return {
        "n_problems": len(items),
        "n_samples": total,
        "requested_k": k,
        "parse_rate": None if total == 0 else parse_ok / total,
        "truncate_rate": None if total == 0 else trunc / total,
        "mean_response_tokens": None if not resp_lens else float(sum(resp_lens) / len(resp_lens)),
        "n_short_response": sum(1 for n in resp_lens if n < 16),
        "avg": None if not avgs else float(sum(avgs) / len(avgs)),
        "pass_at_k": None if not passes else float(sum(passes) / len(passes)),
        "pass_at_k_note": None if passes else (f"n<k={k}" if items else "no items"),
        "problem_success_rate": rate,
        "wilson": {"low": lo if items else None, "high": hi if items else None,
                   "target": "problem_any_success", "successes": successes,
                   "n_problems": len(items), "note": "not an avg@k interval or training-seed interval"},
        "avg_interval": avg_ci,
        "time_to_target": None,
        "hvd": None,
        "time_to_target_note": "not measured; no training discovery history",
        "hvd_note": "not measured; no hard-problem first-success history",
        "per_problem": per,
    }
