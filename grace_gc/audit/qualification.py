"""Pre-registered functional answer-recovery gate for reasoning prefixes."""

from __future__ import annotations

import numpy as np


def score_probe_sample(full_ids, prompt_len, text, gold, max_new_tokens,
                       eos_id=None, finish_reason=None):
    """Score a forced-answer sample using its actual rollout termination."""
    from grace_gc.data.reward import score_prefilled_answer
    from grace_gc.trainer.algorithm import _length_truncated

    truncated = _length_truncated(full_ids, prompt_len, max_new_tokens, False,
                                  eos_id, finish_reason=finish_reason)
    reward = score_prefilled_answer(text, gold, truncated=truncated)
    return {"text": text, "reward": float(reward or 0.0),
            "finish_reason": finish_reason, "truncated": truncated,
            "token_ids": list(map(int, full_ids[prompt_len:]))}


def classify_functional_recovery(rewards, majority: float = 0.5,
                                 threshold: float | None = None) -> dict:
    """Summarize fixed-budget forced-answer probes without changing the rollout."""
    values = np.asarray(rewards, dtype=np.float64).reshape(-1)
    if values.size == 0 or np.any(~np.isfinite(values)) or np.any(~np.isin(values, (0.0, 1.0))):
        raise ValueError("functional qualification rewards must be non-empty binary values")
    if threshold is not None:
        majority = threshold
    majority = float(majority)
    if not 0.0 < majority <= 1.0:
        raise ValueError("qualification threshold must be in (0, 1]")
    mean = float(values.mean())
    successes = int(np.sum(values == 1.0))
    return {
        "qualification_mean_reward": mean,
        "qualification_n": int(values.size),
        "qualification_successes": successes,
        "qualification_failures": int(values.size - successes),
        "qualification_threshold": majority,
        # Backward-compatible alias for old bundle readers.
        "qualification_majority": majority,
        "functional_recoverable": bool(mean >= majority),
        "qualification_threshold_rule": "mean_reward >= configured threshold",
        "qualification_protocol": "forced </think> + Answer:; fixed token budget; threshold mean_reward >= configured threshold; independent probe samples",
    }


def validate_qualification_config(config: dict) -> dict:
    """Validate the optional probe settings before any GPU work starts."""
    n = int(config.get("n", 0))
    budget = int(config.get("max_new_tokens", 0))
    threshold = float(config.get("threshold", config.get("majority", 0.5)))
    if n <= 0 or budget <= 0:
        raise ValueError("functional qualification needs positive n and max_new_tokens")
    if not 0.0 < threshold <= 1.0:
        raise ValueError("functional qualification threshold must be in (0, 1]")
    return {"n": n, "max_new_tokens": budget,
            "threshold": threshold, "majority": threshold}
