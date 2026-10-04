"""Pre-registered functional answer-recovery gate for reasoning prefixes."""

from __future__ import annotations

import numpy as np

from grace_gc.data.reward import rule_reward

FORCED_ANSWER_PREFIX = "</think>\nAnswer:"


def score_functional_probe(text: str, gold: str, truncated: bool = False) -> float:
    """The generated suffix follows an Answer: already present in its prompt."""
    return float(rule_reward("Answer:" + text, gold, truncated=truncated,
                             require_complete=True) or 0.0)


def classify_functional_recovery(rewards, majority: float = 0.5) -> dict:
    """Summarize fixed-budget forced-answer probes without changing the rollout."""
    values = np.asarray(rewards, dtype=np.float64).reshape(-1)
    if values.size == 0 or np.any(~np.isfinite(values)) or np.any(~np.isin(values, (0.0, 1.0))):
        raise ValueError("functional qualification rewards must be non-empty binary values")
    majority = float(majority)
    if not 0.0 < majority <= 1.0:
        raise ValueError("qualification majority must be in (0, 1]")
    mean = float(values.mean())
    return {
        "qualification_mean_reward": mean,
        "qualification_n": int(values.size),
        "qualification_majority": majority,
        "functional_recoverable": bool(mean >= majority),
        "qualification_protocol": "forced </think> + Answer:; score prefilled Answer: and generated suffix; completed outputs only; independent probe samples",
    }


def validate_qualification_config(config: dict) -> dict:
    """Validate the optional probe settings before any GPU work starts."""
    n = int(config.get("n", 0))
    budget = int(config.get("max_new_tokens", 0))
    majority = float(config.get("majority", 0.5))
    if n <= 0 or budget <= 0:
        raise ValueError("functional qualification needs positive n and max_new_tokens")
    if not 0.0 < majority <= 1.0:
        raise ValueError("functional qualification majority must be in (0, 1]")
    return {"n": n, "max_new_tokens": budget, "majority": majority}
