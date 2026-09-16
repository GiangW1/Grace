"""Prefix features. Dimension is derived from the actual tensors."""

from __future__ import annotations

import numpy as np

POOL_LAST = 64


def pool_last_hidden(seq_row: np.ndarray, end: int, width: int = POOL_LAST) -> np.ndarray:
    """Paper φ(h): mean-pool the last 64 tokens of the prefix, not the whole prefix."""
    end = max(min(int(end), int(seq_row.shape[0])), 1)
    start = max(end - int(width), 0)
    return np.asarray(seq_row[start:end], dtype=np.float64).mean(axis=0)


def entropy_stats(token_entropy: np.ndarray) -> np.ndarray:
    """Mean, max, and high-entropy count. Input is entropy, not log-prob."""
    ent = np.asarray(token_entropy, dtype=np.float64).reshape(-1)
    if ent.size == 0:
        return np.zeros(3, dtype=np.float64)
    high = float(np.sum(ent > np.log(2.0)))
    return np.array([float(ent.mean()), float(ent.max()), high], dtype=np.float64)


def prefix_features(
    last_hidden: np.ndarray,
    mid_hidden: np.ndarray,
    pooled_hidden: np.ndarray,
    token_entropy: np.ndarray,
    prefix_len: float,
    baseline: float,
) -> np.ndarray:
    parts = [
        np.asarray(last_hidden, dtype=np.float64).reshape(-1),
        np.asarray(mid_hidden, dtype=np.float64).reshape(-1),
        np.asarray(pooled_hidden, dtype=np.float64).reshape(-1),
        entropy_stats(token_entropy),
        np.array([float(prefix_len), float(baseline)], dtype=np.float64),
    ]
    return np.concatenate(parts, axis=0)


def prompt_slice_features(
    last_seq: np.ndarray,
    mid_seq: np.ndarray,
    token_ent_seq: np.ndarray,
    prompt_lens,
    baselines,
) -> np.ndarray:
    """Same layout as prefix_features, but read only the prompt span."""
    feats = []
    for i, plen in enumerate(prompt_lens):
        end = max(int(plen), 1)
        pos = min(max(end - 1, 0), last_seq.shape[1] - 1)
        pooled = pool_last_hidden(last_seq[i], end)
        ent = token_ent_seq[i, : max(end - 1, 1)]
        feats.append(
            prefix_features(last_seq[i, pos], mid_seq[i, pos], pooled, ent, float(plen), float(baselines[i]))
        )
    return np.stack(feats, axis=0)


def prompt_only_features(prompt_hidden: np.ndarray, baseline: float, prompt_len: float) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(prompt_hidden, dtype=np.float64).reshape(-1),
            np.array([float(baseline), float(prompt_len)], dtype=np.float64),
        ]
    )


def reward_risk_features(q_hat: float, difficulty: float, length: float, baseline: float) -> np.ndarray:
    q = float(np.clip(q_hat, 1e-6, 1 - 1e-6))
    entropy = -(q * np.log(q) + (1 - q) * np.log(1 - q))
    return np.array([q, entropy, float(difficulty), float(length), float(baseline)], dtype=np.float64)
