"""Small-n, full-gradient diagnostics with disjoint layer/q-v blocks."""

from __future__ import annotations

import re

import numpy as np


_LAYER = re.compile(r"(?:^|\.)layers\.(\d+)\.")
_TARGET = re.compile(r"(?:^|\.)(q_proj|v_proj)\.")


def layer_qv_blocks(entries, dimension: int):
    """Return exact per-layer blocks and four depth groups, preserving packed order."""
    layers = {}
    covered = np.zeros(dimension, dtype=np.uint8)
    for entry in entries:
        name = entry["name"]
        layer, target = _LAYER.search(name), _TARGET.search(name)
        if layer is None or target is None or not re.search(r"lora_[aAbB](?:\.|$)", name):
            raise ValueError(f"not a layer q/v LoRA A/B parameter: {name}")
        start, size = int(entry["offset"]), int(entry["numel"])
        end = start + size
        if size <= 0 or start < 0 or end > dimension or np.any(covered[start:end]):
            raise ValueError("layout entries overlap or exceed the replay dimension")
        covered[start:end] = 1
        layers.setdefault((int(layer.group(1)), target.group(1)[0]), []).append((start, end))
    if not np.all(covered):
        raise ValueError("layout entries do not cover the replay dimension")
    depth = sorted({layer for layer, _ in layers})
    if any((layer, target) not in layers for layer in depth for target in ("q", "v")):
        raise ValueError("every layer needs both q and v LoRA parameters")
    per_layer = {f"layer_{layer}_{target}": layers[(layer, target)]
                 for layer in depth for target in ("q", "v") if (layer, target) in layers}
    groups = {}
    n_bins = min(4, len(depth))
    for ordinal, layer in enumerate(depth):
        depth_bin = ordinal * n_bins // len(depth)
        for target in ("q", "v"):
            if (layer, target) in layers:
                groups.setdefault(f"depth_{depth_bin}_{target}", []).extend(layers[(layer, target)])
    return per_layer, groups


def fit_projected_basis(means, train_indices, segments, k: int, reference, block: int = 32768):
    """Gram-SVD coordinates without storing a D-by-k basis or D-by-D covariance."""
    train = np.asarray(train_indices, dtype=np.int64)
    if len(train) == 0 or k <= 0:
        raise ValueError("basis needs training rows and a positive rank")
    gram = np.zeros((len(train), len(train)), dtype=np.float64)
    cross = np.zeros((len(means), len(train)), dtype=np.float64)
    ref_cross = np.zeros(len(train), dtype=np.float64)
    for left, right in segments:
        for start in range(left, right, block):
            end = min(start + block, right)
            fitting = np.asarray(means[train, start:end], dtype=np.float64)
            gram += fitting @ fitting.T
            cross += np.asarray(means[:, start:end], dtype=np.float64) @ fitting.T
            ref_cross += fitting @ reference[start:end]
    eigen, vectors = np.linalg.eigh((gram + gram.T) / 2)
    live = np.flatnonzero(eigen > max(float(eigen[-1]), 0.) * 1e-12)[::-1]
    rank = min(k, len(live))
    if rank == 0:
        return np.zeros((len(means), 0)), np.zeros(0), np.zeros((len(train), 0))
    coeff = vectors[:, live[:rank]] / np.sqrt(eigen[live[:rank]])
    return cross @ coeff, ref_cross @ coeff, coeff


def reference_block_diagnostics(reference, check, blocks):
    result = {}
    total = float(reference @ reference)
    for name, segments in blocks.items():
        first = np.concatenate([reference[a:b] for a, b in segments])
        second = np.concatenate([check[a:b] for a, b in segments])
        denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
        result[name] = {"dimension": len(first),
                        "reference_energy_fraction": None if total == 0 else float(first @ first / total),
                        "reference_check_cosine": None if denominator == 0 else float(first @ second / denominator)}
    return result
