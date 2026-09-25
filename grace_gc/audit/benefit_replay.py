"""Replay frozen-actor audit continuations into exact prefix gradient means."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from time import perf_counter

import numpy as np


def audit_file(path):
    source = Path(path)
    return source / "audit_bundles.jsonl" if source.is_dir() else source


def continuation_digest(record):
    content = {key: record.get(key) for key in
               ("token_ids", "prompt_len", "reward", "baseline", "generated_suffix_tokens")}
    return hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def replay_config(payload, model_path, config_paths=(), bundles=None):
    """Recover numerical policy settings before applying explicit overrides."""
    from grace_gc.config import default_config, merge_configs, load_config, validate_config
    from grace_gc.trainer.methods import apply_method_defaults
    cfg = merge_configs(default_config(), payload.get("run_config") or {})
    if payload.get("lora"):
        cfg["lora"] = merge_configs(cfg.get("lora") or {}, payload["lora"])
    if bundles is not None:
        saved = audit_file(bundles).parent / "config.yaml"
        if saved.is_file():
            cfg = merge_configs(cfg, load_config(saved))
    for path in config_paths:
        cfg = merge_configs(cfg, load_config(path))
    cfg["model_path"] = model_path
    # Source training/audit data may live elsewhere; replay uses saved token IDs.
    cfg.pop("data_path", None)
    cfg.pop("eval_data_path", None)
    return validate_config(apply_method_defaults(cfg))


def audit_rows(path: str | Path, decision_tokens: int | None = 512):
    """Read the small audit fields; the saved JL sketch is not a full gradient."""
    source = audit_file(path)
    if not source.is_file():
        raise FileNotFoundError(f"audit bundles are not readable: {source}")
    # Existing audits normally store only a 256-d sketch. Legacy full-gradient
    # JSONL can be very large, so reuse the project's field-stripping reader.
    from scripts.diagnose_predictor_suite import _without_large_grads

    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(_without_large_grads(line))
            if decision_tokens is not None and int(row["t"]) != int(decision_tokens):
                continue
            yield row


def replay_means(rows, grad_fn, dimension: int, output: str | Path,
                 max_continuations: int | None = 16, seed: int = 17,
                 prompt_feature_fn=None, reference_direction=None) -> dict:
    """Stream exact G labels, storing one FP64 mean per live prefix.

    When ``reference_direction`` is supplied, also store the scalar dot
    product for each sampled continuation so later n-grid stability checks do
    not need to replay the model.

    `grad_fn` receives (token_ids, prompt_len, reward, baseline). This small
    seam also permits a CPU test of the replay bookkeeping without a model.
    """
    if dimension <= 0 or (max_continuations is not None and max_continuations <= 0):
        raise ValueError("dimension and max_continuations must be positive")
    direction = None if reference_direction is None else np.asarray(reference_direction, dtype=np.float64)
    if direction is not None and direction.shape != (dimension,):
        raise ValueError("reference direction and full gradient layout differ")
    rows = list(rows)
    selected = [row for row in rows if not bool(row.get("finished", False))]
    if not selected:
        raise ValueError("no live prefixes at the requested decision point")
    target = Path(output)
    target.mkdir(parents=True, exist_ok=True)
    matrix = np.lib.format.open_memmap(target / "mean_grads.npy", mode="w+",
                                       dtype=np.float64, shape=(len(selected), dimension))
    directional_values = None
    if direction is not None:
        # Keep the per-continuation scalar only.  This is tiny compared with
        # the full gradients and lets the offline audit check n=16/64/256
        # without replaying the model a second time.
        capacities = [min(len(row.get("continuation_records") or []),
                           max_continuations if max_continuations is not None else
                           len(row.get("continuation_records") or []))
                      for row in selected]
        if any(capacity <= 0 for capacity in capacities):
            raise ValueError("directional replay needs at least one continuation per prefix")
        capacity = max(capacities)
        directional_values = np.lib.format.open_memmap(
            target / "directional_values.npy", mode="w+", dtype=np.float64,
            shape=(len(selected), capacity))
        directional_values[:] = np.nan
    metadata = []
    max_norm_relative_error = 0.0
    started = perf_counter()
    with (target / "prefixes.jsonl").open("w", encoding="utf-8") as handle:
        for i, row in enumerate(selected):
            records = row.get("continuation_records") or []
            n = len(records) if max_continuations is None else min(len(records), max_continuations)
            if n <= 0:
                raise ValueError(f"prefix {i} has no saved continuation records")
            # Shuffle even when retaining every suffix so the two halves are
            # randomized independently of acquisition order.
            chosen = np.random.default_rng(np.random.SeedSequence([seed, i])).permutation(len(records))[:n]
            mean = np.zeros(dimension, dtype=np.float64)
            first = np.zeros(dimension, dtype=np.float64)
            second = np.zeros(dimension, dtype=np.float64)
            norm_sum = 0.0
            rewards, costs = [], []
            replay_errors = {}
            expected_norms = row.get("true_grad_norm_sq") or []
            if expected_norms and len(expected_norms) < len(records):
                raise ValueError(f"prefix {i} has fewer stored norms than trajectories")
            for j, original_index in enumerate(chosen):
                rec = records[int(original_index)]
                tokens = rec.get("token_ids")
                if not tokens:
                    raise ValueError(f"prefix {i} continuation {j} has no token IDs")
                prompt_len = int(rec["prompt_len"])
                reward = float(rec["reward"])
                baseline = float(rec["baseline"])
                if not 0 < prompt_len < len(tokens):
                    raise ValueError(f"prefix {i} continuation {j} has an invalid prompt length")
                grad = (np.zeros(dimension, dtype=np.float64) if reward == baseline else
                        np.asarray(grad_fn(tokens, prompt_len, reward, baseline), dtype=np.float64))
                if grad.shape != (dimension,) or not np.all(np.isfinite(grad)):
                    raise ValueError(f"prefix {i} continuation {j} has an invalid gradient")
                norm = float(grad @ grad)
                norm_error = None
                if expected_norms:
                    expected = float(expected_norms[int(original_index)])
                    if not np.isfinite(expected) or expected < 0:
                        raise ValueError(f"prefix {i} continuation {j} has an invalid saved gradient norm")
                    norm_error = abs(norm - expected) / max(expected, 1e-12)
                    max_norm_relative_error = max(max_norm_relative_error, norm_error)
                    replay_errors[str(int(original_index))] = {
                        "norm_relative_error": norm_error,
                    }
                if directional_values is not None:
                    directional_values[i, j] = float(grad @ direction)
                mean += grad
                (first if j < n // 2 else second)[:] += grad
                norm_sum += norm
                rewards.append(reward)
                costs.append(float(rec.get("generated_suffix_tokens", 0)))
            mean /= n
            matrix[i] = mean
            first_n, second_n = n // 2, n - n // 2
            half_dot = (float((first / first_n) @ (second / second_n))
                        if first_n and second_n else None)
            info = {
                "index": i, "problem_id": str(row["problem_id"]),
                "path_id": str(row.get("path_id") or i), "t": int(row["t"]),
                "n_continuations": n, "continuation_indices": chosen.tolist(),
                "continuation_sha256": {str(int(index)): continuation_digest(records[int(index)])
                                         for index in chosen},
                "replay_errors": replay_errors,
                "mean_norm_sq": norm_sum / n,
                "half_mean_dot": half_dot, "mean_reward": float(np.mean(rewards)),
                "half_directional_gain": ([float((first / first_n) @ direction),
                                            float((second / second_n) @ direction)]
                                           if direction is not None and first_n and second_n else None),
                "mean_cost": float(np.mean(costs)), "baseline": float(row.get("baseline") or 0),
                "features": row.get("features"), "prompt_token_ids": row.get("prompt_token_ids"),
                "prefix_token_ids": row.get("prefix_token_ids"),
                "prompt_features": (None if prompt_feature_fn is None else
                                    np.asarray(prompt_feature_fn(row), dtype=np.float64).tolist()),
            }
            metadata.append(info)
            handle.write(json.dumps(info, ensure_ascii=False) + "\n")
            matrix.flush()
            print(f"replay={i + 1}/{len(selected)} problem={info['problem_id']}", flush=True)
    del matrix
    if directional_values is not None:
        directional_values.flush()
        del directional_values
    summary = {"n_prefixes": len(selected), "n_problems": len({x["problem_id"] for x in metadata}),
               "input_prefixes": len(rows), "finished_input_prefixes": len(rows) - len(selected),
               "dimension": dimension, "n_continuations": sum(x["n_continuations"] for x in metadata),
               "wall_seconds": perf_counter() - started,
               "max_norm_relative_error": max_norm_relative_error,
               "directional_values": (None if direction is None else
                                       {"path": "directional_values.npy",
                                        "shape": [len(selected), capacity]}),
               "note": "exact frozen-actor ascent gradients; prefix means stored in FP64"}
    (target / "replay_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
