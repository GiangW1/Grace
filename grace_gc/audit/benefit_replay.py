"""Replay frozen-actor audit continuations into exact prefix gradient means."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from time import perf_counter

import numpy as np
from grace_gc.versions import sha256_array, sha256_file


def audit_file(path):
    source = Path(path)
    return source / "audit_bundles.jsonl" if source.is_dir() else source


def continuation_digest(record):
    content = {key: record.get(key) for key in
               ("token_ids", "prompt_len", "reward", "baseline", "generated_suffix_tokens")}
    return hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def prefix_keys_sha256(rows):
    """Hash replay row identity and order for scalar sidecar alignment checks."""
    keys = [{"problem_id": str(row["problem_id"]),
             "path_id": str(row.get("path_id") or row.get("index") or i),
             "t": int(row["t"])}
            for i, row in enumerate(rows)]
    payload = json.dumps(keys, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


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


def audit_rows(path: str | Path, decision_tokens: int | None = 512,
               require_pre_emit: bool = False, stats: dict | None = None):
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
            if stats is not None:
                stats["input_rows"] = stats.get("input_rows", 0) + 1
            row = json.loads(_without_large_grads(line))
            if decision_tokens is not None and int(row["t"]) != int(decision_tokens):
                if stats is not None:
                    stats["excluded_decision_tokens"] = stats.get("excluded_decision_tokens", 0) + 1
                continue
            if require_pre_emit and bool(row.get("answer_emitted", False)):
                if stats is not None:
                    stats["excluded_answer_emitted"] = stats.get("excluded_answer_emitted", 0) + 1
                continue
            projection = row.get("projection") or {}
            if 0 < int(projection.get("stored_dim") or 0) <= 1024:
                row["grads"] = np.asarray(json.loads(line)["grads"], dtype=np.float32)
            yield row


def check_replayed_gradient(grad, row, index: int) -> tuple[float, float | None, float | None]:
    """Measure BF16 replay drift against the saved norm and audit sketch."""
    grad = np.asarray(grad, dtype=np.float64)
    if grad.ndim != 1 or not np.all(np.isfinite(grad)):
        raise ValueError(f"continuation {index} has an invalid gradient")
    norm = float(grad @ grad)
    if not np.isfinite(norm):
        raise ValueError(f"continuation {index} has a non-finite gradient norm")
    expected_norms = row.get("true_grad_norm_sq") or []
    if not expected_norms:
        return norm, None, None
    expected = float(expected_norms[index])
    if not np.isfinite(expected) or expected < 0:
        raise ValueError(f"continuation {index} has an invalid saved gradient norm")
    norm_error = abs(norm - expected) / max(expected, 1e-12)
    sketches = row.get("grads")
    if sketches is None:
        return norm, norm_error, None
    from grace_gc.audit.prefix_audit import jl_project

    projection = row.get("projection") or {}
    saved = np.asarray(sketches[index], dtype=np.float64)
    actual = jl_project(grad, int(projection["stored_dim"]), int(projection["seed"]))
    if saved.shape != actual.shape or not np.all(np.isfinite(saved)) or not np.all(np.isfinite(actual)):
        raise ValueError(f"continuation {index} has an invalid saved gradient sketch")
    sketch_error = float(np.linalg.norm(actual - saved) / max(np.linalg.norm(saved), 1e-12))
    return norm, norm_error, sketch_error


def replay_means(rows, grad_fn, dimension: int, output: str | Path,
                 max_continuations: int | None = 16, seed: int = 17,
                 prompt_feature_fn=None, feature_bundle_fn=None, reference_direction=None,
                 store_half_means: bool = False, metric_weights=None,
                 metric_name: str = "euclidean", verify_saved_gradients: bool = True,
                 prefix_grad_fn=None, store_trajectory_decomposition: bool = False) -> dict:
    """Stream exact G labels, storing one FP64 mean per live prefix.

    When ``reference_direction`` is supplied, also store the scalar dot
    product for each sampled continuation so later n-grid stability checks do
    not need to replay the model.

    `grad_fn` receives (token_ids, prompt_len, reward, baseline). When
    ``store_trajectory_decomposition`` is enabled, ``prefix_grad_fn`` receives
    the live prefix row and returns the unweighted score gradient ``g_h``. The
    grad_fn must return the full ascent label G=(R-b)(g_h+g_s), with prompt_len
    referring to the original problem prompt. Both legacy and trajectory
    sidecars store this same label; g_h is a separate explanatory basis.
    """
    if dimension <= 0 or (max_continuations is not None and max_continuations <= 0):
        raise ValueError("dimension and max_continuations must be positive")
    if store_trajectory_decomposition and prefix_grad_fn is None:
        raise ValueError("trajectory decomposition needs prefix_grad_fn")
    metric = (np.ones(dimension, dtype=np.float64) if metric_weights is None else
              np.asarray(metric_weights, dtype=np.float64).reshape(-1))
    if (metric.shape != (dimension,) or not np.all(np.isfinite(metric)) or
            np.any(metric < 0.0) or not np.any(metric > 0.0)):
        raise ValueError("metric weights must be finite, nonnegative, and match dimension")
    direction = None if reference_direction is None else np.asarray(reference_direction, dtype=np.float64)
    if direction is not None:
        if direction.shape != (dimension,):
            raise ValueError("reference direction and full gradient layout differ")
        if not np.all(np.isfinite(direction)):
            raise ValueError("reference direction must be finite")
    rows = list(rows)
    selected = [row for row in rows if not bool(row.get("finished", False))
                and not bool(row.get("answer_emitted", False))]
    if not selected:
        raise ValueError("no live pre-answer prefixes at the requested decision point")
    target = Path(output)
    target.mkdir(parents=True, exist_ok=True)
    matrix = np.lib.format.open_memmap(target / "mean_grads.npy", mode="w+",
                                       dtype=np.float64, shape=(len(selected), dimension))
    half_a_matrix = half_b_matrix = half_a_norms = half_b_norms = None
    half_a_metric_norms = half_b_metric_norms = None
    prefix_matrix = full_half_a_matrix = full_half_b_matrix = None
    full_half_a_norms = full_half_b_norms = None
    full_half_a_metric_norms = full_half_b_metric_norms = None
    if store_trajectory_decomposition:
        if not store_half_means:
            raise ValueError("trajectory decomposition needs store_half_means")
        prefix_matrix = np.lib.format.open_memmap(
            target / "prefix_score_gradients.npy", mode="w+", dtype=np.float64,
            shape=(len(selected), dimension))
        full_half_a_matrix = np.lib.format.open_memmap(
            target / "half_mean_full_grads_a.npy", mode="w+", dtype=np.float64,
            shape=(len(selected), dimension))
        full_half_b_matrix = np.lib.format.open_memmap(
            target / "half_mean_full_grads_b.npy", mode="w+", dtype=np.float64,
            shape=(len(selected), dimension))
        full_half_a_norms = np.lib.format.open_memmap(
            target / "half_mean_full_norm_sq_a.npy", mode="w+", dtype=np.float64,
            shape=(len(selected),))
        full_half_b_norms = np.lib.format.open_memmap(
            target / "half_mean_full_norm_sq_b.npy", mode="w+", dtype=np.float64,
            shape=(len(selected),))
        full_half_a_metric_norms = np.lib.format.open_memmap(
            target / "half_full_metric_norm_sq_a.npy", mode="w+", dtype=np.float64,
            shape=(len(selected),))
        full_half_b_metric_norms = np.lib.format.open_memmap(
            target / "half_full_metric_norm_sq_b.npy", mode="w+", dtype=np.float64,
            shape=(len(selected),))
    if store_half_means:
        half_a_matrix = np.lib.format.open_memmap(
            target / "half_mean_grads_a.npy", mode="w+", dtype=np.float64,
            shape=(len(selected), dimension))
        half_b_matrix = np.lib.format.open_memmap(
            target / "half_mean_grads_b.npy", mode="w+", dtype=np.float64,
            shape=(len(selected), dimension))
        half_a_norms = np.lib.format.open_memmap(
            target / "half_mean_norm_sq_a.npy", mode="w+", dtype=np.float64,
            shape=(len(selected),))
        half_b_norms = np.lib.format.open_memmap(
            target / "half_mean_norm_sq_b.npy", mode="w+", dtype=np.float64,
            shape=(len(selected),))
        half_a_metric_norms = np.lib.format.open_memmap(
            target / "half_metric_norm_sq_a.npy", mode="w+", dtype=np.float64,
            shape=(len(selected),))
        half_b_metric_norms = np.lib.format.open_memmap(
            target / "half_metric_norm_sq_b.npy", mode="w+", dtype=np.float64,
            shape=(len(selected),))
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
    max_norm_error = 0.0
    max_sketch_error = 0.0
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
            full_first = np.zeros(dimension, dtype=np.float64)
            full_second = np.zeros(dimension, dtype=np.float64)
            norm_sum = 0.0
            norms, metric_norms, full_norms, full_metric_norms = [], [], [], []
            rewards, baselines, advantages, costs = [], [], [], []
            replay_errors = {}
            expected_norms = row.get("true_grad_norm_sq") or []
            if expected_norms and len(expected_norms) < len(records):
                raise ValueError(f"prefix {i} has fewer stored norms than trajectories")
            prefix_grad = None
            if store_trajectory_decomposition:
                prefix_grad = np.asarray(prefix_grad_fn(row), dtype=np.float64).reshape(-1)
                if prefix_grad.shape != (dimension,) or not np.all(np.isfinite(prefix_grad)):
                    raise ValueError(f"prefix {i} has an invalid score gradient")
                prefix_matrix[i] = prefix_grad
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
                # ``grad_fn`` returns the complete policy-gradient label for
                # the saved full sequence. The prefix score gradient is a
                # separately stored basis vector; adding it here would count
                # the prefix term twice.
                full_grad = grad
                if verify_saved_gradients:
                    try:
                        norm, norm_error, sketch_error = check_replayed_gradient(
                            grad, row, int(original_index))
                    except ValueError as exc:
                        raise ValueError(f"prefix {i} continuation {j}: {exc}") from exc
                else:
                    norm, norm_error, sketch_error = float(grad @ grad), None, None
                max_norm_error = max(max_norm_error, norm_error or 0.0)
                max_sketch_error = max(max_sketch_error, sketch_error or 0.0)
                replay_errors[str(int(original_index))] = {
                    "norm_relative_error": norm_error,
                    "sketch_relative_error": sketch_error,
                }
                if directional_values is not None:
                    directional_values[i, j] = float(grad @ direction)
                mean += grad
                (first if j < n // 2 else second)[:] += grad
                (full_first if j < n // 2 else full_second)[:] += full_grad
                norm_sum += norm
                norms.append(norm)
                metric_norms.append(float(np.sum(metric * grad * grad)))
                full_norms.append(float(full_grad @ full_grad))
                full_metric_norms.append(float(np.sum(metric * full_grad * full_grad)))
                rewards.append(reward)
                baselines.append(baseline)
                advantages.append(reward - baseline)
                costs.append(float(rec.get("generated_suffix_tokens", 0)))
            mean /= n
            matrix[i] = mean
            first_n, second_n = n // 2, n - n // 2
            half_dot = (float((first / first_n) @ (second / second_n))
                        if first_n and second_n else None)
            if store_half_means:
                if not first_n or not second_n:
                    raise ValueError("half means need at least two continuations")
                half_a_matrix[i] = first / first_n
                half_b_matrix[i] = second / second_n
                half_a_norms[i] = float(np.mean(norms[:first_n]))
                half_b_norms[i] = float(np.mean(norms[first_n:]))
                half_a_metric_norms[i] = float(np.mean(metric_norms[:first_n]))
                half_b_metric_norms[i] = float(np.mean(metric_norms[first_n:]))
                if prefix_grad is not None:
                    full_half_a_matrix[i] = full_first / first_n
                    full_half_b_matrix[i] = full_second / second_n
                    full_half_a_norms[i] = float(np.mean(full_norms[:first_n]))
                    full_half_b_norms[i] = float(np.mean(full_norms[first_n:]))
                    full_half_a_metric_norms[i] = float(np.mean(full_metric_norms[:first_n]))
                    full_half_b_metric_norms[i] = float(np.mean(full_metric_norms[first_n:]))
            feature_bundle = None if feature_bundle_fn is None else feature_bundle_fn(row)
            if feature_bundle is not None and set(feature_bundle) != {"features", "prompt_features"}:
                raise ValueError("feature_bundle_fn must return features and prompt_features")
            if feature_bundle is None:
                feature_values = row.get("features")
                prompt_values = (None if prompt_feature_fn is None else
                                 np.asarray(prompt_feature_fn(row), dtype=np.float64).tolist())
            else:
                feature_values = np.asarray(feature_bundle["features"], dtype=np.float64).tolist()
                prompt_values = (None if feature_bundle["prompt_features"] is None else
                                 np.asarray(feature_bundle["prompt_features"], dtype=np.float64).tolist())
            info = {
                "index": i, "problem_id": str(row["problem_id"]),
                "path_id": str(row.get("path_id") or i), "t": int(row["t"]),
                "n_continuations": n, "continuation_indices": chosen.tolist(),
                "continuation_sha256": {str(int(index)): continuation_digest(records[int(index)])
                                         for index in chosen},
                "replay_errors": replay_errors,
                "mean_norm_sq": norm_sum / n,
                "half_mean_dot": half_dot,
                "mean_reward": float(np.mean(rewards)),
                "mean_baseline": float(np.mean(baselines)),
                "mean_advantage": float(np.mean(advantages)),
                "half_counts": ([first_n, second_n] if store_half_means else None),
                "half_reward_sum": ([float(np.sum(rewards[:first_n])),
                                      float(np.sum(rewards[first_n:]))]
                                     if store_half_means else None),
                "half_mean_reward": ([float(np.mean(rewards[:first_n])),
                                       float(np.mean(rewards[first_n:]))]
                                      if store_half_means else None),
                "half_mean_reward_a": (float(np.mean(rewards[:first_n]))
                                        if store_half_means and first_n else None),
                "half_mean_reward_b": (float(np.mean(rewards[first_n:]))
                                        if store_half_means and second_n else None),
                "half_mean_baseline": ([float(np.mean(baselines[:first_n])),
                                         float(np.mean(baselines[first_n:]))]
                                        if store_half_means else None),
                "half_mean_advantage": ([float(np.mean(advantages[:first_n])),
                                          float(np.mean(advantages[first_n:]))]
                                         if store_half_means else None),
                "half_mean_norm_sq": ([float(half_a_norms[i]), float(half_b_norms[i])]
                                       if store_half_means else None),
                "half_metric_norm_sq": ([float(half_a_metric_norms[i]),
                                          float(half_b_metric_norms[i])]
                                         if store_half_means else None),
                "half_directional_gain": ([float((first / first_n) @ direction),
                                            float((second / second_n) @ direction)]
                                           if direction is not None and first_n and second_n else None),
                "trajectory_decomposition": (None if prefix_grad is None else {
                    "target": "full_trajectory",
                    "formula": "G=(reward-baseline)*(g_h+g_s)",
                    "prefix_score_gradient_index": i,
                    "half_full_metric_norm_sq": [float(full_half_a_metric_norms[i]),
                                                  float(full_half_b_metric_norms[i])],
                }),
                "mean_cost": float(np.mean(costs)), "baseline": float(row.get("baseline") or 0),
                "features": feature_values,
                "prompt_token_ids": row.get("prompt_token_ids"),
                "prefix_token_ids": row.get("prefix_token_ids"),
                "prompt_features": prompt_values,
                "prefix_text": row.get("prefix_text"),
                "answer_emitted": row.get("answer_emitted"),
                "finished": bool(row.get("finished", False)),
                "gold": row.get("gold"),
                "difficulty": row.get("difficulty"),
            }
            metadata.append(info)
            handle.write(json.dumps(info, ensure_ascii=False) + "\n")
            matrix.flush()
            print(f"replay={i + 1}/{len(selected)} problem={info['problem_id']}", flush=True)
    del matrix
    if store_half_means:
        for array in (half_a_matrix, half_b_matrix, half_a_norms, half_b_norms,
                      half_a_metric_norms, half_b_metric_norms):
            array.flush()
        del (half_a_matrix, half_b_matrix, half_a_norms, half_b_norms,
             half_a_metric_norms, half_b_metric_norms)
    if store_trajectory_decomposition:
        for array in (prefix_matrix, full_half_a_matrix, full_half_b_matrix,
                      full_half_a_norms, full_half_b_norms,
                      full_half_a_metric_norms, full_half_b_metric_norms):
            array.flush()
        del (prefix_matrix, full_half_a_matrix, full_half_b_matrix,
             full_half_a_norms, full_half_b_norms,
             full_half_a_metric_norms, full_half_b_metric_norms)
        basis = np.load(target / "prefix_score_gradients.npy", mmap_mode="r")
        (target / "score_gradient_provenance.json").write_text(json.dumps({
            "basis_kind": "prefix_score_gradient",
            "shape": [len(metadata), dimension, 1],
            "replay_prefixes_sha256": sha256_file(target / "prefixes.jsonl"),
            "prefix_keys_sha256": prefix_keys_sha256(metadata),
            "score_gradients_sha256": sha256_array(basis),
        }, indent=2), encoding="utf-8")
        del basis
    if directional_values is not None:
        directional_values.flush()
        del directional_values
    summary = {"n_prefixes": len(selected), "n_problems": len({x["problem_id"] for x in metadata}),
               "input_prefixes": len(rows),
               "finished_input_prefixes": sum(bool(row.get("finished", False)) for row in rows),
               "answer_emitted_input_prefixes": sum(bool(row.get("answer_emitted", False)) for row in rows),
               "excluded_prefixes": len(rows) - len(selected),
               "answer_emitted_unknown_prefixes": sum(row.get("answer_emitted") is None for row in selected),
               "selection": "exclude finished or answer_emitted; unknown answer status retained and counted",
               "gradient_label": "G=(reward-baseline)*grad log p(full_response|original_prompt)",
               "dimension": dimension, "n_continuations": sum(x["n_continuations"] for x in metadata),
               "wall_seconds": perf_counter() - started,
               "max_norm_relative_error": max_norm_error,
               "max_sketch_relative_error": max_sketch_error,
               "half_means": bool(store_half_means),
               "trajectory_decomposition": bool(store_trajectory_decomposition),
               "trajectory_files": (None if not store_trajectory_decomposition else {
                   "prefix_score_gradients": "prefix_score_gradients.npy",
                   "half_mean_full_grads_a": "half_mean_full_grads_a.npy",
                   "half_mean_full_grads_b": "half_mean_full_grads_b.npy",
                   "half_mean_full_norm_sq_a": "half_mean_full_norm_sq_a.npy",
                   "half_mean_full_norm_sq_b": "half_mean_full_norm_sq_b.npy",
                   "half_full_metric_norm_sq_a": "half_full_metric_norm_sq_a.npy",
                   "half_full_metric_norm_sq_b": "half_full_metric_norm_sq_b.npy",
               }),
               "metric_name": str(metric_name),
               "metric_weights_sha256": sha256_array(metric),
               "directional_values": (None if direction is None else
                                       {"path": "directional_values.npy",
                                        "shape": [len(selected), capacity],
                                        "direction_sha256": sha256_array(direction),
                                        "prefix_keys_sha256": prefix_keys_sha256(metadata)}),
               "note": "exact frozen-actor ascent gradients; prefix means stored in FP64"}
    if direction is not None:
        (target / "directional_values_provenance.json").write_text(
            json.dumps({"direction_sha256": sha256_array(direction),
                        "prefix_keys_sha256": prefix_keys_sha256(metadata),
                        "shape": [len(metadata), capacity]}, indent=2), encoding="utf-8")
    (target / "replay_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
