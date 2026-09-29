"""Replay full score gradients into O(N) scalar statistics and fixed batches."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np

from grace_gc.audit.benefit_replay import continuation_digest
from grace_gc.audit.learning_stats import _dot, _strict_pre_answer
from grace_gc.versions import sha256_array


def _gram(g, a, b, moments, weight):
    return {"hh": _dot(weight, g, g), "ga": _dot(weight, g, a),
            "gb": _dot(weight, g, b), "aa": _dot(weight, a, a),
            "bb": _dot(weight, b, b), "ab": _dot(weight, a, b),
            "s_a": float(moments[0]), "s_b": float(moments[1])}


def _batch_report(acc, weight):
    a, b, reward_a, reward_b = acc["vectors"]
    n = acc["n"]
    dot = lambda x, y: _dot(weight, x, y) / n**2
    energy = dot(a, b)
    c_energy = dot(a - reward_a, b - reward_b)
    reward_energy = dot(reward_a, reward_b)
    total_variance = acc["second"] / n - energy
    q_a, q_b = acc["q_a"] / n, acc["q_b"] / n
    reward_total_variance = .5 * (q_a + q_b) - q_a * q_b
    return {
        "n_prefixes": n, "mean_energy": energy,
        "cross_term_energy": c_energy,
        "cross_term_fraction": None if energy <= 0 else c_energy / energy,
        "batch_mean_update_energy": energy,
        "batch_cross_update_energy": c_energy,
        "batch_reward_only_energy": reward_energy,
        "batch_cross_update_fraction": None if energy <= 0 else c_energy / energy,
        "reward_only_mean_cosine": (None if energy <= 0 or reward_energy <= 0 else
                                    .5 * (dot(reward_a, b) + dot(reward_b, a)) /
                                    np.sqrt(energy * reward_energy)),
        "batch_cross_reward_cosine": (None if energy <= 0 or reward_energy <= 0 else
                                       .5 * (dot(reward_a, b) + dot(reward_b, a)) /
                                       np.sqrt(energy * reward_energy)),
        "rho_l": None if total_variance <= 0 else acc["conditional_variance"] / n / total_variance,
        "rho_a": None if reward_total_variance <= 0 else acc["reward_variance"] / n / reward_total_variance,
        "conditional_variance": acc["conditional_variance"] / n,
        "total_variance": total_variance,
        "scope": "fixed equal-prefix batch; energies are squared norms of average updates; cross-half estimates are unclipped; no batch bootstrap",
    }


def replay_statistics(rows, score_fn, prefix_score_fn, dimension, output,
                      max_continuations=16, seed=17, metrics=None):
    """Compute all requested metrics in one replay without N-by-D output.

    ``score_fn(tokens,prompt_len)`` returns grad log p of the entire response.
    Even R=b trajectories need their unweighted score for suffix energy.
    Only four sum vectors per predeclared position/subset are retained; memory
    scales with D and the position grid, never the number of prefixes.
    """
    if dimension <= 0 or max_continuations < 2:
        raise ValueError("positive dimension and at least two continuations are required")
    metrics = {"euclidean": np.ones(dimension)} if metrics is None else metrics
    if "euclidean" not in metrics or not np.all(metrics["euclidean"] == 1):
        raise ValueError("streaming replay must include the Euclidean primary metric")
    for weight in metrics.values():
        if (np.asarray(weight).shape != (dimension,) or not np.all(np.isfinite(weight))
                or np.any(weight < 0) or not np.any(weight > 0)):
            raise ValueError("invalid fixed metric weights")
    target = Path(output)
    target.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    batch, n_rows, n_suffixes, zero_keys, pids = {}, 0, 0, [], set()
    with (target / "trajectory_scalars.jsonl").open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            if row.get("finished", False):
                continue
            records = row.get("continuation_records") or []
            n = min(len(records), max_continuations)
            if n < 2:
                raise ValueError("a live prefix needs at least two continuations")
            chosen = np.random.default_rng(np.random.SeedSequence([seed, index])).permutation(len(records))[:n]
            g = np.asarray(prefix_score_fn(row), dtype=np.float64)
            if g.shape != (dimension,) or not np.all(np.isfinite(g)):
                raise ValueError("invalid prefix score gradient")
            if not np.any(g):
                zero_keys.append({key: row.get(key) for key in ("problem_id", "path_id", "t")})
            counts, means = [n // 2, n - n // 2], [np.zeros(dimension), np.zeros(dimension)]
            samples, baseline = [], float(row["baseline"])
            for j, original_index in enumerate(chosen):
                rec = records[int(original_index)]
                prefix, prompt, tokens = row["prefix_token_ids"], row["prompt_token_ids"], rec["token_ids"]
                if rec["prompt_len"] != len(prompt) or tokens[:len(prefix)] != prefix or prefix[:len(prompt)] != prompt:
                    raise ValueError("continuation does not extend the original prompt and prefix")
                reward = float(rec["reward"])
                if reward not in (0., 1.) or float(rec["baseline"]) != baseline or not 0 <= baseline <= 1:
                    raise ValueError("binary rewards and one finite fixed baseline per prefix are required")
                score = np.asarray(score_fn(tokens, len(prompt)), dtype=np.float64)
                if score.shape != (dimension,) or not np.all(np.isfinite(score)):
                    raise ValueError("invalid full score gradient")
                full, suffix = (reward - baseline) * score, score - g
                half = int(j >= counts[0])
                means[half] += full / counts[half]
                sample = {"original_index": int(original_index), "half": half, "reward": reward,
                          "suffix_tokens": len(tokens) - len(prefix), "truncated": rec.get("truncated"),
                          "trajectory_sha256": continuation_digest(rec), "metrics": {}}
                for name, weight in metrics.items():
                    suffix_energy = _dot(weight, suffix, suffix)
                    sample["metrics"][name] = {"g_norm_sq": _dot(weight, full, full),
                                                "g_dot_prefix": _dot(weight, full, g),
                                                "suffix_score_norm_sq": suffix_energy,
                                                "adv_suffix_norm_sq": (reward - baseline)**2 * suffix_energy}
                samples.append(sample)
            info = {key: value for key, value in row.items() if key not in
                    {"continuation_records", "grads", "trajectory_gradients", "true_grad_norm_sq"}}
            info.update(index=n_rows, n_continuations=n, half_counts=counts, samples=samples,
                        half_mean_reward=[float(np.mean([s["reward"] for s in samples if s["half"] == k]))
                                          for k in (0, 1)], trajectory_statistics={})
            for name, weight in metrics.items():
                moments = [float(np.mean([s["metrics"][name]["g_norm_sq"] for s in samples if s["half"] == k]))
                           for k in (0, 1)]
                gram = _gram(g, *means, moments, weight)
                gram["suffix_score_energy_a"] = float(np.mean([
                    s["metrics"][name]["suffix_score_norm_sq"] for s in samples if s["half"] == 0]))
                gram["adv_suffix_energy_a"] = float(np.mean([
                    s["metrics"][name]["adv_suffix_norm_sq"] for s in samples if s["half"] == 0]))
                info["trajectory_statistics"][name] = gram
            position = f"{row.get('enable_thinking')}/{row['t']}"
            for key in [position] + ([position + "/strict"] if _strict_pre_answer(info) else []):
                acc = batch.setdefault(key, {"vectors": [np.zeros(dimension) for _ in range(4)],
                                             "metrics": {name: dict.fromkeys(
                                                 ("second", "conditional_variance", "reward_variance", "q_a", "q_b", "n"), 0.)
                                                         for name in metrics}})
                qa, qb = info["half_mean_reward"]
                for dest, value in zip(acc["vectors"], (*means, (qa - baseline) * g, (qb - baseline) * g)):
                    dest += value
                for name in metrics:
                    values = acc["metrics"][name]
                    gram = info["trajectory_statistics"][name]
                    second = .5 * (gram["s_a"] + gram["s_b"])
                    values["second"] += second
                    values["conditional_variance"] += second - gram["ab"]
                    values["reward_variance"] += .5 * (qa + qb) - qa * qb
                    values["q_a"] += qa; values["q_b"] += qb; values["n"] += 1
            handle.write(json.dumps(info, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            n_rows += 1; n_suffixes += n; pids.add(str(row["problem_id"]))
            print(f"statistics={n_rows} problem={row['problem_id']}", flush=True)
    batches = {key: {name: _batch_report({**acc["metrics"][name], "vectors": acc["vectors"]}, weight)
                     for name, weight in metrics.items()} for key, acc in batch.items()}
    (target / "batch_statistics.json").write_text(json.dumps(batches, indent=2, allow_nan=False), encoding="utf-8")
    summary = {"storage": "sufficient_statistics", "n_prefixes": n_rows, "n_problems": len(pids),
               "n_continuations": n_suffixes, "dimension": dimension, "metrics": list(metrics),
               "metric_hashes": {name: sha256_array(w) for name, w in metrics.items()},
               "zero_prefix_gradient_keys": zero_keys, "wall_seconds": perf_counter() - started,
               "gradient_label": "G=(reward-baseline)*grad log p(full_response|original_prompt)"}
    (target / "replay_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
