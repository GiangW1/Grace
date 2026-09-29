"""Cross-fitted mechanism statistics for the next GRACE audit."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _finite_vector(value, name):
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _dot(weight, first, second):
    return float(np.dot(weight * first, second))


def _half_rewards(row):
    pair = row.get("half_mean_reward")
    if pair is None:
        pair = [row.get("half_mean_reward_a"), row.get("half_mean_reward_b")]
    pair = _finite_vector(pair, "half_mean_reward")
    if pair.shape != (2,) or np.any(pair < 0.0) or np.any(pair > 1.0):
        raise ValueError("half rewards must contain two values in [0, 1]")
    return pair


def mechanism_values(row, stats):
    """Return cross-fitted Gamma statistics for one prefix.

    The formula assumes binary rewards and a score gradient sampled from the
    same policy, so E[g_s | h] = 0. The null keeps the observed suffix second
    moment and removes reward/suffix mean coupling.
    """
    required = ("hh", "ga", "gb", "ab", "s_a", "s_b")
    if any(key not in stats for key in required):
        raise ValueError("mechanism statistics lack a required sufficient statistic")
    q_a, q_b = _half_rewards(row)
    baseline = float(row.get("baseline"))
    if not np.isfinite(baseline) or not 0.0 <= baseline <= 1.0:
        raise ValueError("baseline must be finite and in [0, 1]")
    h, ga, gb, ab = (float(stats[key]) for key in ("hh", "ga", "gb", "ab"))
    s_a, s_b = (float(stats[key]) for key in ("s_a", "s_b"))
    if not np.all(np.isfinite([h, ga, gb, ab, s_a, s_b])) or h < 0.0:
        raise ValueError("mechanism statistics are invalid")
    alpha_a, alpha_b = q_a - baseline, q_b - baseline
    gc_a, gc_b = ga - alpha_a * h, gb - alpha_b * h
    c_energy = ab - alpha_a * gb - alpha_b * ga + alpha_a * alpha_b * h
    cross_coupling = 0.5 * (
        (1.0 - q_a - baseline) * gc_b +
        (1.0 - q_b - baseline) * gc_a
    )
    gamma = c_energy - 2.0 * cross_coupling
    variance = 0.5 * (s_a + s_b) - ab
    parallel = None if h <= 0.0 else gc_a * gc_b / h
    return {
        "c_energy": float(c_energy),
        "g_dot_c": float(0.5 * (gc_a + gc_b)),
        "gamma": float(gamma),
        "variance": float(variance),
        "v_null": float(variance + gamma),
        "c_parallel_energy": None if parallel is None else float(parallel),
        "c_orthogonal_energy": None if parallel is None else float(c_energy - parallel),
        "mean_energy": float(ab),
        "reward_variance": float(0.5 * (q_a + q_b) - q_a * q_b),
        "zero_prefix_gradient": bool(h == 0.0),
    }


def problem_bootstrap_report(rows, metric_name, bootstrap=1000, seed=17,
                             target_ci_half_width=0.05):
    """Aggregate prefixes and bootstrap whole problem clusters."""
    if not rows:
        return {"n_prefixes": 0, "n_problems": 0, "lambda": None,
                "lambda_ci95": None, "bootstrap_defined": 0}
    values = [mechanism_values(row, row["mechanism"][metric_name]) for row in rows]
    by_problem = {}
    for index, row in enumerate(rows):
        by_problem.setdefault(str(row["problem_id"]), []).append(index)

    def ratio(indices):
        gamma = float(np.mean([values[index]["gamma"] for index in indices]))
        null = float(np.mean([values[index]["v_null"] for index in indices]))
        return None if null <= 0.0 else gamma / null

    point = ratio(np.arange(len(rows)))
    problem_ids = sorted(by_problem)
    rng = np.random.default_rng(int(seed))
    draws = []
    if len(problem_ids) >= 2:
        for _ in range(max(0, int(bootstrap))):
            sampled = rng.choice(problem_ids, size=len(problem_ids), replace=True)
            indices = [index for problem_id in sampled for index in by_problem[str(problem_id)]]
            value = ratio(indices)
            if value is not None and np.isfinite(value):
                draws.append(float(value))
    totals = {key: (float(np.mean([value[key] for value in values if value[key] is not None]))
                    if any(value[key] is not None for value in values) else None)
              for key in ("gamma", "variance", "v_null", "c_energy", "g_dot_c",
                          "c_parallel_energy", "c_orthogonal_energy", "mean_energy")}
    influence = []
    denominator = float(np.mean([value["v_null"] for value in values]))
    for indices in by_problem.values():
        influence.append(sum(values[i]["gamma"] - (point or 0.) * values[i]["v_null"]
                             for i in indices))
    sd = (None if len(influence) < 2 or denominator <= 0 else
          float(np.std(influence, ddof=1) / (denominator * len(rows) / len(influence))))
    return {
        "n_prefixes": len(rows), "n_problems": len(problem_ids),
        "weighting": "equal prefix within the reported problem cluster",
        "gamma": totals["gamma"], "v_null": totals["v_null"], "lambda": point,
        "lambda_ci95": (None if not draws else
                         np.quantile(np.asarray(draws), [0.025, 0.975]).tolist()),
        "bootstrap_requested": int(bootstrap), "bootstrap_defined": len(draws),
        "bootstrap_unit": "problem; all prefixes for a sampled problem move together",
        "zero_prefix_gradient_rows": int(sum(value["zero_prefix_gradient"] for value in values)),
        "means": totals,
        "precision_pilot": {"target_ci_half_width": float(target_ci_half_width),
                            "problem_influence_sd": sd,
                            "approx_problems": None if sd is None else
                                int(np.ceil((1.96 * sd / target_ci_half_width)**2)),
                            "scope": "normal approximation on this population; not a suffix-count power analysis"},
    }


def _difficulty_labels(rows, manifest):
    if manifest is None:
        labels = [row.get("difficulty") for row in rows]
    else:
        payload = json.loads(Path(manifest).read_text(encoding="utf-8"))
        mapping = payload.get("difficulty", payload) if isinstance(payload, dict) else None
        if not isinstance(mapping, dict):
            raise ValueError("difficulty manifest must map problem IDs to easy/medium/hard")
        labels = [mapping.get(str(row["problem_id"])) for row in rows]
    labels = [None if value is None else str(value).lower() for value in labels]
    if any(value not in {"easy", "medium", "hard"} for value in labels if value is not None):
        raise ValueError("difficulty labels must be easy, medium, or hard")
    return labels


def _q_label(row):
    value = row.get("qualification_mean_reward")
    independent = value is not None
    if value is None:
        value = _half_rewards(row)[0]
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("qualification reward mean must lie in [0,1]")
    if value == 0.0:
        return "observed_q_zero", independent
    if value == 1.0:
        return "observed_q_one", independent
    return "observed_q_uncertain", independent


def _strict_pre_answer(row):
    q, independent = _q_label(row)
    return bool(independent and q == "observed_q_uncertain" and
                row.get("answer_emitted") is False and
                row.get("functional_recoverable") is False and
                not row.get("finished", False))


def grouped_reports(rows, metric_name, difficulty_manifest=None, bootstrap=1000, seed=17):
    labels = _difficulty_labels(rows, difficulty_manifest)
    groups = {"all": list(range(len(rows))),
              "strict_pre_answer_undecided": [i for i, row in enumerate(rows)
                                               if _strict_pre_answer(row)]}
    for index, row in enumerate(rows):
        q, _independent = _q_label(row)
        groups.setdefault(q, []).append(index)
        position = f"position/{row.get('enable_thinking')}/{row['t']}"
        groups.setdefault(position, []).append(index)
        if _strict_pre_answer(row):
            groups.setdefault(position + "/strict", []).append(index)
        if labels[index] is not None:
            groups.setdefault(f"difficulty/{labels[index]}", []).append(index)
            groups.setdefault(f"q_difficulty/{q}/{labels[index]}", []).append(index)
    return {
        "q_strata": "independent qualification where available; legacy half-A q is exploratory",
        "difficulty_manifest": (None if difficulty_manifest is None else
                                str(Path(difficulty_manifest).resolve())),
        "groups": {name: problem_bootstrap_report(
            [rows[index] for index in indices], metric_name,
            bootstrap=bootstrap, seed=seed,
        ) for name, indices in groups.items()},
    }


def load_legacy_trajectory(replay_dir, metric_file=None, metric_name=None, block=32768):
    """Reduce PR12 trajectory arrays to scalar Gram statistics."""
    root = Path(replay_dir)
    rows = [json.loads(line) for line in (root / "prefixes.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    required = ["prefix_score_gradients.npy", "half_mean_full_grads_a.npy",
                "half_mean_full_grads_b.npy", "half_mean_full_norm_sq_a.npy",
                "half_mean_full_norm_sq_b.npy"]
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise ValueError("trajectory replay lacks: " + ", ".join(missing))
    prefix = np.load(root / "prefix_score_gradients.npy", mmap_mode="r")
    if prefix.ndim == 3 and prefix.shape[2] == 1:
        prefix = prefix[:, :, 0]
    a = np.load(root / "half_mean_full_grads_a.npy", mmap_mode="r")
    b = np.load(root / "half_mean_full_grads_b.npy", mmap_mode="r")
    if prefix.shape != a.shape or a.shape != b.shape or a.ndim != 2 or a.shape[0] != len(rows):
        raise ValueError("trajectory arrays have incompatible shapes")
    metric = np.ones(a.shape[1], dtype=np.float64)
    name = metric_name or "euclidean"
    if metric_file is not None:
        metric = _finite_vector(np.load(metric_file), "metric")
        if metric.shape != (a.shape[1],) or np.any(metric < 0.0) or not np.any(metric > 0.0):
            raise ValueError("metric file has wrong shape or invalid weights")
        name = metric_name or "diagonal_file"
    norm_name = "half_mean_full_norm_sq"
    if not np.all(metric == 1.0):
        norm_name = "half_full_metric_norm_sq"
        metric_path = root / "metric_weights.npy"
        if (not metric_path.is_file() or not np.array_equal(np.load(metric_path), metric)
                or not (root / f"{norm_name}_a.npy").is_file()):
            raise ValueError("weighted mechanism audit needs matching replay second moments")
    s_a = np.load(root / f"{norm_name}_a.npy", mmap_mode="r")
    s_b = np.load(root / f"{norm_name}_b.npy", mmap_mode="r")
    if s_a.shape != (len(rows),) or s_b.shape != s_a.shape:
        raise ValueError("second-moment arrays have incompatible shapes")
    for index, row in enumerate(rows):
        values = {"hh": 0.0, "ga": 0.0, "gb": 0.0, "ab": 0.0,
                  "aa": 0.0, "bb": 0.0, "s_a": float(s_a[index]), "s_b": float(s_b[index])}
        for start in range(0, a.shape[1], int(block)):
            stop = min(start + int(block), a.shape[1])
            weight = metric[start:stop]
            g = np.asarray(prefix[index, start:stop], dtype=np.float64)
            va = np.asarray(a[index, start:stop], dtype=np.float64)
            vb = np.asarray(b[index, start:stop], dtype=np.float64)
            values["hh"] += _dot(weight, g, g)
            values["ga"] += _dot(weight, g, va)
            values["gb"] += _dot(weight, g, vb)
            values["ab"] += _dot(weight, va, vb)
            values["aa"] += _dot(weight, va, va)
            values["bb"] += _dot(weight, vb, vb)
        row["mechanism"] = {name: values}
    return rows, name


def load_scalar_trajectory(replay_dir, metric_name="euclidean"):
    root = Path(replay_dir)
    path = root / "trajectory_scalars.jsonl"
    if not path.is_file():
        path = root / "prefixes.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        stats = row.get("trajectory_statistics", {}).get(metric_name)
        if stats is None:
            raise ValueError(f"replay lacks trajectory statistics for {metric_name}")
        row["mechanism"] = {metric_name: stats}
    return rows
