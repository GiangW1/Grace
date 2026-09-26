#!/usr/bin/env python3
"""Fast audit for prefix-conditioned scalar training value.

The labels are a dot product with one frozen ascent/update direction.  This
keeps the experiment aligned with the training update without asking a
predictor to reconstruct the 5.9M-dimensional gradient.  The command only
uses stored replay means and (when available) the tiny per-continuation scalar
file, so fitting the audit is CPU-cheap after the GPU replay.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grace_gc.audit.benefit_replay import prefix_keys_sha256
from grace_gc.audit.expected_gain import (fit_predict, load_replay, problem_split,
                                           reference_gradient, start_experiment_run)
from grace_gc.versions import sha256_array, sha256_file
from scripts.expected_gain_suite import (_check_replay_identity, _directional_labels,
                                         _indices, _metrics, _provenance)


def _csv_ints(value, name):
    try:
        values = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"{name} must be a comma-separated integer list") from exc
    if not values or any(item <= 0 for item in values):
        raise ValueError(f"{name} must contain positive integers")
    return values


def _split_roles(rows, manifest, seed):
    split = (json.loads(Path(manifest).read_text(encoding="utf-8")) if manifest else
             problem_split(rows, seed=seed))
    roles = {role: _indices(rows, split[role])
             for role in ("train", "validation", "diagnostic")}
    if (any(len(index) == 0 for index in roles.values()) or
            sum(map(len, roles.values())) != len(rows) or
            any(set(split[a]) & set(split[b])
                for a, b in (("train", "validation"), ("train", "diagnostic"),
                             ("validation", "diagnostic")))):
        raise ValueError("split must partition every problem with nonempty roles")
    return split, roles


def _feature_matrix(rows, feature_set):
    def stack(name):
        values = [row.get(name) for row in rows]
        if any(value is None for value in values):
            raise ValueError(f"replay lacks {name}; regenerate it with the matching feature option")
        result = np.asarray(values, dtype=np.float64)
        if result.ndim != 2 or not np.all(np.isfinite(result)):
            raise ValueError(f"{name} features are not a finite matrix")
        return result

    if feature_set == "legacy":
        return stack("features")
    if feature_set == "prompt":
        return stack("prompt_features")
    if feature_set == "legacy_plus_prompt":
        return np.concatenate((stack("features"), stack("prompt_features")), axis=1)
    if feature_set == "cheap":
        result = np.asarray([[float(len(row.get("prefix_token_ids") or [])),
                              float(row.get("mean_cost", 0.)),
                              float(row.get("mean_reward", 0.)),
                              float(row.get("baseline", 0.))]
                             for row in rows], dtype=np.float64)
        if not np.all(np.isfinite(result)):
            raise ValueError("cheap features contain nonfinite values")
        return result
    raise ValueError(f"unknown feature set: {feature_set}")


def _loo_targets(rows, means):
    problem_ids = np.asarray([str(row["problem_id"]) for row in rows])
    unique = sorted(set(problem_ids))
    if len(unique) < 2:
        raise ValueError("leave-one-problem-out direction needs two problems")
    problem_means = {}
    for problem_id in unique:
        indices = np.flatnonzero(problem_ids == problem_id)
        problem_means[problem_id] = np.mean(np.asarray(means[indices], dtype=np.float64), axis=0)
    total = np.sum(np.stack(list(problem_means.values())), axis=0)
    targets = np.empty(len(rows), dtype=np.float64)
    direction_norms = {}
    for problem_id in unique:
        direction = (total - problem_means[problem_id]) / (len(unique) - 1)
        indices = np.flatnonzero(problem_ids == problem_id)
        targets[indices] = _directional_labels(means[indices], direction)
        direction_norms[problem_id] = float(np.linalg.norm(direction))
    return targets, direction_norms


def _rank(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=np.float64)
    result[order] = np.arange(len(values), dtype=np.float64)
    return result


def _correlation(first, second):
    first, second = np.asarray(first), np.asarray(second)
    if len(first) < 2 or np.std(first) == 0 or np.std(second) == 0:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def _ranking_metrics(predicted, actual):
    predicted, actual = np.asarray(predicted), np.asarray(actual)
    count = max(1, len(actual) // 5)
    top_pred = np.argsort(predicted)[-count:]
    top_actual = np.argsort(actual)[-count:]
    bottom_pred = np.argsort(predicted)[:count]
    bottom_actual = np.argsort(actual)[:count]
    return {
        "pearson": _correlation(predicted, actual),
        "spearman": _correlation(_rank(predicted), _rank(actual)),
        "top20_actual_mean": float(np.mean(actual[top_pred])),
        "bottom20_actual_mean": float(np.mean(actual[bottom_pred])),
        "top20_actual_spread": float(np.mean(actual[top_pred]) - np.mean(actual[bottom_pred])),
        "top20_overlap": float(len(set(top_pred) & set(top_actual)) / count),
        "sign_accuracy_nonzero": (None if not np.any(actual != 0) else
                                   float(np.mean(np.sign(predicted[actual != 0]) ==
                                                 np.sign(actual[actual != 0])))),
    }


def _half_reliability(rows, direction_values=None, n=None):
    pairs = []
    if direction_values is not None and n is not None:
        for i, row in enumerate(rows):
            values = np.asarray(direction_values[i], dtype=np.float64)
            values = values[np.isfinite(values)][:n]
            if len(values) >= 2:
                midpoint = len(values) // 2
                pairs.append((str(row["problem_id"]),
                              (float(np.mean(values[:midpoint])),
                               float(np.mean(values[midpoint:])))))
    else:
        for row in rows:
            value = row.get("half_directional_gain")
            if value is not None and len(value) == 2:
                pairs.append((str(row["problem_id"]), (float(value[0]), float(value[1]))))
    if not pairs:
        return {"n_prefixes": 0, "pearson": None, "spearman": None,
                "same_problem_pair_order_agreement": None, "n_comparable_pairs": 0}
    first = np.asarray([value[0] for _, value in pairs])
    second = np.asarray([value[1] for _, value in pairs])
    order = []
    for problem_id in sorted({problem for problem, _ in pairs}):
        group = [value for problem, value in pairs if problem == problem_id]
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                left = np.sign(group[i][0] - group[j][0])
                right = np.sign(group[i][1] - group[j][1])
                if left != 0 and right != 0:
                    order.append(float(left == right))
    return {"n_prefixes": len(pairs), "pearson": _correlation(first, second),
            "spearman": _correlation(_rank(first), _rank(second)),
            "same_problem_pair_order_agreement": (None if not order else float(np.mean(order))),
            "n_comparable_pairs": len(order)}


def _model_settings(names):
    for name in names:
        if name in {"zero", "constant"}:
            yield name, name, 1.0
        elif name == "ridge":
            for l2 in (1., 10., 100.):
                yield f"ridge_l2_{l2:g}", "ridge", l2
        else:
            raise ValueError(f"unknown fast audit model: {name}")


def _fit_rows(features, targets, roles, settings):
    train, validation, diagnostic = (roles[name] for name in ("train", "validation", "diagnostic"))
    rows = []
    for offset, (name, model, scale) in enumerate(settings):
        predicted = fit_predict(model, features[train], targets[train, None], features,
                                ridge_l2=scale if model == "ridge" else 1.)[:, 0]
        rows.append({"model": name,
                     "validation_mse": float(np.mean((predicted[validation] - targets[validation]) ** 2)),
                     "diagnostic": {**_metrics(predicted[diagnostic], targets[diagnostic],
                                               float(np.mean(targets[train]))),
                                    **_ranking_metrics(predicted[diagnostic], targets[diagnostic])},
                     "train_count": int(len(train)), "validation_count": int(len(validation)),
                     "diagnostic_count": int(len(diagnostic)), "offset": offset})
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", required=True)
    parser.add_argument("--direction", choices=("reference", "file", "loo"), default="reference")
    parser.add_argument("--reference-dir", default=None,
                        help="disjoint replay used to form the equal-problem reference direction")
    parser.add_argument("--reference-check-dir", default=None,
                        help="optional second disjoint replay for reference-direction cosine")
    parser.add_argument("--direction-file", default=None,
                        help=".npy direction with one entry per replay gradient coordinate")
    parser.add_argument("--directional-values", default=None,
                        help="optional per-prefix scalar matrix; defaults to replay/directional_values.npy")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--split-manifest", default=None)
    parser.add_argument("--features", choices=("legacy", "prompt", "legacy_plus_prompt", "cheap"),
                        default="legacy")
    parser.add_argument("--models", default="zero,constant,ridge")
    parser.add_argument("--n-grid", default="16,64,256")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)
    if args.direction == "reference" and not args.reference_dir:
        raise ValueError("reference direction needs --reference-dir")
    if args.direction == "file" and not args.direction_file:
        raise ValueError("file direction needs --direction-file")
    if args.direction != "file" and args.direction_file:
        raise ValueError("--direction-file is only valid with --direction file")
    started = perf_counter()
    rows, means = load_replay(args.replay_dir)
    replay_meta = _provenance(args.replay_dir) or {}
    reference = None
    check_cosine = None
    direction_description = args.direction
    if args.direction == "reference":
        reference_rows, reference_means = load_replay(args.reference_dir)
        _check_replay_identity(args.replay_dir, args.reference_dir)
        if {str(row["problem_id"]) for row in rows} & {str(row["problem_id"]) for row in reference_rows}:
            raise ValueError("replay and reference problems overlap")
        reference = reference_gradient(reference_rows, reference_means)
        stored = replay_meta.get("direction_sha256") or replay_meta.get("reference_direction_sha256")
        if stored and stored != sha256_array(reference):
            raise ValueError("replay directional labels use another reference direction")
        if args.reference_check_dir:
            check_rows, check_means = load_replay(args.reference_check_dir)
            _check_replay_identity(args.replay_dir, args.reference_check_dir)
            if ({str(row["problem_id"]) for row in rows + reference_rows} &
                    {str(row["problem_id"]) for row in check_rows}):
                raise ValueError("reference check problems overlap the other replay pools")
            check = reference_gradient(check_rows, check_means)
            denominator = float(np.linalg.norm(reference) * np.linalg.norm(check))
            check_cosine = None if denominator == 0 else float(reference @ check / denominator)
        targets = _directional_labels(means, reference)
        direction_description = "equal_problem_reference_gradient"
    elif args.direction == "file":
        reference = np.asarray(np.load(args.direction_file), dtype=np.float64).reshape(-1)
        if reference.shape != (means.shape[1],) or not np.all(np.isfinite(reference)):
            raise ValueError("direction-file must contain one finite vector with replay dimension entries")
        targets = _directional_labels(means, reference)
        direction_description = "frozen_direction_file"
    else:
        targets, loo_norms = _loo_targets(rows, means)
        direction_description = "leave_one_problem_out_replay_mean"

    if args.direction == "file":
        stored = replay_meta.get("direction_sha256")
        if stored and stored != sha256_array(reference):
            raise ValueError("replay labels use another direction file")
        stored_file = replay_meta.get("direction_file_sha256")
        if stored_file and stored_file != sha256_file(args.direction_file):
            raise ValueError("direction file differs from the replay provenance")

    split, roles = _split_roles(rows, args.split_manifest, args.seed)
    features = _feature_matrix(rows, args.features)
    model_names = [name.strip() for name in args.models.split(",") if name.strip()]
    settings = list(_model_settings(model_names))
    destination = start_experiment_run(args.run_dir, "expected_gain_directional_audit", vars(args))
    (destination / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")

    directional_path = Path(args.directional_values or (Path(args.replay_dir) / "directional_values.npy"))
    directional_values = None
    n_rows = []
    if args.direction != "loo" and directional_path.is_file():
        candidate = np.load(directional_path, mmap_mode="r")
        if candidate.ndim != 2 or candidate.shape[0] != len(rows) or np.any(np.isinf(candidate)):
            raise ValueError("directional-values must be a prefix by continuation matrix")
        sidecar_path = directional_path.parent / "directional_values_provenance.json"
        if args.directional_values and not sidecar_path.is_file():
            raise ValueError("custom directional-values needs directional_values_provenance.json")
        if sidecar_path.is_file():
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            if sidecar.get("prefix_keys_sha256") != prefix_keys_sha256(rows):
                raise ValueError("directional-values rows do not match this replay")
            if sidecar.get("shape") != list(candidate.shape):
                raise ValueError("directional-values sidecar shape disagrees with matrix")
            if reference is not None and sidecar.get("direction_sha256") != sha256_array(reference):
                raise ValueError("directional-values were generated with another direction")
        for row_values in candidate:
            finite = np.isfinite(row_values)
            first_missing = np.flatnonzero(~finite)
            if not len(finite) or not np.any(finite):
                raise ValueError("directional-values rows must contain at least one finite scalar")
            if len(first_missing) and np.any(finite[first_missing[0] + 1:]):
                raise ValueError("directional-values must have only trailing NaN padding")
        stored = replay_meta.get("direction_sha256")
        if stored and reference is not None and stored != sha256_array(reference):
            raise ValueError("directional-values were generated with another direction")
        directional_values = candidate
        for n in _csv_ints(args.n_grid, "n-grid"):
            available = np.asarray([i for i in range(len(rows))
                                    if np.sum(np.isfinite(candidate[i])) >= n], dtype=np.int64)
            if not len(available):
                n_rows.append({"n": n, "available_prefixes": 0})
                continue
            n_targets = np.asarray([float(np.mean(candidate[i][np.isfinite(candidate[i])][:n]))
                                    for i in available])
            sub_roles = {name: np.asarray([pos for pos, original in enumerate(available)
                                           if original in set(roles[name])], dtype=np.int64)
                         for name in roles}
            if any(len(value) == 0 for value in sub_roles.values()):
                n_rows.append({"n": n, "available_prefixes": int(len(available)),
                               "status": "missing_split_role"})
                continue
            n_rows.append({"n": n, "available_prefixes": int(len(available)),
                           "half_reliability": _half_reliability(
                               [rows[int(i)] for i in available], candidate[available], n),
                           "predictors": _fit_rows(features[available], n_targets, sub_roles, settings)})
    reliability = _half_reliability(rows, directional_values, None) if directional_values is None else {
        str(n): _half_reliability(rows, directional_values, n) for n in _csv_ints(args.n_grid, "n-grid")
        if len(directional_values) and np.min(np.sum(np.isfinite(directional_values), axis=1)) >= n
    }
    summary = {
        "direction": direction_description,
        "direction_norm": (None if reference is None else float(np.linalg.norm(reference))),
        "reference_check_cosine": check_cosine,
        "gradient_dimension": int(means.shape[1]),
        "n_prefixes": len(rows), "n_problems": len({str(row["problem_id"]) for row in rows}),
        "feature_set": args.features, "feature_dimension": int(features.shape[1]),
        "split": split, "half_reliability": reliability,
        "predictors_on_mean_label": _fit_rows(features, targets, roles, settings),
        "n_grid": n_rows,
        "directional_values_path": (None if directional_values is None else str(directional_path.resolve())),
        "wall_seconds": perf_counter() - started,
        "note": "scalar direction audit; a held-out label result is not an online scheduler result",
    }
    if args.direction == "loo":
        summary["loo_direction_norms"] = loo_norms
    (destination / "directional_value_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps({"run_dir": str(destination), "direction": direction_description,
                      "half_reliability": reliability, "wall_seconds": summary["wall_seconds"]},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
