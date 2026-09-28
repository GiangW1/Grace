#!/usr/bin/env python3
"""Stage B: predict dynamic score-gradient coefficients from prefix features.

The coefficient labels are fitted on replay half A.  A frozen feature model
then predicts them for held-out problems, and the reconstructed gradient is
scored only against half B.  This keeps the coefficient fit and report half
independent while retaining zero and per-prefix oracle controls.
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

from grace_gc.audit.dynamic_score import (coefficient_diagnostics, feature_matrix,
                                           difficulty_strata, fit_prefix_coefficients, load_dynamic_replay,
                                           predict_coefficients, q_strata, residual_metrics,
                                           split_roles)
from grace_gc.audit.expected_gain import start_experiment_run


def _model_settings(names):
    for name in names:
        if name in {"zero", "constant"}:
            yield name, name, 1.0
        elif name == "ridge":
            for l2 in (1.0, 10.0, 100.0):
                yield f"ridge_l2_{l2:g}", "ridge", l2
        elif name in {"mlp64", "mlp256"}:
            yield name, "mlp", int(name[3:])
        else:
            raise ValueError(f"unknown predictor model: {name}")


def _score_model(model, features, labels, basis, target_b, norm_b, metric, roles,
                 strata, difficulty, l2, args, offset):
    train, validation, diagnostic = (roles[name] for name in ("train", "validation", "diagnostic"))
    predicted = predict_coefficients(
        model, features, labels, train, np.arange(len(features)), l2=l2,
        mlp_hidden=int(l2) if model == "mlp" else 64,
        mlp_epochs=args.mlp_epochs, seed=args.seed + offset, device=args.device)
    rows = []
    for role, indices in (("validation", validation), ("diagnostic", diagnostic)):
        row = residual_metrics(np.asarray(basis[indices]), predicted[indices],
                               np.asarray(target_b[indices]), np.asarray(norm_b[indices]), metric)
        row.update({"role": role, "n_prefixes": int(len(indices)),
                    "coefficient_mse": float(np.mean((predicted[indices] - labels[indices]) ** 2))})
        row["q_strata"] = {
            name: {"n_prefixes": int(len(selected)),
                   **residual_metrics(np.asarray(basis[selected]), predicted[selected],
                                      np.asarray(target_b[selected]),
                                      np.asarray(norm_b[selected]), metric)}
            for name, stratum in strata.items()
            for selected in [np.intersect1d(indices, stratum)]
            if len(selected)
        }
        row["difficulty_strata"] = {
            name: {"n_prefixes": int(len(selected)),
                   **residual_metrics(np.asarray(basis[selected]), predicted[selected],
                                      np.asarray(target_b[selected]),
                                      np.asarray(norm_b[selected]), metric)}
            for name, stratum in difficulty.items()
            for selected in [np.intersect1d(indices, stratum)]
            if len(selected)
        }
        rows.append(row)
    return predicted, rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", required=True)
    parser.add_argument("--score-gradients", default=None,
                        help="score_gradients.npy; trajectory replays can use their stored basis")
    parser.add_argument("--gradient-target", choices=("suffix", "trajectory"), default="suffix",
                        help="select legacy or trajectory sidecars; both store full G=(R-b)(g_h+g_s)")
    parser.add_argument("--metric-file", default=None,
                        help="optional fixed diagonal metric weights; defaults to replay metric")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--split-manifest", default=None)
    parser.add_argument("--features", choices=("legacy", "prompt", "legacy_plus_prompt", "cheap"),
                        default="legacy")
    parser.add_argument("--models", default="zero,constant,ridge")
    parser.add_argument("--mlp-epochs", type=int, default=50)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--difficulty-manifest", default=None,
                        help="predeclared problem_id-to-easy/medium/hard JSON mapping")
    args = parser.parse_args(argv)
    started = perf_counter()

    rows, target_a, target_b, metric_norm_b, basis, metric, metric_name = load_dynamic_replay(
        args.replay_dir, args.score_gradients, args.metric_file, args.gradient_target)
    split, roles = split_roles(rows, args.split_manifest, args.seed)
    strata, strata_available = q_strata(rows)
    difficulty, difficulty_available = difficulty_strata(rows, args.difficulty_manifest)
    features = feature_matrix(rows, args.features)
    labels = fit_prefix_coefficients(basis, target_a, metric)
    settings = list(_model_settings([name.strip() for name in args.models.split(",") if name.strip()]))
    if not settings:
        raise ValueError("choose at least one predictor model")
    destination = start_experiment_run(args.run_dir, "expected_gain_dynamic_predictor", vars(args))
    (destination / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    rows_out, selected, predictions = [], {}, {}
    for offset, (name, model, scale) in enumerate(settings):
        _predicted, reports = _score_model(model, features, labels, basis, target_b, metric_norm_b,
                                           metric, roles, strata, difficulty, scale, args, offset)
        predictions[name] = _predicted
        validation = next(report for report in reports if report["role"] == "validation")
        row = {"model": name, "feature_set": args.features, "validation": validation,
               "diagnostic": next(report for report in reports if report["role"] == "diagnostic"),
               "coefficient_count": int(basis.shape[2])}
        rows_out.append(row)
        if not selected or validation["residual_mean"] < selected["validation_residual_mean"]:
            selected = {"model": name, "validation_residual_mean": validation["residual_mean"]}

    oracle = fit_prefix_coefficients(basis, target_a, metric)
    oracle_reports = {}
    for role, indices in roles.items():
        oracle_reports[role] = residual_metrics(np.asarray(basis[indices]), oracle[indices],
                                                 np.asarray(target_b[indices]),
                                                 np.asarray(metric_norm_b[indices]), metric)
        oracle_reports[role]["q_strata"] = {
            name: {"n_prefixes": int(len(selected)),
                   **residual_metrics(np.asarray(basis[selected]), oracle[selected],
                                      np.asarray(target_b[selected]),
                                      np.asarray(metric_norm_b[selected]), metric)}
            for name, stratum in strata.items()
            for selected in [np.intersect1d(indices, stratum)]
            if len(selected)
        }
        oracle_reports[role]["difficulty_strata"] = {
            name: {"n_prefixes": int(len(selected)),
                   **residual_metrics(np.asarray(basis[selected]), oracle[selected],
                                      np.asarray(target_b[selected]),
                                      np.asarray(metric_norm_b[selected]), metric)}
            for name, stratum in difficulty.items()
            for selected in [np.intersect1d(indices, stratum)]
            if len(selected)
        }
    np.save(destination / "coefficient_labels_a.npy", labels)
    for name, predicted in predictions.items():
        np.save(destination / f"predicted_coefficients_{name}.npy", predicted)
    result = {
        "stage": "B", "basis_kind": "prefix_score_gradient",
        "gradient_target": args.gradient_target,
        "gradient_label": "G=(reward-baseline)*grad log p(full_response|original_prompt)",
        "feature_set": args.features, "basis_shape": list(map(int, basis.shape)),
        "split": split, "oracle_by_role": oracle_reports,
        "q_stratification": {"available": bool(strata_available),
                              "definition": "half A observed reward mean: 0, (0,1), 1"},
        "difficulty_stratification": {"available": bool(difficulty_available),
                                       "manifest": (None if args.difficulty_manifest is None else
                                                     str(Path(args.difficulty_manifest).resolve())),
                                       "definition": "predeclared problem difficulty; no outcome labels used"},
        "metric_name": metric_name,
        "metric_weights": {"min": float(np.min(metric)), "max": float(np.max(metric)),
                            "positive_count": int(np.count_nonzero(metric > 0.0))},
        "basis_diagnostics": coefficient_diagnostics(basis),
        "models": rows_out, "selected_by_validation": selected,
        "wall_seconds": perf_counter() - started,
    }
    (destination / "dynamic_predictor_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps({"run_dir": str(destination), **result}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
