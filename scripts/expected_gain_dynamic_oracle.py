#!/usr/bin/env python3
"""Stage A: measure the dynamic score-gradient representation ceiling.

For every prefix, coefficients are fitted from independent replay half A and
evaluated on half B.  This is an oracle geometry test, not a learned online
predictor and not a claim about training quality.
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

from grace_gc.audit.dynamic_score import (coefficient_diagnostics, fit_prefix_coefficients,
                                           difficulty_strata, load_dynamic_replay, q_strata,
                                           residual_metrics, split_roles)
from grace_gc.audit.expected_gain import start_experiment_run


def _mechanism_report(rows, basis, target_a, target_b, metric_norm_b, metric, indices, coefficients):
    subset_basis = np.asarray(basis[indices], dtype=np.float64)
    subset_target_a = np.asarray(target_a[indices], dtype=np.float64)
    subset_target = np.asarray(target_b[indices], dtype=np.float64)
    subset_norm = np.asarray(metric_norm_b[indices], dtype=np.float64)
    free = residual_metrics(subset_basis, coefficients[indices], subset_target,
                            subset_norm, metric)
    report = {"free": free}
    advantages = [rows[int(index)].get("half_mean_advantage") for index in indices]
    if subset_basis.shape[2] != 1 or any(value is None or len(value) < 1 for value in advantages):
        report["status"] = "missing_half_advantage_or_single_score_basis"
        return report
    advantage_a = np.asarray([float(value[0]) for value in advantages], dtype=np.float64)
    if not np.all(np.isfinite(advantage_a)):
        raise ValueError("half_mean_advantage contains nonfinite values")
    reward_coefficients = advantage_a[:, None]
    reward_only = residual_metrics(subset_basis, reward_coefficients, subset_target,
                                   subset_norm, metric)
    gradient = subset_basis[:, :, 0]
    target_energy = np.sum(metric[None, :] * subset_target * subset_target, axis=1)
    gradient_energy = np.sum(metric[None, :] * gradient * gradient, axis=1)
    dot_a = np.sum(metric[None, :] * subset_target_a * gradient, axis=1)
    dot_b = np.sum(metric[None, :] * subset_target * gradient, axis=1)
    # The conditional-mean energy must be cross-fitted. Using only B makes
    # the orthogonal component positive even when the true mean is exactly
    # aligned with the score-gradient basis, because B contains sampling
    # noise orthogonal to that basis.
    cross_target_energy = np.sum(metric[None, :] * subset_target_a * subset_target, axis=1)
    cross_projected_energy = dot_a * dot_b / np.maximum(gradient_energy, 1e-30)
    cross_orthogonal = cross_target_energy - cross_projected_energy
    valid_cross = (np.isfinite(cross_orthogonal) & np.isfinite(cross_target_energy) &
                   (gradient_energy > 0.0))
    b_projected_energy = dot_b * dot_b / np.maximum(gradient_energy, 1e-30)
    b_orthogonal = target_energy - b_projected_energy
    valid_b = np.isfinite(b_orthogonal) & (target_energy > 0.0) & (gradient_energy > 0.0)
    cross_denominator = float(np.sum(cross_target_energy[valid_cross]))
    b_denominator = float(np.sum(target_energy[valid_b]))
    report.update({
        "reward_only": reward_only,
        "free_minus_reward_residual": float(free["residual_mean"] -
                                              reward_only["residual_mean"]),
        "reward_only_residual_ratio_to_zero": reward_only["residual_ratio_to_zero"],
        # Kept as a compatibility alias for prior summaries.
        "cross_term_residual_ratio": reward_only["residual_ratio_to_zero"],
        "orthogonal_energy_fraction": (None if not np.any(valid_cross) or cross_denominator <= 0.0 else
                                        float(np.sum(cross_orthogonal[valid_cross]) /
                                              cross_denominator)),
        "projected_energy_fraction": (None if not np.any(valid_cross) or cross_denominator <= 0.0 else
                                       float(np.sum(cross_projected_energy[valid_cross]) /
                                             cross_denominator)),
        "cross_mean_target_energy": (None if not np.any(valid_cross) else
                                      float(np.mean(cross_target_energy[valid_cross]))),
        "cross_orthogonal_energy": (None if not np.any(valid_cross) else
                                     float(np.mean(cross_orthogonal[valid_cross]))),
        "cross_projected_energy": (None if not np.any(valid_cross) else
                                    float(np.mean(cross_projected_energy[valid_cross]))),
        "cross_orthogonal_energy_fraction": (None if not np.any(valid_cross) or cross_denominator <= 0.0 else
                                              float(np.sum(cross_orthogonal[valid_cross]) /
                                                    cross_denominator)),
        "cross_projected_energy_fraction": (None if not np.any(valid_cross) or cross_denominator <= 0.0 else
                                             float(np.sum(cross_projected_energy[valid_cross]) /
                                                   cross_denominator)),
        "b_only_orthogonal_energy_fraction": (None if not np.any(valid_b) or b_denominator <= 0.0 else
                                                float(np.sum(b_orthogonal[valid_b]) / b_denominator)),
        "b_only_projected_energy_fraction": (None if not np.any(valid_b) or b_denominator <= 0.0 else
                                               float(np.sum(b_projected_energy[valid_b]) / b_denominator)),
        "n_valid_orthogonal_rows": int(np.count_nonzero(valid_cross)),
        "n_valid_cross_energy": int(np.count_nonzero(valid_cross)),
    })
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", required=True)
    parser.add_argument("--score-gradients", default=None,
                        help="score_gradients.npy; trajectory replays can use their stored basis")
    parser.add_argument("--gradient-target", choices=("suffix", "trajectory"), default="suffix",
                        help="fit/evaluate suffix gradients or full G=(R-b)(g_h+g_s)")
    parser.add_argument("--metric-file", default=None,
                        help="optional fixed diagonal metric weights; defaults to replay metric")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--split-manifest", default=None)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--difficulty-manifest", default=None,
                        help="predeclared problem_id-to-easy/medium/hard JSON mapping")
    args = parser.parse_args(argv)
    started = perf_counter()

    rows, target_a, target_b, metric_norm_b, basis, metric, metric_name = load_dynamic_replay(
        args.replay_dir, args.score_gradients, args.metric_file, args.gradient_target)
    split, roles = split_roles(rows, args.split_manifest, args.seed)
    coefficients = fit_prefix_coefficients(basis, target_a, metric)
    metrics = {}
    strata, strata_available = q_strata(rows)
    difficulty, difficulty_available = difficulty_strata(rows, args.difficulty_manifest)
    for role, indices in roles.items():
        subset_basis = np.asarray(basis[indices], dtype=np.float64)
        subset_coefficients = coefficients[indices]
        metrics[role] = {
            "n_prefixes": int(len(indices)),
            "n_problems": int(len({str(rows[i]["problem_id"]) for i in indices})),
            "oracle": residual_metrics(subset_basis, subset_coefficients,
                                         np.asarray(target_b[indices]), np.asarray(metric_norm_b[indices]), metric),
            "zero": {"residual_mean": float(np.mean(metric_norm_b[indices]))},
            "mechanism": _mechanism_report(rows, basis, target_a, target_b, metric_norm_b,
                                            metric, indices, coefficients),
            "q_strata": {
                name: {"n_prefixes": int(len(role_selected)),
                       "mechanism": _mechanism_report(
                           rows, basis, target_a, target_b, metric_norm_b, metric,
                           role_selected, coefficients)}
                for name, selected in strata.items()
                for role_selected in [np.intersect1d(indices, selected)]
                if len(role_selected)
            },
            "difficulty_strata": {
                name: {"n_prefixes": int(len(role_selected)),
                       "mechanism": _mechanism_report(
                           rows, basis, target_a, target_b, metric_norm_b, metric,
                           role_selected, coefficients)}
                for name, selected in difficulty.items()
                for role_selected in [np.intersect1d(indices, selected)]
                if len(role_selected)
            },
        }
    destination = start_experiment_run(args.run_dir, "expected_gain_dynamic_oracle", vars(args))
    (destination / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    np.save(destination / "oracle_coefficients.npy", coefficients)
    result = {
        "stage": "A",
        "basis_kind": "prefix_score_gradient",
        "gradient_target": args.gradient_target,
        "basis_shape": list(map(int, basis.shape)),
        "gradient_dimension": int(target_a.shape[1]),
        "coefficients": int(basis.shape[2]),
        "metric_name": metric_name,
        "metric_weights": {"min": float(np.min(metric)), "max": float(np.max(metric)),
                            "positive_count": int(np.count_nonzero(metric > 0.0))},
        "basis_diagnostics": coefficient_diagnostics(basis),
        "q_stratification": {"available": bool(strata_available),
                              "definition": "half A observed reward mean: 0, (0,1), 1"},
        "difficulty_stratification": {"available": bool(difficulty_available),
                                       "manifest": (None if args.difficulty_manifest is None else
                                                     str(Path(args.difficulty_manifest).resolve())),
                                       "definition": "predeclared problem difficulty; no outcome labels used"},
        "split": split,
        "roles": metrics,
        "wall_seconds": perf_counter() - started,
    }
    (destination / "dynamic_oracle_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps({"run_dir": str(destination), **result}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
