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

from grace_gc.audit.dynamic_score import (fit_prefix_coefficients, load_dynamic_replay,
                                           residual_metrics, split_roles)
from grace_gc.audit.expected_gain import start_experiment_run


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", required=True)
    parser.add_argument("--score-gradients", required=True,
                        help="score_gradients.npy from extract_prefix_score_gradients.py")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--split-manifest", default=None)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)
    started = perf_counter()

    rows, target_a, target_b, norm_b, basis = load_dynamic_replay(
        args.replay_dir, args.score_gradients)
    split, roles = split_roles(rows, args.split_manifest, args.seed)
    coefficients = fit_prefix_coefficients(basis, target_a)
    metrics = {}
    for role, indices in roles.items():
        subset_basis = np.asarray(basis[indices], dtype=np.float64)
        subset_coefficients = coefficients[indices]
        metrics[role] = {
            "n_prefixes": int(len(indices)),
            "n_problems": int(len({str(rows[i]["problem_id"]) for i in indices})),
            "oracle": residual_metrics(subset_basis, subset_coefficients,
                                         np.asarray(target_b[indices]), np.asarray(norm_b[indices])),
            "zero": {"residual_mean": float(np.mean(norm_b[indices]))},
        }
    destination = start_experiment_run(args.run_dir, "expected_gain_dynamic_oracle", vars(args))
    (destination / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    np.save(destination / "oracle_coefficients.npy", coefficients)
    result = {
        "stage": "A",
        "basis_kind": "prefix_score_gradient",
        "basis_shape": list(map(int, basis.shape)),
        "gradient_dimension": int(target_a.shape[1]),
        "coefficients": int(basis.shape[2]),
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
