#!/usr/bin/env python3
"""Fit predictor candidates on a common frozen prefix audit.

The script intentionally consumes ``audit_bundles.jsonl`` instead of running
another rollout.  Every suffix continuation from one prefix therefore shares
the same stored feature vector, and the train/test split is made at the
problem level.  This keeps the diagnostic about predictability rather than
about changing the actor, seeds, or allocation policy.
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

from grace_gc.audit.prefix_audit import bundle_from_dict


def _load_bundles(path: Path):
    if path.is_dir():
        path = path / "audit_bundles.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"bundle file is not readable: {path}")
    bundles = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            bundles.append(bundle_from_dict(json.loads(line)))
    if not bundles:
        raise ValueError(f"bundle file is empty: {path}")
    return bundles


def _problem_split(problem_ids: list[str], seed: int, test_fraction: float):
    unique = sorted(set(problem_ids))
    if len(unique) < 2:
        raise ValueError("predictor diagnostic needs at least two distinct problem IDs")
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(unique))
    n_test = max(1, int(round(len(unique) * float(test_fraction))))
    n_test = min(n_test, len(unique) - 1)
    test = {unique[int(i)] for i in order[:n_test]}
    train = set(unique) - test
    return train, test


def _rows_from_bundles(bundles):
    rows = []
    feature_dim = None
    target_dim = None
    for bundle_index, bundle in enumerate(bundles):
        if bundle.features is None:
            raise ValueError(
                "bundles do not contain features; rerun the audit with "
                "predictor.store_features: true or audit.store_features: true"
            )
        x = np.asarray(bundle.features, dtype=np.float64).reshape(-1)
        coords = None if bundle.true_grad_coords is None else np.asarray(bundle.true_grad_coords, dtype=np.float64)
        norms = None if bundle.true_grad_norm_sq is None else np.asarray(bundle.true_grad_norm_sq, dtype=np.float64).reshape(-1)
        gram = None if bundle.basis_gram is None else np.asarray(bundle.basis_gram, dtype=np.float64)
        if coords is None or norms is None or gram is None:
            raise ValueError(
                "bundles need true_grad_coords, true_grad_norm_sq, and basis_gram "
                "for full-space predictor residuals"
            )
        if (coords.ndim != 2 or coords.shape[1] <= 0 or len(norms) != coords.shape[0]
                or gram.shape != (coords.shape[1], coords.shape[1])):
            raise ValueError(f"bundle {bundle_index} has inconsistent coordinate labels or Gram matrix")
        if feature_dim is None:
            feature_dim = int(x.size)
            target_dim = int(coords.shape[1])
        if x.size != feature_dim or coords.shape[1] != target_dim:
            raise ValueError("all bundles must share feature and coordinate dimensions")
        if (not np.all(np.isfinite(x)) or not np.all(np.isfinite(coords))
                or not np.all(np.isfinite(norms)) or not np.all(np.isfinite(gram))
                or np.any(norms < 0.0)):
            raise ValueError(f"bundle {bundle_index} contains non-finite diagnostic values")
        weights = np.full(len(norms), 1.0 / max(len(norms), 1), dtype=np.float64)
        raw_grads = None if bundle.grads is None else np.asarray(bundle.grads, dtype=np.float64)
        gram_inverse = np.linalg.pinv(gram)
        for suffix_index in range(len(norms)):
            rows.append({
                "problem_id": str(bundle.problem_id),
                "bundle_index": int(bundle_index),
                "suffix_index": int(suffix_index),
                "x": x,
                # Predictor f is the coefficient in U f.  Stored
                # true_grad_coords is U.T G, so the least-squares target is
                # (U.T G) (U.T U)^+ when U is not exactly orthonormal.
                "target": coords[suffix_index] @ gram_inverse,
                "true_coord": coords[suffix_index],
                "norm": float(norms[suffix_index]),
                "gram": gram,
                "weight": float(weights[suffix_index]),
                "raw_grads": raw_grads,
            })
    if not rows:
        raise ValueError("no continuation labels found in bundles")
    return rows, int(feature_dim), int(target_dim)


def _weighted_mean(values, weights):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    return float(np.sum(values * weights) / max(np.sum(weights), 1e-12))


def _residuals(rows, prediction):
    out = []
    omitted = []
    coordinate_error = []
    zero = []
    for row, f in zip(rows, np.asarray(prediction, dtype=np.float64)):
        true_coord = np.asarray(row["true_coord"], dtype=np.float64)
        gram = np.asarray(row["gram"], dtype=np.float64)
        norm = float(row["norm"])
        captured = max(float(true_coord @ np.linalg.pinv(gram) @ true_coord), 0.0)
        omit = max(norm - captured, 0.0)
        coord = max(
            captured - 2.0 * float(true_coord @ f) + float(f @ gram @ f),
            0.0,
        )
        out.append(omit + coord)
        omitted.append(omit)
        coordinate_error.append(coord)
        zero.append(max(norm, 0.0))
    return {
        "residual": np.asarray(out, dtype=np.float64),
        "subspace_omission": np.asarray(omitted, dtype=np.float64),
        "coordinate_error": np.asarray(coordinate_error, dtype=np.float64),
        "zero_residual": np.asarray(zero, dtype=np.float64),
    }


def _future_randomness(bundles):
    rows = []
    for bundle in bundles:
        if bundle.true_grad_coords is None:
            continue
        coords = np.asarray(bundle.true_grad_coords, dtype=np.float64)
        gram = None if bundle.basis_gram is None else np.asarray(bundle.basis_gram, dtype=np.float64)
        if coords.ndim != 2 or len(coords) < 2 or gram is None:
            continue
        centered = coords - coords.mean(axis=0, keepdims=True)
        coord_energy = np.sum((centered @ np.linalg.pinv(gram)) * centered, axis=1)
        coord_var = float(np.sum(coord_energy) / (len(coords) - 1))
        item = {
            "problem_id": str(bundle.problem_id),
            "t": int(bundle.t),
            "path_id": bundle.path_id,
            "n_suffixes": int(len(coords)),
            "basis_coordinate_future_variance": max(coord_var, 0.0),
        }
        if bundle.grads is not None:
            grads = np.asarray(bundle.grads, dtype=np.float64)
            if grads.ndim == 2 and grads.shape[0] == len(coords):
                g_centered = grads - grads.mean(axis=0, keepdims=True)
                item["stored_gradient_space_future_variance"] = float(max(
                    np.sum(np.sum(g_centered * g_centered, axis=1)) / (len(grads) - 1), 0.0,
                ))
                projection = bundle.projection or {}
                item["stored_gradient_space"] = {
                    "dimension": int(grads.shape[1]),
                    "original_dimension": projection.get("original_dim"),
                }
        rows.append(item)
    if not rows:
        return {"n_prefixes": 0, "mean_basis_coordinate_future_variance": None, "rows": []}
    values = [r["basis_coordinate_future_variance"] for r in rows]
    raw_values = [r["stored_gradient_space_future_variance"] for r in rows if "stored_gradient_space_future_variance" in r]
    result = {
        "n_prefixes": len(rows),
        "mean_basis_coordinate_future_variance": float(np.mean(values)),
        "rows": rows,
        "note": "Unbiased same-prefix suffix variance trace in the stored basis subspace; it is not predictor error.",
    }
    if raw_values:
        result["mean_stored_gradient_space_future_variance"] = float(np.mean(raw_values))
    return result


def _fit_predictions(name, x_train, y_train, w_train, x_eval, args):
    started = perf_counter()
    if name == "zero":
        pred = np.zeros((len(x_eval), y_train.shape[1]), dtype=np.float64)
        fit_loss = None
    elif name == "simple":
        mean = np.average(y_train, axis=0, weights=w_train)
        pred = np.broadcast_to(mean, (len(x_eval), len(mean))).copy()
        fit_loss = float(_weighted_mean(np.sum((y_train - mean) ** 2, axis=1), w_train))
    else:
        try:
            from grace_gc.predictor.heads import PredictorHeads
            from grace_gc.core.rng import seed_all
        except Exception as exc:
            raise RuntimeError("Ridge/MLP diagnostics require the server torch environment") from exc
        kind = "ridge" if name == "ridge" else "mlp"
        seed_all(int(getattr(args, "seed", 17)) + (0 if kind == "ridge" else 1))
        heads = PredictorHeads(
            int(x_train.shape[1]),
            int(y_train.shape[1]),
            hidden_coord=int(args.mlp_hidden),
            coord_kind=kind,
            ridge_l2=float(args.ridge_l2),
            feature_scaler="refit",
            shrink_m=False,
            train_auxiliary=False,
        )
        heads.fit_feature_scaler(x_train)
        heads.scaler.fit_coord_scale(y_train)
        fit_loss = heads.train_coord(
            x_train,
            y_train,
            w_train,
            epochs=1 if kind == "ridge" else int(args.mlp_epochs),
        )
        pred = heads.predict_f(x_eval, shrink=False)
    return np.asarray(pred, dtype=np.float64), fit_loss, perf_counter() - started


def diagnose(bundles, *, seed=17, test_fraction=0.3, models=None, args=None):
    rows, feature_dim, target_dim = _rows_from_bundles(bundles)
    problem_ids = [row["problem_id"] for row in rows]
    train_problems, test_problems = _problem_split(problem_ids, seed, test_fraction)
    train = [row for row in rows if row["problem_id"] in train_problems]
    test = [row for row in rows if row["problem_id"] in test_problems]
    x_train = np.stack([row["x"] for row in train])
    y_train = np.stack([row["target"] for row in train])
    w_train = np.asarray([row["weight"] for row in train], dtype=np.float64)
    w_train = w_train / max(float(np.mean(w_train)), 1e-12)
    x_test = np.stack([row["x"] for row in test])
    test_weights = np.asarray([row["weight"] for row in test], dtype=np.float64)
    model_names = [str(x) for x in (models or ["zero", "simple", "ridge", "mlp"])]
    result_rows = []
    by_problem = []
    descriptions = {
        "zero": "m(h)=0",
        "simple": "constant coefficient equal to the weighted training-label mean",
        "ridge": "weighted ridge on all stored prefix features",
        "mlp": "two-layer coordinate MLP on all stored prefix features",
    }
    for name in model_names:
        pred, fit_loss, fit_seconds = _fit_predictions(name, x_train, y_train, w_train, x_test, args)
        metrics = _residuals(test, pred)
        zero_mean = _weighted_mean(metrics["zero_residual"], test_weights)
        item = {
            "model": name,
            "description": descriptions[name],
            "n_train_labels": len(train),
            "n_test_labels": len(test),
            "n_train_problems": len(train_problems),
            "n_test_problems": len(test_problems),
            "feature_dim": feature_dim,
            "target_dim": target_dim,
            "fit_loss": fit_loss,
            "fit_seconds": float(fit_seconds),
            "test_zero_residual_mean": float(zero_mean),
            "test_residual_mean": _weighted_mean(metrics["residual"], test_weights),
            "test_subspace_omission_mean": _weighted_mean(metrics["subspace_omission"], test_weights),
            "test_coordinate_error_mean": _weighted_mean(metrics["coordinate_error"], test_weights),
        }
        item["test_residual_ratio_to_zero"] = (
            item["test_residual_mean"] / zero_mean if zero_mean > 0 else None
        )
        result_rows.append(item)
        for problem_id in sorted(test_problems):
            indices = [i for i, row in enumerate(test) if row["problem_id"] == problem_id]
            local_weights = test_weights[indices]
            local_zero = _weighted_mean(metrics["zero_residual"][indices], local_weights)
            local_residual = _weighted_mean(metrics["residual"][indices], local_weights)
            by_problem.append({
                "model": name,
                "problem_id": problem_id,
                "n_bundles": len({test[i]["bundle_index"] for i in indices}),
                "n_labels": len(indices),
                "zero_residual_mean": local_zero,
                "residual_mean": local_residual,
                "residual_ratio_to_zero": local_residual / local_zero if local_zero > 0 else None,
                "subspace_omission_mean": _weighted_mean(
                    metrics["subspace_omission"][indices], local_weights,
                ),
                "coordinate_error_mean": _weighted_mean(
                    metrics["coordinate_error"][indices], local_weights,
                ),
            })
    oracle = np.stack([row["target"] for row in test])
    oracle_metrics = _residuals(test, oracle)
    oracle_zero = _weighted_mean(oracle_metrics["zero_residual"], test_weights)
    result = {
        "seed": int(seed),
        "split": {
            "unit": "problem_id",
            "test_fraction": float(test_fraction),
            "train_problem_ids": sorted(train_problems),
            "test_problem_ids": sorted(test_problems),
        },
        "n_bundles": len(bundles),
        "n_labels": len(rows),
        "models": result_rows,
        "by_problem": by_problem,
        "oracle_basis": {
            "test_residual_mean": _weighted_mean(oracle_metrics["residual"], test_weights),
            "test_residual_ratio_to_zero": (
                _weighted_mean(oracle_metrics["residual"], test_weights) / oracle_zero
                if oracle_zero > 0 else None
            ),
            "test_subspace_omission_mean": _weighted_mean(oracle_metrics["subspace_omission"], test_weights),
            "test_coordinate_error_mean": _weighted_mean(oracle_metrics["coordinate_error"], test_weights),
            "note": "Oracle coefficients are computed independently per realized suffix; they are a geometry ceiling, not a learned predictor.",
        },
        "future_randomness": _future_randomness([
            bundle for bundle in bundles if str(bundle.problem_id) in test_problems
        ]),
    }
    return result, result_rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Predictor diagnostics on a common frozen audit")
    parser.add_argument("--bundles", required=True, help="audit_bundles.jsonl or its run directory")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--test-fraction", type=float, default=0.3)
    parser.add_argument("--models", default="zero,simple,ridge,mlp")
    parser.add_argument("--ridge-l2", type=float, default=1.0)
    parser.add_argument("--mlp-hidden", type=int, default=64)
    parser.add_argument("--mlp-epochs", type=int, default=50)
    args = parser.parse_args(argv)
    if not 0.0 < float(args.test_fraction) < 1.0:
        raise ValueError("test-fraction must be in (0, 1)")
    if args.ridge_l2 < 0 or args.mlp_hidden <= 0 or args.mlp_epochs <= 0:
        raise ValueError("ridge-l2 must be nonnegative; mlp-hidden and mlp-epochs must be positive")
    models = [x.strip().lower() for x in str(args.models).split(",") if x.strip()]
    allowed = {"zero", "simple", "ridge", "mlp"}
    if not models or any(x not in allowed for x in models):
        raise ValueError(f"models must be a comma-separated subset of {sorted(allowed)}")
    bundles = _load_bundles(Path(args.bundles))
    result, rows = diagnose(
        bundles,
        seed=args.seed,
        test_fraction=args.test_fraction,
        models=models,
        args=args,
    )
    from grace_gc.logging_util.run_dir import RunDirectory, default_run_dir, resolve_run_dir, utc_now

    requested = args.run_dir or str(default_run_dir("predictor-diagnostic"))
    run = RunDirectory(resolve_run_dir(requested))
    started = utc_now()
    run.write_run_meta(kind="predictor-diagnostic", started=started, requested=requested)
    run.write_json("predictor_diagnostics.json", result)
    run.write_jsonl("predictor_diagnostics_rows.jsonl", rows)
    run.write_jsonl("predictor_diagnostics_by_problem.jsonl", result["by_problem"])
    run.write_json("config.json", {
        "bundles": str(Path(args.bundles).resolve()),
        "seed": int(args.seed),
        "test_fraction": float(args.test_fraction),
        "models": models,
        "ridge_l2": float(args.ridge_l2),
        "mlp_hidden": int(args.mlp_hidden),
        "mlp_epochs": int(args.mlp_epochs),
    })
    result["run_dir"] = str(run.root)
    print(json.dumps(result, indent=2))
    print("run_dir", run.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
