"""Shared CPU helpers for the dynamic score-gradient experiments."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from grace_gc.audit.expected_gain import fit_predict, problem_split
from grace_gc.versions import sha256_array, sha256_file


def load_diagonal_metric(path: str | Path | None, dimension: int):
    """Load a fixed nonnegative diagonal quadratic metric."""
    if path is None:
        return np.ones(int(dimension), dtype=np.float64)
    values = np.asarray(np.load(path), dtype=np.float64).reshape(-1)
    if values.shape != (int(dimension),) or not np.all(np.isfinite(values)):
        raise ValueError("metric file must contain one finite weight per gradient coordinate")
    if np.any(values < 0.0) or not np.any(values > 0.0):
        raise ValueError("metric weights must be nonnegative with at least one positive entry")
    return values


def load_dynamic_replay(replay_dir: str | Path, score_gradients: str | Path,
                        metric_file: str | Path | None = None):
    """Load the replay rows, A/B labels, and per-prefix score bases.

    The A/B arrays are deliberately separate from ``mean_grads.npy``.  A
    coefficient is fitted only on A and evaluated on the independent B draw.
    """
    root = Path(replay_dir)
    rows = [json.loads(line) for line in (root / "prefixes.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()]
    required = {
        "half_mean_grads_a.npy", "half_mean_grads_b.npy",
        "half_mean_norm_sq_b.npy",
    }
    missing = sorted(name for name in required if not (root / name).is_file())
    if missing:
        raise ValueError("replay lacks stage A/B sidecars: " + ", ".join(missing))
    target_a = np.load(root / "half_mean_grads_a.npy", mmap_mode="r")
    target_b = np.load(root / "half_mean_grads_b.npy", mmap_mode="r")
    norm_b = np.load(root / "half_mean_norm_sq_b.npy", mmap_mode="r")
    basis = np.load(score_gradients, mmap_mode="r")
    provenance_path = Path(score_gradients).with_name("score_gradient_provenance.json")
    score_provenance = {}
    if provenance_path.is_file():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        score_provenance = provenance
        expected_hash = sha256_file(root / "prefixes.jsonl")
        if provenance.get("replay_prefixes_sha256") != expected_hash:
            raise ValueError("score-gradient rows do not match replay prefixes")
        if provenance.get("shape") != list(map(int, basis.shape)):
            raise ValueError("score-gradient provenance shape does not match the array")
    replay_provenance_path = root / "replay_provenance.json"
    replay_provenance = (json.loads(replay_provenance_path.read_text(encoding="utf-8"))
                         if replay_provenance_path.is_file() else {})
    for key in ("actor_sha256", "checkpoint_sha256", "layout_dim", "layout_names"):
        expected = replay_provenance.get(key)
        observed = score_provenance.get(key)
        if expected is not None and observed is not None and expected != observed:
            raise ValueError(f"score-gradient and replay provenance differ in {key}")
    if (target_a.ndim != 2 or target_b.shape != target_a.shape or
            norm_b.shape != (len(rows),) or basis.ndim != 3 or
            basis.shape[0] != len(rows) or basis.shape[1] != target_a.shape[1]):
        raise ValueError("dynamic replay arrays have incompatible shapes")
    metric_path = metric_file
    if metric_path is None and (root / "metric_weights.npy").is_file():
        metric_path = root / "metric_weights.npy"
    metric = load_diagonal_metric(metric_path, target_a.shape[1])
    metric_name = str(replay_provenance.get("metric_name") or
                      ("diagonal_file" if metric_path is not None else "euclidean"))
    metric_hash = sha256_array(metric)
    if (replay_provenance.get("metric_name") == "euclidean" and
            not np.all(metric == 1.0)):
        raise ValueError("requested metric differs from the replay metric")
    recorded_metric_hash = replay_provenance.get("metric_weights_sha256")
    if recorded_metric_hash is not None and recorded_metric_hash != metric_hash:
        raise ValueError("metric weights do not match replay provenance")
    metric_norm_path = root / "half_metric_norm_sq_b.npy"
    metric_norm_b = (np.load(metric_norm_path, mmap_mode="r") if metric_norm_path.is_file()
                     else norm_b if np.all(metric == 1.0) else None)
    if metric_norm_b is None:
        raise ValueError("replay lacks half_metric_norm_sq_b.npy for the requested metric")
    if metric_norm_b.shape != (len(rows),):
        raise ValueError("metric second-moment labels have the wrong shape")
    arrays = (target_a, target_b, norm_b, metric_norm_b, basis, metric)
    if any(not np.all(np.isfinite(np.asarray(array))) for array in arrays):
        raise ValueError("dynamic replay arrays must be finite")
    return rows, target_a, target_b, metric_norm_b, basis, metric, metric_name


def split_roles(rows, manifest: str | Path | None, seed: int):
    split = (json.loads(Path(manifest).read_text(encoding="utf-8")) if manifest
             else problem_split(rows, seed=seed))
    ids = {str(row["problem_id"]) for row in rows}
    if (set(split) != {"train", "validation", "diagnostic"} or
            set().union(*(set(map(str, split[name])) for name in split)) != ids or
            any(set(map(str, split[a])) & set(map(str, split[b]))
                for a, b in (("train", "validation"), ("train", "diagnostic"),
                             ("validation", "diagnostic")))):
        raise ValueError("split must partition every replay problem exactly once")
    indices = {}
    for name in ("train", "validation", "diagnostic"):
        wanted = set(map(str, split[name]))
        indices[name] = np.asarray(
            [i for i, row in enumerate(rows) if str(row["problem_id"]) in wanted],
            dtype=np.int64)
        if len(indices[name]) == 0:
            raise ValueError(f"split role {name} is empty")
    if sum(len(value) for value in indices.values()) != len(rows):
        raise ValueError("split must cover every replay row")
    return split, indices


def fit_prefix_coefficients(basis, targets, metric=None):
    """Fit per-prefix coefficients under the same diagonal metric as scoring."""
    basis = np.asarray(basis, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if basis.ndim != 3 or targets.ndim != 2 or basis.shape[:2] != (len(targets), targets.shape[1]):
        raise ValueError("basis and target dimensions disagree")
    weights = (np.ones(targets.shape[1], dtype=np.float64) if metric is None else
               np.asarray(metric, dtype=np.float64).reshape(-1))
    if (weights.shape != (targets.shape[1],) or not np.all(np.isfinite(weights)) or
            np.any(weights < 0.0) or not np.any(weights > 0.0)):
        raise ValueError("metric has the wrong shape or invalid weights")
    root_weights = np.sqrt(weights)
    coefficients = np.empty((len(targets), basis.shape[2]), dtype=np.float64)
    for index, (matrix, target) in enumerate(zip(basis, targets)):
        coefficients[index] = np.linalg.lstsq(matrix * root_weights[:, None],
                                               target * root_weights, rcond=None)[0]
    return coefficients


def reconstruct(basis, coefficients):
    basis = np.asarray(basis, dtype=np.float64)
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if basis.ndim != 3 or coefficients.shape != (basis.shape[0], basis.shape[2]):
        raise ValueError("basis and coefficient dimensions disagree")
    return np.einsum("ndw,nw->nd", basis, coefficients)


def _residual_metrics_with_metric(basis, coefficients, target, norm_sq, metric):
    prediction = reconstruct(basis, coefficients)
    target = np.asarray(target, dtype=np.float64)
    norm_sq = np.asarray(norm_sq, dtype=np.float64).reshape(-1)
    metric = np.asarray(metric, dtype=np.float64).reshape(-1)
    if target.shape != prediction.shape or norm_sq.shape != (len(prediction),):
        raise ValueError("residual labels and predictions disagree")
    if metric.shape != (prediction.shape[1],) or np.any(metric < 0) or not np.all(np.isfinite(metric)):
        raise ValueError("metric has the wrong shape or invalid weights")
    residual = norm_sq - 2.0 * np.sum(metric * prediction * target, axis=1) + np.sum(
        metric * prediction * prediction, axis=1)
    zero = norm_sq
    mean_target_mse = np.mean((prediction - target) ** 2, axis=1)
    return {
        "residual_mean": float(np.mean(residual)),
        "zero_residual_mean": float(np.mean(zero)),
        "residual_ratio_to_zero": (None if float(np.mean(zero)) == 0.0
                                    else float(np.mean(residual) / np.mean(zero))),
        "mean_gradient_mse": float(np.mean(mean_target_mse)),
        "mean_gradient_energy": float(np.mean(np.sum(target * target, axis=1))),
        "metric_weighted_target_energy": float(np.mean(np.sum(metric * target * target, axis=1))),
    }


def residual_metrics(basis, coefficients, target, norm_sq, metric=None):
    metric = (np.ones(np.asarray(target).shape[1], dtype=np.float64)
              if metric is None else np.asarray(metric, dtype=np.float64))
    return _residual_metrics_with_metric(basis, coefficients, target, norm_sq, metric)


def coefficient_diagnostics(basis):
    basis = np.asarray(basis, dtype=np.float64)
    ranks, condition_numbers = [], []
    for matrix in basis:
        singular = np.linalg.svd(matrix, compute_uv=False)
        tolerance = max(float(singular[0]), 0.0) * max(matrix.shape) * np.finfo(np.float64).eps
        rank = int(np.count_nonzero(singular > tolerance)) if len(singular) else 0
        ranks.append(rank)
        condition_numbers.append(None if not len(singular) or singular[-1] <= tolerance
                                else float(singular[0] / singular[-1]))
    finite_conditions = [value for value in condition_numbers if value is not None]
    return {"n_prefixes": int(len(basis)), "rank_min": int(min(ranks, default=0)),
            "rank_max": int(max(ranks, default=0)),
            "condition_number_max": (None if not finite_conditions else max(finite_conditions)),
            "condition_number_median": (None if not finite_conditions else
                                         float(np.median(finite_conditions)))}


def feature_matrix(rows, feature_set: str):
    def stack(name):
        values = [row.get(name) for row in rows]
        if any(value is None for value in values):
            raise ValueError(f"replay lacks {name}; regenerate it with the matching feature option")
        result = np.asarray(values, dtype=np.float64)
        if result.ndim != 2 or not np.all(np.isfinite(result)):
            raise ValueError(f"{name} is not a finite feature matrix")
        return result

    if feature_set == "legacy":
        return stack("features")
    if feature_set == "prompt":
        return stack("prompt_features")
    if feature_set == "legacy_plus_prompt":
        return np.concatenate((stack("features"), stack("prompt_features")), axis=1)
    if feature_set == "cheap":
        result = np.asarray([[float(len(row.get("prefix_token_ids") or [])),
                              float(row.get("baseline", 0.))]
                             for row in rows], dtype=np.float64)
        if not np.all(np.isfinite(result)):
            raise ValueError("cheap features are not finite")
        return result
    raise ValueError(f"unknown feature set: {feature_set}")


def predict_coefficients(model, features, labels, train, evaluate, l2=1.0,
                         mlp_hidden=64, mlp_epochs=50, seed=17, device="cpu"):
    return fit_predict(model, features[train], labels[train], features[evaluate],
                       ridge_l2=l2 if model == "ridge" else 1.0,
                       mlp_hidden=mlp_hidden, mlp_epochs=mlp_epochs,
                       seed=seed, device=device)
