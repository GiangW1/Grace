"""Shared CPU helpers for the dynamic score-gradient experiments."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from grace_gc.audit.expected_gain import fit_predict, problem_split
from grace_gc.versions import sha256_file


def load_dynamic_replay(replay_dir: str | Path, score_gradients: str | Path):
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
    if provenance_path.is_file():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        expected_hash = sha256_file(root / "prefixes.jsonl")
        if provenance.get("replay_prefixes_sha256") != expected_hash:
            raise ValueError("score-gradient rows do not match replay prefixes")
        if provenance.get("shape") != list(map(int, basis.shape)):
            raise ValueError("score-gradient provenance shape does not match the array")
    if (target_a.ndim != 2 or target_b.shape != target_a.shape or
            norm_b.shape != (len(rows),) or basis.ndim != 3 or
            basis.shape[0] != len(rows) or basis.shape[1] != target_a.shape[1]):
        raise ValueError("dynamic replay arrays have incompatible shapes")
    arrays = (target_a, target_b, norm_b, basis)
    if any(not np.all(np.isfinite(np.asarray(array))) for array in arrays):
        raise ValueError("dynamic replay arrays must be finite")
    return rows, target_a, target_b, norm_b, basis


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


def fit_prefix_coefficients(basis, targets):
    """Fit the smallest per-prefix coefficients for ``basis @ coeff ≈ target``."""
    basis = np.asarray(basis, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if basis.ndim != 3 or targets.ndim != 2 or basis.shape[:2] != (len(targets), targets.shape[1]):
        raise ValueError("basis and target dimensions disagree")
    coefficients = np.empty((len(targets), basis.shape[2]), dtype=np.float64)
    for index, (matrix, target) in enumerate(zip(basis, targets)):
        coefficients[index] = np.linalg.lstsq(matrix, target, rcond=None)[0]
    return coefficients


def reconstruct(basis, coefficients):
    basis = np.asarray(basis, dtype=np.float64)
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if basis.ndim != 3 or coefficients.shape != (basis.shape[0], basis.shape[2]):
        raise ValueError("basis and coefficient dimensions disagree")
    return np.einsum("ndw,nw->nd", basis, coefficients)


def residual_metrics(basis, coefficients, target, norm_sq):
    prediction = reconstruct(basis, coefficients)
    target = np.asarray(target, dtype=np.float64)
    norm_sq = np.asarray(norm_sq, dtype=np.float64).reshape(-1)
    if target.shape != prediction.shape or norm_sq.shape != (len(prediction),):
        raise ValueError("residual labels and predictions disagree")
    residual = norm_sq - 2.0 * np.sum(prediction * target, axis=1) + np.sum(prediction * prediction, axis=1)
    zero = norm_sq
    mean_target_mse = np.mean((prediction - target) ** 2, axis=1)
    return {
        "residual_mean": float(np.mean(residual)),
        "zero_residual_mean": float(np.mean(zero)),
        "residual_ratio_to_zero": (None if float(np.mean(zero)) == 0.0
                                    else float(np.mean(residual) / np.mean(zero))),
        "mean_gradient_mse": float(np.mean(mean_target_mse)),
        "mean_gradient_energy": float(np.mean(np.sum(target * target, axis=1))),
    }


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
                              float(row.get("mean_cost", 0.)),
                              float(row.get("mean_reward", 0.)),
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
