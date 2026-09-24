"""CPU-side, full-space diagnostics for frozen prefix gradient means."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from grace_gc.predictor.basis import crossfit_ridge_operator, effective_rank
from grace_gc.predictor.scale import FeatureScaler, fit_weighted_ridge, ridge_predict


def load_replay(path: str | Path):
    root = Path(path)
    rows = [json.loads(line) for line in (root / "prefixes.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    means = np.load(root / "mean_grads.npy", mmap_mode="r")
    if means.ndim != 2 or means.shape[0] != len(rows):
        raise ValueError("replay gradient means and prefix metadata disagree")
    return rows, means


def problem_split(rows, seed: int = 17, counts=(160, 48, 48), prior_train=()):
    """Split by problem; cap requested counts gracefully for smaller pilots."""
    ids = sorted({str(row["problem_id"]) for row in rows})
    if len(ids) < 3:
        raise ValueError("need at least three distinct problems for train/validation/diagnostic")
    prior = set(map(str, prior_train)) & set(ids)
    ordered = list(sorted(prior)) + list(np.random.default_rng(seed).permutation(
        [pid for pid in ids if pid not in prior]))
    if len(ids) >= sum(counts):
        sizes = counts
    else:
        n_val = max(1, round(len(ids) * counts[1] / sum(counts)))
        n_test = max(1, round(len(ids) * counts[2] / sum(counts)))
        sizes = (len(ids) - n_val - n_test, n_val, n_test)
    if sizes[0] < 1 or len(prior) > sizes[0]:
        raise ValueError("not enough training problems to include the prior training set")
    train = set(ordered[:sizes[0]])
    val = set(ordered[sizes[0]:sizes[0] + sizes[1]])
    test = set(ordered[sizes[0] + sizes[1]:])
    return {"train": sorted(train), "validation": sorted(val), "diagnostic": sorted(test)}


def reference_gradient(reference_rows, reference_means, block: int = 32768):
    """Equal weight per reference problem, then equal weight across problems."""
    groups = {}
    for i, row in enumerate(reference_rows):
        groups.setdefault(str(row["problem_id"]), []).append(i)
    out = np.zeros(reference_means.shape[1], dtype=np.float64)
    for start in range(0, out.size, block):
        end = min(start + block, out.size)
        value = np.zeros(end - start, dtype=np.float64)
        for indices in groups.values():
            value += np.mean(reference_means[indices, start:end], axis=0)
        out[start:end] = value / len(groups)
    return out


def _train_gram(means, indices, block: int = 32768):
    gram = np.zeros((len(indices), len(indices)), dtype=np.float64)
    for start in range(0, means.shape[1], block):
        piece = np.asarray(means[indices, start:start + block], dtype=np.float64)
        gram += piece @ piece.T
    return (gram + gram.T) / 2


def fit_global_basis(means, indices, features, problem_ids, variant: str,
                     k: int, output: str | Path, block: int = 32768) -> dict:
    """Small-n Gram fit; never construct a D-by-D covariance matrix."""
    if variant not in {"mean_svd", "predictable_crossfit_signal"} or k not in {8, 64}:
        raise ValueError("basis variant or k is outside the reviewed global experiment")
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) == 0 or np.any(indices < 0) or np.any(indices >= len(means)):
        raise ValueError("basis training indices are invalid")
    gram = _train_gram(means, indices, block)
    eigen, left = np.linalg.eigh(gram)
    live = eigen > max(float(eigen.max()), 0.0) * 1e-12
    if not np.any(live):
        raise ValueError("training prefix means have no nonzero gradient rank")
    values = eigen[live]
    left = left[:, live]
    roots = np.sqrt(values)
    metrics = {"variant": variant, "requested_k": int(k),
               "training_rows": len(indices), "training_mean_energy": float(np.trace(gram))}
    if variant == "mean_svd":
        order = np.argsort(values)[::-1]
        rank = effective_rank(roots[order], min(k, len(order)))
        coeff = (left[:, order[:rank]] / roots[order[:rank]])
    else:
        operator, available = crossfit_ridge_operator(
            np.asarray(features, dtype=np.float64), problem_ids, ridge_l2=1.0)
        if not np.all(available):
            raise ValueError("predictable basis needs other training problems for every row")
        b = (operator + operator.T) / (2 * len(indices))
        small = roots[:, None] * (left.T @ b @ left) * roots[None, :]
        signal, vectors = np.linalg.eigh((small + small.T) / 2)
        positive = np.flatnonzero(signal > max(float(np.max(np.abs(signal))), 0.0) * 1e-12)[::-1]
        rank = min(k, len(positive))
        coeff = (left / roots) @ vectors[:, positive[:rank]]
        metrics["positive_signal_rank"] = len(positive)
    if rank == 0:
        raise ValueError(f"{variant} has no valid basis direction")
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    u = np.lib.format.open_memmap(destination, mode="w+", dtype=np.float64,
                                  shape=(means.shape[1], rank))
    for start in range(0, means.shape[1], block):
        end = min(start + block, means.shape[1])
        piece = np.asarray(means[indices, start:end], dtype=np.float64)
        u[start:end] = piece.T @ coeff
    u.flush()
    metrics.update(actual_rank=int(rank), dimension=int(means.shape[1]),
                   basis_gram=np.asarray(u.T @ u).tolist())
    return metrics


def project_means(means, basis, block: int = 32768):
    if means.shape[1] != basis.shape[0]:
        raise ValueError("gradient and basis dimensions disagree")
    coords = np.zeros((means.shape[0], basis.shape[1]), dtype=np.float64)
    for start in range(0, means.shape[1], block):
        end = min(start + block, means.shape[1])
        coords += np.asarray(means[:, start:end], dtype=np.float64) @ basis[start:end]
    return coords


def fit_predict(model: str, x_train, y_train, x_eval, ridge_l2: float = 1.,
                mlp_hidden: int = 64, mlp_epochs: int = 50, seed: int = 17,
                device: str = "cpu"):
    y = np.asarray(y_train, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    if model == "zero":
        return np.zeros((len(x_eval), y.shape[1]), dtype=np.float64)
    if model == "constant":
        return np.broadcast_to(np.mean(y, axis=0), (len(x_eval), y.shape[1])).copy()
    if model == "mlp":
        from scripts.diagnose_predictor_suite import _fit_mlp
        return _fit_mlp(x_train, y, np.ones(len(y)), x_eval, hidden=mlp_hidden,
                        epochs=mlp_epochs, seed=seed, device=device)
    if model == "ridge":
        scaler = FeatureScaler()
        scaler.fit_features(x_train)
        scale = np.sqrt(np.mean(y * y, axis=0))
        scale = np.where(scale > 1e-8, scale, 1.0)
        coef = fit_weighted_ridge(scaler.transform(x_train), y / scale,
                                  np.ones(len(y)), l2=ridge_l2)
        return ridge_predict(scaler.transform(x_eval), coef) * scale
    raise ValueError(f"unknown predictor: {model}")


def full_space_residual(mean_norm_sq, mean_coords, predicted_coords, basis_gram=None):
    """E||G-Uf||² from exact E||G||² and E[G], no stored suffix vectors."""
    norms = np.asarray(mean_norm_sq, dtype=np.float64).reshape(-1)
    target = np.asarray(mean_coords, dtype=np.float64)
    pred = np.asarray(predicted_coords, dtype=np.float64)
    gram = (np.eye(pred.shape[1]) if basis_gram is None else np.asarray(basis_gram))
    if target.shape != pred.shape or norms.shape != (len(pred),):
        raise ValueError("residual label and prediction shapes disagree")
    return norms - 2 * np.sum(pred * target, axis=1) + np.sum((pred @ gram) * pred, axis=1)


def ridge_oof_by_problem(features, targets, problem_ids, l2=1.0):
    """Train-side out-of-problem predictions for honest residual labels."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    was_vector = y.ndim == 1
    if was_vector:
        y = y[:, None]
    out = np.zeros_like(y)
    pids = np.asarray(problem_ids)
    for pid in np.unique(pids):
        held = np.flatnonzero(pids == pid)
        train = np.flatnonzero(pids != pid)
        if len(train) == 0:
            raise ValueError("out-of-problem prediction needs more than one training problem")
        out[held] = fit_predict("ridge", x[train], y[train], x[held], ridge_l2=l2)
    return out[:, 0] if was_vector else out
