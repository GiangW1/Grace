#!/usr/bin/env python3
"""Run offline predictor experiments on one frozen prefix audit.

The suite separates three effects that were mixed in the first diagnostic:

* suffix randomness: multiple continuations from one prefix are averaged;
* basis/coordinate error: predictions are evaluated inside the stored U space;
* model error: zero, constant, ridge, MLP, and shuffled-label controls share
  the same problem-level splits.

The script does not run another rollout.  Generate one larger audit with
``predictor_suite_audit.yaml`` on the A100, then run this script repeatedly or
in parallel on its ``audit_bundles.jsonl``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grace_gc.audit.prefix_audit import bundle_from_dict
from grace_gc.logging_util.run_dir import RunDirectory, default_run_dir, resolve_run_dir, utc_now
from grace_gc.predictor.scale import FeatureScaler, fit_weighted_ridge, ridge_predict


@dataclass
class PrefixExample:
    key: str
    problem_id: str
    t: int
    features: np.ndarray
    coord_targets: np.ndarray
    coord_mean: np.ndarray
    coord_variance: float
    rewards: np.ndarray
    reward_mean: float
    reward_variance: float
    costs: np.ndarray
    cost_mean: float
    cost_variance: float
    norms: np.ndarray
    gram: np.ndarray


_NEXT_AFTER_GRADS = re.compile(r',\s*"[A-Za-z_][A-Za-z_0-9]*"\s*:')


def _without_large_grads(line: str) -> str:
    """Remove the stored raw gradient array before JSON parsing.

    The exact full-space norm and U.T@G labels remain in later fields.  This
    avoids constructing millions of Python floats per JSONL row when reading
    legacy audits that did not set ``audit.jl_dim``.
    """
    key = re.search(r'"grads"\s*:', line)
    if key is None:
        return line
    value_start = key.end()
    while value_start < len(line) and line[value_start].isspace():
        value_start += 1
    if value_start >= len(line) or line[value_start] != "[":
        return line
    next_field = _NEXT_AFTER_GRADS.search(line, value_start)
    if next_field is not None:
        return line[:value_start] + "null" + line[next_field.start():]
    # A minimal JSONL may put grads last. The large project audits always
    # have later fields, so this fallback is only for small/custom inputs.
    depth = 0
    for index in range(value_start, len(line)):
        if line[index] == "[":
            depth += 1
        elif line[index] == "]":
            depth -= 1
            if depth == 0:
                return line[:value_start] + "null" + line[index + 1:]
    raise ValueError("audit JSONL grads array is not closed")


def _load_bundles(path: Path):
    if path.is_dir():
        path = path / "audit_bundles.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"bundle file is not readable: {path}")
    bundles = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                bundles.append(bundle_from_dict(json.loads(_without_large_grads(line))))
    if not bundles:
        raise ValueError(f"bundle file is empty: {path}")
    return bundles


def _weighted_mean(values, weights=None) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not values.size:
        return 0.0
    if weights is None:
        return float(np.mean(values))
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    return float(np.sum(values * weights) / max(float(np.sum(weights)), 1e-12))


def _sample_variance(values) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size <= 1:
        return 0.0
    return float(np.var(values, ddof=1))


def _prefix_examples(bundles) -> list[PrefixExample]:
    examples = []
    feature_dim = None
    target_dim = None
    for bundle_index, bundle in enumerate(bundles):
        if bundle.features is None or bundle.true_grad_coords is None:
            raise ValueError(
                "every bundle needs features and true_grad_coords; "
                "rerun the audit with store_features: true"
            )
        coords = np.asarray(bundle.true_grad_coords, dtype=np.float64)
        norms = np.asarray(bundle.true_grad_norm_sq, dtype=np.float64).reshape(-1)
        gram = np.asarray(bundle.basis_gram, dtype=np.float64)
        rewards = np.asarray(bundle.rewards, dtype=np.float64).reshape(-1)
        if bundle.suffix_cost is None:
            raise ValueError(f"bundle {bundle_index} has no suffix_cost for cost prediction")
        costs = np.asarray(bundle.suffix_cost, dtype=np.float64).reshape(-1)
        features = np.asarray(bundle.features, dtype=np.float64).reshape(-1)
        if (coords.ndim != 2 or len(coords) == 0 or len(norms) != len(coords)
                or len(rewards) != len(coords) or len(costs) != len(coords)
                or gram.shape != (coords.shape[1], coords.shape[1])):
            raise ValueError(f"bundle {bundle_index} has inconsistent labels")
        if (not np.all(np.isfinite(features)) or not np.all(np.isfinite(coords))
                or not np.all(np.isfinite(norms)) or not np.all(np.isfinite(gram))
                or not np.all(np.isfinite(rewards)) or not np.all(np.isfinite(costs))):
            raise ValueError(f"bundle {bundle_index} contains non-finite values")
        if feature_dim is None:
            feature_dim, target_dim = int(features.size), int(coords.shape[1])
        if features.size != feature_dim or coords.shape[1] != target_dim:
            raise ValueError("all bundles must share feature and coordinate dimensions")
        gram_inverse = np.linalg.pinv(gram)
        targets = coords @ gram_inverse
        coord_mean = targets.mean(axis=0)
        centered = targets - coord_mean
        coord_variance = float(np.sum((centered @ gram) * centered) / max(len(targets) - 1, 1))
        problem_id = str(bundle.problem_id)
        path_id = str(bundle.path_id) if bundle.path_id is not None else str(bundle_index)
        examples.append(PrefixExample(
            key=f"{problem_id}|t={int(bundle.t)}|path={path_id}",
            problem_id=problem_id,
            t=int(bundle.t),
            features=features,
            coord_targets=targets,
            coord_mean=coord_mean,
            coord_variance=max(coord_variance, 0.0),
            rewards=rewards,
            reward_mean=float(rewards.mean()),
            reward_variance=_sample_variance(rewards),
            costs=costs,
            cost_mean=float(costs.mean()),
            cost_variance=_sample_variance(costs),
            norms=norms,
            gram=gram,
        ))
    return examples


def _split(examples: list[PrefixExample], seed: int, test_fraction: float):
    problems = sorted({item.problem_id for item in examples})
    if len(problems) < 2:
        raise ValueError("suite needs at least two distinct problem IDs")
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(problems))
    n_test = min(len(problems) - 1, max(1, int(round(len(problems) * float(test_fraction)))))
    test = {problems[int(i)] for i in order[:n_test]}
    train = {pid for pid in problems if pid not in test}
    return ([item for item in examples if item.problem_id in train],
            [item for item in examples if item.problem_id in test],
            sorted(train), sorted(test))


def _feature_view(features: np.ndarray, name: str) -> np.ndarray:
    """Slice the stored legacy feature vector without another actor forward."""
    x = np.asarray(features, dtype=np.float64).reshape(-1)
    if name == "all":
        return x
    if x.size < 5 or (x.size - 5) % 3:
        raise ValueError("feature ablations require the legacy 3H+5 layout; use --feature-sets all")
    hidden = (x.size - 5) // 3
    views = {
        "last": x[:hidden],
        "middle": x[hidden:2 * hidden],
        "pooled": x[2 * hidden:3 * hidden],
        "hidden": x[:3 * hidden],
        "entropy": x[3 * hidden:3 * hidden + 3],
        "length_baseline": x[-2:],
        "no_baseline": x[:-1],
        "no_length_baseline": x[:-2],
    }
    if name not in views:
        raise ValueError(f"unknown feature set {name!r}; available: all,{','.join(views)}")
    return views[name]


def _fit_mlp(x_train, y_train, weights, x_eval, *, hidden: int, epochs: int, seed: int, device: str):
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:
        raise RuntimeError("MLP suite rows require the torch environment") from exc
    torch.manual_seed(int(seed))
    scaler = FeatureScaler()
    scaler.fit_features(x_train)
    x_fit = scaler.transform(x_train)
    y_fit = np.asarray(y_train, dtype=np.float64)
    y_scale = np.sqrt(np.mean(y_fit * y_fit, axis=0))
    y_scale = np.where(np.isfinite(y_scale) & (y_scale > 1e-8), y_scale, 1.0)
    x_t = torch.as_tensor(x_fit, dtype=torch.float32)
    y_t = torch.as_tensor(y_fit / y_scale, dtype=torch.float32)
    w_t = torch.as_tensor(np.asarray(weights, dtype=np.float32).reshape(-1))
    net = nn.Sequential(nn.Linear(x_fit.shape[1], int(hidden)), nn.ReLU(),
                        nn.Linear(int(hidden), y_fit.shape[1]))
    requested = str(device)
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--device=cuda requested but CUDA is unavailable")
    net = net.to(requested)
    x_t, y_t, w_t = x_t.to(requested), y_t.to(requested), w_t.to(requested)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.0)
    for _ in range(int(epochs)):
        optimizer.zero_grad()
        pred = net(x_t)
        loss = (w_t * ((pred - y_t) ** 2).sum(dim=1)).mean()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        pred = net(torch.as_tensor(scaler.transform(x_eval), dtype=torch.float32, device=requested))
    return pred.detach().cpu().numpy().astype(np.float64) * y_scale


def _fit_model(name, x_train, y_train, x_eval, weights, args, *, target_name, shuffle_seed):
    if name == "zero":
        return np.zeros((len(x_eval), y_train.shape[1]), dtype=np.float64)
    if name == "simple":
        mean = np.average(y_train, axis=0, weights=weights)
        return np.broadcast_to(mean, (len(x_eval), len(mean))).copy()
    train_y = y_train
    if name.startswith("shuffled_"):
        rng = np.random.default_rng(int(shuffle_seed))
        train_y = y_train[rng.permutation(len(y_train))]
        name = name[len("shuffled_"):]
    if name == "ridge":
        scaler = FeatureScaler()
        scaler.fit_features(x_train)
        scaled_x = scaler.transform(x_train)
        scaled_eval = scaler.transform(x_eval)
        y_scale = np.sqrt(np.mean(train_y * train_y, axis=0))
        y_scale = np.where(np.isfinite(y_scale) & (y_scale > 1e-8), y_scale, 1.0)
        coef = fit_weighted_ridge(scaled_x, train_y / y_scale, weights, l2=float(args.ridge_l2))
        return ridge_predict(scaled_eval, coef) * y_scale
    if name == "mlp":
        return _fit_mlp(x_train, train_y, weights, x_eval, hidden=args.mlp_hidden,
                        epochs=args.mlp_epochs, seed=shuffle_seed, device=args.device)
    raise ValueError(f"unknown model {name!r} for {target_name}")


def _coordinate_metrics(test: list[PrefixExample], predictions: np.ndarray) -> dict:
    mean_errors, mean_zero_errors = [], []
    future_residuals, zero_residuals = [], []
    omission, coordinate_error = [], []
    for item, pred in zip(test, predictions):
        pred = np.asarray(pred, dtype=np.float64)
        mean_error = float((pred - item.coord_mean) @ item.gram @ (pred - item.coord_mean))
        mean_errors.append(max(mean_error, 0.0))
        mean_zero_errors.append(max(float(item.coord_mean @ item.gram @ item.coord_mean), 0.0))
        item_residuals, item_zero, item_omission, item_coord_error = [], [], [], []
        for coord, norm in zip(item.coord_targets, item.norms):
            # coord is the projected coefficient (U.T U)^+ U.T G, not U.T G.
            captured = max(float(coord @ item.gram @ coord), 0.0)
            om = max(float(norm) - captured, 0.0)
            delta = coord - pred
            ce = max(float(delta @ item.gram @ delta), 0.0)
            item_residuals.append(om + ce)
            item_zero.append(float(norm))
            item_omission.append(om)
            item_coord_error.append(ce)
        # Equal weight per prefix, including naturally finished one-row prefixes.
        future_residuals.append(_weighted_mean(item_residuals))
        zero_residuals.append(_weighted_mean(item_zero))
        omission.append(_weighted_mean(item_omission))
        coordinate_error.append(_weighted_mean(item_coord_error))
    zero = _weighted_mean(zero_residuals)
    residual = _weighted_mean(future_residuals)
    mean_zero = _weighted_mean(mean_zero_errors)
    mean_pred = _weighted_mean(mean_errors)
    noise_floor = _weighted_mean([
        item.coord_variance / len(item.coord_targets) for item in test
    ])
    return {
        "test_prefix_mean_coordinate_error": mean_pred,
        "test_prefix_mean_zero_error": mean_zero,
        "test_prefix_mean_ratio_to_zero": mean_pred / mean_zero if mean_zero > 0 else None,
        "test_prefix_mean_noise_floor_estimate": noise_floor,
        "test_prefix_mean_error_minus_noise_floor": mean_pred - noise_floor,
        "test_suffix_residual_mean": residual,
        "test_zero_residual_mean": zero,
        "test_subspace_omission_mean": _weighted_mean(omission),
        "test_coordinate_error_mean": _weighted_mean(coordinate_error),
        "test_residual_ratio_to_zero": residual / zero if zero > 0 else None,
    }


def _scalar_metrics(test: list[PrefixExample], predictions: np.ndarray,
                    target_name: str, train_mean: float) -> dict:
    actual = np.asarray([getattr(item, target_name) for item in test], dtype=np.float64)
    pred = np.asarray(predictions, dtype=np.float64).reshape(-1)
    error = pred - actual
    baseline = float(np.mean((actual - float(train_mean)) ** 2))
    return {
        "test_scalar_mse": float(np.mean(error * error)),
        "test_scalar_mae": float(np.mean(np.abs(error))),
        "test_scalar_target_mean": float(actual.mean()),
        "test_scalar_target_variance": _sample_variance(actual),
        "test_scalar_mse_vs_train_mean": baseline,
        "test_scalar_r2_vs_train_mean": 1.0 - float(np.mean(error * error)) / baseline if baseline > 0 else None,
    }


def _run_split(examples, train, test, train_ids, test_ids, *, seed, args,
               feature_set, train_fraction):
    if train_fraction < 1.0:
        shuffled_ids = np.random.default_rng(int(seed) + 7919).permutation(train_ids)
        n_keep = max(1, int(round(len(train_ids) * train_fraction)))
        chosen_ids = set(str(x) for x in shuffled_ids[:n_keep])
        train = [item for item in train if item.problem_id in chosen_ids]
        train_ids = sorted(chosen_ids)
    x_train = np.stack([_feature_view(item.features, feature_set) for item in train])
    x_test = np.stack([_feature_view(item.features, feature_set) for item in test])
    weights = np.ones(len(train), dtype=np.float64)
    rows = []
    coord_y = np.stack([item.coord_mean for item in train])
    scalar_targets = [
        ("reward_mean", np.asarray([[item.reward_mean] for item in train]), "reward_mean"),
        ("cost_mean", np.asarray([[item.cost_mean] for item in train]), "cost_mean"),
    ]
    model_names = [str(x) for x in args.models]
    if args.include_shuffled:
        model_names += [f"shuffled_{x}" for x in model_names if x in {"ridge", "mlp"}]
    for model_name in model_names:
        pred = _fit_model(model_name, x_train, coord_y, x_test, weights, args,
                          target_name="coord_mean", shuffle_seed=seed + 1009)
        row = {"target": "coord_mean", "model": model_name, "split_seed": int(seed),
               "feature_set": feature_set, "train_fraction": float(train_fraction),
               "n_train_prefixes": len(train), "n_test_prefixes": len(test),
               "n_train_problems": len(train_ids), "n_test_problems": len(test_ids),
               "train_problem_ids": train_ids, "test_problem_ids": test_ids}
        row.update(_coordinate_metrics(test, pred))
        rows.append(row)
    for target_name, y_train, _ in scalar_targets:
        for model_name in model_names:
            pred = _fit_model(model_name, x_train, y_train, x_test, weights, args,
                              target_name=target_name, shuffle_seed=seed + 2003)
            row = {"target": target_name, "model": model_name, "split_seed": int(seed),
                   "feature_set": feature_set, "train_fraction": float(train_fraction),
                   "n_train_prefixes": len(train), "n_test_prefixes": len(test),
                   "n_train_problems": len(train_ids), "n_test_problems": len(test_ids)}
            row.update(_scalar_metrics(test, pred, target_name, float(np.mean(y_train))))
            rows.append(row)
    return rows


def _summary(rows, examples):
    grouped = {}
    for row in rows:
        key = (row["target"], row["model"], row["feature_set"], row["train_fraction"])
        grouped.setdefault(key, []).append(row)
    aggregate = []
    metric_keys = sorted({key for row in rows for key, value in row.items()
                          if key.startswith("test_") and isinstance(value, (int, float))})
    for (target, model, feature_set, train_fraction), items in sorted(grouped.items()):
        out = {"target": target, "model": model, "feature_set": feature_set,
               "train_fraction": float(train_fraction), "n_splits": len(items)}
        for key in metric_keys:
            values = [float(item[key]) for item in items if item.get(key) is not None]
            if values:
                out[f"{key}_mean"] = float(np.mean(values))
                out[f"{key}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        aggregate.append(out)
    variance = {
        "n_prefixes": len(examples),
        "n_problems": len({item.problem_id for item in examples}),
        "min_suffixes_per_prefix": min(len(item.coord_targets) for item in examples),
        "max_suffixes_per_prefix": max(len(item.coord_targets) for item in examples),
        "mean_coordinate_future_variance": _weighted_mean([item.coord_variance for item in examples]),
        "mean_reward_future_variance": _weighted_mean([item.reward_variance for item in examples]),
        "mean_cost_future_variance": _weighted_mean([item.cost_variance for item in examples]),
        "by_t": [],
    }
    for t in sorted({item.t for item in examples}):
        subset = [item for item in examples if item.t == t]
        variance["by_t"].append({"t": int(t), "n_prefixes": len(subset),
                                  "mean_coordinate_future_variance": _weighted_mean([x.coord_variance for x in subset]),
                                  "mean_reward_future_variance": _weighted_mean([x.reward_variance for x in subset]),
                                  "mean_cost_future_variance": _weighted_mean([x.cost_variance for x in subset])})
    return {"aggregate": aggregate, "future_randomness": variance}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Offline predictor suite on a frozen prefix audit")
    parser.add_argument("--bundles", required=True, help="audit_bundles.jsonl or its run directory")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--split-repeats", type=int, default=5)
    parser.add_argument("--test-fraction", type=float, default=0.3)
    parser.add_argument("--models", default="zero,simple,ridge,mlp")
    parser.add_argument("--include-shuffled", action="store_true")
    parser.add_argument("--feature-sets", default="all",
                        help="comma-separated legacy feature slices; e.g. all,last,pooled,entropy")
    parser.add_argument("--train-fractions", default="1.0",
                        help="comma-separated fractions of training problems; test problems remain fixed")
    parser.add_argument("--ridge-l2", type=float, default=1.0)
    parser.add_argument("--mlp-hidden", type=int, default=64)
    parser.add_argument("--mlp-epochs", type=int, default=50)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    if not 0.0 < float(args.test_fraction) < 1.0 or args.split_repeats <= 0:
        raise ValueError("test-fraction must be in (0, 1) and split-repeats must be positive")
    if args.ridge_l2 < 0 or args.mlp_hidden <= 0 or args.mlp_epochs <= 0:
        raise ValueError("ridge-l2 must be nonnegative; mlp-hidden and mlp-epochs must be positive")
    args.models = [x.strip().lower() for x in str(args.models).split(",") if x.strip()]
    allowed = {"zero", "simple", "ridge", "mlp"}
    if not args.models or any(x not in allowed for x in args.models):
        raise ValueError(f"models must be a comma-separated subset of {sorted(allowed)}")
    feature_sets = [x.strip().lower() for x in str(args.feature_sets).split(",") if x.strip()]
    train_fractions = [float(x) for x in str(args.train_fractions).split(",") if x.strip()]
    if not feature_sets or not train_fractions or any(not 0.0 < x <= 1.0 for x in train_fractions):
        raise ValueError("feature-sets must be nonempty and train-fractions must be in (0, 1]")
    bundles = _load_bundles(Path(args.bundles))
    examples = _prefix_examples(bundles)
    for name in feature_sets:
        _feature_view(examples[0].features, name)
    rows = []
    started = perf_counter()
    for repeat in range(int(args.split_repeats)):
        split_seed = int(args.seed) + 1009 * repeat
        train, test, train_ids, test_ids = _split(examples, split_seed, args.test_fraction)
        for feature_set in feature_sets:
            for train_fraction in train_fractions:
                rows.extend(_run_split(examples, train, test, train_ids, test_ids,
                                       seed=split_seed, args=args,
                                       feature_set=feature_set, train_fraction=train_fraction))
        print(f"split={repeat + 1}/{args.split_repeats} train={len(train_ids)} test={len(test_ids)}", flush=True)
    requested = args.run_dir or str(default_run_dir("predictor-suite"))
    run = RunDirectory(resolve_run_dir(requested))
    run.write_run_meta(kind="predictor-suite", started=utc_now(), requested=requested)
    run.write_json("config.json", {"bundles": str(Path(args.bundles).resolve()),
                                   "seed": int(args.seed), "split_repeats": int(args.split_repeats),
                                   "test_fraction": float(args.test_fraction), "models": args.models,
                                   "include_shuffled": bool(args.include_shuffled),
                                   "feature_sets": feature_sets, "train_fractions": train_fractions,
                                   "ridge_l2": float(args.ridge_l2), "mlp_hidden": int(args.mlp_hidden),
                                   "mlp_epochs": int(args.mlp_epochs), "device": args.device})
    run.write_jsonl("predictor_suite_rows.jsonl", rows)
    summary = {"n_bundles": len(bundles), "n_prefixes": len(examples),
               "elapsed_fit_seconds": float(perf_counter() - started), **_summary(rows, examples),
               "note": "Prefix means are finite-suffix Monte Carlo estimates of E[G|h]. "
                       "The sample-variance-over-suffix-count noise-floor estimate "
                       "can exceed an observed squared error on finite data."}
    run.write_json("predictor_suite_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    print("run_dir", run.root, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
