#!/usr/bin/env python3
"""Global 8/64-D gradient, benefit and risk experiment on frozen audit replays."""

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

from grace_gc.audit.expected_gain import (
    fit_global_basis, fit_predict, full_space_residual, load_replay,
    problem_split, project_means, reference_gradient, ridge_oof_by_problem,
)
from grace_gc.logging_util.run_dir import resolve_run_dir


def _indices(rows, ids):
    allowed = set(map(str, ids))
    return np.asarray([i for i, row in enumerate(rows) if str(row["problem_id"]) in allowed], dtype=int)


def _provenance(path):
    source = Path(path) / "replay_provenance.json"
    return json.loads(source.read_text(encoding="utf-8")) if source.exists() else None


def _check_replay_identity(replay_dir, reference_dir):
    first, second = _provenance(replay_dir), _provenance(reference_dir)
    if first is None or second is None:
        raise ValueError("both replays need actor/layout provenance from replay_expected_gain.py")
    for key in ("actor_sha256", "layout_names", "layout_dim", "model_path"):
        if first.get(key) != second.get(key):
            raise ValueError(f"predictor and reference replays differ in {key}")


def _mse(pred, truth):
    return float(np.mean((np.asarray(pred).reshape(-1) - np.asarray(truth).reshape(-1)) ** 2))


def _metrics(pred, truth, train_mean):
    truth = np.asarray(truth).reshape(-1)
    pred = np.asarray(pred).reshape(-1)
    base = float(np.mean((truth - float(train_mean)) ** 2))
    return {"mse": _mse(pred, truth), "mean_target": float(truth.mean()),
            "r2_vs_train_mean": None if base == 0 else 1 - _mse(pred, truth) / base,
            "sign_accuracy_nonzero": (None if not np.any(truth != 0) else
                                      float(np.mean(np.sign(pred[truth != 0]) == np.sign(truth[truth != 0]))))}


def _half_gain_reliability(rows):
    pairs = [(row["problem_id"], row["half_directional_gain"])
             for row in rows if row.get("half_directional_gain") is not None]
    if not pairs:
        return {"n_prefixes": 0, "pearson": None,
                "same_problem_pair_order_agreement": None}
    values = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
    pearson = (None if np.std(values[:, 0]) == 0 or np.std(values[:, 1]) == 0 else
               float(np.corrcoef(values[:, 0], values[:, 1])[0, 1]))
    order = []
    for pid in sorted({pair[0] for pair in pairs}):
        group = np.asarray([pair[1] for pair in pairs if pair[0] == pid])
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                first = np.sign(group[i, 0] - group[j, 0])
                second = np.sign(group[i, 1] - group[j, 1])
                if first != 0 and second != 0:
                    order.append(float(first == second))
    return {"n_prefixes": len(values), "pearson": pearson,
            "same_problem_pair_order_agreement": None if not order else float(np.mean(order)),
            "n_comparable_pairs": len(order)}


def _model_settings(names):
    for name in names:
        if name in {"zero", "constant"}:
            yield {"name": name, "model": name}
        elif name == "ridge":
            for l2 in (1., 10., 100.):
                yield {"name": f"ridge_l2_{l2:g}", "model": "ridge", "ridge_l2": l2}
        elif name in {"mlp64", "mlp256"}:
            yield {"name": name, "model": "mlp", "mlp_hidden": int(name[3:])}
        else:
            raise ValueError(f"unknown model {name}")


def _predict(setting, x_train, y_train, x_eval, args, offset=0):
    return fit_predict(setting["model"], x_train, y_train, x_eval,
                       ridge_l2=setting.get("ridge_l2", 1.),
                       mlp_hidden=setting.get("mlp_hidden", 64),
                       mlp_epochs=args.mlp_epochs, seed=args.seed + offset,
                       device=args.device)


def _predict_three(setting, x_train, y_train, x_val, x_test, args, offset=0):
    combined = np.concatenate((x_train, x_val, x_test), axis=0)
    pred = _predict(setting, x_train, y_train, combined, args, offset=offset)
    n_train, n_val = len(x_train), len(x_val)
    return pred[:n_train], pred[n_train:n_train + n_val], pred[n_train + n_val:]


def _directional_labels(means, reference, block=32768):
    scores = np.zeros(len(means), dtype=np.float64)
    for start in range(0, means.shape[1], block):
        end = min(start + block, means.shape[1])
        scores += np.asarray(means[:, start:end], dtype=np.float64) @ reference[start:end]
    return scores


def _fixed_probability(score, cost, lam, p_min):
    values = np.asarray(score, dtype=np.float64)
    values = np.maximum(values, 1e-12 * max(float(np.max(values)), 1.))
    return np.maximum(p_min, np.minimum(1.0, lam * values /
                                      np.sqrt(np.maximum(np.asarray(cost), 1.0))))


def _calibrate_probability(score, cost, beta, p_min):
    if not p_min <= beta <= 1.0:
        raise ValueError("beta must lie in [p_min, 1]")
    if beta == 1.0:
        return float("inf")
    lo, hi = 0., 1.
    weights = np.maximum(np.asarray(cost, dtype=np.float64), 1.)
    while float(np.sum(_fixed_probability(score, weights, hi, p_min) * weights) /
                np.sum(weights)) < beta:
        hi *= 2
        if hi > 1e20:
            break
    for _ in range(70):
        mid = (lo + hi) / 2
        ratio = float(np.sum(_fixed_probability(score, weights, mid, p_min) * weights) /
                      np.sum(weights))
        if ratio < beta:
            lo = mid
        else:
            hi = mid
    return hi


def _allocation(name, val_score, test_score, val_cost, test_cost,
                test_risk, beta, p_min):
    lam = _calibrate_probability(val_score, val_cost, beta, p_min)
    p = (_fixed_probability(test_score, test_cost, lam, p_min) if np.isfinite(lam)
         else np.ones(len(test_score)))
    risk = np.maximum(np.asarray(test_risk), 0.)
    return {"strategy": name, "lambda_from_validation": lam,
            "test_probability_mean": float(np.mean(p)),
            "test_cost_fraction": float(np.sum(p * test_cost) / np.sum(test_cost)),
            "test_extra_variance_per_start": float(np.mean((1 / p - 1) * risk)),
            "test_max_inverse_probability": float(np.max(1 / p))}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", required=True)
    parser.add_argument("--reference-dir", required=True,
                        help="disjoint reference-construction audit replay")
    parser.add_argument("--reference-check-dir", default=None,
                        help="optional second disjoint reference pool")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--split-manifest", default=None,
                        help="existing JSON with train, validation, diagnostic problem IDs")
    parser.add_argument("--prior-train-ids", default=None,
                        help="optional JSON list of already inspected problem IDs")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--models", default="zero,constant,ridge,mlp64,mlp256")
    parser.add_argument("--mlp-epochs", type=int, default=50)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--p-min", type=float, default=0.2)
    parser.add_argument("--n-start", type=int, default=8,
                        help="fixed starts per training step; must match the checkpoint run")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    args = parser.parse_args(argv)
    started = perf_counter()
    rows, means = load_replay(args.replay_dir)
    ref_rows, ref_means = load_replay(args.reference_dir)
    _check_replay_identity(args.replay_dir, args.reference_dir)
    if means.shape[1] != ref_means.shape[1]:
        raise ValueError("reference and prediction gradient layouts differ")
    if set(x["problem_id"] for x in rows) & set(x["problem_id"] for x in ref_rows):
        raise ValueError("reference construction and predictor problems overlap")
    ref = reference_gradient(ref_rows, ref_means)
    check_agreement = None
    if args.reference_check_dir:
        _check_replay_identity(args.replay_dir, args.reference_check_dir)
        check_rows, check_means = load_replay(args.reference_check_dir)
        all_ids = {x["problem_id"] for x in rows + ref_rows}
        if all_ids & {x["problem_id"] for x in check_rows}:
            raise ValueError("reference check pool overlaps predictor or reference construction")
        ref_check = reference_gradient(check_rows, check_means)
        check_agreement = float(ref @ ref_check / max(np.linalg.norm(ref) * np.linalg.norm(ref_check), 1e-20))
    if args.split_manifest:
        split = json.loads(Path(args.split_manifest).read_text(encoding="utf-8"))
    else:
        prior = ([] if not args.prior_train_ids else
                 json.loads(Path(args.prior_train_ids).read_text(encoding="utf-8")))
        split = problem_split(rows, seed=args.seed, prior_train=prior)
    roles = {role: _indices(rows, split[role]) for role in ("train", "validation", "diagnostic")}
    if any(not len(indices) for indices in roles.values()):
        raise ValueError("all three split roles need live prefixes")
    if sum(map(len, roles.values())) != len(rows):
        raise ValueError("split manifest must cover each problem exactly once")
    if any(set(split[a]) & set(split[b]) for a, b in (("train", "validation"),
                                                       ("train", "diagnostic"),
                                                       ("validation", "diagnostic"))):
        raise ValueError("split manifest has overlapping problems")
    if not all(row.get("features") is not None for row in rows):
        raise ValueError("replay lacks stored prefix features; audit needs store_features: true")
    features = np.stack([np.asarray(row["features"], dtype=np.float64) for row in rows])
    if features.ndim != 2 or not np.all(np.isfinite(features)):
        raise ValueError("prefix features are inconsistent or nonfinite")
    norms = np.asarray([row["mean_norm_sq"] for row in rows], dtype=np.float64)
    costs = np.maximum(np.asarray([row["mean_cost"] for row in rows], dtype=np.float64), 1.)
    rewards = np.asarray([row["mean_reward"] for row in rows], dtype=np.float64)
    utilities = _directional_labels(means, ref)
    names = [value.strip() for value in args.models.split(",") if value.strip()]
    settings = list(_model_settings(names))
    destination = resolve_run_dir(args.run_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    tr, va, te = (roles[x] for x in ("train", "validation", "diagnostic"))
    xtr, xva, xte = features[tr], features[va], features[te]
    baseline_features = {"all": features,
                         "length_baseline": features[:, -2:]}
    # Reward is predicted out of problem on fit rows; using their observed
    # rewards here would let the gain head train on an unavailable shortcut.
    reward_features = np.zeros((len(rows), 3), dtype=np.float64)
    reward_features[:, :2] = features[:, -2:]
    reward_features[tr, 2] = ridge_oof_by_problem(
        xtr, rewards[tr], [rows[i]["problem_id"] for i in tr]).reshape(-1)
    reward_features[va, 2] = fit_predict("ridge", xtr, rewards[tr], xva).reshape(-1)
    reward_features[te, 2] = fit_predict("ridge", xtr, rewards[tr], xte).reshape(-1)
    baseline_features["predicted_reward_length"] = reward_features
    if all(row.get("prompt_features") is not None for row in rows):
        prompt_features = np.stack([np.asarray(row["prompt_features"], dtype=np.float64)
                                    for row in rows])
        if prompt_features.shape != features.shape or not np.all(np.isfinite(prompt_features)):
            raise ValueError("prompt-only features are inconsistent with prefix features")
        baseline_features["prompt_only"] = prompt_features
    scalar_rows = []
    scalar_best = {}
    for target_name, target in (("gain_direct", utilities), ("risk0", norms),
                                ("cost", costs), ("reward", rewards)):
        for view_name, x_all in baseline_features.items():
            for setting in settings:
                pred_train, pred_val, pred_test = _predict_three(
                    setting, x_all[tr], target[tr], x_all[va], x_all[te], args, offset=19)
                val_mse = _mse(pred_val, target[va])
                row = {"target": target_name, "feature_set": view_name,
                       "model": setting["name"],
                       "train_mse": _mse(pred_train, target[tr]),
                       "validation_mse": val_mse,
                       "diagnostic": _metrics(pred_test, target[te], np.mean(target[tr]))}
                scalar_rows.append(row)
                if target_name not in scalar_best or val_mse < scalar_best[target_name][0]:
                    scalar_best[target_name] = (val_mse, pred_val.reshape(-1),
                                                pred_test.reshape(-1), row)
    basis_rows = []
    chosen = None
    for variant in ("mean_svd", "predictable_crossfit_signal"):
        for k in (8, 64):
            path = destination / f"basis_{variant}_{k}.npy"
            try:
                info = fit_global_basis(means, tr, xtr,
                                        [rows[i]["problem_id"] for i in tr], variant, k, path)
            except ValueError as exc:
                basis_rows.append({"basis": variant, "k": k, "status": "unavailable",
                                   "reason": str(exc)})
                continue
            u = np.load(path, mmap_mode="r")
            coord = project_means(means, u)
            gram = np.asarray(info["basis_gram"])
            target_energy = np.sum((coord @ np.linalg.pinv(gram)) * coord, axis=1)
            for setting in settings:
                ctrain, cval, ctest = _predict_three(
                    setting, xtr, coord[tr], xva, xte, args, offset=31)
                residual_train = full_space_residual(norms[tr], coord[tr], ctrain, gram)
                residual_val = full_space_residual(norms[va], coord[va], cval, gram)
                residual_test = full_space_residual(norms[te], coord[te], ctest, gram)
                projected_ref = np.asarray(u.T @ ref)
                gain_from_m = ctest @ projected_ref
                row = {"basis": variant, "k": k, "actual_rank": info["actual_rank"],
                       "model": setting["name"],
                       "train_full_residual_mean": float(np.mean(residual_train)),
                       "validation_full_residual_mean": float(np.mean(residual_val)),
                       "diagnostic_full_residual_mean": float(np.mean(residual_test)),
                       "diagnostic_zero_residual_mean": float(np.mean(norms[te])),
                       "diagnostic_projected_mean_energy": float(np.mean(target_energy[te])),
                       "diagnostic_projected_mean_over_observed_energy":
                           float(np.sum(target_energy[te]) / max(float(np.sum(norms[te])), 1e-30)),
                       "diagnostic_gain_from_gradient": _metrics(gain_from_m, utilities[te],
                                                                np.mean(utilities[tr])),
                       "basis_info": {key: value for key, value in info.items()
                                      if key != "basis_gram"}}
                basis_rows.append(row)
                key = float(np.mean(residual_val))
                if chosen is None or key < chosen[0]:
                    chosen = (key, row, cval, ctest, coord, gram)
            print(f"basis={variant} k={k} rank={info['actual_rank']}", flush=True)
    if chosen is None:
        raise ValueError("no global basis was fit; see unavailable basis rows")
    _, selected, cval, ctest, coords, gram = chosen
    residual_val = np.maximum(full_space_residual(norms[va], coords[va], cval, gram), 0.)
    residual_test = np.maximum(full_space_residual(norms[te], coords[te], ctest, gram), 0.)
    # The coordinate predictor was fitted on train only. Validation residuals
    # are therefore honest labels for this fixed predictor; fit the risk head
    # there and evaluate once on the independent diagnostic problems.
    validation_problems = sorted({str(rows[i]["problem_id"]) for i in va})
    if len(validation_problems) < 2:
        raise ValueError("risk-head fit and allocation calibration need two validation problems")
    shuffled = np.random.default_rng(args.seed + 101).permutation(validation_problems)
    fit_problems = set(shuffled[:max(1, len(shuffled) // 2)])
    risk_fit = np.asarray([j for j, i in enumerate(va)
                           if str(rows[i]["problem_id"]) in fit_problems], dtype=int)
    risk_cal = np.asarray([j for j, i in enumerate(va)
                           if str(rows[i]["problem_id"]) not in fit_problems], dtype=int)
    risk_test = fit_predict("ridge", xva[risk_fit], residual_val[risk_fit], xte,
                            ridge_l2=10.).reshape(-1)
    risk_val_for_calibration = fit_predict("ridge", xva[risk_fit],
                                           residual_val[risk_fit], xva[risk_cal],
                                           ridge_l2=10.).reshape(-1)
    direct_val, direct_test = scalar_best["gain_direct"][1:3]
    cost_val, cost_test = scalar_best["cost"][1:3]
    cost_val, cost_test = np.maximum(cost_val, 1.), np.maximum(cost_test, 1.)
    risk0_val, risk0_test = scalar_best["risk0"][1:3]
    gain_scale = max(float(np.sqrt(np.mean(utilities[tr] ** 2))), 1e-8)
    allocations = [
        _allocation("uniform", np.sqrt(cost_val[risk_cal]), np.sqrt(cost_test),
                    cost_val[risk_cal], cost_test, residual_test, args.beta, args.p_min),
        _allocation("risk0", np.sqrt(np.maximum(risk0_val[risk_cal], 0.)),
                    np.sqrt(np.maximum(risk0_test, 0.)), cost_val[risk_cal], cost_test,
                    norms[te], args.beta, args.p_min),
        _allocation("residual_risk", np.sqrt(np.maximum(risk_val_for_calibration, 0.)),
                    np.sqrt(np.maximum(risk_test, 0.)), cost_val[risk_cal], cost_test,
                    residual_test, args.beta, args.p_min),
        _allocation("direct_gain", np.exp(np.clip(direct_val[risk_cal] / gain_scale, -5, 5)),
                    np.exp(np.clip(direct_test / gain_scale, -5, 5)),
                    cost_val[risk_cal], cost_test, residual_test, args.beta, args.p_min),
    ]
    half_dots = [row["half_mean_dot"] for row in rows if row["half_mean_dot"] is not None]
    result = {"source_replay": str(Path(args.replay_dir).resolve()),
              "reference_replay": str(Path(args.reference_dir).resolve()),
              "reference_check_cosine": check_agreement,
              "gradient_dimension": int(means.shape[1]), "n_prefixes": len(rows),
              "n_problems": len(set(row["problem_id"] for row in rows)),
              "split_counts": {key: len(value) for key, value in roles.items()},
              "split_problem_counts": {key: len(value) for key, value in split.items()},
              "mean_half_dot": None if not half_dots else float(np.mean(half_dots)),
              "half_gain_reliability": _half_gain_reliability(rows),
              "risk_validation_fit_problems": sorted(fit_problems),
              "risk_validation_calibration_problems": sorted(set(shuffled) - fit_problems),
              "utility_label": "reference policy-gradient dot product; local SGD direction only",
              "utility_scale_eta_over_n": args.learning_rate / args.n_start,
              "selected_by_validation_residual": {key: selected[key] for key in
                                                  ("basis", "k", "actual_rank", "model")},
              "selected_scalar_by_validation": {key: value[3] for key, value in scalar_best.items()},
              "diagnostic_residual_risk_mse": _mse(risk_test, residual_test),
              "diagnostic_residual_risk_constant_mse": _mse(np.full(len(te), np.mean(residual_val)),
                                                             residual_test),
              "allocation": allocations, "basis_rows": basis_rows,
              "scalar_rows": scalar_rows, "wall_seconds": perf_counter() - started}
    (destination / "expected_gain_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"run_dir": str(destination),
                      "selected": result["selected_by_validation_residual"],
                      "wall_seconds": result["wall_seconds"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
