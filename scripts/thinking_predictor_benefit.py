"""Cross-problem prediction and full-space variance-times-token-cost reports."""

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from grace_gc.audit.dynamic_score import feature_matrix, fit_prefix_coefficients
from grace_gc.audit.expected_gain import fit_predict
from grace_gc.audit.learning_stats import _strict_pre_answer


def moment_product(norms, gram, residuals, prefix, suffix, probability, weights=None):
    weights = np.ones(len(norms)) if weights is None else np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    variance = max(float(weights @ norms - weights @ gram @ weights), 0.0)
    full_cost = float(weights @ (prefix + suffix))
    extra = float((1.0 / probability - 1.0) * (weights @ residuals))
    cost = float(weights @ (prefix + probability * suffix))
    denominator = variance * full_cost
    return {"full_variance": variance, "extra_variance": extra,
            "full_token_cost": full_cost, "actual_token_cost": cost,
            "token_cost_ratio": None if full_cost == 0.0 else cost / full_cost,
            "variance_cost_ratio": None if denominator == 0.0 else (variance + extra) * cost / denominator}


def bootstrap_product(norms, gram, residuals, prefix, suffix, probability, problem_ids, seed):
    ids = sorted(set(problem_ids))
    if len(ids) < 2:
        return {"ratio_ci95": None, "defined": 0, "n_problem_clusters": len(ids)}
    draws = np.random.default_rng(seed).integers(0, len(ids), size=(1000, len(ids)))
    counts = np.asarray([np.bincount(draw, minlength=len(ids)) for draw in draws])
    weights = counts[:, [ids.index(pid) for pid in problem_ids]].astype(float)
    weights /= weights.sum(axis=1, keepdims=True)
    variance = np.maximum(weights @ norms - np.einsum("bi,ij,bj->b", weights, gram, weights), 0.0)
    full_cost = weights @ (prefix + suffix)
    cost = weights @ (prefix + probability * suffix)
    extra = (1.0 / probability - 1.0) * (weights @ residuals)
    denominator = variance * full_cost
    ratios = np.divide((variance + extra) * cost, denominator,
                       out=np.full(1000, np.nan), where=denominator > 0)
    finite = ratios[np.isfinite(ratios)]
    return {"ratio_ci95": None if not len(finite) else np.quantile(finite, [.025, .975]).tolist(),
            "defined": len(finite), "undefined": 1000 - len(finite),
            "n_problem_clusters": len(ids), "unit": "problem_id with all its paths and positions"}


def report_benefit(replay, destination, seed=17):
    replay, destination = Path(replay), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in (replay / "prefixes.jsonl").read_text().splitlines()]
    pids = np.asarray([row["problem_id"] for row in rows])
    a = np.load(replay / "half_mean_full_grads_a.npy", mmap_mode="r")
    b = np.load(replay / "half_mean_full_grads_b.npy", mmap_mode="r")
    basis = np.load(replay / "prefix_score_gradients.npy", mmap_mode="r")
    norms = np.load(replay / "half_mean_full_norm_sq_b.npy")
    coefficients = fit_prefix_coefficients(basis, a)[:, 0]
    hh, gb, gram = np.zeros(len(rows)), np.zeros(len(rows)), np.zeros((len(rows), len(rows)))
    for start in range(0, a.shape[1], 65536):
        score = np.asarray(basis[:, start:start+65536, 0])
        target = np.asarray(b[:, start:start+65536])
        hh += np.sum(score * score, axis=1)
        gb += np.sum(score * target, axis=1)
        gram += target @ target.T
    prefix = np.asarray([row["prefix_tokens"] for row in rows], dtype=float)
    suffix = np.asarray([row["half_mean_suffix_cost"][1] for row in rows])
    predictions = {"independent_half_oracle": coefficients}
    settings = [("zero", "zero", 1., 64), ("constant", "constant", 1., 64),
                ("ridge_l2_1", "ridge", 1., 64), ("ridge_l2_10", "ridge", 10., 64),
                ("ridge_l2_100", "ridge", 100., 64), ("mlp64", "mlp", 1., 64), ("mlp256", "mlp", 1., 256)]
    for feature_set in ("legacy", "cheap"):
        features = feature_matrix(rows, feature_set)
        for offset, (name, model, l2, width) in enumerate(settings):
            predicted = np.zeros(len(rows))
            for problem_index, pid in enumerate(sorted(set(pids))):
                held, train = np.flatnonzero(pids == pid), np.flatnonzero(pids != pid)
                if not len(train):
                    continue
                predicted[held] = fit_predict(model, features[train], coefficients[train, None], features[held],
                    ridge_l2=l2, mlp_hidden=width, mlp_epochs=50, seed=seed+offset*101+problem_index, device="cpu")[:, 0]
            predictions[f"{feature_set}/{name}"] = predicted
            np.save(destination / f"lopo_{feature_set}_{name}.npy", predicted)
            print(f"phase=lopo feature_set={feature_set} model={name}", flush=True)
    selections = {"all": np.arange(len(rows))}
    for t in sorted({row["t"] for row in rows}):
        selections[f"t={t}"] = np.flatnonzero(np.asarray([row["t"] for row in rows]) == t)
    strict = [i for i, row in enumerate(rows) if _strict_pre_answer(row)]
    if strict:
        selections["strict_pre_answer_undecided"] = np.asarray(strict)
    reports = []
    for name, predicted in predictions.items():
        residuals = norms - 2 * predicted * gb + predicted * predicted * hh
        for selection, indices in selections.items():
            energy = float(np.mean(norms[indices]))
            item = {"model": name, "metric": "euclidean", "selection": selection,
                    "evaluation": "leave_one_problem_out" if name != "independent_half_oracle" else "fit_on_A_evaluate_on_B",
                    "n_prefixes": len(indices), "n_problems": len(set(pids[indices])),
                    "residual_ratio_to_zero": None if energy == 0 else float(np.mean(residuals[indices])) / energy,
                    "variance_cost": []}
            for probability in (.2, .5, .8, 1.):
                inputs = (norms[indices], gram[np.ix_(indices, indices)], residuals[indices],
                          prefix[indices], suffix[indices], probability)
                item["variance_cost"].append({"p": probability, **moment_product(*inputs),
                    **bootstrap_product(*inputs, pids[indices].tolist(), seed)})
            reports.append(item)
    output = {"status": "completed", "finished_utc": datetime.now(timezone.utc).isoformat(),
              "model": "Qwen/Qwen3-4B", "enable_thinking": True, "n_original_problems": 32,
              "n_observed_problems": len(set(pids)), "n_prefixes": len(rows), "lopo": reports,
              "functional_qualification_available": True,
              "strict_pre_answer_undecided_prefixes": len(strict),
              "adam_metric": "unavailable: fresh frozen LoRA has no optimizer history",
              "scope": {"cost": "prefix plus probability-weighted suffix tokens; excludes GPU and feature overhead",
                        "variance": "empirical pooled B gradients, conditional on surviving decision-point prefixes",
                        "oracle": "noisy A-fit/B-score reference; not a theoretical bound",
                        "bootstrap": "problem clusters, conditional on fitted predictions",
                        "single_problem_fallback": "cross-problem prediction unavailable; only zero predictions recorded",
                        "policy": "frozen original post-trained model with fresh zero-B LoRA, not old Base policy"}}
    (destination / "predictor_benefit_summary.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
