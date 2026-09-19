"""Batch-end IPW updates; frozen diagnostics never fit on their labels."""

from __future__ import annotations

import numpy as np

from grace_gc.core.allocation import allocate_continuation
from grace_gc.predictor.heads import PredictorHeads
from grace_gc.predictor.features import reward_risk_features
from grace_gc.predictor.ipw import assign_new_problems, ipw_weights
from grace_gc.predictor.reservoir import GradientReservoir, ReservoirItem, age_weights
from grace_gc.predictor.risk import full_space_residual
from grace_gc.predictor.scale import design_shrink_diagnostics, design_shrink_from_projections


def prepare_predictor_split(heads, reservoir, rng, basis_fit_only=False, *,
                            current_step=None, max_age_steps=None, age_half_life=None,
                            missing_step_policy="exclude") -> dict:
    """Assign whole problems before constructing U, and persist their sides.

    Holdout calibration excludes its labels from U and coordinate fitting.
    Risk fitting can indirectly depend on past calibration through gamma; this
    calibration set is therefore not an independent evaluation of the pipeline.
    """
    if reservoir.fit_problem_ids is None:
        reservoir.fit_problem_ids = set()
    if reservoir.hold_problem_ids is None:
        reservoir.hold_problem_ids = set()
    if reservoir.calibration_problem_ids is None:
        reservoir.calibration_problem_ids = set()
    fit_ids, hold_ids = reservoir.fit_problem_ids, reservoir.hold_problem_ids
    cal_ids = reservoir.calibration_problem_ids
    pids = reservoir.problem_ids()
    if heads.shrink_calibration == "holdout":
        for pid in dict.fromkeys(pids):
            if pid in fit_ids or pid in hold_ids or pid in cal_ids:
                continue
            # As with the legacy two-way split, seed each side on arrival.
            # Empty sides after eviction remain empty: never move labels.
            if not fit_ids:
                fit_ids.add(pid)
            elif not hold_ids:
                hold_ids.add(pid)
            elif not cal_ids or rng.random() < heads.calibration_fraction:
                cal_ids.add(pid)
            elif rng.random() < .5:
                fit_ids.add(pid)
            else:
                hold_ids.add(pid)
    else:
        assign_new_problems(fit_ids, hold_ids, [p for p in pids if p not in cal_ids], rng)
    fit = np.array([pid in fit_ids for pid in pids], dtype=bool)
    hold = np.array([pid in hold_ids for pid in pids], dtype=bool)
    cal = np.array([pid in cal_ids for pid in pids], dtype=bool)
    recency, freshness = age_weights(reservoir.items, current_step, max_age_steps,
                                     age_half_life, missing_step_policy)
    active = recency > 0
    return {"fit": fit & active, "hold": hold & active, "calibration": cal & active,
            "basis": (fit if basis_fit_only else ~cal) & active,
            "age_weight": recency, "freshness": freshness}


def _residuals(items, f, u):
    # Do not assemble another reservoir-sized n x D gradient or prediction.
    return np.array([float(full_space_residual(it.g.reshape(-1), row, u))
                     for it, row in zip(items, f)], dtype=np.float64)


def _current_reward_features(heads, feats, items):
    q = heads.forward_success(feats)
    return np.stack([reward_risk_features(prob, 1. - prob, it.reward_feat[3], it.reward_feat[4])
                     for prob, it in zip(q, items)])


def _supervision_summary(items, weights, split, current_step):
    norms = np.array([float(np.dot(it.g.reshape(-1), it.g.reshape(-1))) for it in items])
    ages = [int(current_step) - it.observed_step for it in items
            if current_step is not None and it.observed_step is not None]
    summary = {
        "n": len(items), "n_problems": len({it.problem_id for it in items}),
        "n_zero_gradient": int(np.count_nonzero(norms == 0)),
        "ipw_effective_n": float(weights.sum() ** 2 / np.dot(weights, weights)) if np.any(weights > 0) else 0.,
        "weight_definition": "inverse propensity multiplied by configured age weight",
        "age_observed_n": len(ages),
        "min_age_steps": min(ages) if ages else None,
        "max_age_steps": max(ages) if ages else None,
        "mean_age_steps": float(np.mean(ages)) if ages else None,
    }
    for side in ("fit", "hold", "calibration", "basis"):
        mask = split[side]
        selected = [it for it, keep in zip(items, mask) if keep]
        counts = {}
        for it in selected:
            counts[it.problem_id] = counts.get(it.problem_id, 0) + 1
        summary[side] = {"n": len(selected), "n_problems": len(counts),
                         "n_nonzero_gradient": int(np.count_nonzero(norms[mask])),
                         "n_prefix_finished": sum(it.prefix_finished is True for it in selected),
                         "n_prefix_unfinished": sum(it.prefix_finished is False for it in selected),
                         "n_prefix_finished_unknown": sum(it.prefix_finished is None for it in selected),
                         "n_singleton_problems": sum(n == 1 for n in counts.values())}
    return summary


def diagnose_frozen_predictor(heads: PredictorHeads, new_items: list[ReservoirItem],
                              u: np.ndarray, only_unseen: bool = True, risk_mode: str = "full") -> dict:
    """Evaluate BEFORE any basis/head update; leave heads, split and RNG unchanged.

    Unseen means no previous supervised predictor update on this problem. Old
    checkpoints without that history are explicitly marked incomplete.
    """
    items = [it for it in new_items if not only_unseen or it.problem_id not in heads.seen_problem_ids]
    out = {"n": len(items), "n_problems": len({it.problem_id for it in items}),
           "only_unseen_problems": bool(only_unseen),
           "history_complete": bool(heads.diagnostic_history_complete),
           "residual_mean": None, "m0_residual_mean": None,
           "residual_to_m0_ratio": None, "risk_residual_correlation": None}
    if not items:
        return out
    pred = heads.forward_numpy(np.stack([it.features for it in items]))
    e = _residuals(items, pred.f, u)
    e0 = np.array([np.dot(it.g.reshape(-1), it.g.reshape(-1)) for it in items])
    w = ipw_weights(np.array([it.p for it in items]), np.array([it.s for it in items]))
    emean, e0mean = float(np.average(e, weights=w)), float(np.average(e0, weights=w))
    out.update(residual_mean=emean, m0_residual_mean=e0mean,
               residual_to_m0_ratio=emean / e0mean if e0mean > 0 else None)
    risk = pred.r_hat
    if risk_mode == "reward":
        valid = np.array([it.reward_feat is not None for it in items])
        out["risk_missing_features_n"] = int((~valid).sum())
        if valid.any():
            risk_items = [it for it, keep in zip(items, valid) if keep]
            risk = heads.forward_reward_risk(_current_reward_features(
                heads, np.stack([it.features for it in risk_items]), risk_items))
        else:
            risk = np.array([])
        e = e[valid]
    if len(e) > 1 and np.std(e) > 0 and np.std(risk) > 0:
        out["risk_residual_correlation"] = float(np.corrcoef(e, risk)[0, 1])
    return out


def update_predictor_from_reservoir(
    heads: PredictorHeads, reservoir: GradientReservoir, u: np.ndarray,
    rng: np.random.Generator, epochs: int = 2, *, current_step=None,
    split=None, calibration_p=None, calibration_beta=.5, calibration_p_min=.2,
    calibration_risk_mode="full", calibration_uniform_shrink=0.0,
) -> dict:
    if not reservoir.items:
        return {"coord_loss": None, "risk_loss": None, "n": 0}
    items = reservoir.items
    feats = np.stack([it.features for it in items])
    p = np.array([it.p for it in items], dtype=np.float64)
    weights = ipw_weights(p, np.array([it.s for it in items]))
    # G remains in its existing reservoir arrays. Only n x k coordinates copy.
    coords = np.stack([it.g.reshape(-1) @ u for it in items])
    if split is None:
        split = prepare_predictor_split(heads, reservoir, rng, current_step=current_step)
    weights *= split.get("age_weight", np.ones(len(items)))
    fit, hold, cal = (split[key] for key in ("fit", "hold", "calibration"))
    coord_loss = risk_loss = reward_loss = None
    if np.any(fit):
        heads.fit_feature_scaler(feats[fit])
        if heads.coord_kind == "ridge" or heads.feature_scaler == "refit" or heads.scaler.coord_scale is None:
            heads.scaler.fit_coord_scale(coords[fit])
        coord_loss = heads.train_coord(feats[fit], coords[fit], weights[fit], epochs=epochs)
        if heads.train_auxiliary:
            rewards = np.array([0. if it.reward is None else float(it.reward) for it in items])
            heads.train_success(feats[fit], rewards[fit], weights[fit], epochs=epochs)
    gamma = gamma_diagnostics = None
    selected = cal if heads.shrink_calibration == "holdout" else fit
    calibration_missing_reward_features_n = 0
    if heads.shrink_calibration == "holdout" and calibration_risk_mode == "reward" and calibration_p is None:
        valid = np.array([it.reward_feat is not None for it in items])
        calibration_missing_reward_features_n = int(np.count_nonzero(selected & ~valid))
        selected = selected & valid
    candidate_p = None
    calibration_missing_cost_n = calibration_unknown_finished_n = 0
    if heads.shrink_m and np.any(selected):
        if heads.shrink_calibration == "holdout":
            if calibration_p is None:
                # This risk/cost design is frozen BEFORE their updates below;
                # no calibration gradient/reward enters its construction.
                cost_feat = np.stack([np.zeros(3) if it.cost_feat is None else it.cost_feat
                                      for it, keep in zip(items, selected) if keep])
                design = heads.forward_numpy(feats[selected], cost_feat)
                cal_items = [it for it, keep in zip(items, selected) if keep]
                risk = design.r_hat
                if calibration_risk_mode == "reward":
                    risk = heads.forward_reward_risk(_current_reward_features(heads, feats[selected], cal_items))
                cost = design.c_hat
                if heads.constant_cost:
                    calibration_missing_cost_n = sum(it.remaining_cost is None for it in cal_items)
                    cost = np.array([1. if it.remaining_cost is None else it.remaining_cost for it in cal_items])
                calibration_unknown_finished_n = sum(it.prefix_finished is None for it in cal_items)
                finished = np.array([it.prefix_finished is True for it in cal_items])
                # Historical audits are selected. IPW also estimates the
                # starts-population cost budget, while r/c stays unchanged.
                budget_w = weights[selected] / weights[selected].mean()
                candidate_p = (np.ones(len(cal_items)) if calibration_beta == 1.0 else
                               allocate_continuation(risk * budget_w, cost * budget_w,
                                                    calibration_beta, calibration_p_min, finished=finished,
                                                    uniform_shrink=calibration_uniform_shrink).p)
            else:
                candidate_p = np.asarray(calibration_p, dtype=np.float64).reshape(-1)
                if len(candidate_p) == len(items):
                    candidate_p = candidate_p[selected]
                if len(candidate_p) != int(selected.sum()):
                    raise ValueError("calibration p dimension does not match calibration rows")
        else:
            candidate_p = p[selected]
        unshrunk_f = heads.predict_f(feats[selected], shrink=False)
        gram = u.T @ u
        gamma = design_shrink_from_projections(coords[selected], unshrunk_f, gram,
                                               candidate_p, weights[selected])
        gamma_diagnostics = design_shrink_diagnostics(coords[selected], unshrunk_f, gram,
                                                       candidate_p, weights[selected])
        if gamma is not None:
            heads.m_shrink = gamma
    e = None
    if np.any(hold):
        hold_items = [it for it, keep in zip(items, hold) if keep]
        e = _residuals(hold_items, heads.predict_f(feats[hold]), u)
        if heads.feature_scaler == "refit" or not heads.risk_scale_fitted:
            heads.risk_scale_fitted = heads.scaler.fit_risk_scale(e)
        risk_loss = heads.train_risk(feats[hold], e, weights[hold], epochs=epochs)
    cost_loss = 0.
    cost_idx = [i for i, it in enumerate(items)
                if (fit[i] or hold[i]) and it.cost_feat is not None and it.realized_cost is not None]
    if cost_idx and not heads.constant_cost:
        cost_loss = heads.train_cost(np.stack([items[i].cost_feat for i in cost_idx]),
                                    np.array([items[i].realized_cost for i in cost_idx]),
                                    weights[cost_idx], epochs=epochs)
    if e is not None and heads.train_auxiliary:
        reward_idx = [i for i, it in enumerate(hold_items) if it.reward_feat is not None]
        if reward_idx:
            # Match online/calibration inputs after this update's success fit.
            # Historical length and baseline remain the labels' own context.
            x = _current_reward_features(heads, feats[hold][reward_idx],
                                          [hold_items[i] for i in reward_idx])
            reward_loss = heads.train_reward_risk(x, e[reward_idx], weights[hold][reward_idx], epochs=epochs)
    heads.seen_problem_ids.update(reservoir.problem_ids())
    return {"coord_loss": coord_loss, "risk_loss": risk_loss, "cost_loss": cost_loss,
            "reward_risk_loss": reward_loss, "n": len(items), "m_shrink": float(heads.m_shrink),
            "coord_kind": heads.coord_kind, "calibration_mode": heads.shrink_calibration,
            "calibration_n": int(selected.sum()), "calibration_gamma": gamma,
            "calibration_diagnostics": gamma_diagnostics,
            "calibration_p_mean": None if candidate_p is None else float(candidate_p.mean()),
            "calibration_uniform_shrink": float(calibration_uniform_shrink),
            "calibration_missing_cost_n": calibration_missing_cost_n,
            "calibration_unknown_finished_n": calibration_unknown_finished_n,
            "calibration_missing_reward_features_n": calibration_missing_reward_features_n,
            "risk_numerics": heads.risk_diagnostics(feats),
            "risk_scale": float(heads.scaler.risk_scale), "risk_scale_fitted": bool(heads.risk_scale_fitted),
            "freshness": split.get("freshness"),
            "supervision": _supervision_summary(items, weights, split, current_step)}
