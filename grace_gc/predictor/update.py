"""Batch-end IPW predictor update. Uses only historical audited labels."""

from __future__ import annotations

import numpy as np

from grace_gc.predictor.heads import PredictorHeads
from grace_gc.predictor.ipw import assign_new_problems, ipw_weights
from grace_gc.predictor.reservoir import GradientReservoir
from grace_gc.predictor.risk import full_space_residual


def update_predictor_from_reservoir(
    heads: PredictorHeads,
    reservoir: GradientReservoir,
    u: np.ndarray,
    rng: np.random.Generator,
    epochs: int = 2,
) -> dict:
    if not reservoir.items:
        return {"coord_loss": None, "risk_loss": None, "n": 0}
    feats = np.stack([it.features for it in reservoir.items], axis=0)
    grads = np.stack([it.g.reshape(-1) for it in reservoir.items], axis=0)
    p = np.array([it.p for it in reservoir.items], dtype=np.float64)
    s = np.array([it.s for it in reservoir.items], dtype=np.float64)
    weights = ipw_weights(p, s)
    coords = grads @ u
    pids = reservoir.problem_ids()
    if reservoir.fit_problem_ids is None:
        reservoir.fit_problem_ids = set()
    if reservoir.hold_problem_ids is None:
        reservoir.hold_problem_ids = set()
    assign_new_problems(reservoir.fit_problem_ids, reservoir.hold_problem_ids, pids, rng)
    fit = np.array([pid in reservoir.fit_problem_ids for pid in pids])
    hold = ~fit
    # Eviction can leave only hold items. Do not dump them into coord; the
    # problem assignment is persistent. Skip a head when its side is empty.
    coord_loss = None
    risk_loss = None
    rewards = np.array([0.0 if it.reward is None else float(it.reward) for it in reservoir.items], dtype=np.float64)
    if np.any(fit):
        coord_loss = heads.train_coord(feats[fit], coords[fit], weights[fit], epochs=epochs)
        heads.train_success(feats[fit], rewards[fit], weights[fit], epochs=epochs)
    e = None
    risk_idx = hold
    if np.any(hold):
        pred = heads.forward_numpy(feats[hold])
        e = full_space_residual(grads[hold], pred.f, u)
        risk_loss = heads.train_risk(feats[hold], e, weights[hold], epochs=epochs)
    cost_loss = 0.0
    cost_feat = []
    cost_y = []
    cost_w = []
    for it, w in zip(reservoir.items, weights):
        if it.cost_feat is None or it.realized_cost is None:
            continue
        cost_feat.append(it.cost_feat)
        cost_y.append(it.realized_cost)
        cost_w.append(w)
    if cost_feat and not heads.constant_cost:
        cost_loss = heads.train_cost(np.stack(cost_feat, axis=0), np.asarray(cost_y), np.asarray(cost_w), epochs=epochs)
    reward_loss = None
    reward_x = []
    reward_e = []
    reward_w = []
    if e is not None:
        hold_items = [it for i, it in enumerate(reservoir.items) if risk_idx[i]]
        hold_e = np.atleast_1d(np.asarray(e, dtype=np.float64))
        hold_w = weights[risk_idx]
        for it, residual, w in zip(hold_items, hold_e, hold_w):
            if it.reward_feat is None:
                continue
            reward_x.append(it.reward_feat)
            reward_e.append(float(residual))
            reward_w.append(w)
    if reward_x:
        reward_loss = heads.train_reward_risk(
            np.stack(reward_x, axis=0), np.asarray(reward_e), np.asarray(reward_w), epochs=epochs
        )
    return {
        "coord_loss": coord_loss,
        "risk_loss": risk_loss,
        "cost_loss": cost_loss,
        "reward_risk_loss": reward_loss,
        "n": len(reservoir.items),
    }
