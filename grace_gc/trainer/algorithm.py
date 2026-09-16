"""One Algorithm-1 training step, shared by CPU tiny and GPU engines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from grace_gc.core.layout import ParamLayout, collect_lora_layout
from grace_gc.core.rng import IsolatedRNG
from grace_gc.predictor.heads import PredictorHeads
from grace_gc.predictor.reservoir import GradientReservoir, ReservoirItem
from grace_gc.predictor.risk import full_space_residual
from grace_gc.data.reward import extract_answer
from grace_gc.logging_util.ledger import Timer
from grace_gc.predictor.update import update_predictor_from_reservoir
from grace_gc.trainer.actor_update import (
    apply_correction_clip_step,
    audit_one,
    real_stream_backward_each,
    restore_grads,
    snapshot_grads,
)
from grace_gc.trainer.advantages import advantages_for_method
from grace_gc.trainer.baseline import HistoricalBaseline
from grace_gc.trainer.grace_step import (
    StartRecord,
    answer_first_token,
    batch_token_costs,
    decide_continuation,
    finish_reason,
    next_start_count,
)
from grace_gc.trainer.methods import MethodSpec


@dataclass
class StepEngines:
    generate_prefix: Callable
    continue_selected: Callable
    prefix_features: Callable
    logprob_sums: Callable
    logprob_one: Callable
    named_lora: Callable
    trainable_params: Callable
    reward_fn: Callable
    decode: Callable | None = None
    eos_id: int | list[int] | None = None
    last_rollout: dict | None = None


@dataclass
class TrainState:
    spec: MethodSpec
    baseline: HistoricalBaseline
    rng: IsolatedRNG
    layout: ParamLayout
    u: np.ndarray
    predictor: PredictorHeads | None
    reservoir: GradientReservoir
    basis_id: int = 0
    n_ref: int = 0
    history_costs: list[float] = field(default_factory=list)
    step: int = 0


def _reward_features(q_hat: float, length: float, baseline: float | None = None):
    from grace_gc.predictor.features import reward_risk_features

    q = float(np.clip(q_hat, 1e-6, 1.0 - 1e-6))
    b = q if baseline is None else float(np.clip(baseline, 1e-6, 1.0 - 1e-6))
    return reward_risk_features(q, 1.0 - q, length, b)


def _length_truncated(
    full_ids: list[int],
    prompt_len: int,
    max_new: int,
    finished: bool,
    eos_id: int | list[int] | None,
) -> bool:
    """True only for length hits; natural EOS at the budget is not truncation."""
    from grace_gc.data.tokenize import is_stop_token

    if finished:
        return False
    if len(full_ids) < int(prompt_len) + int(max_new):
        return False
    if eos_id is not None and full_ids and is_stop_token(full_ids[-1], eos_id):
        return False
    return True


def _traj_natural_finish(
    prefix_finished: bool,
    full_ids,
    eos_id,
    generated: int | None = None,
    requested: int | None = None,
) -> bool:
    """EOS on the prefix, on the continuation, or a short continue after vLLM omitted the stop id."""
    from grace_gc.data.tokenize import is_stop_token

    if prefix_finished:
        return True
    if not full_ids:
        return False
    if eos_id is not None and is_stop_token(full_ids[-1], eos_id):
        return True
    if generated is not None and requested is not None and generated < int(requested):
        return True
    return False


def _meta_at(prompt_meta, i: int, key: str, default=None):
    if not prompt_meta or i >= len(prompt_meta) or not isinstance(prompt_meta[i], dict):
        return default
    return prompt_meta[i].get(key, default)


def _rollout_finish(engines: StepEngines, i: int, z: float):
    roll = getattr(engines, "last_rollout", None) or {}
    cont = roll.get("continue_finish_reasons") or {}
    if float(z) >= 1.0 and i in cont:
        return cont[i]
    prefix = roll.get("prefix_finish_reasons") or []
    if i < len(prefix):
        return prefix[i]
    return None


def _rollout_stop(engines: StepEngines, i: int, z: float):
    roll = getattr(engines, "last_rollout", None) or {}
    cont = roll.get("continue_stop_reasons") or {}
    if float(z) >= 1.0 and i in cont:
        return cont[i]
    prefix = roll.get("prefix_stop_reasons") or []
    if i < len(prefix):
        return prefix[i]
    return None


def _pad_features(feats: np.ndarray, k: int) -> np.ndarray:
    if feats.shape[1] >= k:
        return feats[:, :k]
    return np.pad(feats, ((0, 0), (0, k - feats.shape[1])))


def prescan_unseen_baselines(
    engines: StepEngines,
    state: TrainState,
    prompt_ids: list[list[int]],
    problem_ids: list[str],
    golds: list[str],
    max_new: int,
    n_prescan: int,
) -> int:
    """Paper §6.3: unseen problems get a 4-sample pass rate, not the 0.5 default."""
    if n_prescan <= 0:
        return 0
    done = 0
    seen: set[str] = set()
    for i, pid in enumerate(problem_ids):
        if pid in state.baseline.values or pid in seen:
            continue
        seen.add(pid)
        prompts = [list(prompt_ids[i])] * int(n_prescan)
        fulls, finished = engines.generate_prefix(prompts, int(max_new), state.rng, "token")
        finished = np.asarray(finished, dtype=bool).reshape(-1)
        if len(fulls) != int(n_prescan) or finished.shape[0] != int(n_prescan):
            raise ValueError(
                f"prescan generate_prefix returned {len(fulls)} sequences for {n_prescan} samples"
            )
        rewards = []
        plen = len(prompt_ids[i])
        for j, full in enumerate(fulls):
            if full is None:
                raise ValueError("prescan generate_prefix returned no sequence")
            resp = full[plen:]
            text = engines.decode(resp) if engines.decode else ""
            truncated = _length_truncated(full, plen, int(max_new), bool(finished[j]), engines.eos_id)
            scored = engines.reward_fn(full, golds[i], truncated=truncated, text=text)
            rewards.append(0.0 if scored is None else float(scored))
        state.baseline.values[pid] = float(np.mean(rewards)) if rewards else 0.5
        done += 1
    return done


def continuation_remainings(prefixes, prompt_lens, finished, max_new: int) -> np.ndarray:
    """Leftover tokens against the method budget, from the actual prefix length."""
    rem = np.zeros(len(prefixes), dtype=np.int64)
    fin = np.asarray(finished, dtype=bool).reshape(-1)
    budget = int(max_new)
    for i, prefix in enumerate(prefixes):
        if fin[i]:
            continue
        generated = max(len(prefix) - int(prompt_lens[i]), 0)
        rem[i] = max(0, budget - generated)
    return rem


def run_algorithm1_step(
    engines: StepEngines,
    state: TrainState,
    prompt_ids: list[list[int]],
    problem_ids: list[str],
    golds: list[str],
    cfg: dict[str, Any],
    optimizer,
    prompt_meta: list | None = None,
) -> dict[str, Any]:
    torch = __import__("torch")
    n = len(prompt_ids)
    if n <= 0:
        raise ValueError("need at least one start")
    decision = int(cfg.get("decision_tokens", 16))
    max_new = int(cfg.get("max_new_tokens", 32))
    if decision > max_new:
        raise ValueError(f"decision_tokens {decision} exceeds max_new_tokens {max_new}")
    if state.n_ref <= 0:
        state.n_ref = n
    alloc = cfg.get("allocation", {})
    pred_cfg = cfg.get("predictor", {})
    warmup = state.step < int(pred_cfg.get("warmup_steps", 0))
    audit_s = float(pred_cfg.get("audit_s", 0.125))
    n_prescan = int((cfg.get("baseline") or {}).get("prescan", 0) or 0)
    timer = Timer()
    n_prescanned = prescan_unseen_baselines(
        engines, state, prompt_ids, problem_ids, golds, max_new, n_prescan
    )
    timings = {"prescan": timer.lap()}

    prefixes, finished = engines.generate_prefix(prompt_ids, decision, state.rng, "token")
    finished = np.asarray(finished, dtype=bool).reshape(-1)
    if len(prefixes) != n or finished.shape[0] != n:
        raise ValueError(f"generate_prefix returned {len(prefixes)} sequences for {n} starts")
    prompt_lens = np.array([len(p) for p in prompt_ids], dtype=np.int64)
    k = int(state.u.shape[1])
    need_features = state.spec.use_predictor or state.spec.feature_mode != "none"
    if need_features:
        feat_bundle = engines.prefix_features(
            prefixes, prompt_lens, [state.baseline.get(pid) for pid in problem_ids]
        )
        feat = feat_bundle["features"]
        cost_feat = feat_bundle.get("cost_feat")
        if state.spec.feature_mode == "prompt":
            if "prompt_features" not in feat_bundle:
                raise ValueError("Prompt-CV requires prompt_features from the actor engine")
            feat = feat_bundle["prompt_features"]
    else:
        feat = np.zeros((n, max(k, 1)), dtype=np.float64)
        cost_feat = None

    if state.predictor is not None and not warmup and state.spec.use_predictor:
        pred = state.predictor.forward_numpy(feat, cost_feat)
        f, r_hat, c_hat = pred.f, pred.r_hat, pred.c_hat
    else:
        f = _pad_features(feat, k)
        r_hat = np.ones(n, dtype=np.float64)
        c_hat = np.ones(n, dtype=np.float64)
    lengths = [float(max(len(prefixes[i]) - int(prompt_lens[i]), 1)) for i in range(n)]
    q_hat_out = None
    if state.spec.risk_mode == "reward" and state.predictor is not None and not warmup:
        q_hat = state.predictor.forward_success(feat)
        q_hat_out = q_hat
        reward_feats = np.stack(
            [
                _reward_features(float(q_hat[i]), lengths[i], state.baseline.get(problem_ids[i]))
                for i in range(n)
            ]
        )
        r_hat = state.predictor.forward_reward_risk(reward_feats)
    else:
        reward_feats = np.stack(
            [_reward_features(state.baseline.get(problem_ids[i]), lengths[i]) for i in range(n)]
        )
    timings["prefix"] = timer.lap()

    p, deviation = decide_continuation(
        state.spec,
        r_hat,
        c_hat,
        finished,
        float(alloc.get("beta", 0.5)),
        float(alloc.get("p_min", 0.2)),
        warmup,
        iters=int(alloc.get("bisection_iters", 20)),
    )
    if warmup or state.spec.name in {"full_pg", "grpo", "grpo_short"}:
        z = np.ones(n, dtype=np.float64)
    else:
        z = state.rng.bernoulli("selection", p)
    z[finished] = 1.0
    p[finished] = 1.0
    f = np.asarray(f, dtype=np.float64)
    f[finished] = 0.0
    timings["allocate"] = timer.lap()

    remainings = continuation_remainings(prefixes, prompt_lens, finished, max_new)
    selected = (z >= 1.0) & (~finished) & (remainings > 0)
    full_ids: list[list[int] | None] = [None] * n
    for rem in sorted({int(remainings[i]) for i in range(n) if selected[i]}):
        mask = selected & (remainings == rem)
        part = engines.continue_selected(prefixes, mask, rem, state.rng)
        for i in range(n):
            if mask[i]:
                if part[i] is None:
                    raise ValueError("continuation returned no sequence for a selected start")
                full_ids[i] = part[i]
    for i in range(n):
        if z[i] >= 1.0 and full_ids[i] is None:
            full_ids[i] = prefixes[i]
    rewards: list[float | None] = []
    texts: list[str | None] = []
    traj_finished: list[bool] = []
    truncated_flags: list[bool] = []
    prefix_texts: list[str | None] = []
    extracted_list: list[str | None] = []
    first_tokens: list[int | None] = []
    reasons: list[str] = []
    for i in range(n):
        prefix_resp = prefixes[i][len(prompt_ids[i]) :]
        prefix_text = engines.decode(prefix_resp) if engines.decode else ""
        prefix_texts.append(prefix_text)
        if z[i] < 1.0 or full_ids[i] is None:
            rewards.append(None)
            texts.append(None)
            traj_finished.append(False)
            truncated_flags.append(False)
            extracted = extract_answer(prefix_text)
            extracted_list.append(extracted)
            first_tokens.append(None if extracted is None else answer_first_token(engines.decode, prefix_resp))
            reasons.append("stopped")
            continue
        resp = full_ids[i][len(prompt_ids[i]) :]
        text = engines.decode(resp) if engines.decode else ""
        texts.append(text)
        gen_cont = max(len(full_ids[i]) - len(prefixes[i]), 0)
        traj_fin = _traj_natural_finish(
            bool(finished[i]),
            full_ids[i],
            engines.eos_id,
            generated=None if finished[i] else gen_cont,
            requested=None if finished[i] else int(remainings[i]),
        )
        traj_finished.append(traj_fin)
        truncated = _length_truncated(
            full_ids[i], int(prompt_lens[i]), max_new, traj_fin, engines.eos_id
        )
        truncated_flags.append(truncated)
        extracted = extract_answer(text)
        extracted_list.append(extracted)
        first_tokens.append(None if extracted is None else answer_first_token(engines.decode, resp))
        reasons.append(finish_reason(float(z[i]), traj_fin, truncated))
        reward = engines.reward_fn(full_ids[i], golds[i], truncated=truncated, text=text)
        if reward is None:
            reward = 0.0
        rewards.append(reward)
    timings["continue"] = timer.lap()

    adv = advantages_for_method(state.spec.objective, rewards, problem_ids, state.baseline)
    named = engines.named_lora()
    params = engines.trainable_params()
    optimizer.zero_grad()
    chosen = z >= 1.0
    loss = real_stream_backward_each(
        lambda i: None if full_ids[i] is None else engines.logprob_one(full_ids[i], int(prompt_lens[i])),
        adv,
        p,
        z,
        n,
        chosen,
    )
    saved = snapshot_grads(params)
    timings["backward"] = timer.lap()

    audited = 0
    audit_draw = (
        state.rng.bernoulli("audit", np.full(n, audit_s))
        if state.spec.use_predictor
        else np.zeros(n, dtype=np.float64)
    )
    records = []
    for i in range(n):
        rec = StartRecord(
            problem_id=problem_ids[i],
            finished=traj_finished[i],
            p=float(p[i]),
            z=float(z[i]),
            f=f[i],
            r_hat=float(r_hat[i]),
            c_hat=float(c_hat[i]),
            reward=rewards[i],
            advantage=float(adv[i]) if rewards[i] is not None else None,
            g=None,
            audited=False,
            prompt_len=int(prompt_lens[i]),
            prefix_tokens=int(max(len(prefixes[i]) - int(prompt_lens[i]), 0)),
            response_tokens=0
            if full_ids[i] is None
            else int(max(len(full_ids[i]) - int(prompt_lens[i]), 0)),
            suffix_tokens=0
            if full_ids[i] is None
            else int(max(len(full_ids[i]) - len(prefixes[i]), 0)),
            finish_reason=reasons[i],
            truncated=truncated_flags[i],
            text=texts[i],
            prefix_text=prefix_texts[i],
            gold=golds[i],
            extracted=extracted_list[i],
            answer_first_token=first_tokens[i],
            baseline_b=state.baseline.get(problem_ids[i]),
            q_hat=None if q_hat_out is None else float(q_hat_out[i]),
            prompt_token_ids=[int(x) for x in prompt_ids[i]],
            prefix_token_ids=[int(x) for x in prefixes[i]],
            full_token_ids=None if full_ids[i] is None else [int(x) for x in full_ids[i]],
            prompt_truncated=bool(_meta_at(prompt_meta, i, "prompt_truncated", False)),
            untruncated_prompt_len=_meta_at(prompt_meta, i, "untruncated_len"),
            used_chat_template=_meta_at(prompt_meta, i, "used_chat_template"),
            thinking_closed=_meta_at(prompt_meta, i, "thinking_closed"),
            vllm_finish_reason=_rollout_finish(engines, i, float(z[i])),
            vllm_stop_reason=_rollout_stop(engines, i, float(z[i])),
        )
        if state.spec.use_predictor and rec.z >= 1.0 and rec.reward is not None and audit_draw[i] >= 1.0:
            scalar = adv[i] * engines.logprob_one(full_ids[i], int(prompt_lens[i]))
            rec.g = audit_one(scalar, params, [nm for nm, _p in named], state.layout)
            rec.audited = True
            rec.g_norm_sq = float(np.dot(rec.g, rec.g))
            audited += 1
            realized = None if full_ids[i] is None else float(max(len(full_ids[i]) - len(prefixes[i]), 1))
            cf = None if cost_feat is None else np.asarray(cost_feat[i], dtype=np.float64)
            state.reservoir.add(
                ReservoirItem(
                    problem_id=rec.problem_id,
                    g=rec.g,
                    p=rec.p,
                    s=audit_s,
                    features=feat[i],
                    reward=rec.reward,
                    basis_id=state.basis_id,
                    cost_feat=cf,
                    realized_cost=realized,
                    reward_feat=np.asarray(reward_feats[i], dtype=np.float64),
                )
            )
        records.append(rec)
    restore_grads(params, saved)
    timings["audit"] = timer.lap()

    update_stats: dict[str, Any] = {}
    packed, clipped = apply_correction_clip_step(
        named,
        state.layout,
        state.u,
        f,
        z,
        p,
        n,
        optimizer,
        clip=float(cfg.get("optim", {}).get("grad_clip", 1.0)),
        use_correction=state.spec.use_ht_correction and state.spec.use_predictor and not warmup,
        stats=update_stats,
    )
    _ = packed
    _ = torch
    timings["update"] = timer.lap()

    batch_rewards: dict[str, list[float]] = {}
    for rec in records:
        if rec.reward is None:
            continue
        batch_rewards.setdefault(rec.problem_id, []).append(float(rec.reward))
    for pid, rs in batch_rewards.items():
        state.baseline.update_mean(pid, rs)

    u_frozen = np.asarray(state.u, dtype=np.float64).copy()
    pred_metrics = {"coord_loss": None, "risk_loss": None}
    if state.spec.use_predictor and state.predictor is not None and len(state.reservoir.items) >= 2:
        if audited and state.reservoir.items:
            g_mat = state.reservoir.recent_matrix()
            if g_mat.shape[0] >= k and (state.step + 1) % int(pred_cfg.get("refresh_every", 32)) == 0:
                from grace_gc.predictor.basis import refresh_basis

                basis = refresh_basis(
                    g_mat,
                    k=k,
                    basis_id=state.basis_id + 1,
                    seed=state.step,
                    problem_ids=state.reservoir.problem_ids(),
                )
                state.u = basis.u
                state.basis_id = basis.basis_id
        pred_metrics = update_predictor_from_reservoir(
            state.predictor,
            state.reservoir,
            state.u,
            state.rng.generator("predictor"),
            epochs=int(pred_cfg.get("epochs", 2)),
        )

    timings["predictor"] = timer.lap()
    used, full = batch_token_costs(prompt_lens, prefixes, finished, z, max_new)
    ratio = 1.0 if full <= 0.0 else used / full
    state.history_costs.append(ratio)
    state.step += 1
    group = int(getattr(state.spec, "starts_per_prompt", 0) or 0)
    if group <= 0:
        group = max(1, n // max(1, len(set(problem_ids))))
    next_n = next_start_count(state.history_costs, state.n_ref or n, c_full=1.0, group=group)
    return {
        "records": records,
        "n": n,
        "n_completed": int(z.sum()),
        "n_stopped": int((z < 1.0).sum()),
        "p": p,
        "z": z,
        "deviation": deviation,
        "loss": float(loss.detach()) if hasattr(loss, "detach") else float(loss),
        "n_audited": audited,
        "clip_triggered": clipped,
        "next_n": next_n,
        "texts": texts,
        "predictor": pred_metrics,
        "warmup": warmup,
        "n_prescan": n_prescanned,
        "token_cost_used": float(used),
        "token_cost_full": float(full),
        "timings": timings,
        "u_frozen": u_frozen,
        "grad_norm": update_stats.get("grad_norm"),
        "grad_norm_preclip": update_stats.get("grad_norm_preclip"),
        "sampling": (getattr(engines, "last_rollout", None) or {}).get("sampling"),
        "residual_mean": None
        if not any(r.g is not None for r in records)
        else float(
            np.mean(
                [
                    full_space_residual(r.g, r.f, u_frozen)
                    for r in records
                    if r.g is not None
                ]
            )
        ),
    }
