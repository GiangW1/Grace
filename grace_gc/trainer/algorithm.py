"""One Algorithm-1 training step, shared by CPU tiny and GPU engines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable
import copy
import time

import numpy as np

from grace_gc.core.layout import ParamLayout, collect_lora_layout
from grace_gc.core.rng import IsolatedRNG
from grace_gc.predictor.heads import PredictorHeads
from grace_gc.predictor.reservoir import GradientReservoir, ReservoirItem
from grace_gc.predictor.risk import full_space_residual
from grace_gc.data.reward import extract_answer
from grace_gc.logging_util.ledger import Timer
from grace_gc.predictor.basis import align_basis, refresh_basis, should_refresh_basis
from grace_gc.predictor.update import diagnose_frozen_predictor, prepare_predictor_split, update_predictor_from_reservoir
from grace_gc.trainer.actor_update import (
    apply_correction_clip_step,
    real_stream_backward_and_audit,
)
from grace_gc.trainer.advantages import advantages_for_method
from grace_gc.trainer.baseline import HistoricalBaseline
from grace_gc.trainer.cost_control import start_count_mode
from grace_gc.trainer.grace_step import (
    StartRecord,
    answer_first_token,
    batch_token_costs,
    control_variate_coordinates,
    decide_continuation,
    finish_reason,
    incremental_token_costs,
    maybe_mark_predictor_synced,
    next_start_count,
    neyman_ready,
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
    before_update: Callable | None = None
    last_prescan: list | None = None


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
    predictor_synced_basis_id: int = -1
    prescan_rng: IsolatedRNG | None = None
    n_ref: int = 0
    history_costs: list[float] = field(default_factory=list)
    step: int = 0
    cost_control: dict = field(default_factory=dict)
    offline_predictor: dict | None = None


def ensure_prescan_rng(state: TrainState) -> IsolatedRNG:
    """Prescan uses its own IsolatedRNG so it does not consume the actor token stream."""
    if state.prescan_rng is None:
        state.prescan_rng = IsolatedRNG.create(int(state.rng.seed) + 1)
    return state.prescan_rng


def _scalar_float(value) -> float:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _rollout_logprob_sum(engines: StepEngines, i: int, z: float) -> float | None:
    roll = getattr(engines, "last_rollout", None) or {}
    prefix_lps = list(roll.get("prefix_logprob_sums") or [])
    cont = roll.get("continue_logprob_sums") or {}
    pref = prefix_lps[i] if i < len(prefix_lps) else None
    if float(z) >= 1.0 and i in cont:
        suf = cont[i]
        if pref is None or suf is None:
            return None
        return float(pref) + float(suf)
    if pref is None:
        return None
    return float(pref)


def _rollout_token_logprobs(engines: StepEngines, i: int, z: float):
    roll = engines.last_rollout or {}
    prefix = roll.get("prefix_token_logprobs") or []
    if i >= len(prefix):
        return None
    values = list(prefix[i])
    if z >= 1.0:
        values += list((roll.get("continue_token_logprobs") or {}).get(i, []))
    return values


def _rollout_request_seeds(engines: StepEngines, i: int):
    roll = engines.last_rollout or {}
    prefix = roll.get("prefix_request_seeds") or []
    return {"prefix": prefix[i] if i < len(prefix) else None,
            "continuation": (roll.get("continue_request_seeds") or {}).get(i)}


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
    finish_reason=None,
) -> bool:
    """True only for length hits; natural EOS at the budget is not truncation."""
    from grace_gc.backends.vllm_two_phase import _is_length_finish, _is_stop_finish
    from grace_gc.data.tokenize import is_stop_token

    if _is_length_finish(finish_reason):
        return True
    if _is_stop_finish(finish_reason) or finished:
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
    finish_reason=None,
) -> bool:
    """EOS on the prefix or continuation. vLLM length is not a natural stop."""
    from grace_gc.backends.vllm_two_phase import _is_length_finish, _is_stop_finish
    from grace_gc.data.tokenize import is_stop_token

    if _is_length_finish(finish_reason):
        return False
    if _is_stop_finish(finish_reason):
        return True
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


def _gpu_progress(cfg: dict[str, Any] | None, msg: str) -> None:
    if str((cfg or {}).get("backend", "")) == "gpu_verl":
        print(msg, flush=True)


def prescan_unseen_baselines(
    engines: StepEngines,
    state: TrainState,
    prompt_ids: list[list[int]],
    problem_ids: list[str],
    golds: list[str],
    max_new: int,
    n_prescan: int,
    batch: bool = False,
) -> int:
    """Independent full samples for unseen problems; retain their source evidence."""
    engines.last_prescan = []
    if n_prescan <= 0 or state.spec.objective == "grpo" or not state.baseline.uses_prescan:
        return 0
    unseen = []
    seen = set(state.baseline.values)
    for i, pid in enumerate(problem_ids):
        if pid not in seen:
            unseen.append(i)
            seen.add(pid)
    groups = [unseen] if batch and unseen else [[i] for i in unseen]
    for group in groups:
        source = [i for i in group for _ in range(int(n_prescan))]
        prompts = [list(prompt_ids[i]) for i in source]
        fulls, finished = engines.generate_prefix(prompts, int(max_new), ensure_prescan_rng(state), "token")
        finished = np.asarray(finished, dtype=bool).reshape(-1)
        if len(fulls) != len(prompts) or len(finished) != len(prompts):
            raise ValueError("prescan generate_prefix returned the wrong number of samples")
        roll = getattr(engines, "last_rollout", None) or {}
        reasons = list(roll.get("prefix_finish_reasons") or [])
        seeds = (roll.get("sampling") or {}).get("request_seeds") or []
        rewards = {i: [] for i in group}
        for j, (i, full) in enumerate(zip(source, fulls)):
            if full is None:
                raise ValueError("prescan generate_prefix returned no sequence")
            plen = len(prompt_ids[i])
            text = engines.decode(full[plen:]) if engines.decode else ""
            fr = reasons[j] if j < len(reasons) else None
            truncated = _length_truncated(full, plen, int(max_new), bool(finished[j]), engines.eos_id, finish_reason=fr)
            scored = engines.reward_fn(full, golds[i], truncated=truncated, text=text)
            reward = 0.0 if scored is None else float(scored)
            rewards[i].append(reward)
            engines.last_prescan.append({
                "problem_id": problem_ids[i], "gold": golds[i], "text": text,
                "reward": reward, "truncated": truncated, "finish_reason": fr,
                "prompt_token_ids": list(prompt_ids[i]), "full_token_ids": list(full),
                "request_seed": seeds[j] if j < len(seeds) else None,
            })
        for i, values in rewards.items():
            state.baseline.initialize_from_prescan(problem_ids[i], values)
        for row in engines.last_prescan[-len(source):]:
            row["baseline_after_prescan"] = state.baseline.get(row["problem_id"])
            row["baseline_configuration"] = state.baseline.configuration()
    return len(unseen)


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
    if state.reservoir.fixed_basis_id is not None:
        if not pred_cfg.get("fixed_basis", False):
            raise ValueError("compact supervision requires predictor.fixed_basis=true on resume")
        if state.reservoir.fixed_basis_id != state.basis_id:
            raise ValueError("compact supervision does not match the actor basis")
    from grace_gc.versions import sha256_array, sha256_mapping, sha256_named

    behavior_context = {
        "snapshot_sha": sha256_named(engines.named_lora()),
        "basis_sha": (state.offline_predictor or {}).get('basis_sha') or sha256_array(state.u),
        "predictor_sha": None if state.predictor is None else sha256_mapping(state.predictor),
        "rng_counters": dict(state.rng.counters),
        "timing": "before_rollout_and_update",
    }
    new_audit_items = []
    frozen = state.offline_predictor is not None
    warmup = not frozen and state.step < int(pred_cfg.get("warmup_steps", 0))
    audit_s = float(pred_cfg.get("audit_s", 0.125))
    n_prescan = int((cfg.get("baseline") or {}).get("prescan", 0) or 0)
    if state.spec.objective == "grpo" or not state.baseline.uses_prescan:
        n_prescan = 0
    control_variate_enabled = bool(pred_cfg.get("control_variate", True))
    timer = Timer()
    if n_prescan > 0:
        _gpu_progress(
            cfg,
            f"phase=prescan n={n} n_prescan={n_prescan} max_new={max_new} "
            f"(unseen problems, {n_prescan} full samples each)",
        )
    n_prescanned = prescan_unseen_baselines(
        engines, state, prompt_ids, problem_ids, golds, max_new, n_prescan,
        batch=bool((cfg.get("baseline") or {}).get("batch_prescan", False)),
    )
    timings = {"prescan": timer.lap()}
    if n_prescanned:
        _gpu_progress(cfg, f"phase=prescan_done n_prescanned={n_prescanned} wall_s={timings['prescan']:.1f}")
    _gpu_progress(cfg, f"phase=prefix n={n} decision={decision}")

    detail_started = time.perf_counter()
    prefixes, finished = engines.generate_prefix(prompt_ids, decision, state.rng, "token")
    timing_details = {"prefix_generate": time.perf_counter() - detail_started}
    finished = np.asarray(finished, dtype=bool).reshape(-1)
    if len(prefixes) != n or finished.shape[0] != n:
        raise ValueError(f"generate_prefix returned {len(prefixes)} sequences for {n} starts")
    prompt_lens = np.array([len(p) for p in prompt_ids], dtype=np.int64)
    k = int(state.u.shape[1])
    need_features = state.spec.use_predictor or state.spec.feature_mode != "none"
    detail_started = time.perf_counter()
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
    timing_details["prefix_features"] = time.perf_counter() - detail_started

    detail_started = time.perf_counter()
    remainings = continuation_remainings(prefixes, prompt_lens, finished, max_new)
    if state.predictor is not None and not warmup and state.spec.use_predictor:
        pred = state.predictor.forward_numpy(feat, cost_feat)
        f, r_hat, c_hat = pred.f, pred.r_hat, pred.c_hat
        if state.predictor.constant_cost:
            c_hat = incremental_token_costs(remainings, finished)
    else:
        f = _pad_features(feat, k)
        r_hat = np.ones(n, dtype=np.float64)
        c_hat = incremental_token_costs(remainings, finished)
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
    timing_details["predictor_forward"] = time.perf_counter() - detail_started
    timings["prefix"] = timer.lap()

    basis_at_allocate = int(state.basis_id)
    synced_at_allocate = int(state.predictor_synced_basis_id)
    basis_ready = (not state.spec.use_allocation) or neyman_ready(basis_at_allocate, synced_at_allocate)
    p, deviation = decide_continuation(
        state.spec,
        r_hat,
        c_hat,
        finished,
        float(alloc.get("beta", 0.5)),
        float(alloc.get("p_min", 0.2)),
        warmup,
        iters=int(alloc.get("bisection_iters", 20)),
        basis_ready=basis_ready,
        uniform_shrink=float(alloc.get("uniform_shrink", 0.0)),
    )
    force_complete = warmup or not basis_ready or state.spec.name in {"full_pg", "grpo", "grpo_short"}
    if force_complete:
        z = np.ones(n, dtype=np.float64)
    else:
        z = state.rng.bernoulli("selection", p)
    z[finished] = 1.0
    p[finished] = 1.0
    f = control_variate_coordinates(f, enabled=control_variate_enabled)
    f[finished] = 0.0
    timings["allocate"] = timer.lap()

    selected = (z >= 1.0) & (~finished) & (remainings > 0)
    detail_started = time.perf_counter()
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
    timing_details["suffix_generate"] = time.perf_counter() - detail_started
    # Read only after selection, and snapshot before auxiliary rollouts.
    rollout = engines.last_rollout or {}
    rollout_execution = {key: copy.deepcopy(rollout[key]) for key in (
        "prefix_execution", "continue_execution", "prefix_num_cached_tokens", "continue_num_cached_tokens"
    ) if key in rollout}
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
        vllm_fr = _rollout_finish(engines, i, float(z[i]))
        traj_fin = _traj_natural_finish(
            bool(finished[i]),
            full_ids[i],
            engines.eos_id,
            generated=None if finished[i] else gen_cont,
            requested=None if finished[i] else int(remainings[i]),
            finish_reason=vllm_fr,
        )
        traj_finished.append(traj_fin)
        truncated = _length_truncated(
            full_ids[i],
            int(prompt_lens[i]),
            max_new,
            traj_fin,
            engines.eos_id,
            finish_reason=vllm_fr,
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
    _gpu_progress(cfg, f"phase=continue_done n_rewards={len(rewards)} wall_s={timings['continue']:.1f}")

    adv = advantages_for_method(state.spec.objective, rewards, problem_ids, state.baseline)
    named = engines.named_lora()
    optimizer.zero_grad()
    chosen = z >= 1.0
    audit_draw = (
        state.rng.bernoulli("audit", np.full(n, audit_s))
        if state.spec.use_predictor and not frozen
        else np.zeros(n, dtype=np.float64)
    )
    loss, audited_gradients = real_stream_backward_and_audit(
        lambda i: None if full_ids[i] is None else engines.logprob_one(full_ids[i], int(prompt_lens[i])),
        adv,
        p,
        z,
        n,
        chosen,
        named,
        state.layout,
        audit_draw >= 1.0,
    )
    timings["backward"] = timer.lap()

    audited = 0
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
            rollout_logprob_sum=_rollout_logprob_sum(engines, i, float(z[i])),
            rollout_token_logprobs=_rollout_token_logprobs(engines, i, float(z[i])),
            request_seeds=_rollout_request_seeds(engines, i),
        )
        if rec.rollout_logprob_sum is None and rec.z >= 1.0 and full_ids[i] is not None:
            roll = getattr(engines, "last_rollout", None) or {}
            if not roll.get("has_generate_logprobs"):
                rec.rollout_logprob_sum = _scalar_float(
                    engines.logprob_one(full_ids[i], int(prompt_lens[i]))
                )
        if state.spec.use_predictor and rec.z >= 1.0 and rec.reward is not None and audit_draw[i] >= 1.0:
            rec.g = audited_gradients[i]
            rec.audited = True
            rec.g_norm_sq = float(np.dot(rec.g, rec.g))
            audited += 1
            realized = None if full_ids[i] is None else float(max(len(full_ids[i]) - len(prefixes[i]), 1))
            cf = None if cost_feat is None else np.asarray(cost_feat[i], dtype=np.float64)
            item = ReservoirItem(
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
                    observed_step=state.step + 1,
                    remaining_cost=float(max(remainings[i], 1)),
                    prefix_finished=bool(finished[i]),
                )
            state.reservoir.add(item, u=state.u)
            new_audit_items.append(item)
        records.append(rec)
    timings["audit"] = timer.lap()

    from grace_gc.trainer.supervision import collect_fresh_supervision

    fresh_items, fresh_rows, fresh_metrics = ([], [], {"n": 0, "generated_tokens": 0}) if frozen else collect_fresh_supervision(
        engines, state, prompt_ids, problem_ids, golds, cfg,
    )
    for item in fresh_items:
        state.reservoir.add(item, u=state.u)
    new_audit_items.extend(fresh_items)
    for row in fresh_rows:
        row["behavior_context"] = behavior_context
    timings["fresh_supervision"] = timer.lap()

    logprob_probe = None if engines.before_update is None else engines.before_update(records)
    frozen_predictor = (None if state.predictor is None or frozen else
                        diagnose_frozen_predictor(state.predictor, new_audit_items, state.u,
                                                  risk_mode=state.spec.risk_mode, basis_id=state.basis_id))
    if frozen_predictor is not None:
        frozen_predictor["used_in_actor_update"] = bool(state.spec.use_ht_correction and not warmup and control_variate_enabled)
        frozen_predictor["basis_id"] = int(state.basis_id)
    timings["behavior_probe"] = timer.lap()

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
        use_correction=state.spec.use_ht_correction and state.spec.use_predictor and not warmup and control_variate_enabled,
        stats=update_stats,
        log_geometry=bool((cfg.get("optim") or {}).get("log_update_geometry", False)),
    )
    _ = packed
    _ = torch
    timings["update"] = timer.lap()

    batch_rewards: dict[str, list] = {}
    batch_probabilities: dict[str, list[float]] = {}
    for rec in records:
        batch_rewards.setdefault(rec.problem_id, []).append(rec.reward)
        batch_probabilities.setdefault(rec.problem_id, []).append(float(rec.p))
    if state.spec.objective != "grpo":
        for pid, rs in batch_rewards.items():
            state.baseline.update_ht_mean(pid, rs, batch_probabilities[pid])

    # Basis refresh assigns a new array; retaining the frozen reference avoids
    # another D-by-k copy without allowing this batch's correction to change.
    u_frozen = state.u
    pred_metrics = {"coord_loss": None, "risk_loss": None}
    basis_rank = None
    basis_changed = False
    if not frozen and state.spec.use_predictor and state.predictor is not None and len(state.reservoir.items) >= 2:
        variant = str(pred_cfg.get("basis_variant", "pca"))
        if variant not in {"pca", "predictable_crossfit", "predictable_crossfit_signal"}:
            raise ValueError("basis_variant must be pca, predictable_crossfit or predictable_crossfit_signal")
        crossfit_basis = variant in {"predictable_crossfit", "predictable_crossfit_signal"}
        split = prepare_predictor_split(
            state.predictor, state.reservoir, state.rng.generator("predictor"),
            basis_fit_only=bool(pred_cfg.get("basis_fit_only", False)) or crossfit_basis,
            current_step=state.step + 1, max_age_steps=pred_cfg.get("max_age_steps"),
            age_half_life=pred_cfg.get("age_half_life"),
            missing_step_policy=pred_cfg.get("missing_step_policy", "exclude"),
        )
        basis_mask = split["fit"] if crossfit_basis else split["basis"]
        basis_items = [it for it, keep in zip(state.reservoir.items, basis_mask) if keep]
        basis_metrics = {"variant": variant, "refreshed": False}
        if state.reservoir.fixed_basis_id is None and should_refresh_basis(
            len(basis_items),
            min(k, len(basis_items)) if crossfit_basis else k,
            state.step,
            state.basis_id,
            int(pred_cfg.get("refresh_every", 32)),
            warmup_steps=int(pred_cfg.get("warmup_steps", 0)),
            refresh_after_warmup=bool(pred_cfg.get("refresh_after_warmup", False)),
        ):
            from grace_gc.predictor.ipw import ipw_weights

            basis_weights = split["age_weight"][basis_mask].copy()
            if pred_cfg.get("basis_ipw", False):
                basis_weights *= ipw_weights(np.array([it.p for it in basis_items]), np.array([it.s for it in basis_items]))
            basis_args = dict(grads=np.stack([it.g for it in basis_items]), k=k,
                              basis_id=state.basis_id + 1,
                              problem_ids=[it.problem_id for it in basis_items], weights=basis_weights)
            if crossfit_basis:
                from grace_gc.predictor.basis import refresh_predictable_basis

                basis, details = refresh_predictable_basis(
                    **basis_args, features=np.stack([it.features for it in basis_items]),
                    ridge_l2=float(pred_cfg.get("basis_ridge_l2", pred_cfg.get("ridge_l2", 1.))),
                    objective="cross_moment" if variant == "predictable_crossfit_signal" else "prediction_energy",
                )
                basis_metrics.update(details)
            else:
                basis = refresh_basis(**basis_args, seed=state.step,
                                      solver=str(pred_cfg.get("basis_solver", "auto")),
                                      center=str(pred_cfg.get("basis_center", "problem")))
            if basis is not None:
                u_new = basis.u
                if int(state.basis_id) > 0 and bool(pred_cfg.get("align_basis", True)):
                    u_new = align_basis(u_new, state.u)
                state.u = u_new
                if int(state.basis_id) != int(basis.basis_id):
                    basis_changed = True
                state.basis_id = basis.basis_id
                basis_rank = int(basis.rank)
                basis_metrics["refreshed"] = True
                state.reservoir.stamp_basis_id(state.basis_id)
        if (pred_cfg.get("fixed_basis", False) and state.reservoir.fixed_basis_id is None
                and state.step + 1 >= int(pred_cfg.get("warmup_steps", 0)) and state.basis_id > 0):
            state.reservoir.compact(state.u, state.basis_id)
        basis_metrics.update(fixed=state.reservoir.fixed_basis_id is not None,
                             fixed_basis_id=state.reservoir.fixed_basis_id)
        pred_metrics = update_predictor_from_reservoir(
            state.predictor,
            state.reservoir,
            state.u,
            state.rng.generator("predictor"),
            epochs=int(pred_cfg.get("epochs", 2)),
            current_step=state.step + 1,
            split=split,
            calibration_beta=float(alloc.get("beta", .5)),
            calibration_p_min=float(alloc.get("p_min", .2)),
            calibration_p=(np.array([1. if it.prefix_finished else max(float(alloc.get("beta", .5)), float(alloc.get("p_min", .2)))
                                     for it in state.reservoir.items]) if not state.spec.use_allocation else None),
            calibration_risk_mode=state.spec.risk_mode,
            calibration_uniform_shrink=float(alloc.get("uniform_shrink", 0.0)),
            basis_id=state.basis_id,
        )
        pred_metrics["frozen_new_problems"] = frozen_predictor
        pred_metrics["basis"] = basis_metrics
        maybe_mark_predictor_synced(state, pred_metrics, basis_changed)

    pred_metrics["fresh_supervision"] = fresh_metrics

    timings["predictor"] = timer.lap()
    used, full = batch_token_costs(prompt_lens, prefixes, finished, z, max_new)
    ratio = 1.0 if full <= 0.0 else used / full
    actual_response = [
        float(rec.response_tokens) for rec in records if float(rec.z) >= 1.0
    ]
    mean_actual_response_tokens = float(np.mean(actual_response)) if actual_response else 0.0
    state.history_costs.append(ratio)
    state.step += 1
    from grace_gc.trainer.methods import start_group_size

    group = start_group_size(state.spec, int(cfg.get("n_prompts", 4)), state.n_ref or n)
    next_n = (state.n_ref or n) if start_count_mode(cfg) == "fixed" else next_start_count(
        state.history_costs, state.n_ref or n, c_full=1.0, group=group)
    prefix_done = np.asarray(finished, dtype=bool).reshape(-1)
    n_continued = int(
        sum(
            1
            for i in range(n)
            if full_ids[i] is not None and len(full_ids[i]) > len(prefixes[i])
        )
    )
    return {
        "records": records,
        "behavior_context": behavior_context,
        "logprob_probe": logprob_probe,
        "prescan_records": engines.last_prescan or [],
        "supervision_records": fresh_rows,
        "n_problems": len(set(problem_ids)),
        "n": n,
        "n_completed": int(z.sum()),
        "n_stopped": int((z < 1.0).sum()),
        "n_prefix_finished": int(prefix_done.sum()),
        "n_eligible": int(((~prefix_done) & (remainings > 0)).sum()),
        "n_continued": n_continued,
        "mean_suffix_tokens": float(
            np.mean(
                [
                    0 if full_ids[i] is None else max(len(full_ids[i]) - len(prefixes[i]), 0)
                    for i in range(n)
                ]
            )
        ),
        "p": p,
        "z": z,
        "deviation": deviation,
        "loss": float(loss.detach()) if hasattr(loss, "detach") else float(loss),
        "n_audited": audited,
        "audit_gradient_source": "mandatory_actor_backward",
        "clip_triggered": clipped,
        "next_n": next_n,
        "texts": texts,
        "predictor": pred_metrics,
        "predictor_frozen": frozen,
        "warmup": warmup,
        "n_prescan": n_prescanned,
        "token_cost_used": float(used),
        "token_cost_full": float(full),
        "token_cost_proxy_used": float(used),
        "token_cost_proxy_full": float(full),
        "token_cost_proxy_ratio": float(ratio),
        "mean_actual_response_tokens": mean_actual_response_tokens,
        "allocation_ready": bool((not warmup) and basis_ready and state.spec.use_allocation),
        "allocation_uniform_shrink": float(alloc.get("uniform_shrink", 0.0)),
        "control_variate_enabled": control_variate_enabled,
        "control_variate_used": bool(state.spec.use_ht_correction and state.spec.use_predictor and not warmup and control_variate_enabled),
        "baseline_configuration": state.baseline.configuration(),
        "basis_id_at_allocate": basis_at_allocate,
        "predictor_synced_basis_id": int(state.predictor_synced_basis_id),
        "basis_rank": basis_rank,
        "timings": timings,
        "timing_details": {"wall_seconds": timing_details,
                           "scope": "nested in prefix/continue phases; not additional ledger charges"},
        "rollout_execution": rollout_execution or None,
        "u_frozen": u_frozen,
        "grad_norm": update_stats.get("grad_norm"),
        "grad_norm_preclip": update_stats.get("grad_norm_preclip"),
        "parameter_update_norm": update_stats.get("parameter_update_norm"),
        "update_ascent_cosine": update_stats.get("update_ascent_cosine"),
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
