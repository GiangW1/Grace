"""Additional current-policy labels, paid separately from the actor sample."""

from __future__ import annotations

import copy

import numpy as np

from grace_gc.core.rng import IsolatedRNG
from grace_gc.predictor.reservoir import ReservoirItem
from grace_gc.trainer.actor_update import audit_one, restore_grads, snapshot_grads


def collect_fresh_supervision(engines, state, prompts, problem_ids, golds, cfg):
    """Sample before the actor update; labels can only train the NEXT predictor.

    A separate per-step RNG is reproducible on resume without consuming any actor
    stream. These are new trajectories, never rescored old-policy trajectories.
    This function does not mutate the baseline, reservoir, heads, or actor weights.
    """
    from grace_gc.trainer.algorithm import (
        _length_truncated, _reward_features, _rollout_finish,
        _rollout_request_seeds, _rollout_token_logprobs, continuation_remainings,
    )

    pcfg = cfg.get("predictor") or {}
    count = int(pcfg.get("fresh_samples_per_problem", 0))
    every = int(pcfg.get("fresh_every", 1))
    if count < 0 or every < 1:
        raise ValueError("fresh_samples_per_problem must be nonnegative and fresh_every positive")
    if not state.spec.use_predictor or not count or (state.step + 1) % every:
        return [], [], {"n": 0, "generated_tokens": 0}
    first = {}
    for i, pid in enumerate(problem_ids):
        first.setdefault(pid, i)
    source = [i for i in first.values() for _ in range(count)]
    fresh_prompts = [list(prompts[i]) for i in source]
    pids = [problem_ids[i] for i in source]
    baselines = [state.baseline.get(pid) for pid in pids]
    seed = int(np.random.SeedSequence([state.rng.seed, 0x47524346, state.step]).generate_state(1)[0])
    rng = IsolatedRNG.create(seed)
    max_new = int(cfg.get("max_new_tokens", 32))
    decision = int(cfg.get("decision_tokens", 16))
    saved_rollout = copy.deepcopy(engines.last_rollout)
    params, named = engines.trainable_params(), engines.named_lora()
    saved_grads = snapshot_grads(params)
    items, rows = [], []
    try:
        prefixes, finished = engines.generate_prefix(fresh_prompts, decision, rng, "token")
        finished = np.asarray(finished, dtype=bool)
        if len(prefixes) != len(source) or finished.shape != (len(source),):
            raise ValueError("fresh supervision prefix count does not match requests")
        lens = np.array([len(p) for p in fresh_prompts])
        bundle = engines.prefix_features(prefixes, lens, baselines)
        feats = bundle["prompt_features"] if state.spec.feature_mode == "prompt" else bundle["features"]
        cost_feats = bundle.get("cost_feat")
        q_hat = (state.predictor.forward_success(feats)
                 if state.spec.risk_mode == "reward" and state.predictor is not None
                 and state.step >= int(pcfg.get("warmup_steps", 0)) else baselines)
        remaining = continuation_remainings(prefixes, lens, finished, max_new)
        fulls = [list(ids) for ids in prefixes]
        for left in sorted(set(remaining[remaining > 0])):
            selected = remaining == left
            part = engines.continue_selected(prefixes, selected, int(left), rng)
            for j in np.flatnonzero(selected):
                if part[j] is None:
                    raise ValueError("fresh supervision continuation returned no sequence")
                fulls[j] = part[j]
        for j, (i, full) in enumerate(zip(source, fulls)):
            if len(full) <= lens[j]:
                raise ValueError("fresh supervision has no response tokens")
            text = engines.decode(full[lens[j]:]) if engines.decode else ""
            reason = _rollout_finish(engines, j, 1.)
            truncated = _length_truncated(full, int(lens[j]), max_new, bool(finished[j]),
                                         engines.eos_id, finish_reason=reason)
            scored = engines.reward_fn(full, golds[i], truncated=truncated, text=text)
            reward = 0. if scored is None else float(scored)
            advantage = reward - baselines[j]
            g = (np.zeros(state.layout.dim) if advantage == 0. else
                 audit_one(advantage * engines.logprob_one(full, int(lens[j])), params,
                           [name for name, _ in named], state.layout))
            prefix_n = len(prefixes[j]) - int(lens[j])
            item = ReservoirItem(
                problem_id=pids[j], g=g, p=1., s=1., features=np.asarray(feats[j]),
                reward=reward, basis_id=state.basis_id,
                cost_feat=None if cost_feats is None else np.asarray(cost_feats[j]),
                realized_cost=float(max(len(full) - len(prefixes[j]), 1)),
                reward_feat=_reward_features(float(q_hat[j]), float(max(prefix_n, 1)), baselines[j]),
                observed_step=state.step + 1, remaining_cost=float(max(remaining[j], 1)),
                prefix_finished=bool(finished[j]), source="fresh_policy",
            )
            items.append(item)
            rows.append({
                "problem_id": pids[j], "source": "fresh_policy", "used_in_actor_update": False,
                "observation_step": state.step + 1, "rng_seed": seed,
                "prompt_token_ids": fresh_prompts[j], "prefix_token_ids": list(prefixes[j]),
                "full_token_ids": list(full), "gold": golds[i], "text": text,
                "reward": reward, "baseline_b": baselines[j], "q_hat": float(q_hat[j]), "advantage": advantage,
                "finish_reason": reason, "truncated": bool(truncated),
                "request_seeds": _rollout_request_seeds(engines, j),
                "token_logprobs": _rollout_token_logprobs(engines, j, 1.),
                "response_tokens": len(full) - int(lens[j]), "gradient_norm_sq": float(g @ g),
            })
    finally:
        engines.last_rollout = saved_rollout
        restore_grads(params, saved_grads)
    return items, rows, {
        "n": len(items), "n_problems": len(first), "rng_seed": seed,
        "generated_tokens": sum(row["response_tokens"] for row in rows),
        "scope": "additional full current-policy trajectories; all generation and gradients paid",
        "used_in_actor_update": False,
    }
