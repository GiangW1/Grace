"""Generate same-prefix independent continuations, then summarize."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from grace_gc.audit.prefix_audit import PrefixBundle, audit_bundles, bundle_to_dict, jl_project
from grace_gc.core.layout import collect_lora_layout, pack_grads
from grace_gc.data.format_prompt import apply_solve_instruction
from grace_gc.data.math_data import MathRecord, select_records, selection_manifest
from grace_gc.data.reward import REWARD_PROTOCOL_VERSION, answer_already_emitted, extract_answer, rule_reward
from grace_gc.logging_util.forensics import persist_load_report, write_failed
from grace_gc.logging_util.ledger import ComputeLedger, Timer
from grace_gc.logging_util.run_dir import RunDirectory, resolve_run_dir, utc_now
from grace_gc.logging_util.run_log import RunLog
from grace_gc.versions import collect_environment
from grace_gc.trainer.algorithm import _length_truncated, _traj_natural_finish
from grace_gc.trainer.checkpoint import load_checkpoint
from grace_gc.trainer.baseline import baseline_from_config
from grace_gc.trainer.grace_step import allocation_ready_from_checkpoint, control_variate_coordinates, incremental_token_costs
from grace_gc.trainer.state_io import load_numpy_module_state


def _prefix_finish_reasons(engines) -> list:
    roll = getattr(engines, "last_rollout", None) or {}
    return list(roll.get("prefix_finish_reasons") or [])


def _continue_finish_reason(engines, index: int = 0):
    """Read the finish_reason written by the continue that just ran."""
    roll = getattr(engines, "last_rollout", None) or {}
    cont = roll.get("continue_finish_reasons") or {}
    if index in cont:
        return cont[index]
    return cont.get(str(index))


def _request_seeds(engines, n: int) -> list:
    sampling = (getattr(engines, "last_rollout", None) or {}).get("sampling") or {}
    seeds = list(sampling.get("request_seeds") or [])
    return seeds if len(seeds) == n else [None] * n


def _independent_pass_rate(engines, rec: MathRecord, encode_fn, n_base: int, max_new: int, rng, eos_id, samples=None) -> float:
    """b(x) uses n_base independent full answers, never the audit suffixes."""
    if n_base <= 0:
        return 0.5
    prompts, _pids, _golds = encode_fn(rec, int(n_base))
    fulls, finished = engines.generate_prefix(prompts, int(max_new), rng, "eval")
    finished = np.asarray(finished, dtype=bool).reshape(-1)
    if len(fulls) != len(prompts) or finished.shape[0] != len(prompts):
        raise ValueError(f"audit pass-rate generate_prefix returned {len(fulls)} for {len(prompts)} samples")
    prefix_frs = _prefix_finish_reasons(engines)
    seeds = _request_seeds(engines, len(fulls))
    rewards = []
    for i, full in enumerate(fulls):
        resp = full[len(prompts[i]) :]
        text = engines.decode(resp) if engines.decode else ""
        fr = prefix_frs[i] if i < len(prefix_frs) else None
        truncated = _length_truncated(
            full, len(prompts[i]), int(max_new), bool(finished[i]), eos_id, finish_reason=fr
        )
        scored = rule_reward(text, rec.answer, truncated=truncated)
        rewards.append(0.0 if scored is None else float(scored))
        if samples is not None:
            samples.append({"text": text, "gold": rec.answer, "reward": rewards[-1],
                            "extracted": extract_answer(text), "token_ids": [int(x) for x in full],
                            "prompt_len": len(prompts[i]), "seed": seeds[i],
                            "rng_stream": "eval", "truncated": truncated,
                            "finish_reason": fr, "natural_finish": bool(finished[i])})
    return float(np.mean(rewards)) if rewards else 0.5


def _policy_grad_vec(engines, layout, token_ids, prompt_len: int, reward: float, baseline: float) -> np.ndarray:
    torch = __import__("torch")
    lp = engines.logprob_one(token_ids, int(prompt_len))
    params = engines.trainable_params()
    named = engines.named_lora()
    grads = torch.autograd.grad(lp, params, allow_unused=True)
    packed = []
    for (name, param), gi in zip(named, grads):
        packed.append(
            (
                name,
                np.zeros(tuple(param.shape), dtype=np.float64)
                if gi is None
                else gi.detach().float().cpu().numpy(),
            )
        )
    return pack_grads(packed, layout) * (float(reward) - float(baseline))


def _audit_method_spec(cfg: dict[str, Any] | None, payload=None):
    from grace_gc.trainer.methods import method_spec

    name = "grace" if not cfg else cfg.get("method", "grace")
    ckpt = None if not cfg else (cfg.get("checkpoint") or cfg.get("resume"))
    if ckpt:
        payload = load_checkpoint(ckpt) if payload is None else payload
        name = payload.get("spec") or name
    return method_spec(str(name))


def _predictor_audit_features(feat_bundle: dict, spec) -> Any:
    if spec is not None and spec.feature_mode == "prompt":
        if "prompt_features" not in feat_bundle:
            raise ValueError("Prompt-CV audit requires prompt_features")
        return feat_bundle["prompt_features"]
    return feat_bundle["features"]


def _decision_list(decision: int | list[int]) -> list[int]:
    if isinstance(decision, (list, tuple, np.ndarray)):
        out = [int(t) for t in decision]
    else:
        out = [int(decision)]
    if not out or any(t < 0 for t in out):
        raise ValueError("decision tokens must be non-negative")
    return out


def _prefix_at_t(long_ids, prompt_len: int, t: int, finished_long: bool, eos_id: int | list[int] | None):
    """Slice a longer rollout so decision_grid t values share one path."""
    gen = list(long_ids[int(prompt_len) :])
    cut = gen[: int(t)]
    prefix = list(long_ids[: int(prompt_len)]) + cut
    if finished_long and len(gen) <= int(t):
        return list(long_ids), True
    from grace_gc.data.tokenize import is_stop_token

    if eos_id is not None and cut and is_stop_token(cut[-1], eos_id):
        return prefix, True
    if len(cut) < int(t):
        return prefix, bool(finished_long)
    return prefix, False


def _keep_prefix_at_t(prefix, prompt_len: int, t: int, finished: bool) -> bool:
    """Keep length-t survivors. A finish before t is not a prefix of length t."""
    actual = max(len(prefix) - int(prompt_len), 0)
    return not (finished and actual < int(t))


def _bundles_from_engines(
    records: list[MathRecord],
    engines,
    layout,
    n_prefixes: int,
    n_cont: int,
    decision: int | list[int],
    max_new: int,
    seed: int,
    encode_fn,
    baseline_fn=None,
    predictor=None,
    u=None,
    spec=None,
    jl_dim: int | None = None,
    jl_seed: int = 0,
    n_baseline: int | None = None,
    control_variate: bool = True,
    baseline_policy=None,
) -> list[PrefixBundle]:
    from grace_gc.core.rng import IsolatedRNG

    rng = IsolatedRNG.create(seed)
    decisions = _decision_list(decision)
    tmax = max(decisions)
    if baseline_fn is None:
        baseline_fn = lambda _pid: None
    eos_id = getattr(engines, "eos_id", None)
    bundles = []
    _bundles_from_engines.last = []
    gram = None if u is None else np.asarray(u).T @ np.asarray(u)
    gram_inverse = None if gram is None else np.linalg.pinv(gram)
    for record_index, rec in enumerate(records, start=1):
        print(
            f"phase=audit_problem_begin problem={record_index}/{len(records)} "
            f"problem_id={rec.problem_id} n_bundles={len(bundles)}",
            flush=True,
        )
        baseline_samples = []
        baseline_rng_state = rng.streams["eval"].bit_generator.state
        explicit = None if baseline_fn is None else baseline_fn(rec.problem_id)
        if explicit is not None:
            b_grad, baseline_source = float(explicit), "explicit_audit_baseline"
        elif baseline_policy is not None and not baseline_policy.uses_prescan:
            b_grad, baseline_source = baseline_policy.get(rec.problem_id), "configured_fixed_baseline"
        else:
            b_grad = _independent_pass_rate(engines, rec, encode_fn, n_cont if n_baseline is None else n_baseline,
                                            max_new, rng, eos_id, samples=baseline_samples)
            if baseline_policy is not None:
                b_grad = baseline_policy.prescan_estimate([row["reward"] for row in baseline_samples])
            baseline_source = "independent_prescan"
        b_feat = b_grad
        prompts, _pids, _golds = encode_fn(rec, n_prefixes)
        encode_meta = getattr(encode_fn, "last_meta", None) or []
        prompt_truncated = (
            None
            if not encode_meta
            else any(bool(m.get("prompt_truncated")) for m in encode_meta if isinstance(m, dict))
        )
        prefix_rng_state = rng.streams["token"].bit_generator.state
        longs, finished_long = engines.generate_prefix(prompts, tmax, rng, "token")
        prefix_seeds = _request_seeds(engines, len(longs))
        finished_long = np.asarray(finished_long, dtype=bool).reshape(-1)
        if len(longs) != len(prompts) or finished_long.shape[0] != len(prompts):
            raise ValueError(f"audit generate_prefix returned {len(longs)} for {len(prompts)} prefixes")
        prompt_lens = [len(p) for p in prompts]
        _bundles_from_engines.last.append({
            "problem_id": rec.problem_id, "prompt": rec.prompt, "gold": rec.answer,
            "baseline": b_grad, "baseline_feature_value": b_feat, "baseline_samples": baseline_samples,
            "baseline_source": baseline_source,
            "baseline_configuration": None if baseline_policy is None else baseline_policy.configuration(),
            "baseline_rng_state": baseline_rng_state, "prefix_rng_state": prefix_rng_state,
            "paths": [{"path_id": f"{rec.problem_id}:{i}", "token_ids": [int(x) for x in seq],
                       "prompt_len": prompt_lens[i], "natural_finish": bool(finished_long[i]),
                       "seed": prefix_seeds[i]} for i, seq in enumerate(longs)],
            "decision_grid": decisions, "max_new_tokens": max_new,
        })
        for t in decisions:
            prefixes = []
            finished = []
            prompt_lens_t = []
            path_idx = []
            for idx, long in enumerate(longs):
                pref, fin = _prefix_at_t(long, prompt_lens[idx], t, bool(finished_long[idx]), eos_id)
                if not _keep_prefix_at_t(pref, prompt_lens[idx], t, fin):
                    continue
                prefixes.append(pref)
                finished.append(fin)
                prompt_lens_t.append(prompt_lens[idx])
                path_idx.append(idx)
            if not prefixes:
                continue
            m_list: list[np.ndarray | None] = [None] * len(prefixes)
            r_list: list[float | None] = [None] * len(prefixes)
            c_list: list[float | None] = [None] * len(prefixes)
            f_list: list[np.ndarray | None] = [None] * len(prefixes)
            effective_f_list: list[np.ndarray | None] = [None] * len(prefixes)
            if predictor is not None and u is not None:
                feat_bundle = engines.prefix_features(
                    prefixes, prompt_lens_t, [b_feat for _ in prefixes]
                )
                feat_in = _predictor_audit_features(feat_bundle, spec)
                pred = predictor.forward_numpy(feat_in, feat_bundle.get("cost_feat"))
                effective_f = control_variate_coordinates(pred.f, enabled=control_variate).copy()
                effective_f[np.asarray(finished, dtype=bool)] = 0.
                mapped = effective_f @ np.asarray(u, dtype=np.float64).T
                if mapped.shape[1] != layout.dim:
                    raise ValueError("checkpoint predictor maps to a different G dimension than the audit actor")
                m_list = [mapped[i] for i in range(len(prefixes))]
                if spec is not None and spec.risk_mode == "reward":
                    from grace_gc.trainer.algorithm import _reward_features

                    q_hat = predictor.forward_success(feat_in)
                    reward_feats = np.stack(
                        [
                            _reward_features(
                                float(q_hat[i]),
                                float(max(len(prefixes[i]) - int(prompt_lens_t[i]), 1)),
                                b_feat,
                            )
                            for i in range(len(prefixes))
                        ]
                    )
                    r_hat = predictor.forward_reward_risk(reward_feats)
                    r_list = [float(r_hat[i]) for i in range(len(prefixes))]
                else:
                    r_list = [float(pred.r_hat[i]) for i in range(len(prefixes))]
                remain_c = [
                    max(0, int(max_new) - (len(prefixes[i]) - int(prompt_lens_t[i])))
                    for i in range(len(prefixes))
                ]
                if getattr(predictor, "constant_cost", True):
                    c_list = [float(x) for x in incremental_token_costs(remain_c, finished)]
                else:
                    c_list = [float(pred.c_hat[i]) for i in range(len(prefixes))]
                f_list = [np.asarray(pred.f[i], dtype=np.float64) for i in range(len(prefixes))]
                effective_f_list = [np.asarray(effective_f[i], dtype=np.float64) for i in range(len(prefixes))]
            for loc, prefix in enumerate(prefixes):
                prompt_len = prompt_lens_t[loc]
                idx = path_idx[loc]
                rem = max(0, int(max_new) - (len(prefix) - prompt_len))
                if finished[loc] or rem <= 0:
                    # One observation: cloning n_cont identical rows would overweight this prefix.
                    # Do not reuse a previous continue's finish_reason.
                    fulls = [prefix]
                    finish_reasons = [None]
                    suffix_seeds = [None]
                else:
                    fulls = []
                    finish_reasons = []
                    suffix_seeds = []
                    for _ in range(n_cont):
                        one = engines.continue_selected([prefix], np.ones(1, dtype=bool), rem, rng)
                        if one[0] is None:
                            raise ValueError("continuation returned no sequence for a selected audit prefix")
                        fulls.append(one[0])
                        finish_reasons.append(_continue_finish_reason(engines, 0))
                        suffix_seeds.append(_request_seeds(engines, 1)[0])
                rewards = []
                grads = []
                costs = []
                suffix_texts = []
                continuation_records = []
                for full, cont_fr, suffix_seed in zip(fulls, finish_reasons, suffix_seeds):
                    if full is None:
                        raise ValueError("audit continuation produced an empty sequence")
                    resp = full[prompt_len:]
                    text = engines.decode(resp) if engines.decode else ""
                    gen_cont = max(len(full) - len(prefix), 0)
                    traj_fin = _traj_natural_finish(
                        bool(finished[loc]),
                        full,
                        eos_id,
                        generated=None if finished[loc] else gen_cont,
                        requested=None if finished[loc] else rem,
                        finish_reason=cont_fr,
                    )
                    truncated = _length_truncated(
                        full,
                        prompt_len,
                        int(max_new),
                        traj_fin,
                        eos_id,
                        finish_reason=cont_fr,
                    )
                    r = rule_reward(text, rec.answer, truncated=truncated)
                    reward = 0.0 if r is None else float(r)
                    rewards.append(reward)
                    suffix_texts.append(text)
                    continuation_records.append({"text": text, "gold": rec.answer,
                        "token_ids": [int(x) for x in full], "prompt_len": prompt_len,
                        "seed": suffix_seed, "rng_stream": "continuation", "reward": reward,
                        "extracted": extract_answer(text), "truncated": truncated,
                        "natural_finish": traj_fin, "finish_reason": cont_fr,
                        "generated_suffix_tokens": gen_cont, "baseline": b_grad})
                    grads.append(
                        _policy_grad_vec(engines, layout, full, prompt_len, reward, b_grad)
                    )
                    costs.append(float(gen_cont))
                prefix_text = engines.decode(prefix[prompt_len:]) if engines.decode else ""
                grads_arr = np.stack(grads, axis=0)
                full_norm = np.sum(grads_arr * grads_arr, axis=1).tolist()
                coord_array = None if u is None else grads_arr @ u
                full_coords = None if coord_array is None else coord_array.tolist()
                full_ortho = None if coord_array is None else float(np.mean(np.maximum(
                    np.asarray(full_norm) - np.sum((coord_array @ gram_inverse) * coord_array, axis=1), 0.0)))
                original_dim = int(grads_arr.shape[1])
                m_pred = None if m_list[loc] is None else np.asarray(m_list[loc], dtype=np.float64)
                if jl_dim and grads_arr.shape[1] > int(jl_dim):
                    grads_arr = jl_project(grads_arr, int(jl_dim), int(jl_seed))
                    if m_pred is not None:
                        m_pred = jl_project(m_pred, int(jl_dim), int(jl_seed))
                bundles.append(
                    PrefixBundle(
                        rec.problem_id,
                        t,
                        np.asarray(rewards),
                        grads_arr,
                        path_id=f"{rec.problem_id}:{idx}",
                        suffix_cost=np.asarray(costs, dtype=np.float64),
                        m_pred=m_pred,
                        coords=f_list[loc],
                        r_hat=r_list[loc],
                        c_hat=c_list[loc],
                        finished=bool(finished[loc]),
                        prefix_tokens=max(len(prefix) - prompt_len, 0),
                        answer_emitted=answer_already_emitted(prefix_text),
                        prefix_text=prefix_text,
                        suffix_texts=suffix_texts,
                        prompt_truncated=prompt_truncated,
                        gold=rec.answer, baseline=b_grad, baseline_samples=baseline_samples,
                        prompt_token_ids=[int(x) for x in prefix[:prompt_len]],
                        prefix_token_ids=[int(x) for x in prefix],
                        continuation_records=continuation_records, prefix_request_seed=prefix_seeds[idx],
                        true_grad_norm_sq=full_norm, true_grad_coords=full_coords,
                        full_orthogonal_energy=full_ortho,
                        basis_gram=None if gram is None else gram.tolist(),
                        effective_coords=None if effective_f_list[loc] is None else effective_f_list[loc].tolist(),
                        control_variate_enabled=control_variate,
                        projection={"version": 2, "original_dim": original_dim,
                                    "stored_dim": int(grads_arr.shape[1]), "seed": int(jl_seed),
                                    "normalization": "norm_preserving_in_expectation"},
                    )
                )
                print(
                    f"phase=audit_prefix_done problem={record_index}/{len(records)} "
                    f"t={t} path={idx} n_bundles={len(bundles)}",
                    flush=True,
                )
    return bundles


def _baseline_fn(cfg: dict[str, Any] | None):
    """Only an explicit audit.baseline is a study override. Holdout problems use the independent pass rate."""
    if not cfg or (cfg.get("audit") or {}).get("baseline") is None:
        return lambda _pid: None

    default = float((cfg.get("audit") or {}).get("baseline"))

    def _fn(pid: str) -> float:
        _ = pid
        return default

    return _fn


def _audit_u(bundles: list[PrefixBundle], cfg: dict[str, Any]) -> np.ndarray:
    # No saved basis means unavailable, not an invented coordinate subspace.
    u = np.zeros((bundles[0].grads.shape[1], 0))
    ckpt = cfg.get("checkpoint") or cfg.get("resume")
    if not ckpt:
        return u
    payload = load_checkpoint(ckpt)
    stored = (payload.get("basis") or {}).get("u")
    if stored is None:
        return u
    arr = np.asarray(stored, dtype=np.float64)
    gdim = int(bundles[0].grads.shape[1])
    if arr.ndim != 2:
        raise ValueError("checkpoint basis U does not match audit G dimension")
    if arr.shape[0] == gdim:
        return arr
    jl_dim = int((cfg.get("audit") or {}).get("jl_dim") or 0)
    seed = int((cfg.get("audit") or {}).get("jl_seed", cfg.get("seed", 0)))
    if jl_dim and gdim == jl_dim and arr.shape[0] > gdim:
        return jl_project(arr.T, gdim, seed).T
    raise ValueError("checkpoint basis U does not match audit G dimension")


def _tiny_actor(cfg: dict[str, Any] | None = None, payload=None):
    from grace_gc.core.rng import seed_all
    from grace_gc.trainer.cpu_tiny import TinyLoRAActor

    seed_all(int((cfg or {}).get("seed", 17)))
    actor = TinyLoRAActor()
    ckpt = None if cfg is None else (cfg.get("checkpoint") or cfg.get("resume"))
    if ckpt:
        payload = load_checkpoint(ckpt) if payload is None else payload
        if not payload.get("actor_full"):
            raise ValueError("tiny audit needs actor_full; this checkpoint is LoRA-only")
        load_numpy_module_state(actor.named_all_params(), payload["actor_full"])
    return actor


def _frozen_predictor(cfg: dict[str, Any] | None, payload=None):
    if not cfg:
        return None, None
    ckpt = cfg.get("checkpoint") or cfg.get("resume")
    if not ckpt:
        return None, None
    payload = load_checkpoint(ckpt) if payload is None else payload
    raw = payload.get("predictor")
    stored = (payload.get("basis") or {}).get("u")
    if not raw or stored is None:
        return None, None
    from grace_gc.predictor.heads import predictor_from_spec

    heads = predictor_from_spec(int(raw.get("in_dim")), int(raw.get("k")), raw)
    heads.load_state_dict(raw)
    return heads, np.asarray(stored, dtype=np.float64)


def generate_bundles_tiny(
    records: list[MathRecord],
    n_prefixes: int,
    n_cont: int,
    decision: int | list[int],
    max_new: int,
    seed: int,
    cfg: dict[str, Any] | None = None,
    actor=None,
) -> list[PrefixBundle]:
    from grace_gc.data.tokenize import encode_records_tiny
    from grace_gc.trainer.tiny_engine import make_tiny_engines

    actor = actor or _tiny_actor(cfg)
    layout = collect_lora_layout(actor.named_lora_params())
    engines = make_tiny_engines(actor, actor.vocab)

    prompt_max = 1024 if cfg is None else int(cfg.get("prompt_max_tokens", 1024))

    def encode(rec: MathRecord, n: int):
        meta: list = []
        out = encode_records_tiny([rec] * n, actor.vocab, prompt_max, prompt_meta=meta)
        encode.last_meta = meta
        return out

    predictor, u = _frozen_predictor(cfg)
    spec = _audit_method_spec(cfg)
    if spec is not None and not spec.use_predictor:
        predictor, u = None, None
    audit_cfg = {} if cfg is None else (cfg.get("audit") or {})
    return _bundles_from_engines(
        records,
        engines,
        layout,
        n_prefixes,
        n_cont,
        decision,
        max_new,
        seed,
        encode,
        baseline_fn=_baseline_fn(cfg),
        predictor=predictor,
        u=u,
        spec=spec,
        jl_dim=int(audit_cfg["jl_dim"]) if audit_cfg.get("jl_dim") else None,
        jl_seed=int(audit_cfg.get("jl_seed", seed)),
        n_baseline=audit_cfg.get("n_baseline"),
        control_variate=bool(((cfg or {}).get("predictor") or {}).get("control_variate", True)),
        baseline_policy=baseline_from_config((cfg or {}).get("baseline")),
    )


def generate_bundles_gpu(
    records: list[MathRecord],
    n_prefixes: int,
    n_cont: int,
    decision: int | list[int],
    max_new: int,
    seed: int,
    cfg: dict[str, Any],
    work_dir: str | Path | None = None,
    engines=None,
    layout=None,
    cache: dict | None = None,
    payload=None,
) -> list[PrefixBundle]:
    if cache and engines is None and cache.get("engines") is not None:
        engines = cache["engines"]
        layout = cache["layout"]
    if engines is None or layout is None:
        model_path = cfg.get("model_path")
        if not model_path:
            raise ValueError("model_path is required for GPU prefix-audit generation")
        path = Path(str(model_path))
        if path.is_absolute() and not path.exists():
            raise FileNotFoundError(f"model path is not readable: {model_path}")

        from grace_gc.backends.gpu_engine import make_gpu_engines
        from grace_gc.backends.hf_actor import named_lora_params
        from grace_gc.backends.verl_trainer import build_vllm_engine, load_lora_actor, vllm_needed_max_model_len
        from grace_gc.data.reward import require_math_verify
        from grace_gc.data.tokenize import encode_records_hf, load_hf_tokenizer, tokenizer_inventory
        from grace_gc.trainer.state_io import check_snapshot_identity

        ckpt = cfg.get("checkpoint") or cfg.get("resume")
        if not ckpt:
            raise ValueError("GPU prefix audit needs a training checkpoint")
        require_math_verify()
        tokenizer = load_hf_tokenizer(str(model_path))
        if work_dir is not None:
            from grace_gc.logging_util.run_dir import RunDirectory

            RunDirectory(Path(work_dir).parent).write_json("tokenizer.json", tokenizer_inventory(tokenizer))
        actor = load_lora_actor(str(model_path), cfg.get("lora", {}))
        payload = load_checkpoint(ckpt) if payload is None else payload
        check_snapshot_identity(payload, cfg)
        load_numpy_module_state(named_lora_params(actor), payload.get("actor") or {})
        if work_dir is not None:
            from grace_gc.versions import sha256_named

            RunDirectory(Path(work_dir).parent).write_json("actor_source.json", {
                "checkpoint": str(ckpt), "checkpoint_step": payload.get("step"),
                "method": payload.get("spec"), "actor_sha256": sha256_named(named_lora_params(actor)),
                "hash_stage": "loaded_checkpoint_before_audit", "layout": "all_qv_lora_A_B",
            })
        vllm_cfg = dict(cfg.get("vllm") or {})
        vllm_cfg.setdefault("seed", int(cfg.get("seed", 17)))
        vllm_cfg["max_model_len"] = vllm_needed_max_model_len(cfg, _audit_max_new(cfg))
        llm = build_vllm_engine(str(model_path), vllm_cfg, int(cfg.get("lora", {}).get("rank", 16)))
        if work_dir is not None and getattr(build_vllm_engine, "last", None):
            from grace_gc.logging_util.run_dir import RunDirectory

            RunDirectory(Path(work_dir).parent).write_json("vllm_engine.json", build_vllm_engine.last)
        adapter = Path(work_dir or "runs/audit-lora")
        engines, extra = make_gpu_engines(actor, llm, tokenizer, cfg, adapter)
        extra["sync"]()
        layout = collect_lora_layout(named_lora_params(actor))
        max_prompt = int(cfg.get("prompt_max_tokens", 1024))

        def encode(rec: MathRecord, n: int):
            meta: list = []
            out = encode_records_hf([rec] * n, tokenizer, max_prompt, prompt_meta=meta)
            encode.last_meta = meta
            return out

        encode_fn = encode
        if cache is not None:
            cache["engines"] = engines
            cache["layout"] = layout
            cache["encode_fn"] = encode_fn
    else:
        from grace_gc.data.tokenize import encode_records_hf, load_hf_tokenizer

        tokenizer = load_hf_tokenizer(str(cfg["model_path"]))
        max_prompt = int(cfg.get("prompt_max_tokens", 1024))

        def encode_fn(rec: MathRecord, n: int):
            meta: list = []
            out = encode_records_hf([rec] * n, tokenizer, max_prompt, prompt_meta=meta)
            encode_fn.last_meta = meta
            return out

    # A caller may initialize this frozen engine once for another audit design.
    if not records:
        return []
    predictor, u = _frozen_predictor(cfg)
    spec = _audit_method_spec(cfg)
    if spec is not None and not spec.use_predictor:
        predictor, u = None, None
    audit_cfg = cfg.get("audit") or {}
    return _bundles_from_engines(
        records,
        engines,
        layout,
        n_prefixes,
        n_cont,
        decision,
        max_new,
        seed,
        encode_fn,
        baseline_fn=_baseline_fn(cfg),
        predictor=predictor,
        u=u,
        spec=spec,
        jl_dim=int(audit_cfg["jl_dim"]) if audit_cfg.get("jl_dim") else None,
        jl_seed=int(audit_cfg.get("jl_seed", seed)),
        n_baseline=audit_cfg.get("n_baseline"),
        control_variate=bool((cfg.get("predictor") or {}).get("control_variate", True)),
        baseline_policy=baseline_from_config(cfg.get("baseline")),
    )


def _audit_max_new(cfg: dict[str, Any]) -> int:
    """Audit continuation length. GRPO-short's train 1024 must not bind this."""
    audit = cfg.get("audit") or {}
    if audit.get("max_new_tokens") is not None:
        return int(audit["max_new_tokens"])
    ev = cfg.get("eval") or {}
    if ev.get("max_new_tokens") is not None:
        return int(ev["max_new_tokens"])
    if cfg.get("audit_max_new_tokens") is not None:
        return int(cfg["audit_max_new_tokens"])
    if str(cfg.get("backend", "cpu_tiny")) == "gpu_verl":
        return 4096
    return int(cfg.get("max_new_tokens", 8))


def run_audit(records: list[MathRecord], cfg: dict[str, Any], run_dir: str | Path) -> dict:
    timer = Timer()
    started = utc_now()
    cfg = dict(cfg)
    requested = Path(run_dir)
    run_dir = resolve_run_dir(run_dir)
    run = RunDirectory(run_dir)
    backend = cfg.get("backend", "cpu_tiny")
    n_gpu = int((cfg.get("hardware") or {}).get("n_gpu", 1 if backend == "gpu_verl" else 0))
    hardware = str((cfg.get("hardware") or {}).get("name", "gpu" if backend == "gpu_verl" else "cpu"))
    ledger = ComputeLedger(n_gpu=n_gpu, hardware=hardware)
    versions, status = {}, "failed"
    try:
        spec = _audit_method_spec(cfg)
        cfg["method"] = spec.name
        n_pref = int(cfg.get("audit", {}).get("n_prefixes", cfg.get("n_prefixes", 2)))
        n_cont = int(cfg.get("audit", {}).get("n_continuations", cfg.get("n_cont", 16)))
        decision = int(cfg.get("decision_tokens", 4))
        max_new = _audit_max_new(cfg)
        records = apply_solve_instruction(list(records))
        seed = int(cfg.get("seed", 17))
        raw_grid = list(cfg.get("audit", {}).get("decision_grid") or [decision])
        grid = [int(t) for t in raw_grid]
        if any(t < 0 for t in grid):
            raise ValueError("decision tokens must be non-negative")
        dropped = [t for t in grid if t > max_new]
        kept = [t for t in grid if t <= max_new]
        if not kept:
            raise ValueError(f"audit decision exceeds max_new_tokens {max_new}")
        grid = kept
        n_problems = int(cfg.get("audit", {}).get("n_problems", 0) or 0)
        audit_cfg = cfg.get("audit") or {}
        selection = str(audit_cfg.get("selection", "first"))
        selection_seed = int(audit_cfg.get("selection_seed", cfg.get("split_seed", seed)))
        recs = select_records(records, n_problems, selection, selection_seed)
        versions = collect_environment(cfg)
        versions["started"] = started
        run.write_run_meta(kind="audit", started=started, requested=requested)
        run.write_yaml("config.yaml", cfg)
        run.write_json("environment.json", versions)
        manifest = selection_manifest(recs, selection, selection_seed)
        manifest.update(n_baseline=int(audit_cfg.get("n_baseline", n_cont)), n_continuations=n_cont,
                        max_new_tokens=max_new, decision_grid=grid, method=spec.name,
                        reward_protocol_version=REWARD_PROTOCOL_VERSION,
                        baseline_configuration=baseline_from_config(cfg.get("baseline")).configuration(),
                        baseline_protocol="Independent prescan with configured prior; explicit audit.baseline or configured fixed mode overrides both gradient and feature baseline.")
        run.write_json("audit_manifest.json", manifest)
        persist_load_report(run, data_path=cfg.get("data_path"))
        with RunLog(run.root / "run.log"):
            print(f"audit start {started} backend={backend} n_problems={len(recs)} grid={grid} dir={run.root}")
            if Path(run.root).resolve() != requested.resolve():
                print(f"run-dir {requested} already had artifacts; writing to {run.root}")
            if backend == "gpu_verl":
                bundles = generate_bundles_gpu(
                    recs,
                    n_pref,
                    n_cont,
                    grid,
                    max_new,
                    seed,
                    cfg,
                    work_dir=Path(run.root) / "audit_lora",
                )
            else:
                bundles = generate_bundles_tiny(recs, n_pref, n_cont, grid, max_new, seed, cfg=cfg)
            run.write_jsonl("audit_raw_problems.jsonl", getattr(_bundles_from_engines, "last", []))
            run.write_jsonl("audit_bundles.jsonl", [bundle_to_dict(b) for b in bundles])
            if not bundles:
                result = {
                    "n_bundles": 0,
                    "note": "no prefixes",
                    "audit_manifest": manifest,
                    "started": started,
                    "run_dir": str(run.root),
                }
                result["finished"] = utc_now()
                run.write_json("audit_summary.json", result)
                status = "completed"
                return result
            u = _audit_u(bundles, cfg)
            rng = np.random.default_rng(seed)
            analysis = dict(cfg.get("analysis", {}))
            alloc = cfg.get("allocation", {})
            analysis.setdefault("beta", alloc.get("beta", 0.5))
            analysis.setdefault("p_min", alloc.get("p_min", 0.2))
            analysis.setdefault("uniform_shrink", alloc.get("uniform_shrink", 0.))
            analysis.setdefault("control_variate", (cfg.get("predictor") or {}).get("control_variate", True))
            analysis["method"] = spec.name
            analysis["max_new_tokens"] = max_new
            ckpt = cfg.get("checkpoint") or cfg.get("resume")
            payload = load_checkpoint(ckpt) if ckpt else {}
            analysis["allocation_ready"] = allocation_ready_from_checkpoint(spec, payload, cfg)
            analysis["warmup"] = int(payload.get("step", 0)) < int((cfg.get("predictor") or {}).get("warmup_steps", 0))
            if cfg.get("audit", {}).get("jl_dim"):
                analysis["jl_dim"] = int(cfg["audit"]["jl_dim"])
            result = audit_bundles(bundles, u, analysis, rng)
            result["seed"] = seed
            result["audit_manifest"] = manifest
            result["checkpoint"] = cfg.get("checkpoint") or cfg.get("resume")
            result["max_new_tokens"] = max_new
            result["decision_grid"] = grid
            if dropped:
                result["decision_grid_dropped"] = dropped
            observed = {int(b.t) for b in bundles}
            unobserved = [t for t in grid if t not in observed]
            if unobserved:
                result["decision_grid_unobserved"] = unobserved
            result["started"] = started
            result["run_dir"] = str(run.root)
            from grace_gc.backends.vllm_two_phase import build_sampling_params

            if getattr(build_sampling_params, "last", None):
                run.write_json("sampling.json", build_sampling_params.last)
            print(f"audit done n_bundles={result.get('n_bundles')}")
        result["finished"] = utc_now()
        run.write_json("audit_summary.json", result)
        status = "completed"
        return result
    except Exception as exc:
        status = "failed"
        write_failed(run, exc, versions, started)
        raise
    finally:
        ledger.add("audit", timer.elapsed(), cpu_s=timer.cpu_elapsed(), status=status,
                   timing_scope="Function entry through setup, computation and result/failure persistence; excludes final ledger publication and caller-side CLI/data loading.")
        run.write_json("compute_ledger.json", ledger.summary())
        run.append_jsonl("compute_ledger.jsonl", ledger.rows[-1])
