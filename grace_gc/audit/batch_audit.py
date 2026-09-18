"""Fixed-prompt, fixed-N full-space audit. Every suffix is actually generated.

Selection is simulated independently of suffixes; its hypothetical token cost
is never substituted for the measured cost of this diagnostic experiment.
"""
from __future__ import annotations

import copy

import numpy as np
from grace_gc.audit.mechanism import paired_geometry, summarize_geometries

from grace_gc.audit.run import (
    _audit_method_spec, _continue_finish_reason, _frozen_predictor,
    _independent_pass_rate, _policy_grad_vec, _prefix_finish_reasons,
    _predictor_audit_features, _request_seeds, _tiny_actor, generate_bundles_gpu,
)
from grace_gc.core.estimator import ht_estimate, optimizer_grad_from_ghat
from grace_gc.core.layout import collect_lora_layout, pack_grads, unpack_to_dict
from grace_gc.core.rng import IsolatedRNG
from grace_gc.data.format_prompt import apply_solve_instruction
from grace_gc.data.math_data import select_records, selection_manifest
from grace_gc.data.reward import REWARD_PROTOCOL_VERSION, extract_answer, rule_reward
from grace_gc.logging_util.forensics import persist_load_report, write_failed
from grace_gc.logging_util.ledger import ComputeLedger, Timer
from grace_gc.logging_util.run_dir import RunDirectory, resolve_run_dir, utc_now
from grace_gc.trainer.actor_update import apply_correction_clip_step
from grace_gc.trainer.algorithm import _length_truncated, _reward_features, _traj_natural_finish, continuation_remainings
from grace_gc.trainer.checkpoint import load_checkpoint
from grace_gc.trainer.baseline import HistoricalBaseline, baseline_from_config
from grace_gc.trainer.grace_step import control_variate_coordinates, decide_continuation, incremental_token_costs, neyman_ready
from grace_gc.trainer.state_io import load_optimizer_state
from grace_gc.versions import collect_environment, sha256_file, sha256_mapping, sha256_named


def _audit_seed(seed, index):
    """Dedicated domain, rather than restarting a training seed's token stream."""
    return int(np.random.SeedSequence([int(seed), 0xBA7C2026, int(index)]).generate_state(1)[0])


class PairedMoments:
    """O(D) memory, unbiased trace sample covariance across replicate batches."""
    def __init__(self, dim):
        self.n = 0
        self.mean = np.zeros((2, dim), dtype=np.float64)
        self.m2 = np.zeros(2)
        self.diff_m2 = 0.

    def add(self, full, actual):
        self.n += 1
        old_difference = self.mean[1] - self.mean[0]
        for i, value in enumerate((full, actual)):
            delta = value - self.mean[i]
            self.mean[i] += delta / self.n
            self.m2[i] += float(delta @ (value - self.mean[i]))
        difference = actual - full
        self.diff_m2 += float((difference-old_difference) @ (difference-(self.mean[1]-self.mean[0])))

    def summary(self):
        difference = self.mean[1] - self.mean[0]
        result = {"replicates": self.n,
                  "full_mean_norm": float(np.linalg.norm(self.mean[0])),
                  "actual_mean_norm": float(np.linalg.norm(self.mean[1])),
                  "paired_mean_difference_norm": float(np.linalg.norm(difference)),
                  "paired_trace_sample_cov": self.diff_m2/(self.n-1) if self.n > 1 else None}
        for i, name in enumerate(("full", "actual")):
            result[f"{name}_trace_sample_cov"] = float(self.m2[i]/(self.n-1)) if self.n > 1 else None
        result["note"] = "Conditional on the fixed prompts, frozen baseline and checkpoint; replicate batches are the units. No training-seed or task-population CI."
        return result


class OptimizerReplay:
    """Fresh CPU copies for each one-step update; live actor/moments are untouched."""
    def __init__(self, named, layout, snapshot, optim):
        if not snapshot:
            raise ValueError("batch update audit needs the checkpoint optimizer state")
        self.initial = [(name, value.detach().cpu().clone()) for name, value in named]
        self.layout, self.snapshot = layout, copy.deepcopy(snapshot)
        self.clip = float(optim.get("grad_clip", 1.))
        self.kind = "AdamW" if "betas" in snapshot["param_groups"][0] else "SGD"

    def step(self, ascent, *, with_clipped=False):
        import torch
        named = [(name, torch.nn.Parameter(value.clone())) for name, value in self.initial]
        parameters = [value for _, value in named]
        snapshot = copy.deepcopy(self.snapshot)
        # CPU execution flags differ; all numerical hyperparameters and moments
        # are the checkpoint's. This is not bitwise CUDA optimizer validation.
        for group in snapshot["param_groups"]:
            if "capturable" in group: group["capturable"] = False
            if "fused" in group: group["fused"] = False
            if "foreach" in group: group["foreach"] = False
        groups = [{**{k:v for k,v in group.items() if k != "params"},
                   "params": [parameters[int(index)] for index in group["params"]]}
                  for group in snapshot["param_groups"]]
        optimizer = getattr(torch.optim, self.kind)(groups)
        load_optimizer_state(optimizer, snapshot)
        descent = unpack_to_dict(optimizer_grad_from_ghat(ascent), self.layout)
        for name, value in named:
            value.grad = torch.as_tensor(descent[name], dtype=value.dtype).clone()
        stats = {}
        apply_correction_clip_step(named, self.layout, np.zeros((self.layout.dim, 0)), np.zeros((1, 0)),
                                   np.ones(1), np.ones(1), 1, optimizer, self.clip, use_correction=False, stats=stats)
        update = pack_grads([(name, (value.detach()-old).float().numpy())
                             for (name, value), (_, old) in zip(named, self.initial)], self.layout)
        stats["parameter_update_norm"] = float(np.linalg.norm(update))
        if with_clipped:
            # SGD/AdamW leave .grad unchanged: capture the actual dtype-rounded
            # optimizer input, rather than the float64 pre-writeback clip vector.
            clipped_ascent = -pack_grads([(name, value.grad.detach().float().numpy()) for name, value in named], self.layout)
            return update, stats, clipped_ascent
        return update, stats


def _allocation(prefixes, lengths, finished, baselines, engines, predictor, u, spec, payload, cfg):
    n = len(prefixes)
    warmup = int(payload.get("step", 0)) < int((cfg.get("predictor") or {}).get("warmup_steps", 0))
    remaining = continuation_remainings(prefixes, lengths, finished, int(cfg["max_new_tokens"]))
    f = np.zeros((n, u.shape[1]))
    risk, cost = np.ones(n), incremental_token_costs(remaining, finished)
    if spec.use_predictor and not warmup:
        if predictor is None:
            raise ValueError("actual CV audit needs the frozen checkpoint predictor")
        bundle = engines.prefix_features(prefixes, lengths, baselines)
        features = _predictor_audit_features(bundle, spec)
        pred = predictor.forward_numpy(features, bundle.get("cost_feat"))
        f, risk = np.asarray(pred.f).copy(), pred.r_hat
        if not predictor.constant_cost:
            cost = pred.c_hat
        if spec.risk_mode == "reward":
            q = predictor.forward_success(features)
            risk = predictor.forward_reward_risk(np.stack([
                _reward_features(q[i], max(len(prefixes[i])-lengths[i], 1), baselines[i]) for i in range(n)]))
    basis = payload.get("basis") or {}
    ready = not spec.use_allocation or neyman_ready(basis.get("basis_id", 0), basis.get("predictor_synced_basis_id", -1))
    alloc = cfg.get("allocation") or {}
    p, deviation = decide_continuation(spec, risk, cost, finished, float(alloc.get("beta", .5)),
                                     float(alloc.get("p_min", .2)), warmup,
                                     int(alloc.get("bisection_iters", 20)), basis_ready=ready,
                                     uniform_shrink=float(alloc.get("uniform_shrink", 0.)))
    f[finished] = 0
    if not spec.use_ht_correction or warmup:
        f[:] = 0
    f = control_variate_coordinates(f, enabled=bool((cfg.get("predictor") or {}).get("control_variate", True)))
    return f, np.asarray(risk), np.asarray(cost), p, remaining, deviation


def audit_fixed_batches(records, engines, layout, encode, payload, predictor, u, spec, cfg, run):
    """Generate independent full batches, then simulate the frozen policy's HT/CV."""
    if spec.objective != "raw_pg":
        raise ValueError("fixed-batch R-b audit supports mechanism methods; GRPO has a different objective")
    audit = cfg.get("batch_audit") or {}
    repeats, starts = int(audit.get("replicates", 8)), int(audit.get("starts_per_prompt", 4))
    decision, horizon, seed = int(cfg.get("decision_tokens", 512)), int(cfg.get("max_new_tokens", 2048)), int(cfg.get("seed", 17))
    if repeats <= 0 or starts <= 0 or not records or not 0 <= decision <= horizon or horizon <= 0:
        raise ValueError("positive replicate/start/horizon counts and 0 <= decision <= horizon are required")
    u = np.zeros((layout.dim, 0)) if u is None else np.asarray(u, dtype=np.float64)
    if u.ndim != 2 or u.shape[0] != layout.dim:
        raise ValueError("batch audit U and actor dimensions disagree")
    if payload.get("layout_names") and payload["layout_names"] != layout.names():
        raise ValueError("checkpoint parameter ordering does not match batch audit actor")
    named = list(engines.named_lora())
    source_hash = sha256_named(named)
    replay = OptimizerReplay(named, layout, payload.get("optimizer"), cfg.get("optim") or {})
    mode = str(audit.get("baseline_mode", "checkpoint"))
    history = HistoricalBaseline()
    if payload.get("baseline"):
        history.load_state_dict({**history.state_dict(), **payload["baseline"]})
    prescan_policy = baseline_from_config(cfg.get("baseline"))
    prescan_rng = IsolatedRNG.create(int(audit.get("prescan_seed", _audit_seed(seed, 0))))
    baseline, baseline_raw = {}, []
    prescan_start = prescan_rng.state_dict()
    for rec in records:
        samples = []
        if mode == "fixed":
            baseline[rec.problem_id] = float(audit.get("baseline", .5))
        elif mode == "checkpoint":
            baseline[rec.problem_id] = history.get(rec.problem_id)
        elif mode == "prescan":
            count = int(audit.get("prescan_samples", (cfg.get("baseline") or {}).get("prescan", 4)))
            if prescan_policy.uses_prescan:
                if count <= 0:
                    raise ValueError("independent prescan requires a positive sample count")
                _independent_pass_rate(engines, rec, encode, count, horizon, prescan_rng, engines.eos_id, samples)
                baseline[rec.problem_id] = prescan_policy.prescan_estimate([row["reward"] for row in samples])
            else:
                baseline[rec.problem_id] = prescan_policy.get(rec.problem_id)
        else:
            raise ValueError("baseline_mode must be checkpoint, fixed, or prescan")
        baseline_raw.append({"problem_id": rec.problem_id, "baseline": baseline[rec.problem_id], "samples": samples,
                             "checkpoint_default_used": mode == "checkpoint" and history.uses_prescan and rec.problem_id not in history.values,
                             "prescan_skipped_fixed_training_baseline": mode == "prescan" and not prescan_policy.uses_prescan})
    run.write_json("batch_audit_baselines.json", {"mode": mode, "frozen_for_all_replicates": True,
                   "configuration": (history.configuration() if mode == "checkpoint" else prescan_policy.configuration())
                                    if mode != "fixed" else {"mode": "fixed", "fixed_value": float(audit.get("baseline", .5))},
                   "rng_before": prescan_start, "rng_after": prescan_rng.state_dict(), "problems": baseline_raw})
    frozen = {"basis": u, "predictor": predictor, "optimizer": payload.get("optimizer"), "baseline": baseline}
    frozen_hash = sha256_mapping(frozen)
    prompts, pids, golds = [], [], []
    for rec in records:
        tokens, ids, answers = encode(rec, starts)
        prompts.extend(tokens); pids.extend(ids); golds.extend(answers)
    n, lengths = len(prompts), np.asarray([len(x) for x in prompts])
    if n != len(records)*starts:
        raise ValueError("encoded starts do not match the predetermined batch size")
    frozen_b = [baseline[pid] for pid in pids]
    gradient, updates, clipped = PairedMoments(layout.dim), PairedMoments(layout.dim), PairedMoments(layout.dim)
    geometry_rows = {key: [] for key in ("raw_gradient", "clipped_gradient", "optimizer_update")}
    prescan_tokens = sum(len(x["token_ids"])-x["prompt_len"] for row in baseline_raw for x in row["samples"])
    generated, teacher_forced, conditional_extra = prescan_tokens, 0, 0.
    clip_count = {"full": 0, "actual": 0}
    for rep in range(repeats):
        timer = Timer()
        replicate_seed = _audit_seed(seed, rep+1)
        rng = IsolatedRNG.create(replicate_seed)
        rng_before = rng.state_dict()
        prefixes, finished = engines.generate_prefix(prompts, decision, rng, "token")
        finished = np.asarray(finished, dtype=bool)
        if len(prefixes) != n or finished.shape != (n,):
            raise ValueError("prefix generator changed fixed batch cardinality")
        prefix_seeds, prefix_reasons = _request_seeds(engines, n), _prefix_finish_reasons(engines)
        f, risk, cost, p, remaining, deviation = _allocation(prefixes, lengths, finished, frozen_b, engines,
                                                            predictor, u, spec, payload, cfg)
        # Draw only from the prefix-measurable p before seeing full rewards.
        z = rng.bernoulli("selection", p); z[finished] = 1.
        phases = {"prefix_and_allocation_wall_seconds": timer.lap()}
        fulls = [list(x) for x in prefixes]
        suffix_seeds, suffix_reasons = [None]*n, [None]*n
        all_unfinished = (~finished) & (remaining > 0)
        for rem in sorted(set(remaining[all_unfinished].astype(int))):
            mask = all_unfinished & (remaining == rem)
            completed = engines.continue_selected(prefixes, mask, int(rem), rng)
            request = (getattr(engines, "last_rollout", None) or {}).get("continue_request_seeds") or {}
            for i in np.flatnonzero(mask):
                if completed[i] is None:
                    raise ValueError("diagnostic audit requires every suffix, including simulated stops")
                fulls[i] = completed[i]
                suffix_seeds[i] = request.get(int(i), request.get(str(i)))
                suffix_reasons[i] = _continue_finish_reason(engines, int(i))
        phases["all_suffix_generation_wall_seconds"] = timer.lap()
        full_gradient, actual_gradient = np.zeros(layout.dim), np.zeros(layout.dim)
        extra, actual_tokens, simulated_tokens, expected_tokens = 0., 0, 0., 0.
        for i in range(n):
            prefix_len, full_len = len(prefixes[i])-int(lengths[i]), len(fulls[i])-int(lengths[i])
            suffix_len = full_len-prefix_len
            reason = prefix_reasons[i] if finished[i] and i < len(prefix_reasons) else suffix_reasons[i]
            natural = _traj_natural_finish(bool(finished[i]), fulls[i], engines.eos_id,
                                           generated=full_len, requested=horizon, finish_reason=reason)
            truncated = _length_truncated(fulls[i], int(lengths[i]), horizon, natural, engines.eos_id, reason)
            text = engines.decode(fulls[i][int(lengths[i]):]) if engines.decode else ""
            reward = float(rule_reward(text, golds[i], truncated=truncated))
            g = _policy_grad_vec(engines, layout, fulls[i], int(lengths[i]), reward, frozen_b[i])
            m = u @ f[i]
            full_gradient += g/n
            actual_gradient += ht_estimate(g, m, p[i:i+1], z[i:i+1])[0]/n
            extra += float((1./p[i]-1.)*((g-m) @ (g-m)))/(n*n)
            actual_tokens += full_len
            simulated_tokens += prefix_len+z[i]*suffix_len
            expected_tokens += prefix_len+p[i]*suffix_len
            teacher_forced += full_len
            run.append_jsonl("batch_audit_samples.jsonl", {
                "replicate": rep, "replicate_seed": replicate_seed, "start": i, "problem_id": pids[i], "gold": golds[i],
                "baseline": frozen_b[i], "p": float(p[i]), "z": float(z[i]), "r_hat": float(risk[i]), "c_hat": float(cost[i]),
                "f": f[i], "full_reward": reward, "selected_reward": reward if z[i] else None,
                "full_text": text, "extracted": extract_answer(text), "truncated": truncated, "natural_finish": natural,
                "prefix_token_ids": prefixes[i], "full_token_ids": fulls[i], "prompt_len": int(lengths[i]),
                "prefix_request_seed": prefix_seeds[i], "suffix_request_seed": suffix_seeds[i],
                "prefix_finish_reason": prefix_reasons[i] if i < len(prefix_reasons) else None,
                "suffix_finish_reason": suffix_reasons[i], "full_g_norm_sq": float(g@g), "m_norm_sq": float(m@m),
                "residual_norm_sq": float((g-m)@(g-m)), "scope": "full diagnostic answer; selected_reward alone represents simulated training observation"})
        generated += actual_tokens; conditional_extra += extra
        phases["all_full_space_gradients_wall_seconds"] = timer.lap()
        full_update, full_stats, full_clip = replay.step(full_gradient, with_clipped=True)
        actual_update, actual_stats, actual_clip = replay.step(actual_gradient, with_clipped=True)
        gradient.add(full_gradient, actual_gradient); updates.add(full_update, actual_update)
        clipped.add(full_clip, actual_clip)
        geometry = {"raw_gradient": paired_geometry(full_gradient, actual_gradient),
                    "clipped_gradient": paired_geometry(full_clip, actual_clip),
                    "optimizer_update": paired_geometry(full_update, actual_update)}
        for key, row in geometry.items():
            geometry_rows[key].append(row)
        clip_count["full"] += int(full_stats["clip_triggered"]); clip_count["actual"] += int(actual_stats["clip_triggered"])
        phases["cpu_optimizer_replays_wall_seconds"] = timer.lap()
        run.append_jsonl("batch_audit_replicates.jsonl", {
            "replicate": rep, "seed": replicate_seed, "fixed_n": n, "rng_before": rng_before, "rng_after": rng.state_dict(),
            "actual_generation_tokens": actual_tokens, "hypothetical_selected_tokens": simulated_tokens,
            "hypothetical_expected_tokens": expected_tokens, "conditional_selection_extra_variance_trace": extra,
            "allocation_deviation": deviation, "full_update": full_stats, "actual_update": actual_stats,
            "gradient_difference_norm": float(np.linalg.norm(actual_gradient-full_gradient)),
            "update_difference_norm": float(np.linalg.norm(actual_update-full_update)),
            "paired_geometry": geometry, "phases": phases})
        print(f"batch audit replicate={rep+1}/{repeats} N={n} all_generated_tokens={actual_tokens}", flush=True)
    source_after = sha256_named(engines.named_lora())
    frozen_after = sha256_mapping(frozen)
    if source_hash != source_after or frozen_hash != frozen_after:
        raise RuntimeError("batch audit changed a frozen source state")
    np.savez_compressed(run.root / "batch_audit_means.npz", full_gradient=gradient.mean[0], actual_gradient=gradient.mean[1],
                        full_update=updates.mean[0], actual_update=updates.mean[1],
                        full_clipped_gradient=clipped.mean[0], actual_clipped_gradient=clipped.mean[1])
    return {"replicates": repeats, "fixed_n": n, "n_prompts": len(records), "starts_per_prompt": starts,
            "batch_design": "Configured fixed diagnostic N and prompt set, not a reconstruction of the last adaptive training batch.",
            "method": spec.name, "baseline_mode": mode, "gradient": gradient.summary(), "update": updates.summary(),
            "estimator_ablation": {"control_variate": bool((cfg.get("predictor") or {}).get("control_variate", True)),
                                   "uniform_shrink": float((cfg.get("allocation") or {}).get("uniform_shrink", 0.)),
                                   "note": "CV switch changes only m; risk predictions are retained."},
            "clipped_gradient": clipped.summary(),
            "paired_geometry": {key: summarize_geometries(rows) for key, rows in geometry_rows.items()},
            "geometry_scope": "Each replicate compares actual against Full-PG on the same complete labels. Clipped ascent vectors are the negative of actual optimizer-input .grad after shared clipping and parameter-dtype writeback. Zero-vector cosine and zero-reference relative error are undefined. Differences are empirical, not unbiased-update claims.",
            "conditional_selection_extra_variance_trace_mean": conditional_extra/repeats,
            "clip_counts": clip_count, "actual_generation_tokens": generated, "prescan_generation_tokens": prescan_tokens,
            "full_gradient_response_tokens": teacher_forced, "full_gradient_calls": n*repeats,
            "actor_sha256_before": source_hash, "actor_sha256_after": source_after,
            "basis_head_baseline_optimizer_sha256_before": frozen_hash,
            "basis_head_baseline_optimizer_sha256_after": frozen_after,
            "optimizer_replay": {"class": replay.kind, "source": "checkpoint moments and numerical hyperparameters",
                "device": "cpu", "parameter_dtypes": sorted({str(x.dtype) for _, x in replay.initial}),
                "grad_clip": replay.clip,
                "param_groups": [{key:value for key,value in group.items() if key != "params"}
                                 for group in replay.snapshot["param_groups"]],
                "note": "Fresh copies per arm and replicate; same shared clip function and torch optimizer. Not bitwise CUDA execution validation."},
            "scope": "Full-space fixed-N R-b batch gradients; frozen prompts/actor/U/heads/baseline/optimizer. All full suffixes and gradients are paid. HT/CV selection is a simulation, not measured training savings.",
            "memory": "O(D) online moments plus frozen U and optimizer snapshots; no replicate-by-D array",
            "rng_design": "SeedSequence([seed, 0xBA7C2026, index]); index=0 prescan, index=1..replicates independent batches. Separate token/continuation/selection streams; full RNG states persisted.",
            "uncertainty": "A single replicate produces point values with covariance unavailable; no minimum sample gate. Independent prefixes and suffixes across replicates, conditional on this one prompt set."}


def run_batch_audit(records, cfg, run_dir):
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("max_new_tokens", 2048 if cfg.get("backend") == "gpu_verl" else 8)
    cfg.setdefault("decision_tokens", min(512, int(cfg["max_new_tokens"])))
    ckpt = cfg.get("checkpoint") or cfg.get("resume")
    if not ckpt:
        raise ValueError("batch audit requires a frozen training checkpoint")
    backend = cfg.get("backend", "cpu_tiny")
    n_gpu = int((cfg.get("hardware") or {}).get("n_gpu", 1 if backend == "gpu_verl" else 0))
    if backend == "gpu_verl" and n_gpu != 1:
        raise ValueError("batch audit GPU backend currently supports exactly one GPU")
    audit = cfg.setdefault("batch_audit", {})
    n_prompts = int(audit.get("n_prompts", cfg.get("n_prompts", 2)))
    if n_prompts <= 0:
        raise ValueError("batch audit n_prompts must be positive")
    audit.setdefault("starts_per_prompt", max(1, int(cfg.get("n_start", 8))//max(n_prompts, 1)))
    selected = select_records(apply_solve_instruction(list(records)), n_prompts,
                              str(audit.get("selection", "seeded")), int(audit.get("selection_seed", cfg.get("seed", 17))))
    run = RunDirectory(resolve_run_dir(run_dir))
    started, timer = utc_now(), Timer()
    ledger = ComputeLedger(n_gpu=n_gpu if backend == "gpu_verl" else 0, hardware=backend)
    run.write_run_meta(kind="batch_audit", started=started, requested=run_dir)
    run.write_yaml("config.yaml", cfg)
    versions = collect_environment(cfg); run.write_json("environment.json", versions)
    try:
        payload = load_checkpoint(ckpt)
        spec = _audit_method_spec(cfg, payload)
        cfg["method"] = spec.name
        stored = payload.get("run_config") or {}
        behavior_keys = ("max_new_tokens", "decision_tokens", "temperature", "prompt_max_tokens",
                         "allocation", "predictor", "baseline", "optim", "lora")
        source = {}
        for key in behavior_keys:
            if key in stored:
                cfg[key] = copy.deepcopy(stored[key])
                source[key] = "checkpoint.run_config"
            else:
                source[key] = "requested_config_fallback_legacy_checkpoint"
        run.write_yaml("config.yaml", cfg)
        if backend == "gpu_verl":
            cache = {}
            setup_cfg = {**cfg, "audit": {"max_new_tokens": int(cfg.get("max_new_tokens", 2048))}}
            generate_bundles_gpu([], 1, 1, 0, int(setup_cfg["audit"]["max_new_tokens"]), int(cfg.get("seed", 17)),
                                 setup_cfg, work_dir=run.root/"batch_audit_lora", cache=cache, payload=payload)
            engines, layout, encode = cache["engines"], cache["layout"], cache["encode_fn"]
        else:
            from grace_gc.data.tokenize import encode_records_tiny
            from grace_gc.trainer.tiny_engine import make_tiny_engines
            actor = _tiny_actor(cfg, payload)
            engines = make_tiny_engines(actor, actor.vocab)
            layout = collect_lora_layout(actor.named_lora_params())
            encode = lambda rec,n: encode_records_tiny([rec]*n, actor.vocab, int(cfg.get("prompt_max_tokens", 1024)))
        predictor, u = _frozen_predictor(cfg, payload)
        manifest = selection_manifest(selected, str(audit.get("selection", "seeded")),
                                       int(audit.get("selection_seed", cfg.get("seed", 17))))
        manifest.update(checkpoint=str(ckpt), checkpoint_sha256=sha256_file(ckpt), checkpoint_step=payload.get("step"),
                        method=spec.name, reward_protocol_version=REWARD_PROTOCOL_VERSION,
                        requested_n_prompts=n_prompts, actual_n_prompts=len(selected),
                        training_settings_sources=source, diagnostic_batch_settings=audit,
                        horizon=int(cfg.get("max_new_tokens", 2048)), decision_tokens=int(cfg.get("decision_tokens", 512)))
        run.write_json("batch_audit_manifest.json", manifest); persist_load_report(run, data_path=cfg.get("data_path"))
        result = audit_fixed_batches(selected, engines, layout, encode, payload, predictor, u, spec, cfg, run)
        result.update(status="completed", started=started, finished=utc_now(), run_dir=str(run.root), manifest=manifest)
        ledger.add("batch_audit", timer.elapsed(), cpu_s=timer.cpu_elapsed(), status="completed")
        result["cost_scope"] = "Whole audit envelope: initialization, optional full prescan, every prefix/suffix, every full gradient, CPU optimizer replay and evidence persistence. GPU time is reserved envelope, not utilization."
        run.write_json("batch_audit_summary.json", result)
        run.write_json("compute_ledger.json", ledger.summary()); run.append_jsonl("compute_ledger.jsonl", ledger.rows[-1])
        return result
    except Exception as exc:
        ledger.add("batch_audit", timer.elapsed(), cpu_s=timer.cpu_elapsed(), status="failed")
        run.write_json("compute_ledger.json", ledger.summary()); run.append_jsonl("compute_ledger.jsonl", ledger.rows[-1])
        write_failed(run, exc, versions, started)
        raise
