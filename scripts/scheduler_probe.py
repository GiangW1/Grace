#!/usr/bin/env python3
"""Frozen-actor scheduler probe: legacy two-phase vs request-level refill.

This is a measurement entry point, not a training shortcut.  It keeps the
actor and LoRA snapshot fixed, runs a predetermined workload, and records all
starts including fixed-probability early stops.  ``fixed_ht`` uses m=0 and a
scalar continuation probability so scheduler time can be measured without
introducing predictor features or risk allocation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grace_gc.backends.gpu_engine import make_gpu_engines
from grace_gc.backends.vllm_continuous import (
    ContinuousRecord,
    ContinuousRollout,
    make_request_plan,
    record_to_dict,
)
from grace_gc.backends.vllm_two_phase import continue_selected, generate_phase
from grace_gc.backends.verl_trainer import (
    _shutdown_vllm_engine,
    build_vllm_engine,
    load_lora_actor,
    vllm_needed_max_model_len,
)
from grace_gc.backends.weight_sync import reset_vllm_prefix_cache
from grace_gc.core.rng import IsolatedRNG
from grace_gc.data.math_data import load_math_records, select_records
from grace_gc.data.tokenize import collect_stop_token_ids, encode_records_hf, load_hf_tokenizer
from grace_gc.logging_util.run_dir import RunDirectory, default_run_dir, resolve_run_dir
from grace_gc.trainer.loop import build_run_config
from grace_gc.trainer.state_io import check_snapshot_identity, load_checkpoint, load_numpy_module_state
from grace_gc.versions import collect_environment


def _legacy_probe(llm, lora_request, eos_id, prompts, problem_ids, p, decision_tokens,
                  max_new_tokens, seed, workload, request_plan):
    started = perf_counter()
    rng = IsolatedRNG.create(int(seed))
    n = len(prompts)
    if workload == "full_pg":
        phase = generate_phase(
            llm, prompts, int(max_new_tokens), 1.0, eos_id, rng, "token",
            lora_request=lora_request, request_seeds=request_plan["prefix_seeds"],
        )
        records = []
        for i, full in enumerate(phase.token_ids):
            response_tokens = max(len(full) - len(prompts[i]), 0)
            records.append(ContinuousRecord(
                start_index=i, problem_id=str(problem_ids[i]), prompt_token_ids=list(prompts[i]),
                prefix_token_ids=list(full), full_token_ids=list(full), stage="complete",
                natural_finish=bool(phase.natural_finish[i]),
                truncated=bool(not phase.natural_finish[i]),
                finish_reason=(phase.finish_reasons or [None] * n)[i],
                stop_reason=(phase.stop_reasons or [None] * n)[i],
                prefix_tokens=response_tokens,
                prefix_request_id=f"legacy-prefix-{i}", request_count=1,
            ))
        summary = {
            "mode": "legacy_two_phase", "workload": workload, "p": 1.0,
            "n_starts": n, "decision_tokens": None, "max_new_tokens": int(max_new_tokens),
            "wall_seconds": perf_counter() - started,
            "prefix_tokens": int(sum(r.prefix_tokens for r in records)),
            "suffix_tokens": 0, "generated_tokens": int(sum(r.prefix_tokens for r in records)),
            "stopped": 0, "continued": n,
            "phase_execution": phase.execution,
            "scope": "external legacy LLM.generate wall time; includes prefix/full generation, excludes model construction",
        }
        return summary, records

    prefix = generate_phase(
        llm, prompts, int(decision_tokens), 1.0, eos_id, rng, "token",
        lora_request=lora_request, request_seeds=request_plan["prefix_seeds"],
    )
    selected = np.asarray(request_plan["selected"], dtype=bool)
    prefix_at_horizon = np.asarray([
        max(len(pref) - len(prompts[i]), 0) >= int(max_new_tokens)
        for i, pref in enumerate(prefix.token_ids)
    ], dtype=bool)
    selected_for_suffix = selected & ~np.asarray(prefix.natural_finish, dtype=bool) & ~prefix_at_horizon
    fulls = continue_selected(
        llm, prefix.token_ids, selected_for_suffix,
        max(0, int(max_new_tokens) - int(decision_tokens)), 1.0, eos_id, rng,
        lora_request=lora_request, request_seeds=request_plan["suffix_seeds"],
    )
    suffix_phase = getattr(continue_selected, "last_phase", None)
    suffix_idx = list(getattr(continue_selected, "last_idx", None) or [])
    suffix_reason = {}
    suffix_stop = {}
    suffix_natural = {}
    suffix_exec = None
    if suffix_phase is not None:
        suffix_exec = suffix_phase.execution
        suffix_reason = {i: suffix_phase.finish_reasons[j] for j, i in enumerate(suffix_idx)}
        suffix_stop = {i: suffix_phase.stop_reasons[j] for j, i in enumerate(suffix_idx)}
        suffix_natural = {i: bool(suffix_phase.natural_finish[j]) for j, i in enumerate(suffix_idx)}
    records = []
    for i, pref in enumerate(prefix.token_ids):
        prefix_tokens = max(len(pref) - len(prompts[i]), 0)
        if bool(prefix.natural_finish[i]):
            records.append(ContinuousRecord(
                start_index=i, problem_id=str(problem_ids[i]), prompt_token_ids=list(prompts[i]),
                prefix_token_ids=list(pref), full_token_ids=list(pref), stage="complete",
                selected=None, natural_finish=True, truncated=False, prefix_tokens=prefix_tokens,
                finish_reason=(prefix.finish_reasons or [None] * n)[i],
                stop_reason=(prefix.stop_reasons or [None] * n)[i],
                prefix_request_id=f"legacy-prefix-{i}", request_count=1,
            ))
            continue
        if bool(prefix_at_horizon[i]):
            records.append(ContinuousRecord(
                start_index=i, problem_id=str(problem_ids[i]), prompt_token_ids=list(prompts[i]),
                prefix_token_ids=list(pref), full_token_ids=list(pref), stage="complete",
                selected=None, natural_finish=False, truncated=True, prefix_tokens=prefix_tokens,
                finish_reason=(prefix.finish_reasons or [None] * n)[i],
                stop_reason=(prefix.stop_reasons or [None] * n)[i],
                prefix_request_id=f"legacy-prefix-{i}", request_count=1,
            ))
            continue
        if not bool(selected[i]):
            records.append(ContinuousRecord(
                start_index=i, problem_id=str(problem_ids[i]), prompt_token_ids=list(prompts[i]),
                prefix_token_ids=list(pref), full_token_ids=None, stage="stopped",
                selected=False, prefix_tokens=prefix_tokens,
                finish_reason=(prefix.finish_reasons or [None] * n)[i],
                stop_reason=(prefix.stop_reasons or [None] * n)[i],
                prefix_request_id=f"legacy-prefix-{i}", request_count=1,
            ))
            continue
        full = pref if fulls[i] is None else fulls[i]
        suffix_tokens = max(len(full) - len(pref), 0)
        natural = bool(suffix_natural.get(i, False))
        records.append(ContinuousRecord(
            start_index=i, problem_id=str(problem_ids[i]), prompt_token_ids=list(prompts[i]),
            prefix_token_ids=list(pref), full_token_ids=list(full), stage="complete",
            selected=True, natural_finish=natural, truncated=not natural,
            finish_reason=suffix_reason.get(i) or (prefix.finish_reasons or [None] * n)[i],
            stop_reason=suffix_stop.get(i), prefix_tokens=prefix_tokens,
            suffix_tokens=suffix_tokens, prefix_request_id=f"legacy-prefix-{i}",
            request_count=2 if i in suffix_idx else 1,
        ))
    summary = {
        "mode": "legacy_two_phase", "workload": workload, "p": float(p),
        "n_starts": n, "decision_tokens": int(decision_tokens), "max_new_tokens": int(max_new_tokens),
        "wall_seconds": perf_counter() - started,
        "prefix_tokens": int(sum(r.prefix_tokens for r in records)),
        "suffix_tokens": int(sum(r.suffix_tokens for r in records)),
        "generated_tokens": int(sum(r.prefix_tokens + r.suffix_tokens for r in records)),
        "stopped": int(sum(r.stage == "stopped" for r in records)),
        "continued": int(sum(r.selected is True for r in records)),
        "phase_execution": prefix.execution,
        "continue_execution": suffix_exec,
        "scope": "external legacy two-phase LLM.generate wall time; includes prefix and selected suffix, excludes model construction",
    }
    return summary, records


def _load_resources(cfg, run_root):
    model_path = cfg.get("model_path")
    if not model_path:
        raise ValueError("--model-path or model_path in YAML is required")
    tokenizer = load_hf_tokenizer(str(model_path))
    actor = load_lora_actor(str(model_path), cfg.get("lora") or {})
    checkpoint = cfg.get("checkpoint")
    if checkpoint:
        payload = load_checkpoint(checkpoint)
        check_snapshot_identity(payload, cfg)
        from grace_gc.backends.hf_actor import named_lora_params

        load_numpy_module_state(named_lora_params(actor), payload.get("actor") or {})
    vllm_cfg = dict(cfg.get("vllm") or {})
    vllm_cfg.setdefault("seed", int(cfg.get("seed", 17)))
    vllm_cfg["max_model_len"] = vllm_needed_max_model_len(cfg, int(cfg["max_new_tokens"]))
    llm = build_vllm_engine(str(model_path), vllm_cfg, int((cfg.get("lora") or {}).get("rank", 16)))
    engines, extra = make_gpu_engines(actor, llm, tokenizer, cfg, Path(run_root) / "adapter")
    extra["sync"]()
    return tokenizer, actor, llm, engines, extra


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Frozen-actor legacy vs continuous refill probe")
    parser.add_argument("--config", action="append", default=[])
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--selection", choices=("first", "seeded"), default="seeded")
    parser.add_argument("--selection-seed", type=int, default=None)
    parser.add_argument("--n-problems", type=int, default=4)
    parser.add_argument("--starts-per-problem", type=int, default=16)
    parser.add_argument("--capacity", type=int, default=None)
    parser.add_argument("--decision-tokens", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--p", type=float, action="append", default=None,
                        help="fixed-HT continuation probability; repeat the flag (default: 0.5, 0.75, 1.0)")
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--mode", choices=("legacy", "continuous", "both"), default=None)
    parser.add_argument("--workload", choices=("full_pg", "fixed_ht", "both"), default=None)
    args = parser.parse_args(argv)
    seed = int(args.seed if args.seed is not None else 17)
    overrides = {
        "backend": "gpu_verl", "model_path": args.model_path, "checkpoint": args.checkpoint,
        "seed": seed,
        "hardware": {"n_gpu": 1, "name": "a100"},
        "vllm": {"tensor_parallel": 1},
    }
    if args.max_new_tokens is not None:
        overrides["max_new_tokens"] = int(args.max_new_tokens)
    if args.decision_tokens is not None:
        overrides["decision_tokens"] = int(args.decision_tokens)
    cfg = build_run_config(args.config, {k: v for k, v in overrides.items() if v is not None})
    probe_cfg = cfg.get("scheduler_probe") or {}
    capacity = int(args.capacity if args.capacity is not None else probe_cfg.get("capacity", 8))
    decision_tokens = int(
        args.decision_tokens if args.decision_tokens is not None
        else probe_cfg.get("decision_tokens", cfg.get("decision_tokens", 512))
    )
    max_new_tokens = int(
        args.max_new_tokens if args.max_new_tokens is not None
        else probe_cfg.get("max_new_tokens", cfg.get("max_new_tokens", 2048))
    )
    configured_ps = [float(x) for x in (
        args.p if args.p is not None
        else probe_cfg.get("continuation_probabilities", [0.5, 0.75, 1.0])
    )]
    if any(not 0.0 <= p <= 1.0 for p in configured_ps):
        raise ValueError("each --p must be in [0, 1]")
    repeats = int(args.repeats if args.repeats is not None else probe_cfg.get("repeats", 1))
    schedulers = list(probe_cfg.get("schedulers", ["both"]))
    configured_workloads = list(probe_cfg.get("workloads", ["both"]))
    mode = args.mode or ("both" if set(("legacy", "continuous")).issubset(schedulers) else str(schedulers[0]))
    workload = args.workload or (
        "both" if set(("full_pg", "fixed_ht")).issubset(configured_workloads)
        else str(configured_workloads[0])
    )
    if capacity <= 0 or repeats <= 0:
        raise ValueError("capacity and repeats must be positive")
    if decision_tokens <= 0 or max_new_tokens < decision_tokens:
        raise ValueError("decision_tokens must be positive and <= max_new_tokens")
    cfg["model_path"] = args.model_path or cfg.get("model_path")
    cfg["checkpoint"] = args.checkpoint or cfg.get("checkpoint")
    cfg["max_new_tokens"] = max_new_tokens
    cfg["decision_tokens"] = decision_tokens
    if not cfg.get("model_path"):
        raise ValueError("--model-path or model_path in YAML is required")
    if int(args.n_problems) <= 0 or int(args.starts_per_problem) <= 0:
        raise ValueError("n-problems and starts-per-problem must be positive")
    requested = args.run_dir or str(default_run_dir("scheduler-probe"))
    run = RunDirectory(resolve_run_dir(requested))
    run.write_yaml("config.yaml", {**cfg, "scheduler_probe": {
        **probe_cfg,
        "capacity": capacity,
        "decision_tokens": decision_tokens,
        "max_new_tokens": max_new_tokens,
        "continuation_probabilities": configured_ps,
        "repeats": repeats,
        "mode": mode,
        "workload": workload,
        "cli": vars(args),
    }})
    run.write_json("environment.json", collect_environment(cfg))
    records = select_records(
        load_math_records(args.data_path), int(args.n_problems), args.selection,
        int(args.selection_seed if args.selection_seed is not None else seed),
    )
    if not records:
        raise ValueError("the selected data set is empty")
    starts = []
    for rec in records:
        starts.extend([rec] * int(args.starts_per_problem))
    run.write_json("workload.json", {
        "n_problems": len(records), "starts_per_problem": int(args.starts_per_problem),
        "n_starts": len(starts), "problem_ids": [rec.problem_id for rec in records],
    })
    tokenizer = actor = llm = engines = extra = None
    try:
        tokenizer, actor, llm, engines, extra = _load_resources(cfg, run.root)
        prompt_meta = []
        prompt_ids, problem_ids, _golds = encode_records_hf(
            starts, tokenizer, int(cfg.get("prompt_max_tokens", 1024)), prompt_meta=prompt_meta,
        )
        run.write_jsonl("prompt_manifest.jsonl", [
            {"start_index": i, "problem_id": problem_ids[i], "prompt_tokens": len(prompt_ids[i]),
             "prompt_meta": prompt_meta[i] if i < len(prompt_meta) else None}
            for i in range(len(prompt_ids))
        ])
        eos_id = collect_stop_token_ids(tokenizer)
        workloads = ["full_pg", "fixed_ht"] if workload == "both" else [workload]
        modes = ["legacy", "continuous"] if mode == "both" else [mode]
        summaries = []
        comparisons = []
        for workload_index, workload in enumerate(workloads):
            ps = [1.0] if workload == "full_pg" else [float(x) for x in configured_ps]
            for p_index, p in enumerate(ps):
                for repeat in range(repeats):
                    trial_seed = seed + 100003 * repeat + 1009 * p_index + 10000019 * workload_index
                    request_plan = make_request_plan(len(prompt_ids), p, trial_seed)
                    run.append_jsonl("scheduler_plans.jsonl", {
                        "workload": workload, "p": float(p), "repeat": repeat, **request_plan,
                    })
                    paired = {}
                    ordered_modes = modes if repeat % 2 == 0 else list(reversed(modes))
                    for mode_name in ordered_modes:
                        reset_vllm_prefix_cache(llm)
                        if mode_name == "legacy":
                            summary, trial_records = _legacy_probe(
                                llm, extra.get("lora_request"), eos_id, prompt_ids, problem_ids, p,
                                decision_tokens, max_new_tokens, trial_seed, workload, request_plan,
                            )
                        else:
                            runner = ContinuousRollout(
                                llm, eos_id, lora_request=extra.get("lora_request"), capacity=capacity,
                            )
                            summary, trial_records = runner.run(
                                [{"problem_id": problem_ids[i], "prompt_token_ids": prompt_ids[i]}
                                 for i in range(len(prompt_ids))],
                                mode="full_pg" if workload == "full_pg" else "fixed_ht",
                                decision_tokens=decision_tokens, max_new_tokens=max_new_tokens,
                                p=p, seed=trial_seed, request_plan=request_plan,
                            )
                        summary.update({"workload": workload, "scheduler": mode_name, "p": float(p), "repeat": repeat,
                                        "trial_seed": trial_seed})
                        summaries.append(summary)
                        paired[mode_name] = (summary, trial_records)
                        run.append_jsonl("scheduler_trials.jsonl", summary)
                        run.write_jsonl("scheduler_records.jsonl", [
                            {**record_to_dict(row), "workload": workload, "scheduler": mode_name,
                             "p": float(p), "repeat": repeat}
                            for row in trial_records
                        ], append=True)
                        print(summary, flush=True)
                    if "legacy" in paired and "continuous" in paired:
                        legacy_summary, legacy_records = paired["legacy"]
                        continuous_summary, continuous_records = paired["continuous"]
                        comparable_fields = (
                            "start_index", "problem_id", "prompt_token_ids", "prefix_token_ids",
                            "full_token_ids", "stage", "selected", "natural_finish", "truncated",
                            "finish_reason", "stop_reason", "prefix_tokens", "suffix_tokens",
                            "request_count",
                        )
                        exact = []
                        for left, right in zip(legacy_records, continuous_records):
                            left_row, right_row = record_to_dict(left), record_to_dict(right)
                            exact.append(all(left_row[key] == right_row[key] for key in comparable_fields))
                        comparison = {
                            "workload": workload, "p": float(p), "repeat": repeat,
                            "trial_seed": trial_seed, "n_starts": len(exact),
                            "exact_record_matches": int(sum(exact)),
                            "all_records_match": bool(all(exact)),
                            "generated_token_difference": int(
                                continuous_summary["generated_tokens"] - legacy_summary["generated_tokens"]
                            ),
                            "legacy_wall_seconds": float(legacy_summary["wall_seconds"]),
                            "continuous_wall_seconds": float(continuous_summary["wall_seconds"]),
                            "legacy_over_continuous_speedup": (
                                float(legacy_summary["wall_seconds"]) / float(continuous_summary["wall_seconds"])
                                if float(continuous_summary["wall_seconds"]) > 0 else None
                            ),
                        }
                        comparisons.append(comparison)
                        run.append_jsonl("scheduler_comparisons.jsonl", comparison)
        run.write_json("scheduler_summary.json", {
            "trials": summaries,
            "comparisons": comparisons,
            "scope": "Frozen actor scheduler probe; no actor update or predictor features; compare external wall time and token/queue records.",
        })
        print("run_dir", run.root)
        return 0
    finally:
        if llm is not None:
            _shutdown_vllm_engine(llm)


if __name__ == "__main__":
    raise SystemExit(main())
