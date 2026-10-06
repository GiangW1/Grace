#!/usr/bin/env python3
"""GRACE-ST smoke, paired mechanism audit and efficient training controls."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "audit", "train"), required=True)
    parser.add_argument("--cpu", action="store_true", help="math-only smoke; GPU smoke also checks actual loss/engines")
    parser.add_argument("--config", action="append", default=[])
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--data-path")
    parser.add_argument("--eval-data-path")
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--gpus", help="visible GPUs: first HF actor, remaining independent vLLM workers")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--method", choices=("full_pg", "donor_only", "suffix_transport"))
    parser.add_argument("--group-size", type=int)
    parser.add_argument("--decision-tokens", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--n-problems", type=int)
    parser.add_argument("--groups-per-problem", type=int)
    parser.add_argument("--audit-draws", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--strength", type=float, help="fixed before any outcomes; 0=paired donor, 1=ST")
    parser.add_argument("--budget-seconds", type=float, help="training wall budget, including setup/sync/checkpoints")
    parser.add_argument("--no-full-rb", action="store_true", help="skip expensive all-receiver backward in audit")
    args = parser.parse_args(argv)
    if args.cpu and args.mode != "smoke":
        parser.error("--cpu is only a math smoke")
    if args.gpus:
        devices = [device.strip() for device in args.gpus.split(",")]
        if any(not device for device in devices) or len(devices) != len(set(devices)):
            parser.error("--gpus must list distinct nonempty device IDs")
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(device.strip() for device in devices)
    from grace_gc.audit.expected_gain import finish_experiment_run, start_experiment_run
    from grace_gc.config import default_config, load_config, merge_configs
    from grace_gc.logging_util.run_dir import RunDirectory
    from grace_gc.trainer.suffix_transport import Experiment, validate_experiment
    from grace_gc.versions import collect_versions

    defaults = {"suffix_transport": {"method": "suffix_transport", "group_size": 4, "strength": 1.,
                 "baseline": .5, "n_problems": 32, "selection_seed": 17, "groups_per_problem": 2,
                 "audit_draws": 8, "full_rb": True, "bootstrap": 1000, "train_steps": 100,
                 "prompts_per_step": 1, "checkpoint_every": 10, "split": "audit"},
                "decision_tokens": 512, "max_new_tokens": 8192, "temperature": 1.0,
                "lora": {"dropout": 0., "targets": ["q_proj", "v_proj"]},
                "hardware": {"n_gpu": 1}, "rollout": {"workers": 0}}
    cfg = merge_configs(default_config(), defaults, *(load_config(p) for p in args.config))
    for key in ("model_path", "data_path", "eval_data_path", "init_checkpoint", "seed", "decision_tokens", "max_new_tokens"):
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value
    st = cfg["suffix_transport"]
    for key in ("method", "group_size", "n_problems", "groups_per_problem", "audit_draws", "strength"):
        value = getattr(args, key)
        if value is not None:
            st[key] = value
    if args.steps is not None:
        st["train_steps"] = args.steps
    if args.no_full_rb:
        st["full_rb"] = False
    if args.gpus:
        cfg["hardware"]["n_gpu"] = len(devices)
        cfg["rollout"]["workers"] = len(devices) - 1
        cfg.setdefault("vllm", {})["tensor_parallel"] = 1
    if args.budget_seconds is not None:
        if not math.isfinite(args.budget_seconds) or args.budget_seconds <= 0 or args.mode != "train":
            parser.error("positive --budget-seconds is training only")
        st["budget_seconds"] = args.budget_seconds
    # Smoke overrides are explicit in saved config, and never reused as effect estimates.
    if args.mode == "smoke" and not args.cpu:
        cfg["decision_tokens"] = 8 if args.decision_tokens is None else args.decision_tokens
        cfg["max_new_tokens"] = 40 if args.max_new_tokens is None else args.max_new_tokens
        st.update(n_problems=1, group_size=2, audit_draws=1, groups_per_problem=1)
    cfg["method"] = st["method"]
    cfg["experiment_variant"] = "suffix_transport"
    cfg["n_prompts"] = int(st["prompts_per_step"]) if args.mode == "train" else 1
    starts_per_prompt = 1 if args.mode == "train" and st["method"] == "donor_only" else int(st["group_size"])
    cfg["n_start"] = cfg["n_prompts"] * starts_per_prompt
    cfg["baseline"] = {"mode": "fixed", "fixed_value": float(st["baseline"]), "prescan_n": 0}
    cfg["lora"]["compute_dtype"] = "bfloat16"
    if not args.cpu:
        cfg["backend"] = "gpu_verl"
    validate_experiment(cfg)
    destination = start_experiment_run(args.run_dir, f"suffix_transport_{args.mode}", {"args": vars(args), "config": cfg})
    run, exp = RunDirectory(destination), None
    started = perf_counter()
    try:
        run.write_yaml("config.yaml", cfg)
        # Distribution metadata suffices here; setup explicitly validates GPU deps.
        run.write_json("versions.json", collect_versions(extra_modules=("numpy", "yaml")))
        if args.cpu:
            from grace_gc.audit.suffix_transport_smoke import math_smoke
            result = {"mode": "cpu_math_smoke", **math_smoke(), "gpu_checks": "not executed"}
        else:
            if not cfg.get("model_path") or not cfg.get("data_path"):
                raise ValueError("GPU modes require --model-path and --data-path (or YAML values)")
            exp = Experiment(cfg, run)
            exp.run_started = started
            exp.timed("setup", lambda: exp.setup(args.mode))
            run.write_json("versions.json", collect_versions())
            if args.mode == "smoke":
                from grace_gc.audit.suffix_transport_smoke import gpu_smoke
                result = gpu_smoke(exp)
            else:
                result = exp.audit() if args.mode == "audit" else exp.train()
            result.update(stage_wall_seconds=dict(exp.timings), generated_tokens=exp.generated_tokens,
                          actual_completions=exp.actual_completions, actual_successes=exp.actual_successes,
                          first_success_problem_ids=sorted(exp.first_successes))
            exp.close()
            exp = None
        elapsed = perf_counter() - started
        result.update(actual_wall_seconds=elapsed,
                      reserved_gpu_seconds=0 if args.cpu else elapsed * int(cfg["hardware"]["n_gpu"]),
                      timing_note="Setup contains initial sync; stage totals overlap. Reserved GPU time is n_gpu*elapsed, not summed stages.")
        run.write_json("summary.json", result)
        finish_experiment_run(destination)
    except BaseException as exc:
        run.write_json("failed_execution.json", {"error_type": type(exc).__name__, "error": str(exc),
                       "wall_seconds": perf_counter() - started,
                       "reserved_gpu_seconds": 0 if args.cpu else (perf_counter() - started) * int(cfg["hardware"]["n_gpu"]),
                       "stage_wall_seconds": dict(exp.timings) if exp is not None else {}})
        finish_experiment_run(destination, exc)
        raise
    finally:
        if exp is not None:
            exp.close()
    print(json.dumps({"run_dir": str(destination), "mode": result["mode"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
