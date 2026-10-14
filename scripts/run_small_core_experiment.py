#!/usr/bin/env python3
"""Sequential comparison on the existing train/evaluate/audit entry points."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grace_gc.config import load_config, merge_configs
from grace_gc.logging_util.run_dir import atomic_write_text, utc_now, utc_stamp


ARMS = {"lora16_full_pg": (None, "full_pg"), "lora16_grace": (None, "grace"),
        "core8_full_pg": (8, "full_pg"), "core8_grace": (8, "grace"),
        "core16_full_pg": (16, "full_pg"), "core16_grace": (16, "grace"),
        "core8_uniform_ht": (8, "uniform_ht"), "core16_uniform_ht": (16, "uniform_ht")}
FOLLOWUP_ARMS = ["core8_full_pg", "core8_uniform_ht", "core8_grace"]


def suite_directory(requested):
    """A plan is also an artifact: never overwrite a previous suite.json."""
    root = Path(requested).resolve()
    candidate, index = root, 1
    while candidate.exists() and any(candidate.iterdir()):
        candidate = root.with_name(f"{root.name}-{utc_stamp()}-{index}")
        index += 1
    candidate.mkdir(parents=True, exist_ok=True)
    return candidate


def arm_config(common, arm, seed, *, smoke=False, run_wall_seconds=None,
               core_basis="svd", include_actor_state=True):
    rank, method = ARMS[arm]
    cfg = merge_configs(common, {"seed": seed, "method": method})
    if rank:
        cfg = merge_configs(cfg, load_config(ROOT / f"configs/experiments/small_core_r{rank}.yaml"))
        cfg["lora"]["basis"] = core_basis
        cfg["predictor"]["include_actor_state"] = include_actor_state
    if smoke:
        cfg = merge_configs(cfg, {"num_steps": 4, "n_start": 4, "n_prompts": 2,
                                 "predictor": {"warmup_steps": 2},
                                 "eval": {"n_problems": 2, "n": 1, "k": 1},
                                 "audit": {"n_problems": 2, "n_prefixes": 1, "n_continuations": 4}})
    if run_wall_seconds is not None:
        cfg["run_wall_seconds"] = run_wall_seconds
    return cfg


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True,
                        help="explicit Qwen3-4B non-Base checkpoint; non-thinking by default; no automatic download")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--eval-data-path", required=True)
    parser.add_argument("--hardware", help="default: RTX5090 three-card for --followup, otherwise single A100")
    parser.add_argument("--config", action="append", default=[], help="shared overrides, e.g. LR/budget")
    parser.add_argument("--arms", nargs="+", choices=list(ARMS))
    parser.add_argument("--followup", action="store_true",
                        help="fast diagnostic: core8 three-way, seed 17, 50 steps, initial/final eval and frozen audit")
    parser.add_argument("--initial-eval", action="store_true", help="evaluate one shared zero-adapter start")
    parser.add_argument("--eval-seed", type=int, default=17, help="shared answer-sampling seed, independent of training seed")
    parser.add_argument("--eval-steps", nargs="+", type=int, help="saved intermediate steps; final is always evaluated")
    parser.add_argument("--seeds", nargs="+", type=int,
                        help="default: 17 for --followup; 17 23 42 otherwise")
    parser.add_argument("--run-dir", default="runs/small-core")
    parser.add_argument("--smoke", action="store_true", help="4 steps; keeps long-context length, not an efficacy test")
    parser.add_argument("--run-wall-seconds", type=float)
    parser.add_argument("--core-basis", choices=["svd", "random"], default="svd")
    parser.add_argument("--without-actor-state", action="store_true", help="ablate z features, keep prefix features")
    parser.add_argument("--audit", action="store_true", help="also run paid repeated-continuation audits")
    parser.add_argument("--no-audit", action="store_true", help="omit follow-up audits")
    parser.add_argument("--execute", action="store_true", help="otherwise save configs and print commands only")
    args = parser.parse_args(argv)
    if args.audit and args.no_audit:
        parser.error("--audit and --no-audit are mutually exclusive")
    if args.eval_steps and any(s <= 0 for s in args.eval_steps):
        parser.error("--eval-steps must be positive")
    audit = (args.audit or args.followup) and not args.no_audit
    initial_eval = args.initial_eval or args.followup
    import yaml
    common = {}
    hardware = args.hardware or ("configs/hardware/rtx5090_rollout_3.yaml" if args.followup else
                                 "configs/hardware/a100_1.yaml")
    paths = [ROOT / "configs/default.yaml", ROOT / hardware,
             ROOT / "configs/experiments/small_core_comparison.yaml"]
    if args.followup:
        paths.append(ROOT / "configs/experiments/small_core_followup.yaml")
    if audit:
        paths.append(ROOT / "configs/experiments/small_core_audit.yaml")
        if args.followup:
            paths.append(ROOT / "configs/experiments/small_core_followup_audit.yaml")
    for path in [*paths, *args.config]:
        common = merge_configs(common, load_config(path))
    common.update(model_path=args.model_path, data_path=args.data_path, eval_data_path=args.eval_data_path)
    if initial_eval:
        if (int((common.get("format_warmup") or {}).get("steps", 0)) or common.get("resume")
                or common.get("init_checkpoint")):
            parser.error("shared initial evaluation requires zero-adapter initialization, without SFT/resume/initial actor")
        common["save_initial_checkpoint"] = True
    root = suite_directory(args.run_dir)
    manifest = {"status": "planned", "smoke_only": args.smoke, "jobs": [],
                "protocol": "pr17_followup" if args.followup else "small_core_comparison",
                "created_at": utc_now(), "eval_seed": args.eval_seed,
                "note": "No experimental results are implied by a generated plan. Full-PG first within each parameterization."}
    manifest_path = root / "suite.json"
    # Avoid a non-Full-PG first GPU job when both are requested.
    arms = sorted(dict.fromkeys(args.arms or (FOLLOWUP_ARMS if args.followup else list(ARMS)[:6])),
                  key=lambda arm: (ARMS[arm][0] or 0, ARMS[arm][1] != "full_pg"))
    initial_pending = initial_eval
    seeds = args.seeds or ([17] if args.followup else [17, 23, 42])
    for seed in dict.fromkeys(seeds):
        for arm in arms:
            dest = root / f"{arm}-seed{seed}"
            config = root / f"{arm}-seed{seed}.yaml"
            cfg = arm_config(common, arm, seed, smoke=args.smoke,
                             run_wall_seconds=args.run_wall_seconds,
                             core_basis=args.core_basis,
                             include_actor_state=not args.without_actor_state)
            atomic_write_text(config, yaml.safe_dump(cfg, sort_keys=False))
            train = [sys.executable, str(ROOT / "scripts/train.py"), "--config", str(config), "--run-dir", str(dest)]
            evaluate = [sys.executable, str(ROOT / "scripts/evaluate.py"), "--generate", "--config", str(config),
                        "--data-path", args.eval_data_path, "--checkpoint", str(dest / "checkpoint.npz"),
                        "--seed", str(args.eval_seed), "--run-dir", str(dest) + "-eval"]
            commands = [("train", train)]
            if initial_pending:
                initial = evaluate.copy()
                initial[initial.index("--checkpoint") + 1] = str(dest / "checkpoints/step_0.npz")
                initial[initial.index("--run-dir") + 1] = str(root / "initial-eval")
                commands.append(("initial_eval", initial))
                initial_pending = False
            eval_steps = args.eval_steps or []  # Intermediate evaluations are opt-in, including follow-up.
            for step in sorted(set(eval_steps)):
                if step >= int(cfg["num_steps"]):
                    continue
                if int(cfg.get("checkpoint_every", 0)) <= 0 or step % int(cfg["checkpoint_every"]):
                    parser.error(f"eval step {step} is not saved by checkpoint_every={cfg.get('checkpoint_every')}")
                intermediate = evaluate.copy()
                intermediate[intermediate.index("--checkpoint") + 1] = str(dest / f"checkpoints/step_{step}.npz")
                intermediate[intermediate.index("--run-dir") + 1] = str(dest) + f"-eval-step{step}"
                commands.append(("eval_snapshot", intermediate))
            commands.append(("eval", evaluate))
            if audit and (not args.followup or ARMS[arm][1] == "grace"):
                commands.append(("audit", [sys.executable, str(ROOT / "scripts/audit.py"), "--generate",
                    "--config", str(config), "--data-path", args.data_path, "--checkpoint", str(dest / "checkpoint.npz"),
                    "--run-dir", str(dest) + "-audit"]))
                if args.followup and ARMS[arm][0] is not None:
                    commands.append(("prediction_report", [sys.executable, str(ROOT / "scripts/analyze_small_core_prediction.py"),
                        "--audit-dir", str(dest) + "-audit"]))
            for phase, command in commands:
                manifest["jobs"].append({"arm": arm, "seed": seed, "phase": phase,
                                         "command": command, "status": "planned", "train_dir": str(dest),
                                         "run_dir": command[command.index("--run-dir") + 1] if "--run-dir" in command else str(dest) + "-audit"})
    atomic_write_text(manifest_path, json.dumps(manifest, indent=2) + "\n")
    for job in manifest["jobs"]:
        print(shlex.join(job["command"]), flush=True)
        if not args.execute:
            continue
        if job["phase"] == "eval_snapshot" and not Path(job["command"][job["command"].index("--checkpoint") + 1]).is_file():
            job.update(status="skipped", reason="checkpoint not produced (e.g. wall-time budget ended before this step)")
            atomic_write_text(manifest_path, json.dumps(manifest, indent=2) + "\n")
            continue
        started = perf_counter()
        job.update(started_at=utc_now(), status="running")
        manifest["status"] = "running"
        atomic_write_text(manifest_path, json.dumps(manifest, indent=2) + "\n")
        result = subprocess.run(job["command"], cwd=ROOT, check=False)
        job.update(returncode=result.returncode, command_wall_seconds=perf_counter() - started,
                   finished_at=utc_now(),
                   status="complete" if result.returncode == 0 else "failed")
        manifest["status"] = "running" if result.returncode == 0 else "failed"
        atomic_write_text(manifest_path, json.dumps(manifest, indent=2) + "\n")
        if result.returncode:
            return result.returncode
    manifest["status"] = "complete" if args.execute else "planned"
    atomic_write_text(manifest_path, json.dumps(manifest, indent=2) + "\n")
    if args.execute:
        from scripts.summarize_small_core_experiment import summarize_suite
        atomic_write_text(root / "comparison_summary.json", json.dumps(summarize_suite(manifest_path), indent=2) + "\n")
    print(f"suite: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
