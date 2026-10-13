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
from grace_gc.logging_util.run_dir import resolve_run_dir


ARMS = {"lora16_full_pg": (None, "full_pg"), "lora16_grace": (None, "grace"),
        "core8_full_pg": (8, "full_pg"), "core8_grace": (8, "grace"),
        "core16_full_pg": (16, "full_pg"), "core16_grace": (16, "grace"),
        "core8_uniform_ht": (8, "uniform_ht"), "core16_uniform_ht": (16, "uniform_ht")}


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
    parser.add_argument("--hardware", default="configs/hardware/a100_1.yaml")
    parser.add_argument("--config", action="append", default=[], help="shared overrides, e.g. LR/budget")
    parser.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS)[:6])
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 23, 42])
    parser.add_argument("--run-dir", default="runs/small-core")
    parser.add_argument("--smoke", action="store_true", help="4 steps; keeps long-context length, not an efficacy test")
    parser.add_argument("--run-wall-seconds", type=float)
    parser.add_argument("--core-basis", choices=["svd", "random"], default="svd")
    parser.add_argument("--without-actor-state", action="store_true", help="ablate z features, keep prefix features")
    parser.add_argument("--audit", action="store_true", help="also run paid repeated-continuation audits")
    parser.add_argument("--execute", action="store_true", help="otherwise save configs and print commands only")
    args = parser.parse_args(argv)
    import yaml
    common = {}
    for path in [ROOT / "configs/default.yaml", ROOT / args.hardware,
                 ROOT / "configs/experiments/small_core_comparison.yaml", *args.config]:
        common = merge_configs(common, load_config(path))
    if args.audit:
        common = merge_configs(common, load_config(ROOT / "configs/experiments/small_core_audit.yaml"))
    common.update(model_path=args.model_path, data_path=args.data_path, eval_data_path=args.eval_data_path)
    root = Path(resolve_run_dir(args.run_dir)).resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest = {"status": "planned", "smoke_only": args.smoke, "jobs": [],
                "note": "No experimental results are implied by a generated plan. Full-PG first within each parameterization."}
    manifest_path = root / "suite.json"
    # Avoid a non-Full-PG first GPU job when both are requested.
    arms = sorted(args.arms, key=lambda arm: (ARMS[arm][0] or 0, ARMS[arm][1] != "full_pg"))
    for seed in args.seeds:
        for arm in arms:
            dest = root / f"{arm}-seed{seed}"
            config = root / f"{arm}-seed{seed}.yaml"
            config.write_text(yaml.safe_dump(arm_config(common, arm, seed, smoke=args.smoke,
                                                      run_wall_seconds=args.run_wall_seconds,
                                                      core_basis=args.core_basis,
                                                      include_actor_state=not args.without_actor_state), sort_keys=False))
            train = [sys.executable, str(ROOT / "scripts/train.py"), "--config", str(config), "--run-dir", str(dest)]
            evaluate = [sys.executable, str(ROOT / "scripts/evaluate.py"), "--generate", "--config", str(config),
                        "--data-path", args.eval_data_path, "--checkpoint", str(dest / "checkpoint.npz"),
                        "--run-dir", str(dest) + "-eval"]
            commands = [("train", train), ("eval", evaluate)]
            if args.audit:
                commands.append(("audit", [sys.executable, str(ROOT / "scripts/audit.py"), "--generate",
                    "--config", str(config), "--data-path", args.data_path, "--checkpoint", str(dest / "checkpoint.npz"),
                    "--run-dir", str(dest) + "-audit"]))
            for phase, command in commands:
                manifest["jobs"].append({"arm": arm, "seed": seed, "phase": phase,
                                         "command": command, "status": "planned"})
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    for job in manifest["jobs"]:
        print(shlex.join(job["command"]), flush=True)
        if not args.execute:
            continue
        started = perf_counter()
        result = subprocess.run(job["command"], cwd=ROOT, check=False)
        job.update(returncode=result.returncode, command_wall_seconds=perf_counter() - started,
                   status="complete" if result.returncode == 0 else "failed")
        manifest["status"] = "running" if result.returncode == 0 else "failed"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        if result.returncode:
            return result.returncode
    manifest["status"] = "complete" if args.execute else "planned"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"suite: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
