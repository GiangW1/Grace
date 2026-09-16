#!/usr/bin/env python3
"""Train GRACE or a core baseline. CPU tiny by default; GPU uses --backend gpu_verl."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grace_gc.logging_util.run_dir import default_run_dir
from grace_gc.trainer.loop import build_run_config, run_training


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GRACE-GC training")
    parser.add_argument("--config", action="append", default=[], help="YAML config path, repeatable")
    parser.add_argument("--method", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--backend", default=None, help="cpu_tiny or gpu_verl")
    parser.add_argument("--model-path", dest="model_path", default=None)
    parser.add_argument("--data-path", dest="data_path", default=None)
    parser.add_argument("--run-dir", dest="run_dir", default=None, help="default: runs/train-UTC")
    parser.add_argument("--num-steps", dest="num_steps", type=int, default=None)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args(argv)
    overrides = {}
    if args.method:
        overrides["method"] = args.method
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.backend:
        overrides["backend"] = args.backend
    if args.model_path:
        overrides["model_path"] = args.model_path
    if args.data_path:
        overrides["data_path"] = args.data_path
    if args.num_steps is not None:
        overrides["num_steps"] = args.num_steps
    if args.resume:
        overrides["resume"] = args.resume
    cfg = build_run_config(args.config, overrides)
    run_dir = args.run_dir or str(default_run_dir("train"))
    payload = run_training(cfg, run_dir)
    print(payload["run_status"], payload["summary"])
    print("run_dir", payload.get("run_dir"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
