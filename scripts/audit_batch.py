#!/usr/bin/env python3
"""Frozen fixed-N batch audit: full generation plus simulated HT/CV selection."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grace_gc.audit.batch_audit import run_batch_audit
from grace_gc.data.math_data import load_math_records, records_for_split
from grace_gc.logging_util.run_dir import default_run_dir
from grace_gc.trainer.loop import build_run_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true", help="accepted for consistency with audit.py")
    parser.add_argument("--config", action="append", default=[])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-path", help="training data; defaults to merged configuration")
    parser.add_argument("--model-path")
    parser.add_argument("--backend")
    parser.add_argument("--method")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--split", default="train", help="fixed batch is drawn from this split")
    parser.add_argument("--batch-shape", choices=("configured", "training"), help="match recorded completed training N, or use explicit diagnostic shape")
    parser.add_argument("--training-run", help="source steps.jsonl directory for --batch-shape training")
    parser.add_argument("--n-prompts", type=int, help="explicit diagnostic prompt count")
    parser.add_argument("--starts-per-prompt", type=int, help="explicit diagnostic starts per prompt")
    parser.add_argument("--run-dir")
    args = parser.parse_args(argv)
    overrides = {key: getattr(args, key) for key in ("checkpoint", "data_path", "model_path", "backend", "method", "seed")
                 if getattr(args, key) is not None}
    cfg = build_run_config(args.config, overrides)
    cfg.setdefault("batch_audit", {}).update({key: value for key, value in {
        "shape": args.batch_shape, "training_run": args.training_run,
        "n_prompts": args.n_prompts, "starts_per_prompt": args.starts_per_prompt}.items() if value is not None})
    cfg["batch_audit_split"] = args.split
    if not cfg.get("data_path"):
        raise ValueError("batch audit requires --data-path or configuration data_path")
    records = records_for_split(load_math_records(cfg["data_path"]), args.split, seed=int(cfg.get("split_seed", 17)))
    result = run_batch_audit(records, cfg, args.run_dir or default_run_dir("batch-audit"))
    print("fixed_n", result["fixed_n"], "replicates", result["replicates"])
    print("run_dir", result["run_dir"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
