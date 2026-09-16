#!/usr/bin/env python3
"""Independent full-answer evaluation from JSONL answers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grace_gc.evaluation.eval_full import EvalItem, evaluate_items
from grace_gc.logging_util.run_dir import RunDirectory, default_run_dir, resolve_run_dir, utc_now


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GRACE-GC evaluation")
    parser.add_argument("--answers", default=None, help="JSONL with problem_id, gold, answers, truncated")
    parser.add_argument("--data-path", dest="data_path", default=None)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--backend", default=None, help="cpu_tiny or gpu_verl; default keeps YAML")
    parser.add_argument("--config", action="append", default=[])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--model-path", dest="model_path", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--split", default="eval", help="split_records bucket used with --generate")
    parser.add_argument("--run-dir", dest="run_dir", default=None, help="default: runs/eval-UTC")
    args = parser.parse_args(argv)
    if args.generate:
        from grace_gc.data.math_data import load_math_records, records_for_split
        from grace_gc.evaluation.generate import run_eval
        from grace_gc.trainer.loop import build_run_config

        if not args.data_path:
            raise ValueError("--data-path is required with --generate")
        overrides = {"data_path": args.data_path, "eval_split": args.split}
        if args.backend:
            overrides["backend"] = args.backend
        if args.seed is not None:
            overrides["seed"] = args.seed
        if args.model_path:
            overrides["model_path"] = args.model_path
        if args.checkpoint:
            overrides["checkpoint"] = args.checkpoint
        if args.k is not None:
            overrides["eval_k"] = args.k
        cfg = build_run_config(args.config, overrides)
        recs = records_for_split(load_math_records(args.data_path), args.split, seed=int(cfg.get("split_seed", 17)))
        if not recs:
            raise ValueError(f"split {args.split!r} is empty")
        run_dir = args.run_dir or str(default_run_dir("eval"))
        result = run_eval(recs, cfg, run_dir)
        # Paper MATH-500 headline is avg@k, not "any of n correct".
        print(result["n_problems"], result["avg"], result["pass_at_k"], result["parse_rate"], result["truncate_rate"])
        print("run_dir", result.get("run_dir"))
        return 0
    if not args.answers:
        raise ValueError("provide --answers or --generate --data-path")
    path = Path(args.answers)
    if not path.is_file():
        raise FileNotFoundError(f"answers file is not readable: {path}")
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        items.append(
            EvalItem(
                problem_id=str(raw["problem_id"]),
                gold=str(raw["gold"]),
                answers=list(raw["answers"]),
                truncated=list(raw.get("truncated", [False] * len(raw["answers"]))),
            )
        )
    k = args.k
    if k is None and args.config:
        from grace_gc.trainer.loop import build_run_config

        cfg = build_run_config(args.config, {})
        ev = cfg.get("eval") or {}
        k = int(cfg.get("eval_k", ev.get("k", 1)))
    if k is None and items:
        ns = {len(it.answers) for it in items}
        if len(ns) == 1:
            k = int(ns.pop())
    result = evaluate_items(items, k=k or 1)
    requested = args.run_dir or str(default_run_dir("eval"))
    run_dir = resolve_run_dir(requested)
    started = utc_now()
    run = RunDirectory(run_dir)
    run.write_run_meta(kind="eval", started=started, requested=requested)
    result["started"] = started
    result["finished"] = utc_now()
    result["run_dir"] = str(run.root)
    run.write_json("eval_summary.json", result)
    print(result["n_problems"], result["avg"], result["pass_at_k"], result["parse_rate"], result["truncate_rate"])
    print("run_dir", result["run_dir"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
