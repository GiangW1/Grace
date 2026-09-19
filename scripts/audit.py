#!/usr/bin/env python3
"""Prefix audit from stored grads/rewards. Analysis filters are options."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grace_gc.audit.prefix_audit import audit_bundles, bundle_from_dict
from grace_gc.logging_util.run_dir import RunDirectory, default_run_dir, resolve_run_dir, utc_now
from grace_gc.trainer.loop import build_run_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GRACE-GC prefix audit")
    parser.add_argument("--bundles", default=None, help="JSONL with problem_id,t,rewards,grads")
    parser.add_argument("--data-path", dest="data_path", default=None)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--config", action="append", default=[])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--model-path", dest="model_path", default=None)
    parser.add_argument("--backend", default=None, help="cpu_tiny or gpu_verl; default keeps YAML")
    parser.add_argument("--method", default=None)
    parser.add_argument("--split", default="audit", help="split_records bucket used with --generate")
    parser.add_argument("--run-dir", dest="run_dir", default=None, help="default: runs/audit-UTC")
    args = parser.parse_args(argv)
    overrides = {}
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.checkpoint:
        overrides["checkpoint"] = args.checkpoint
    if args.model_path:
        overrides["model_path"] = args.model_path
    if args.data_path:
        overrides["data_path"] = args.data_path
    if args.backend:
        overrides["backend"] = args.backend
    if args.method:
        overrides["method"] = args.method
    cfg = build_run_config(args.config, overrides)
    if args.generate:
        from grace_gc.audit.run import run_audit
        from grace_gc.data.math_data import load_math_records, records_for_split

        if not args.data_path:
            raise ValueError("--data-path is required with --generate")
        recs = records_for_split(load_math_records(args.data_path), args.split, seed=int(cfg.get("split_seed", 17)))
        if not recs:
            raise ValueError(f"split {args.split!r} is empty")
        run_dir = args.run_dir or str(default_run_dir("audit"))
        result = run_audit(recs, cfg, run_dir)
        print(result)
        print("run_dir", result.get("run_dir"))
        return 0
    if not args.bundles:
        raise ValueError("provide --bundles or --generate --data-path")
    path = Path(args.bundles)
    if not path.is_file():
        raise FileNotFoundError(f"bundle file is not readable: {path}")
    bundles = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        bundles.append(bundle_from_dict(raw))
    if not bundles:
        requested = args.run_dir or str(default_run_dir("audit"))
        run_dir = resolve_run_dir(requested)
        started = utc_now()
        run = RunDirectory(run_dir)
        run.write_run_meta(kind="audit", started=started, requested=requested)
        result = {
            "n_bundles": 0,
            "note": "no prefixes",
            "started": started,
            "finished": utc_now(),
            "run_dir": str(run.root),
        }
        run.write_json("audit_summary.json", result)
        print(result)
        print("run_dir", result["run_dir"])
        return 0
    from grace_gc.audit.run import _audit_u

    u = _audit_u(bundles, cfg)
    rng = np.random.default_rng(int(cfg.get("seed", 17)))
    analysis = dict(cfg.get("analysis", {}))
    alloc = cfg.get("allocation", {})
    analysis.setdefault("beta", alloc.get("beta", 0.5))
    analysis.setdefault("p_min", alloc.get("p_min", 0.2))
    analysis.setdefault("uniform_shrink", alloc.get("uniform_shrink", 0.0))
    analysis.setdefault("control_variate", (cfg.get("predictor") or {}).get("control_variate", True))
    analysis.setdefault("method", cfg.get("method", "grace"))
    if cfg.get("max_new_tokens") is not None:
        analysis.setdefault("max_new_tokens", int(cfg["max_new_tokens"]))
    result = audit_bundles(bundles, u, analysis, rng)
    requested = args.run_dir or str(default_run_dir("audit"))
    run_dir = resolve_run_dir(requested)
    started = utc_now()
    run = RunDirectory(run_dir)
    run.write_run_meta(kind="audit", started=started, requested=requested)
    result["started"] = started
    result["finished"] = utc_now()
    result["run_dir"] = str(run.root)
    run.write_json("audit_summary.json", result)
    print(result)
    print("run_dir", result["run_dir"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
