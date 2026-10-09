#!/usr/bin/env python3
"""Experiment A: population learning completion F(t) versus answer resolution rho_A(t).

Reads trajectory replays written with --store-half-means
--store-trajectory-decomposition (one directory per shard or per decision
position).  CPU only; no model is loaded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grace_gc.audit.expected_gain import finish_experiment_run, start_experiment_run
from grace_gc.audit.learning_completion import (analyze, load_probe_rescores, load_replay_rows,
                                                position_grams)
from grace_gc.versions import sha256_file


def _format(value):
    return "    -" if value is None else f"{value:7.3f}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", nargs="+", required=True,
                        help="trajectory replay directories; rows must not repeat across them")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--metric-file", default=None,
                        help="fixed diagonal metric for all inner products; Euclidean when omitted")
    parser.add_argument("--metric-name", default=None)
    parser.add_argument("--probe-rescores", default=None,
                        help="JSONL with problem_id, path_id, t, functional_recoverable")
    parser.add_argument("--gram-cache", default=None,
                        help="directory for per-position Gram matrices; reused when inputs match")
    parser.add_argument("--block-mb", type=float, default=1024.0,
                        help="memory for one coordinate block of stacked vectors")
    parser.add_argument("--batch-problems", type=int, nargs="+", default=[1, 8, 32, 128])
    parser.add_argument("--epsilon", type=float, default=None,
                        help="pre-registered learning threshold: F_pop >= 1-eps or residual <= eps")
    parser.add_argument("--delta", type=float, default=None,
                        help="pre-registered answer threshold: rho_A <= delta")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)
    if args.bootstrap < 0 or args.block_mb <= 0 or any(m < 1 for m in args.batch_problems):
        raise ValueError("bootstrap must be nonnegative; block-mb and batch sizes must be positive")
    if (args.epsilon is None) != (args.delta is None):
        raise ValueError("pass --epsilon and --delta together")
    out = start_experiment_run(args.run_dir, "learning_completion", vars(args))
    try:
        rows, sources = load_replay_rows(args.replay_dir)
        grams = position_grams(rows, sources, args.metric_file, args.gram_cache, args.block_mb,
                               log=lambda message: print(message, flush=True))
        probe = None if args.probe_rescores is None else load_probe_rescores(args.probe_rescores)
        report = analyze(rows, grams, probe, args.batch_problems, args.bootstrap, args.seed,
                         args.epsilon, args.delta)
        report["inputs"] = {
            "replay_dirs": [str(source["root"].resolve()) for source in sources],
            "prefixes_sha256": [source["prefixes_sha256"] for source in sources],
            "metric_name": args.metric_name or ("euclidean" if args.metric_file is None else "diagonal_file"),
            "metric_file_sha256": None if args.metric_file is None else sha256_file(args.metric_file),
            "probe_rescores_sha256": (None if args.probe_rescores is None
                                      else sha256_file(args.probe_rescores)),
        }
        report["assumptions"] = ("binary rewards; baseline fixed per prefix; suffixes sampled from the "
                                 "policy whose scores are stored; finished prefixes are not replayed")
        (out / "learning_completion_summary.json").write_text(
            json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        finish_experiment_run(out)
    except BaseException as exc:
        finish_experiment_run(out, exc)
        raise
    print(f"run_dir={out}")
    for group, entry in report["groups"].items():
        if not entry.get("available"):
            continue
        print(f"[{group}] view={entry['view']}")
        print("      t  probs   F_pop  resid_L   rho_A     gap  F_prefix")
        for row in entry["by_t"]:
            values = row.get("estimates") or {}
            print(f"{row['t']:7d} {row['n_problems']:6d} " + " ".join(
                _format(values.get(key)) for key in
                ("F_pop", "residual_pop", "rho_A", "gap", "F_prefix")))
        if "crossings" in entry:
            crossing = entry["crossings"]
            print(f"  t_L_projection={crossing['t_L_projection']} "
                  f"t_L_residual={crossing['t_L_residual']} t_A={crossing['t_A']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
