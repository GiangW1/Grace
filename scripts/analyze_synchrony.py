#!/usr/bin/env python3
"""Analysis paper: update vs answer uncertainty and the HT-completion ceiling.

Reads streaming replays (``replay_expected_gain.py --statistics-only``), one
directory per decision position. CPU only; no model is loaded.
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
from grace_gc.audit.synchrony import analyze


def _fmt(value):
    return "     -" if value is None else f"{value:6.3f}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", nargs="+", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--metric", default="euclidean")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--w-train", type=float, nargs="+", default=[0.5, 1.0, 2.0],
                        help="training pass cost per token relative to one decoded token")
    parser.add_argument("--w-prefill", type=float, default=0.05)
    parser.add_argument("--p-min", type=float, default=0.05)
    args = parser.parse_args(argv)
    out = start_experiment_run(args.run_dir, "synchrony_analysis", vars(args))
    try:
        report = analyze(args.replay_dir, args.metric, args.bootstrap, args.seed,
                         tuple(args.w_train), args.w_prefill, args.p_min)
        (out / "synchrony_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("   t  probs  rho_L  rho_A    gap  alpha  cross  oracle  q-only  (w_train=1 if run)")
        for t, row in report["positions"].items():
            alloc = row["allocation"].get("1.0") or next(iter(row["allocation"].values()), {})
            print(f"{t:>4} {row.get('n_problems', 0):6d} {_fmt(row.get('rho_L'))} {_fmt(row.get('rho_A'))} "
                  f"{_fmt(row.get('gap'))} {_fmt(row.get('alpha_upper'))} {_fmt(row.get('suffix_cross_share'))} "
                  f"{_fmt((alloc.get('oracle_in_sample') or {}).get('ratio'))}  "
                  f"{_fmt((alloc.get('q_only_reward_cv') or {}).get('ratio'))}")
    except Exception as exc:
        finish_experiment_run(out, exc)
        raise
    finish_experiment_run(out)
    print(json.dumps({"run_dir": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
