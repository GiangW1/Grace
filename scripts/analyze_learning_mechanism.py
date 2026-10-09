#!/usr/bin/env python3
"""CPU Gamma/Lambda audit of PR12 arrays or streaming scalar replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grace_gc.audit.expected_gain import start_experiment_run, finish_experiment_run
from grace_gc.audit.learning_stats import (grouped_reports, load_legacy_trajectory,
                                           load_scalar_trajectory, mechanism_values)
from grace_gc.versions import sha256_file


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--metric-file")
    parser.add_argument("--metric-name", default="euclidean")
    parser.add_argument("--difficulty-manifest")
    parser.add_argument("--split-manifest")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)
    if args.bootstrap < 0:
        raise ValueError("bootstrap must be nonnegative")
    out = start_experiment_run(args.run_dir, "learning_mechanism", vars(args))
    try:
        root = Path(args.replay_dir)
        scalar = (root / "trajectory_scalars.jsonl").is_file()
        if scalar:
            rows = load_scalar_trajectory(root, args.metric_name)
            name = args.metric_name
        else:
            rows, name = load_legacy_trajectory(root, args.metric_file, args.metric_name)
        report = grouped_reports(rows, name, args.difficulty_manifest, args.bootstrap, args.seed)
        report["metric_name"] = name
        report["source_format"] = "streaming_scalars" if scalar else "PR12_trajectory_arrays"
        report["assumptions"] = "binary rewards, fixed b and metric, matching full-softmax sampling/score policy"
        report["null_definition"] = "observed suffix second moment with mean coupling removed; not independently permuted reward"
        report["strict_claim"] = "rho_L < rho_A before answer emission and reliable functional recovery"
        report["difficulty_manifest_sha256"] = (None if not args.difficulty_manifest else sha256_file(args.difficulty_manifest))
        if args.split_manifest:
            split = json.loads(Path(args.split_manifest).read_text(encoding="utf-8"))
            seen = set()
            for ids in split.values():
                if seen.intersection(map(str, ids)):
                    raise ValueError("problem split roles overlap")
                seen.update(map(str, ids))
            if seen != {str(row["problem_id"]) for row in rows}:
                raise ValueError("split must cover every replay problem")
            report["roles"] = {role: grouped_reports(
                [row for row in rows if str(row["problem_id"]) in set(map(str, ids))], name,
                args.difficulty_manifest, args.bootstrap, args.seed,
            ) for role, ids in split.items()}
        details = [{"problem_id": row["problem_id"], "path_id": row.get("path_id"), "t": row["t"],
                    **mechanism_values(row, row["mechanism"][name])} for row in rows]
        (out / "prefix_mechanism.jsonl").write_text(
            "".join(json.dumps(row, allow_nan=False) + "\n" for row in details), encoding="utf-8")
        (out / "mechanism_summary.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        finish_experiment_run(out)
    except BaseException as exc:
        finish_experiment_run(out, exc)
        raise
    print(json.dumps({"run_dir": str(out), "all": report["groups"]["all"]}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
