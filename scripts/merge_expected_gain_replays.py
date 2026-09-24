#!/usr/bin/env python3
"""Merge old and new full-gradient replays only when actor and layout agree."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grace_gc.audit.expected_gain import load_replay
from grace_gc.logging_util.run_dir import resolve_run_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", action="append", required=True)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)
    if len(args.replay) < 2:
        raise ValueError("provide at least two disjoint replay directories")
    sources = []
    identity = None
    seen = set()
    for path in args.replay:
        root = Path(path)
        provenance = json.loads((root / "replay_provenance.json").read_text(encoding="utf-8"))
        common = {key: provenance.get(key) for key in
                  ("actor_sha256", "layout_names", "layout_dim", "model_path",
                   "decision_tokens", "max_continuations")}
        if identity is None:
            identity = common
        elif identity != common:
            raise ValueError(f"replay settings or frozen actor differ: {root}")
        rows, means = load_replay(root)
        for row in rows:
            pid = str(row["problem_id"])
            if pid in seen:
                raise ValueError(f"problem {pid} appears in multiple replay inputs")
        seen.update(str(row["problem_id"]) for row in rows)
        sources.append((root, rows, means))
    total = sum(len(rows) for _, rows, _ in sources)
    dim = int(identity["layout_dim"])
    out = resolve_run_dir(args.run_dir)
    out.mkdir(parents=True, exist_ok=True)
    combined = np.lib.format.open_memmap(out / "mean_grads.npy", mode="w+",
                                         dtype=np.float64, shape=(total, dim))
    offset = 0
    with (out / "prefixes.jsonl").open("w", encoding="utf-8") as handle:
        for _root, rows, means in sources:
            if means.shape != (len(rows), dim):
                raise ValueError("replay matrix shape disagrees with provenance")
            for i, row in enumerate(rows):
                combined[offset] = means[i]
                handle.write(json.dumps({**row, "index": offset}, ensure_ascii=False) + "\n")
                offset += 1
            combined.flush()
    del combined
    provenance = {**identity, "merged_replay_dirs": [str(root.resolve()) for root, _, _ in sources],
                  "prompt_features_replayed": all(all(row.get("prompt_features") is not None
                                                      for row in rows) for _, rows, _ in sources)}
    (out / "replay_provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "replay_summary.json").write_text(
        json.dumps({"n_prefixes": total, "n_problems": len(seen), "dimension": dim,
                    "merged_sources": len(sources)}, indent=2), encoding="utf-8")
    print(json.dumps({"run_dir": str(out), "n_prefixes": total, "n_problems": len(seen)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
