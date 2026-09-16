#!/usr/bin/env python3
"""Write simple CSV tables for later plotting. No fabricated GPU numbers."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GRACE-GC plot tables")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--out", default="runs/plot_tables")
    args = parser.parse_args(argv)
    path = Path(args.summary)
    if not path.is_file():
        raise FileNotFoundError(f"summary is not readable: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = payload.get("per_problem") or payload.get("summary", {}).get("per_problem") or []
    with (out / "prefix_audit_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=sorted({k for row in rows for k in row.keys()}) or ["empty"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
