"""Measure complete subprocess lifetimes; keep these separate from inner ledgers."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time


def timing_summary(rows):
    spans = sorted((r["started_monotonic"], r["finished_monotonic"]) for r in rows)
    covered = 0.0
    end = None
    for start, stop in spans:
        covered += max(0., stop - max(start, end if end is not None else start))
        end = max(end if end is not None else stop, stop)
    envelope = max(stop for _, stop in spans) - spans[0][0] if spans else 0.0
    return {"n_commands": len(rows), "command_wall_seconds": sum(r["wall_seconds"] for r in rows),
            "observed_envelope_seconds": envelope, "unattributed_gap_seconds": max(0., envelope - covered),
            "scope": "first measured child launch through last child exit; excludes work before/after this envelope",
            "note": "Includes subprocess startup, data loading and shutdown. Gaps are unclassified orchestration/idle time, not measured GPU compute. Do not add inner ledger totals."}


def measure_command(command, log, stage):
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    start = time.monotonic()
    code, error = None, None
    try:
        code = subprocess.run(command, check=False).returncode
        return code
    except BaseException as exc:
        error = type(exc).__name__
        raise
    finally:
        end = time.monotonic()
        row = {"stage": stage, "command": command, "started": started,
               "finished": datetime.now(timezone.utc).isoformat(),
               "started_monotonic": start, "finished_monotonic": end,
               "wall_seconds": end - start, "exit_code": code, "error_type": error}
        with log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
        path = log.with_suffix(".summary.json")
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(timing_summary(rows), indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--stage", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a subprocess command is required")
    raise SystemExit(measure_command(command, args.log, args.stage))
