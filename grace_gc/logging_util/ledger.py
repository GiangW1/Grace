"""Deployment ledger: GPU reserved wall-clock, separate CPU time."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class ComputeLedger:
    n_gpu: int
    hardware: str
    rows: list[dict] = field(default_factory=list)
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def add(
        self,
        name: str,
        wall_s: float,
        cpu_s: float | None = None,
        overlap_excluded: bool = True,
        **extra,
    ) -> None:
        gpu_reserved = float(wall_s) * max(int(self.n_gpu), 0)
        row = {
            "name": name,
            "hardware": self.hardware,
            "wall_seconds": float(wall_s),
            "gpu_reserved_seconds": gpu_reserved,
            "cpu_seconds": None if cpu_s is None else float(cpu_s),
            "cpu_scope": "current_process_excludes_workers",
            "session_id": self.session_id,
            "overlap_excluded": overlap_excluded,
            "at": datetime.now(timezone.utc).isoformat(),
        }
        row.update(extra)
        self.rows.append(row)

    def exclusive_rows(self) -> list[dict]:
        """Skip nested phase/step rows when an envelope already covers them."""
        envelopes = {"train", "eval", "audit"}
        covered = {r.get("session_id") for r in self.rows if str(r["name"]) in envelopes}
        out = []
        for row in self.rows:
            name = str(row["name"])
            if name.startswith("phase_"):
                continue
            if row.get("session_id") in covered and name not in envelopes:
                continue
            out.append(row)
        if out:
            return out
        return [r for r in self.rows if not str(r["name"]).startswith("phase_")]

    def summary(self) -> dict:
        rows = self.exclusive_rows()
        cpu_values = [r.get("cpu_seconds") for r in rows]
        return {
            "hardware": self.hardware,
            "n_gpu": self.n_gpu,
            "gpu_reserved_seconds": sum(r["gpu_reserved_seconds"] for r in rows),
            "wall_seconds": sum(r["wall_seconds"] for r in rows),
            "cpu_seconds": sum(cpu_values) if all(value is not None for value in cpu_values) else None,
            "cpu_scope": "current_process_excludes_workers",
            "sessions": len({r.get("session_id") for r in rows}),
            "note": (
                "A100 and 5090 hours are recorded separately and never converted. "
                "Totals skip nested phases and steps only within the same session envelope. "
                "CPU seconds are null when unmeasured and exclude worker processes."
            ),
            "rows": list(self.rows),
        }


class Timer:
    def __init__(self):
        self.t0 = time.perf_counter()
        self.mark = self.t0
        self.cpu0 = time.process_time()

    def elapsed(self) -> float:
        return time.perf_counter() - self.t0

    def cpu_elapsed(self) -> float:
        return time.process_time() - self.cpu0

    def lap(self) -> float:
        now = time.perf_counter()
        dt = now - self.mark
        self.mark = now
        return dt
