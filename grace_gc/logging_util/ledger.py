"""Deployment ledger: GPU reserved wall-clock, separate CPU time."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class ComputeLedger:
    n_gpu: int
    hardware: str
    rows: list[dict] = field(default_factory=list)

    def add(
        self,
        name: str,
        wall_s: float,
        cpu_s: float = 0.0,
        overlap_excluded: bool = True,
        **extra,
    ) -> None:
        gpu_reserved = float(wall_s) * max(int(self.n_gpu), 0)
        row = {
            "name": name,
            "hardware": self.hardware,
            "wall_seconds": float(wall_s),
            "gpu_reserved_seconds": gpu_reserved,
            "cpu_seconds": float(cpu_s),
            "overlap_excluded": overlap_excluded,
            "at": datetime.now(timezone.utc).isoformat(),
        }
        row.update(extra)
        self.rows.append(row)

    def summary(self) -> dict:
        return {
            "hardware": self.hardware,
            "n_gpu": self.n_gpu,
            "gpu_reserved_seconds": sum(r["gpu_reserved_seconds"] for r in self.rows),
            "cpu_seconds": sum(r["cpu_seconds"] for r in self.rows),
            "note": "A100 and 5090 hours are recorded separately and never converted.",
            "rows": list(self.rows),
        }


class Timer:
    def __init__(self):
        self.t0 = time.perf_counter()
        self.mark = self.t0

    def elapsed(self) -> float:
        return time.perf_counter() - self.t0

    def lap(self) -> float:
        now = time.perf_counter()
        dt = now - self.mark
        self.mark = now
        return dt
