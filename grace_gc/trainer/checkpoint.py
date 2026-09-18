"""Save/restore actor, optimizer, predictor, baseline, basis, reservoir, RNG."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import os
import shutil
import tempfile
import time

import numpy as np


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> dict[str, float | int]:
    """Publish only a complete archive; a failed write preserves the old file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    started, cpu = time.perf_counter(), time.process_time()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, payload=np.array(payload, dtype=object))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {"serialize_write_wall_seconds": time.perf_counter() - started,
            "serialize_write_cpu_seconds": time.process_time() - cpu,
            "checkpoint_bytes": path.stat().st_size}


def copy_checkpoint(source: str | Path, target: str | Path) -> dict[str, float]:
    """Reuse compressed bytes when publishing latest, without a second dump."""
    source, target = Path(source), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    started, cpu = time.perf_counter(), time.process_time()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            with source.open("rb") as original:
                shutil.copyfileobj(original, stream, length=1024 * 1024)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {"latest_copy_wall_seconds": time.perf_counter() - started,
            "latest_copy_cpu_seconds": time.process_time() - cpu}


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint is not readable: {path}")
    with np.load(path, allow_pickle=True) as data:
        payload = data["payload"].item()
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload is not a mapping")
    return payload


def required_keys() -> tuple[str, ...]:
    return (
        "actor",
        "optimizer",
        "predictor",
        "baseline",
        "basis",
        "reservoir",
        "rng",
        "step",
    )
