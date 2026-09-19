"""Save/restore actor, optimizer, predictor, baseline, basis, reservoir, RNG."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import os
import shutil
import tempfile
import time
import zipfile

import numpy as np


def _compact_gradients(payload):
    """Store only exactly representable float64 reservoir G in float32."""
    stats = {"reservoir_g_compacted": 0, "reservoir_g_original_bytes": 0, "reservoir_g_storage_bytes": 0}
    reservoir = payload.get("reservoir")
    if not isinstance(reservoir, dict):
        return payload, stats
    items = []
    for raw in reservoir.get("items", []):
        item = dict(raw)
        g = item.get("g")
        if isinstance(g, np.ndarray):
            stats["reservoir_g_original_bytes"] += g.nbytes
            if g.dtype == np.dtype("float64") and np.isfinite(g).all():
                with np.errstate(over="ignore", invalid="ignore"):
                    compact = g.astype(np.float32)
                    restored = compact.astype(g.dtype)
                if np.array_equal(g, restored) and np.array_equal(np.signbit(g), np.signbit(restored)):
                    item["g"] = compact
                    item["_grace_checkpoint_g_dtype"] = g.dtype.str
                    stats["reservoir_g_compacted"] += 1
            stats["reservoir_g_storage_bytes"] += item["g"].nbytes
        items.append(item)
    if items:
        payload = {**payload, "reservoir": {**reservoir, "items": items}}
    return payload, stats


def _write_archive(stream, payload):
    # A standard NPZ member at the fastest DEFLATE level; no payload format or
    # float precision change is needed to avoid the default level-6 CPU work.
    with zipfile.ZipFile(stream, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        with archive.open("payload.npy", "w", force_zip64=True) as member:
            np.lib.format.write_array(member, np.array(payload, dtype=object), allow_pickle=True)


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> dict[str, float | int]:
    """Publish only a complete archive; a failed write preserves the old file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    started, cpu = time.perf_counter(), time.process_time()
    temporary = None
    try:
        stored, stats = _compact_gradients(payload)
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            _write_archive(stream, stored)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {**stats, "serialize_write_wall_seconds": time.perf_counter() - started,
            "serialize_write_cpu_seconds": time.process_time() - cpu,
            "checkpoint_bytes": path.stat().st_size, "checkpoint_compression_level": 1}


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
    reservoir = payload.get("reservoir")
    for item in reservoir.get("items", []) if isinstance(reservoir, dict) else []:
        dtype = item.pop("_grace_checkpoint_g_dtype", None)
        if dtype is not None:
            item["g"] = np.asarray(item["g"]).astype(dtype)
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
