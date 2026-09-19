"""Save/restore actor, optimizer, predictor, baseline, basis, reservoir, RNG."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import os
import re
import shutil
import tempfile
import time
import zipfile

import numpy as np

from grace_gc.versions import sha256_array


def _basis_digest(name):
    match = re.fullmatch(r"basis-([0-9a-f]{64})\.npy", str(name))
    if match is None:
        raise ValueError(f"invalid basis artifact basename: {name}")
    return match.group(1)


def _load_basis_artifact(directory, reference):
    name = reference["path"]
    digest = _basis_digest(name)
    path = directory / name
    if digest != reference["sha256"]:
        raise ValueError(f"basis artifact reference hash mismatch: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"basis artifact is not readable: {path}")
    try:
        u = np.load(path, allow_pickle=False)
    except (OSError, ValueError, EOFError) as exc:
        raise ValueError(f"basis artifact is not a valid NPY array: {path}") from exc
    if not isinstance(u, np.ndarray):
        u.close()
        raise ValueError(f"basis artifact is not an NPY array: {path}")
    if sha256_array(u) != digest:
        raise ValueError(f"basis artifact content hash mismatch: {path}")
    if ("dtype" in reference and u.dtype.str != reference["dtype"]
            or "shape" in reference and list(u.shape) != reference["shape"]):
        raise ValueError(f"basis artifact dtype or shape mismatch: {path}")
    return u


def _publish_basis_artifact(directory, reference, write):
    """Publish an immutable NPY once, validating any existing same-name file."""
    path = directory / reference["path"]
    if path.exists():
        _load_basis_artifact(directory, reference)
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        # A hard-link publishes the complete bytes atomically without replacing
        # another writer's immutable artifact if it appeared in the meantime.
        try:
            os.link(temporary, path)
        except FileExistsError:
            _load_basis_artifact(directory, reference)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _externalize_fixed_basis(directory, payload):
    basis = payload.get("basis")
    if not isinstance(basis, dict) or not basis.get("fixed", False):
        return payload, None
    u = np.asarray(basis["u"])
    digest = sha256_array(u)
    reference = {"path": f"basis-{digest}.npy", "sha256": digest,
                 "dtype": u.dtype.str, "shape": list(u.shape)}
    _publish_basis_artifact(directory, reference, lambda stream: np.save(stream, u, allow_pickle=False))
    stored_basis = {key: value for key, value in basis.items() if key != "u"}
    stored_basis["u_artifact"] = reference
    return {**payload, "basis": stored_basis}, reference["path"]


def _copy_basis_artifact(source, target, name):
    reference = {"path": name, "sha256": _basis_digest(name)}
    u = _load_basis_artifact(source, reference)
    if source.resolve() == target.resolve():
        return
    reference.update(dtype=u.dtype.str, shape=list(u.shape))
    del u

    def write(stream):
        with (source / name).open("rb") as original:
            shutil.copyfileobj(original, stream, length=1024 * 1024)

    _publish_basis_artifact(target, reference, write)


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


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> dict[str, float | int | str]:
    """Publish only a complete archive; a failed write preserves the old file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    started, cpu = time.perf_counter(), time.process_time()
    temporary = None
    try:
        stored, stats = _compact_gradients(payload)
        stored = dict(stored)
        offline = stored.pop("_offline_predictor_source", None)
        if offline is not None:
            from grace_gc.predictor.offline import copy_predictor
            reference = copy_predictor(offline, path.parent)
            stored = {**stored, "predictor": None,
                      "basis": {key: value for key, value in stored["basis"].items() if key != "u"},
                      "offline_predictor": reference}
            stats["offline_predictor"] = reference
            basis_artifact = None
        else:
            stored, basis_artifact = _externalize_fixed_basis(path.parent, stored)
        if basis_artifact is not None:
            stats["basis_artifact"] = basis_artifact
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


def copy_checkpoint(source: str | Path, target: str | Path, *, basis_artifact: str | None = None,
                    offline_predictor: dict | None = None) -> dict[str, float]:
    """Reuse compressed bytes when publishing latest, without a second dump."""
    source, target = Path(source), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    started, cpu = time.perf_counter(), time.process_time()
    temporary = None
    try:
        if basis_artifact is not None:
            _copy_basis_artifact(source.parent, target.parent, basis_artifact)
        if offline_predictor is not None:
            from grace_gc.predictor.offline import copy_predictor
            copy_predictor({**offline_predictor, "path": str(source.parent/offline_predictor["path"])}, target.parent)
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
    basis = payload.get("basis")
    if payload.get("offline_predictor") is not None:
        from grace_gc.predictor.offline import read_predictor
        reference = payload["offline_predictor"]
        if Path(reference["path"]).name != reference["path"]:
            raise ValueError("offline predictor reference must be a basename")
        source = path.parent/reference["path"]
        body = read_predictor(source, reference["sha256"])
        payload["predictor"] = body["predictor"]
        basis["u"] = body["u"]
        payload["_offline_predictor_source"] = {**reference, "path": str(source.resolve())}
    if isinstance(basis, dict) and "u_artifact" in basis:
        basis["u"] = _load_basis_artifact(path.parent, basis["u_artifact"])
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
