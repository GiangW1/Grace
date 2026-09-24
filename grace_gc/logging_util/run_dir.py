"""One run directory: config, versions, logs, results."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def atomic_write_text(path: str | Path, text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def default_run_dir(kind: str) -> Path:
    return Path("runs") / f"{kind}-{utc_stamp()}"


def has_run_artifacts(root: str | Path) -> bool:
    root = Path(root)
    if not root.is_dir():
        return False
    names = (
        "summary.json",
        "config.yaml",
        "eval_summary.json",
        "audit_summary.json",
        "trajectories.jsonl",
        "run_meta.json",
        "checkpoints.json",
        "compute_ledger.json",
        "persistence.jsonl",
        "replay_provenance.json",
        "replay_summary.json",
        "mean_grads.npy",
        "expected_gain_summary.json",
        "expected_gain_data_manifest.json",
        "optimizer_probe_summary.json",
    )
    return any((root / name).is_file() for name in names)


def resolve_run_dir(path: str | Path, *, resume: bool = False) -> Path:
    """Keep a new or resumed directory. A second write to the same name gets a UTC suffix."""
    requested = Path(path)
    if resume or not has_run_artifacts(requested):
        return requested
    stamp = utc_stamp()
    stamped = requested.parent / f"{requested.name}-{stamp}"
    extra = 2
    while has_run_artifacts(stamped):
        stamped = requested.parent / f"{requested.name}-{stamp}-{extra}"
        extra += 1
    return stamped


class RunDirectory:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def write_run_meta(self, *, kind: str, started: str, requested: str | Path | None = None) -> Path:
        requested_s = None if requested is None else str(Path(requested))
        return self.write_json(
            "run_meta.json",
            {
                "kind": kind,
                "started": started,
                "run_dir": str(self.root.resolve()),
                "requested_run_dir": requested_s,
                "redirected": requested_s is not None and Path(requested).resolve() != self.root.resolve(),
            },
        )

    def write_json(self, name: str, payload: Any) -> Path:
        path = self.root / name
        atomic_write_text(path, json.dumps(payload, indent=2, default=_json_default))
        return path

    def write_yaml(self, name: str, payload: Any) -> Path:
        path = self.root / name
        atomic_write_text(path, yaml.safe_dump(payload, sort_keys=False))
        return path

    def write_jsonl(self, name: str, rows: list[dict[str, Any]], append: bool = False) -> Path:
        path = self.root / name
        with path.open("a" if append else "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, default=_json_default) + "\n")
        return path

    def append_jsonl(self, name: str, row: dict[str, Any]) -> None:
        path = self.root / name
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=_json_default) + "\n")

    def write_text(self, name: str, text: str) -> Path:
        path = self.root / name
        path.write_text(text, encoding="utf-8")
        return path


def _json_default(obj: Any):
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
