"""One run directory: config, versions, logs, results."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


class RunDirectory:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def write_json(self, name: str, payload: Any) -> Path:
        path = self.root / name
        path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
        return path

    def write_yaml(self, name: str, payload: Any) -> Path:
        path = self.root / name
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
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
