"""Collect available software versions. Missing optional metadata is recorded."""

from __future__ import annotations

import hashlib
import platform
import sys
from pathlib import Path
from typing import Any


def collect_versions(
    extra_modules: tuple[str, ...] = ("numpy", "torch", "yaml", "vllm", "verl", "math_verify"),
    files_to_hash: dict[str, str] | None = None,
) -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {},
        "file_hashes": {},
        "missing": [],
    }
    for name in extra_modules:
        try:
            mod = __import__(name)
            info["packages"][name] = getattr(mod, "__version__", "present-no-version")
        except Exception:
            info["packages"][name] = None
            info["missing"].append(name)
    for label, path in (files_to_hash or {}).items():
        p = Path(path)
        if not p.is_file():
            info["file_hashes"][label] = None
            info["missing"].append(label)
            continue
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        info["file_hashes"][label] = digest
    return info
