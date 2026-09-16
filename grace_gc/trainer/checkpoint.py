"""Save/restore actor, optimizer, predictor, baseline, basis, reservoir, RNG."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, payload=np.array(payload, dtype=object))


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint is not readable: {path}")
    data = np.load(path, allow_pickle=True)
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
