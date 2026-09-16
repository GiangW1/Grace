"""Enumerate q/v LoRA A/B parameters from the live module."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


DEFAULT_TARGETS = ("q_proj", "v_proj")
LORA_MARKERS = ("lora_a", "lora_b", "lora_A", "lora_B")


@dataclass
class LayoutEntry:
    name: str
    shape: tuple[int, ...]
    numel: int
    offset: int


@dataclass
class ParamLayout:
    entries: list[LayoutEntry]
    dim: int

    def names(self) -> list[str]:
        return [e.name for e in self.entries]


def _is_target_lora(name: str, targets: tuple[str, ...] = DEFAULT_TARGETS) -> bool:
    lname = name.replace("-", ".")
    if not any(t in lname for t in targets):
        return False
    return any(m in lname for m in LORA_MARKERS)


def collect_lora_layout(named_params, targets: tuple[str, ...] = DEFAULT_TARGETS) -> ParamLayout:
    entries: list[LayoutEntry] = []
    offset = 0
    for name, param in named_params:
        if not _is_target_lora(name, targets):
            continue
        shape = tuple(int(s) for s in param.shape)
        numel = int(np.prod(shape))
        entries.append(LayoutEntry(name=name, shape=shape, numel=numel, offset=offset))
        offset += numel
    if not entries:
        raise ValueError("no q/v LoRA A/B parameters found")
    return ParamLayout(entries=entries, dim=offset)


def pack_grads(named_grads, layout: ParamLayout) -> np.ndarray:
    vec = np.zeros(layout.dim, dtype=np.float64)
    grads = dict(named_grads)
    for entry in layout.entries:
        g = grads.get(entry.name)
        if g is None:
            raise ValueError(f"missing grad for {entry.name}")
        arr = np.asarray(g, dtype=np.float64).reshape(-1)
        if arr.size != entry.numel:
            raise ValueError(f"grad size mismatch for {entry.name}")
        vec[entry.offset : entry.offset + entry.numel] = arr
    return vec


def unpack_to_dict(vec: np.ndarray, layout: ParamLayout) -> dict[str, np.ndarray]:
    vec = np.asarray(vec, dtype=np.float64).reshape(-1)
    if vec.size != layout.dim:
        raise ValueError(f"vector dim {vec.size} != layout dim {layout.dim}")
    out = {}
    for entry in layout.entries:
        out[entry.name] = vec[entry.offset : entry.offset + entry.numel].reshape(entry.shape)
    return out


def layout_hash(layout: ParamLayout) -> str:
    payload = "|".join(f"{e.name}:{e.shape}:{e.offset}" for e in layout.entries)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
