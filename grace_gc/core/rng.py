"""Isolated RNG streams for tokens, continuation, selection, and audit."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


STREAMS = ("token", "continuation", "selection", "audit", "eval", "predictor")


def seed_all(seed: int) -> None:
    """Recorded run seed also controls numpy/torch module initialization."""
    np.random.seed(int(seed))
    try:
        import torch

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except Exception:
        pass


@dataclass
class IsolatedRNG:
    seed: int
    streams: dict[str, np.random.Generator]
    counters: dict[str, int]

    @classmethod
    def create(cls, seed: int) -> IsolatedRNG:
        streams = {}
        counters = {}
        ss = np.random.SeedSequence(int(seed))
        children = ss.spawn(len(STREAMS))
        for name, child in zip(STREAMS, children):
            streams[name] = np.random.default_rng(child)
            counters[name] = 0
        return cls(seed=int(seed), streams=streams, counters=counters)

    def generator(self, name: str) -> np.random.Generator:
        if name not in self.streams:
            raise KeyError(f"unknown RNG stream: {name}")
        self.counters[name] = self.counters.get(name, 0) + 1
        return self.streams[name]

    def random(self, name: str, size: int | tuple[int, ...] | None = None) -> np.ndarray:
        self.counters[name] = self.counters.get(name, 0) + 1
        return self.streams[name].random(size)

    def integers(self, name: str, low: int, high: int | None = None, size=None) -> np.ndarray:
        self.counters[name] = self.counters.get(name, 0) + 1
        return self.streams[name].integers(low, high, size=size)

    def bernoulli(self, name: str, p: np.ndarray) -> np.ndarray:
        self.counters[name] = self.counters.get(name, 0) + 1
        p = np.asarray(p, dtype=np.float64)
        if np.any(p < 0.0) or np.any(p > 1.0):
            raise ValueError("p must be in [0, 1]")
        return (self.streams[name].random(p.shape) < p).astype(np.float64)

    def state_dict(self) -> dict:
        return {
            "seed": self.seed,
            "counters": dict(self.counters),
            "bit_generators": {k: v.bit_generator.state for k, v in self.streams.items()},
        }

    def load_state_dict(self, state: dict) -> None:
        bit = state.get("bit_generators") or {}
        missing = [name for name in STREAMS if name not in bit]
        if missing:
            raise ValueError(f"checkpoint RNG missing streams {missing}")
        self.seed = int(state["seed"])
        loaded = {k: int(v) for k, v in state["counters"].items()}
        for name in STREAMS:
            self.counters[name] = loaded.get(name, self.counters.get(name, 0))
        for name, bg_state in bit.items():
            if name in self.streams:
                self.streams[name].bit_generator.state = bg_state
