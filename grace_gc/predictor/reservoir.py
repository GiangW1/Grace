"""Host-side reservoir of audited full-space gradients."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ReservoirItem:
    problem_id: str
    g: np.ndarray
    p: float
    s: float
    features: np.ndarray
    reward: float | None
    basis_id: int
    cost_feat: np.ndarray | None = None
    realized_cost: float | None = None
    reward_feat: np.ndarray | None = None


@dataclass
class GradientReservoir:
    capacity: int
    items: list[ReservoirItem] = field(default_factory=list)
    fit_problem_ids: set[str] | None = None
    hold_problem_ids: set[str] | None = None

    def add(self, item: ReservoirItem) -> None:
        self.items.append(item)
        if len(self.items) > self.capacity:
            self.items = self.items[-self.capacity :]

    def recent_matrix(self, n: int | None = None) -> np.ndarray:
        if not self.items:
            raise ValueError("reservoir is empty")
        take = self.items if n is None else self.items[-n:]
        return np.stack([it.g.reshape(-1) for it in take], axis=0)

    def problem_ids(self) -> list[str]:
        return [it.problem_id for it in self.items]

    def state_dict(self) -> dict:
        return {
            "capacity": self.capacity,
            "fit_problem_ids": None if self.fit_problem_ids is None else sorted(self.fit_problem_ids),
            "hold_problem_ids": None if self.hold_problem_ids is None else sorted(self.hold_problem_ids),
            "items": [
                {
                    "problem_id": it.problem_id,
                    "g": it.g,
                    "p": it.p,
                    "s": it.s,
                    "features": it.features,
                    "reward": it.reward,
                    "basis_id": it.basis_id,
                    "cost_feat": it.cost_feat,
                    "realized_cost": it.realized_cost,
                    "reward_feat": it.reward_feat,
                }
                for it in self.items
            ],
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> GradientReservoir:
        obj = cls(capacity=int(state["capacity"]))
        if state.get("fit_problem_ids") is not None:
            obj.fit_problem_ids = set(state["fit_problem_ids"])
        if state.get("hold_problem_ids") is not None:
            obj.hold_problem_ids = set(state["hold_problem_ids"])
        for raw in state["items"]:
            obj.items.append(
                ReservoirItem(
                    problem_id=raw["problem_id"],
                    g=np.asarray(raw["g"], dtype=np.float64),
                    p=float(raw["p"]),
                    s=float(raw["s"]),
                    features=np.asarray(raw["features"], dtype=np.float64),
                    reward=raw["reward"],
                    basis_id=int(raw["basis_id"]),
                    cost_feat=None if raw.get("cost_feat") is None else np.asarray(raw["cost_feat"], dtype=np.float64),
                    realized_cost=None if raw.get("realized_cost") is None else float(raw["realized_cost"]),
                    reward_feat=None if raw.get("reward_feat") is None else np.asarray(raw["reward_feat"], dtype=np.float64),
                )
            )
        return obj
