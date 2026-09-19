"""Audited gradient labels, optionally compacted after freezing the basis."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ReservoirItem:
    problem_id: str
    g: np.ndarray | None
    p: float
    s: float
    features: np.ndarray
    reward: float | None
    basis_id: int
    cost_feat: np.ndarray | None = None
    realized_cost: float | None = None
    reward_feat: np.ndarray | None = None
    observed_step: int | None = None
    remaining_cost: float | None = None
    prefix_finished: bool | None = None
    source: str = "training_audit"
    coords: np.ndarray | None = None
    g_norm_sq: float | None = None

    def coordinates(self, u: np.ndarray, basis_id: int | None = None) -> np.ndarray:
        if self.g is not None:
            return np.asarray(self.g, dtype=np.float64).reshape(-1) @ u
        if basis_id is None or self.basis_id != basis_id:
            raise ValueError("compact label requires its fixed basis id")
        if self.coords is None or self.coords.shape != (u.shape[1],):
            raise ValueError("compact label coordinate dimension does not match basis")
        return self.coords

    def gradient_norm_sq(self) -> float:
        if self.g is not None:
            g = np.asarray(self.g, dtype=np.float64).reshape(-1)
            return float(np.dot(g, g))
        if self.g_norm_sq is None or not np.isfinite(self.g_norm_sq) or self.g_norm_sq < 0:
            raise ValueError("compact label requires a finite nonnegative gradient norm")
        return self.g_norm_sq

    def residual(self, f: np.ndarray, u: np.ndarray, basis_id: int | None = None,
                 gram: np.ndarray | None = None) -> float:
        if self.g is not None:
            from grace_gc.predictor.risk import full_space_residual
            return float(full_space_residual(self.g.reshape(-1), f, u))
        coords = self.coordinates(u, basis_id)
        gram = u.T @ u if gram is None else gram
        norm = self.gradient_norm_sq()
        cross, prediction = float(2. * np.dot(f, coords)), float(f @ gram @ f)
        value = norm - cross + prediction
        # The Gram identity includes energy outside U. Only roundoff-sized
        # negative values may be clamped; no statistical floor is introduced.
        tolerance = 64 * np.finfo(np.float64).eps * (norm + abs(cross) + abs(prediction))
        if not np.isfinite(value) or value < -tolerance:
            raise ValueError("compact label has an invalid full-space residual")
        return max(0., value)

    def compact(self, u: np.ndarray, basis_id: int) -> None:
        if self.g is not None:
            self.coords = np.asarray(self.coordinates(u), dtype=np.float64)
            self.g_norm_sq = self.gradient_norm_sq()
            self.basis_id = int(basis_id)
            self.g = None
        self.coordinates(u, basis_id)
        self.gradient_norm_sq()


@dataclass
class GradientReservoir:
    capacity: int
    items: list[ReservoirItem] = field(default_factory=list)
    fit_problem_ids: set[str] | None = None
    hold_problem_ids: set[str] | None = None
    calibration_problem_ids: set[str] | None = None
    fixed_basis_id: int | None = None

    def add(self, item: ReservoirItem, *, u: np.ndarray | None = None) -> None:
        if self.fixed_basis_id is not None:
            if u is None or item.basis_id != self.fixed_basis_id:
                raise ValueError("new compact label requires the fixed basis")
            item.compact(u, self.fixed_basis_id)
        self.items.append(item)
        if len(self.items) > self.capacity:
            self.items = self.items[-self.capacity :]

    def recent_matrix(self, n: int | None = None) -> np.ndarray:
        if not self.items:
            raise ValueError("reservoir is empty")
        take = self.items if n is None else self.items[-n:]
        if any(it.g is None for it in take):
            raise ValueError("compact labels cannot reconstruct full gradients")
        return np.stack([it.g.reshape(-1) for it in take], axis=0)

    def problem_ids(self) -> list[str]:
        return [it.problem_id for it in self.items]

    def stamp_basis_id(self, basis_id: int) -> None:
        bid = int(basis_id)
        if self.fixed_basis_id is not None and bid != self.fixed_basis_id:
            raise ValueError("compact labels cannot change basis")
        for it in self.items:
            it.basis_id = bid

    def compact(self, u: np.ndarray, basis_id: int) -> None:
        if self.fixed_basis_id is not None and self.fixed_basis_id != basis_id:
            raise ValueError("compact labels cannot change basis")
        for item in self.items:
            item.compact(u, basis_id)
        self.fixed_basis_id = int(basis_id)

    def state_dict(self) -> dict:
        return {
            "capacity": self.capacity,
            "fixed_basis_id": self.fixed_basis_id,
            "fit_problem_ids": None if self.fit_problem_ids is None else sorted(self.fit_problem_ids),
            "hold_problem_ids": None if self.hold_problem_ids is None else sorted(self.hold_problem_ids),
            "calibration_problem_ids": None if self.calibration_problem_ids is None else sorted(self.calibration_problem_ids),
            "items": [
                {
                    "problem_id": it.problem_id,
                    "g": it.g,
                    "coords": it.coords,
                    "g_norm_sq": it.g_norm_sq,
                    "p": it.p,
                    "s": it.s,
                    "features": it.features,
                    "reward": it.reward,
                    "basis_id": it.basis_id,
                    "cost_feat": it.cost_feat,
                    "realized_cost": it.realized_cost,
                    "reward_feat": it.reward_feat,
                    "observed_step": it.observed_step,
                    "remaining_cost": it.remaining_cost,
                    "prefix_finished": it.prefix_finished,
                    "source": it.source,
                }
                for it in self.items
            ],
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> GradientReservoir:
        obj = cls(capacity=int(state["capacity"]))
        if state.get("fixed_basis_id") is not None:
            obj.fixed_basis_id = int(state["fixed_basis_id"])
        if state.get("fit_problem_ids") is not None:
            obj.fit_problem_ids = set(state["fit_problem_ids"])
        if state.get("hold_problem_ids") is not None:
            obj.hold_problem_ids = set(state["hold_problem_ids"])
        if state.get("calibration_problem_ids") is not None:
            obj.calibration_problem_ids = set(state["calibration_problem_ids"])
        for raw in state["items"]:
            obj.items.append(
                ReservoirItem(
                    problem_id=raw["problem_id"],
                    g=None if raw["g"] is None else np.asarray(raw["g"], dtype=np.float64),
                    coords=None if raw.get("coords") is None else np.asarray(raw["coords"], dtype=np.float64),
                    g_norm_sq=None if raw.get("g_norm_sq") is None else float(raw["g_norm_sq"]),
                    p=float(raw["p"]),
                    s=float(raw["s"]),
                    features=np.asarray(raw["features"], dtype=np.float64),
                    reward=raw["reward"],
                    basis_id=int(raw["basis_id"]),
                    cost_feat=None if raw.get("cost_feat") is None else np.asarray(raw["cost_feat"], dtype=np.float64),
                    realized_cost=None if raw.get("realized_cost") is None else float(raw["realized_cost"]),
                    reward_feat=None if raw.get("reward_feat") is None else np.asarray(raw["reward_feat"], dtype=np.float64),
                    observed_step=None if raw.get("observed_step") is None else int(raw["observed_step"]),
                    remaining_cost=None if raw.get("remaining_cost") is None else float(raw["remaining_cost"]),
                    prefix_finished=None if raw.get("prefix_finished") is None else bool(raw["prefix_finished"]),
                    source=str(raw.get("source", "training_audit")),
                )
            )
            item = obj.items[-1]
            if item.g is None:
                if (obj.fixed_basis_id is None or item.basis_id != obj.fixed_basis_id
                        or item.coords is None or item.coords.ndim != 1
                        or not np.all(np.isfinite(item.coords))):
                    raise ValueError("invalid compact label or fixed basis id")
                item.gradient_norm_sq()
        return obj


def age_weights(items, current_step=None, max_age_steps=None, age_half_life=None,
                missing_step_policy="exclude") -> tuple[np.ndarray, dict]:
    """Label recency weights; never relabel historical G as current-policy G.

    Age zero means collected at current_step. The window is inclusive; decay
    is 2**(-age/half_life). Missing dates are kept only by explicit policy when
    controls are enabled. With no age controls the legacy sample set is intact.
    """
    if missing_step_policy not in {"exclude", "keep"}:
        raise ValueError("missing_step_policy must be exclude or keep")
    if max_age_steps is not None and (not np.isfinite(max_age_steps) or max_age_steps < 0):
        raise ValueError("max_age_steps must be finite and nonnegative")
    if age_half_life is not None and (not np.isfinite(age_half_life) or age_half_life <= 0):
        raise ValueError("age_half_life must be finite and positive")
    enabled = max_age_steps is not None or age_half_life is not None
    weights = np.ones(len(items), dtype=np.float64)
    missing = expired = future = 0
    ages = []
    sources = {}
    for i, item in enumerate(items):
        sources[item.source] = sources.get(item.source, 0) + 1
        age = None if current_step is None or item.observed_step is None else int(current_step) - int(item.observed_step)
        if age is None:
            missing += 1
            if enabled and missing_step_policy == "exclude":
                weights[i] = 0.
        elif age < 0:
            # Rewound or malformed history cannot be called fresh supervision.
            future += 1
            if enabled:
                weights[i] = 0.
        else:
            ages.append(age)
            if max_age_steps is not None and age > max_age_steps:
                weights[i] = 0.
                expired += 1
            elif age_half_life is not None:
                weights[i] = np.exp2(-age / float(age_half_life))
    return weights, {"enabled": enabled, "current_step": current_step,
                     "max_age_steps": max_age_steps, "age_half_life": age_half_life,
                     "missing_step_policy": missing_step_policy, "missing_step_n": missing,
                     "future_step_n": future, "expired_n": expired,
                     "active_n": int(np.count_nonzero(weights)), "sources": sources,
                     "min_known_age": min(ages) if ages else None,
                     "max_known_age": max(ages) if ages else None,
                     "parameter_memory": "age policy weights labels; existing head parameters retain prior training"}
