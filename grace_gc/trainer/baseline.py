"""History-only EMA baseline. Frozen inside a batch."""

from __future__ import annotations


class HistoricalBaseline:
    def __init__(self, alpha: float = 0.7):
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = float(alpha)
        self.values: dict[str, float] = {}

    def get(self, problem_id: str, default: float = 0.5) -> float:
        return float(self.values.get(problem_id, default))

    def update(self, problem_id: str, reward: float) -> None:
        if reward is None:
            return
        prev = self.values.get(problem_id, 0.5)
        self.values[problem_id] = self.alpha * float(reward) + (1.0 - self.alpha) * prev

    def update_mean(self, problem_id: str, rewards) -> None:
        """One EMA step per problem using this batch's mean, not the last start."""
        vals = [float(r) for r in rewards if r is not None]
        if not vals:
            return
        self.update(problem_id, sum(vals) / len(vals))

    def state_dict(self) -> dict:
        return {"alpha": self.alpha, "values": dict(self.values)}

    def load_state_dict(self, state: dict) -> None:
        self.alpha = float(state["alpha"])
        self.values = {k: float(v) for k, v in state["values"].items()}
