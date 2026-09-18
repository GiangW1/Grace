"""History-only EMA or explicit fixed baseline. Frozen inside a batch."""

from __future__ import annotations

import math


class HistoricalBaseline:
    def __init__(self, alpha: float = 0.7, mode: str = "ema", fixed_value: float = 0.5,
                 prescan_prior_strength: float = 0.0, prescan_prior_mean: float = 0.5):
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        if mode not in {"ema", "fixed"}:
            raise ValueError(f"unknown baseline mode: {mode}")
        if not math.isfinite(fixed_value):
            raise ValueError("fixed baseline must be finite")
        if not math.isfinite(prescan_prior_strength) or prescan_prior_strength < 0:
            raise ValueError("prescan prior strength must be finite and nonnegative")
        if not math.isfinite(prescan_prior_mean) or not 0 <= prescan_prior_mean <= 1:
            raise ValueError("prescan prior mean must be in [0, 1]")
        self.alpha = float(alpha)
        self.mode = mode
        self.fixed_value = float(fixed_value)
        self.prescan_prior_strength = float(prescan_prior_strength)
        self.prescan_prior_mean = float(prescan_prior_mean)
        self.values: dict[str, float] = {}

    @property
    def uses_prescan(self) -> bool:
        return self.mode == "ema"

    def get(self, problem_id: str, default: float = 0.5) -> float:
        if self.mode == "fixed":
            return self.fixed_value
        return float(self.values.get(problem_id, default))

    def prescan_estimate(self, rewards) -> float:
        """Read-only estimate from independent completed prescan rewards.

        The optional prior applies only here, never to an HT history update.
        Missing observations are not converted to failures.
        """
        if self.mode == "fixed":
            return self.fixed_value
        vals = [float(r) for r in rewards if r is not None]
        denominator = len(vals) + self.prescan_prior_strength
        if denominator == 0:
            return 0.5
        return (sum(vals) + self.prescan_prior_strength * self.prescan_prior_mean) / denominator

    def initialize_from_prescan(self, problem_id: str, rewards) -> float:
        """Initialize an unseen problem without replacing existing history."""
        if self.mode == "fixed" or problem_id in self.values:
            return self.get(problem_id)
        vals = [float(r) for r in rewards if r is not None]
        estimate = self.prescan_estimate(vals)
        if vals:
            self.values[problem_id] = estimate
        return estimate

    def update(self, problem_id: str, reward: float) -> None:
        if self.mode == "fixed" or reward is None:
            return
        prev = self.values.get(problem_id, 0.5)
        self.values[problem_id] = self.alpha * float(reward) + (1.0 - self.alpha) * prev

    def update_mean(self, problem_id: str, rewards) -> None:
        """One EMA step per problem using this batch's mean, not the last start."""
        vals = [float(r) for r in rewards if r is not None]
        if not vals:
            return
        self.update(problem_id, sum(vals) / len(vals))

    def update_ht_mean(self, problem_id: str, rewards, probabilities) -> None:
        """History for the next batch: fixed-start HT mean, including stoppers.

        Missing rewards stay missing. Their observed HT contribution is zero;
        the denominator remains the number of starts, not the completions.
        """
        rewards, probabilities = list(rewards), list(probabilities)
        if len(rewards) != len(probabilities):
            raise ValueError("baseline rewards/probabilities length mismatch")
        if not rewards:
            return
        if any(not math.isfinite(float(p)) or not 0 < float(p) <= 1 for p in probabilities):
            raise ValueError("baseline inclusion probabilities must be in (0, 1]")
        estimate = sum(float(r) / float(p) for r, p in zip(rewards, probabilities) if r is not None) / len(rewards)
        self.update(problem_id, estimate)

    def configuration(self) -> dict:
        return {"ema_alpha": self.alpha, "mode": self.mode, "fixed_value": self.fixed_value,
                "prescan_prior_strength": self.prescan_prior_strength,
                "prescan_prior_mean": self.prescan_prior_mean}

    def state_dict(self) -> dict:
        config = self.configuration()
        config["alpha"] = config.pop("ema_alpha")
        return {**config, "values": dict(self.values)}

    def load_state_dict(self, state: dict) -> None:
        self.__init__(alpha=float(state["alpha"]), mode=state.get("mode", "ema"),
                      fixed_value=float(state.get("fixed_value", 0.5)),
                      prescan_prior_strength=float(state.get("prescan_prior_strength", 0.0)),
                      prescan_prior_mean=float(state.get("prescan_prior_mean", 0.5)))
        self.values = {k: float(v) for k, v in state["values"].items()}


def baseline_from_config(config: dict | None = None) -> HistoricalBaseline:
    config = config or {}
    return HistoricalBaseline(alpha=float(config.get("ema_alpha", 0.7)), mode=config.get("mode", "ema"),
                              fixed_value=float(config.get("fixed_value", 0.5)),
                              prescan_prior_strength=float(config.get("prescan_prior_strength", 0.0)),
                              prescan_prior_mean=float(config.get("prescan_prior_mean", 0.5)))
