"""Mechanism-track methods share GRACE's target. GRPO keeps its own."""

from __future__ import annotations

from dataclasses import dataclass


METHOD_NAMES = (
    "full_pg",
    "uniform_ht",
    "uniform_cv",
    "reward_cv",
    "prompt_cv",
    "grace",
    "grpo",
    "grpo_short",
)


@dataclass(frozen=True)
class MethodSpec:
    name: str
    track: str
    use_predictor: bool
    use_allocation: bool
    use_ht_correction: bool
    feature_mode: str
    risk_mode: str
    objective: str
    max_new_tokens: int | None = None
    n_start: int | None = None
    starts_per_prompt: int | None = None


def method_spec(name: str, **overrides) -> MethodSpec:
    key = name.replace("-", "_").lower()
    table = {
        "full_pg": MethodSpec("full_pg", "mechanism", False, False, False, "none", "none", "raw_pg"),
        "uniform_ht": MethodSpec("uniform_ht", "mechanism", False, False, True, "none", "none", "raw_pg"),
        "uniform_cv": MethodSpec("uniform_cv", "mechanism", True, False, True, "prefix", "full", "raw_pg"),
        "reward_cv": MethodSpec("reward_cv", "mechanism", True, True, True, "prefix", "reward", "raw_pg"),
        "prompt_cv": MethodSpec("prompt_cv", "mechanism", True, True, True, "prompt", "full", "raw_pg"),
        "grace": MethodSpec("grace", "mechanism", True, True, True, "prefix", "full", "raw_pg"),
        "grpo": MethodSpec("grpo", "practical", False, False, False, "none", "none", "grpo"),
        "grpo_short": MethodSpec(
            "grpo_short", "practical", False, False, False, "none", "none", "grpo",
            max_new_tokens=1024, starts_per_prompt=16,
        ),
    }
    if key not in table:
        raise ValueError(f"unknown method {name}; expected one of {METHOD_NAMES}")
    spec = table[key]
    if overrides:
        data = spec.__dict__.copy()
        data.update(overrides)
        spec = MethodSpec(**data)
    return spec


def start_group_size(spec: MethodSpec, n_prompts: int, n_start: int) -> int:
    """Starts that must stay together so next_n does not silently drop leftovers."""
    if spec.starts_per_prompt is not None:
        return int(spec.starts_per_prompt)
    n_prompts = min(max(1, int(n_prompts)), max(1, int(n_start)))
    return max(1, int(n_start) // n_prompts)


def apply_method_defaults(cfg: dict) -> dict:
    """Named methods keep their paper budgets in the run config that is saved."""
    spec = method_spec(cfg.get("method", "grace"))
    out = dict(cfg)
    if spec.max_new_tokens is not None:
        out["max_new_tokens"] = int(spec.max_new_tokens)
    if spec.starts_per_prompt is not None:
        n_prompts = max(1, int(out.get("n_prompts", 4)))
        out["n_start"] = n_prompts * int(spec.starts_per_prompt)
    elif spec.n_start is not None:
        out["n_start"] = int(spec.n_start)
    return out


def uniform_p(n: int, beta: float, p_min: float) -> float:
    """Same expected continuation rate as β, still clipped to p_min."""
    return float(min(1.0, max(p_min, beta)))
