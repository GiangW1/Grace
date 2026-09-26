"""Plain YAML/JSON config loading. Only hard errors abort a run."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml


HARD_ERROR_KEYS = ("p_min", "beta", "rank", "max_new_tokens")


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"config is not readable: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    return data


def merge_configs(*configs: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for cfg in configs:
        out = _deep_merge(out, cfg)
    return out


def _deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def validate_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Raise only when the program cannot run or the math is undefined."""
    method = cfg.get("method", "grace")
    alloc = cfg.get("allocation", {})
    p_min = float(alloc.get("p_min", cfg.get("p_min", 0.2)))
    beta = float(alloc.get("beta", cfg.get("beta", 0.5)))
    if p_min <= 0.0 or p_min > 1.0:
        raise ValueError(f"p_min must be in (0, 1], got {p_min}")
    if beta <= 0.0 or beta > 1.0:
        raise ValueError(f"beta must be in (0, 1], got {beta}")
    n_start = int(cfg.get("n_start", cfg.get("N", 1)))
    if n_start <= 0:
        raise ValueError(f"n_start must be positive, got {n_start}")
    decision = int(cfg.get("decision_tokens", 16))
    max_new = int(cfg.get("max_new_tokens", 32))
    if decision < 0 or max_new <= 0:
        raise ValueError("decision_tokens must be nonnegative and max_new_tokens must be positive")
    if decision == 0:
        grid = (cfg.get("audit") or {}).get("decision_grid") or []
        if not any(int(value) == 0 for value in grid):
            raise ValueError("decision_tokens=0 is only valid for an audit decision_grid containing 0")
    if decision > max_new:
        raise ValueError(f"decision_tokens {decision} exceeds max_new_tokens {max_new}")
    fw_steps = int((cfg.get("format_warmup") or {}).get("steps", 0) or 0)
    if fw_steps < 0:
        raise ValueError(f"format_warmup.steps must be >= 0, got {fw_steps}")
    method = str(cfg.get("method", "grace")).replace("-", "_").lower()
    model_path = cfg.get("model_path")
    if model_path is not None and str(model_path):
        path = Path(str(model_path))
        if path.is_absolute() and not path.exists():
            raise FileNotFoundError(f"model path is not readable: {model_path}")
    data_path = cfg.get("data_path")
    if data_path is not None and str(data_path) and Path(str(data_path)).suffix:
        if not Path(str(data_path)).exists():
            raise FileNotFoundError(f"data path is not readable: {data_path}")
    _ = method
    return cfg


def default_config() -> dict[str, Any]:
    return {
        "method": "grace",
        "seed": 17,
        "split_seed": 17,
        "n_start": 8,
        "n_prompts": 4,
        "num_steps": 1,
        "prompt_max_tokens": 1024,
        "decision_tokens": 16,
        "max_new_tokens": 32,
        "temperature": 1.0,
        "eval_temperature": 0.6,
        "eval_top_p": 0.95,
        "allocation": {"p_min": 0.2, "beta": 0.5, "bisection_iters": 20},
        "predictor": {
            "k": 8,
            "hidden_coord": 256,
            "hidden_risk": 64,
            "lr": 1e-3,
            "epochs": 2,
            "warmup_steps": 0,
            "audit_s": 0.125,
            "reservoir_size": 512,
            "refresh_every": 32,
            "constant_cost": True,
            "coord_kind": "mlp",
            "ridge_l2": 1.0,
            "use_affine": True,
            "shrink_m": True,
            "align_basis": True,
        },
        "lora": {"rank": 16, "alpha": 32.0, "dropout": 0.0, "targets": ["q_proj", "v_proj"]},
        "optim": {"lr": 1e-4, "betas": [0.9, 0.99], "weight_decay": 0.0, "grad_clip": 1.0},
        "format_warmup": {"steps": 0, "batch_size": 4},
        "baseline": {"ema_alpha": 0.7, "prescan": 4},
        "hardware": {"name": "cpu", "n_gpu": 0},
        "analysis": {
            "answer_undecided": 0.15,
            "nonzero_signal_kappa": 0.05,
            "delta_answer": 0.05,
            "epsilon_learn": 0.2,
            "wilson_lo": 0.05,
            "wilson_hi": 0.95,
            "require_pre_emit": True,
        },
    }
