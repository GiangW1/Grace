"""Serialize live train state so a run can actually resume."""

from __future__ import annotations

from typing import Any

import numpy as np

from grace_gc.core.layout import ParamLayout, collect_lora_layout
from grace_gc.core.rng import IsolatedRNG
from grace_gc.predictor.heads import predictor_from_spec
from grace_gc.predictor.reservoir import GradientReservoir
from grace_gc.trainer.algorithm import TrainState
from grace_gc.trainer.baseline import HistoricalBaseline
from grace_gc.trainer.checkpoint import load_checkpoint, save_checkpoint
from grace_gc.trainer.methods import method_spec


def numpy_module_state(named_params) -> dict[str, np.ndarray]:
    return {name: param.detach().cpu().numpy() for name, param in named_params}


def load_numpy_module_state(named_params, payload: dict[str, np.ndarray]) -> None:
    torch = __import__("torch")
    lookup = dict(named_params)
    if not payload:
        raise ValueError("checkpoint actor weights are empty")
    missing = [name for name in lookup if name not in payload]
    if missing:
        raise ValueError(f"checkpoint missing {len(missing)} actor params, e.g. {missing[0]}")
    for name, array in payload.items():
        if name not in lookup:
            continue
        tensor = lookup[name]
        arr = np.asarray(array)
        if tuple(arr.shape) != tuple(tensor.shape):
            raise ValueError(f"checkpoint {name} shape {tuple(arr.shape)} != {tuple(tensor.shape)}")
        tensor.data.copy_(torch.as_tensor(arr, dtype=tensor.dtype))


def optimizer_state(optimizer) -> dict[str, Any]:
    torch = __import__("torch")
    raw = optimizer.state_dict()
    out = {"param_groups": raw["param_groups"], "state": {}}
    for key, item in raw["state"].items():
        out["state"][key] = {
            k: (v.detach().cpu().numpy() if hasattr(v, "detach") else v) for k, v in item.items()
        }
    _ = torch
    return out


def load_optimizer_state(optimizer, payload: dict[str, Any]) -> None:
    torch = __import__("torch")
    state = {"param_groups": payload["param_groups"], "state": {}}
    for key, item in payload.get("state", {}).items():
        conv = {}
        for k, v in item.items():
            conv[k] = torch.as_tensor(v) if isinstance(v, np.ndarray) else v
        state["state"][int(key) if str(key).isdigit() else key] = conv
    optimizer.load_state_dict(state)


def check_snapshot_identity(payload: dict[str, Any], cfg: dict[str, Any]) -> None:
    """Same A/B tensors are not the same policy under a different base or LoRA scale."""
    stored_model = payload.get("model_path")
    live_model = cfg.get("model_path")
    if stored_model and live_model and str(stored_model) != str(live_model):
        raise ValueError(f"checkpoint model_path {stored_model} != {live_model}")
    stored_lora = payload.get("lora") or {}
    live_lora = cfg.get("lora") or {}
    for key in ("rank", "alpha", "targets"):
        if key in stored_lora and key in live_lora and stored_lora[key] != live_lora[key]:
            raise ValueError(f"checkpoint lora.{key} {stored_lora[key]!r} != {live_lora[key]!r}")


def dump_train_state(path, state: TrainState, actor_named, optimizer, actor_full=None, extra=None) -> None:
    payload = {
        "actor": numpy_module_state(actor_named),
        "actor_full": None if actor_full is None else numpy_module_state(actor_full),
        "optimizer": optimizer_state(optimizer),
        "predictor": None if state.predictor is None else state.predictor.state_dict(),
        "baseline": state.baseline.state_dict(),
        "basis": {
            "u": state.u,
            "basis_id": state.basis_id,
            "predictor_synced_basis_id": state.predictor_synced_basis_id,
        },
        "prescan_rng": None if state.prescan_rng is None else state.prescan_rng.state_dict(),
        "reservoir": state.reservoir.state_dict(),
        "rng": state.rng.state_dict(),
        "step": state.step,
        "n_ref": state.n_ref,
        "history_costs": state.history_costs,
        "spec": state.spec.name,
        "layout_names": state.layout.names(),
        "layout_dim": state.layout.dim,
    }
    if extra:
        payload.update(extra)
    save_checkpoint(path, payload)


def restore_train_state(path, actor, optimizer, in_dim: int, k: int, spec_name: str) -> TrainState:
    payload = load_checkpoint(path)
    if hasattr(actor, "named_all_params"):
        if not payload.get("actor_full"):
            raise ValueError("this checkpoint has no actor_full; CPU tiny resume would change the frozen base")
        load_numpy_module_state(actor.named_all_params(), payload["actor_full"])
        named_lora = actor.named_lora_params()
    elif hasattr(actor, "named_lora_params"):
        named_lora = actor.named_lora_params()
        load_numpy_module_state(named_lora, payload.get("actor") or {})
    else:
        from grace_gc.backends.hf_actor import named_lora_params

        named_lora = named_lora_params(actor)
        load_numpy_module_state(named_lora, payload.get("actor") or {})
    layout = collect_lora_layout(named_lora)
    stored_names = payload.get("layout_names")
    if stored_names is not None and list(stored_names) != layout.names():
        raise ValueError("checkpoint LoRA layout names do not match the actor")
    stored_dim = payload.get("layout_dim")
    if stored_dim is not None and int(stored_dim) != layout.dim:
        raise ValueError(f"checkpoint LoRA dim {stored_dim} != actor {layout.dim}")
    if payload.get("optimizer"):
        load_optimizer_state(optimizer, payload["optimizer"])
    ckpt_spec = str(payload.get("spec", spec_name))
    if ckpt_spec != spec_name:
        raise ValueError(f"checkpoint method {ckpt_spec} != requested {spec_name}")
    predictor = None
    if payload.get("predictor"):
        raw = payload["predictor"]
        if spec_name == "reward_cv" and not raw.get("reward_risk"):
            raise ValueError("Reward-CV checkpoint missing reward_risk")
        if spec_name == "reward_cv" and not raw.get("success"):
            raise ValueError("Reward-CV checkpoint missing success")
        predictor = predictor_from_spec(
            int(raw.get("in_dim", in_dim)),
            int(raw.get("k", k)),
            raw,
        )
        predictor.load_state_dict(raw)
    spec = method_spec(spec_name)
    if spec.use_predictor and predictor is None:
        raise ValueError(f"{spec_name} checkpoint missing predictor")
    rng = IsolatedRNG.create(0)
    rng.load_state_dict(payload["rng"])
    prescan_rng = None
    if payload.get("prescan_rng"):
        prescan_rng = IsolatedRNG.create(0)
        prescan_rng.load_state_dict(payload["prescan_rng"])
    reservoir = GradientReservoir.from_state_dict(payload["reservoir"])
    baseline = HistoricalBaseline()
    baseline.load_state_dict(payload["baseline"])
    u = np.asarray(payload["basis"]["u"], dtype=np.float64)
    if u.ndim != 2 or u.shape[0] != layout.dim:
        raise ValueError(f"checkpoint basis U shape {u.shape} does not match LoRA dim {layout.dim}")
    if predictor is not None and u.shape[1] != predictor.k:
        raise ValueError(f"checkpoint basis k {u.shape[1]} != predictor k {predictor.k}")
    state = TrainState(
        spec=spec,
        baseline=baseline,
        rng=rng,
        layout=layout,
        u=u,
        predictor=predictor,
        reservoir=reservoir,
        basis_id=int(payload["basis"].get("basis_id", 0)),
        predictor_synced_basis_id=int(payload["basis"].get("predictor_synced_basis_id", -1)),
        prescan_rng=prescan_rng,
        history_costs=list(payload.get("history_costs", [])),
        step=int(payload.get("step", 0)),
        n_ref=int(payload.get("n_ref", 0)),
    )
    return state
