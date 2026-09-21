"""Export and load a frozen predictor, independently of actor/optimizer state."""

from pathlib import Path
import re

import numpy as np

from grace_gc.versions import sha256_array, sha256_file, sha256_mapping


def feature_protocol(cfg):
    from grace_gc.trainer.baseline import baseline_from_config
    from grace_gc.trainer.methods import method_spec
    spec = method_spec(cfg.get('method', 'grace'))
    return {"backend": cfg.get("backend", "cpu_tiny"),
            "method_features": spec.feature_mode, "risk_mode": spec.risk_mode,
            "feature_mode": (cfg.get("predictor") or {}).get("feature_mode", "legacy"),
            "model_path": cfg.get("model_path"), "lora": cfg.get("lora", {}),
            "decision_tokens": cfg.get("decision_tokens"), "max_new_tokens": cfg.get("max_new_tokens"),
            "baseline": baseline_from_config(cfg.get("baseline")).configuration()}


def check_feature_protocol(protocol, cfg):
    for key, value in feature_protocol(cfg).items():
        if protocol.get(key) != value:
            raise ValueError(f"offline predictor {key} does not match the consumer")


def export_predictor(checkpoint, directory):
    from grace_gc.trainer.checkpoint import load_checkpoint, save_checkpoint
    payload = load_checkpoint(checkpoint)
    if payload.get("predictor") is None:
        raise ValueError("source checkpoint has no predictor")
    predictor = {key: value for key, value in payload["predictor"].items()
                 if not key.startswith("opt_")}
    body = {"format": "grace_offline_predictor_v1", "predictor": predictor,
            "u": payload["basis"]["u"],
            "basis_id": payload["basis"].get("basis_id", 0),
            "predictor_synced_basis_id": payload["basis"].get("predictor_synced_basis_id", -1),
            "layout_names": payload.get("layout_names"),
            "protocol": feature_protocol(payload.get("run_config") or {}),
            "source": {"checkpoint_sha256": sha256_file(checkpoint), "step": payload.get("step"),
                       "actor_sha256": sha256_mapping(payload.get("actor")),
                       "run_config": payload.get("run_config"), "identity": payload.get("identity")}}
    path = Path(directory)/f"predictor-{sha256_mapping(body)}.npz"
    if not path.exists():
        save_checkpoint(path, body)
    read_predictor(path)
    return path


def read_predictor(path, digest=None):
    path = Path(path)
    match = re.fullmatch(r"predictor-([0-9a-f]{64})\.npz", path.name)
    if match is None:
        raise ValueError("invalid offline predictor artifact basename")
    if digest is not None and sha256_file(path) != digest:
        raise ValueError("offline predictor file hash mismatch")
    with np.load(path, allow_pickle=True) as data:
        body = data["payload"].item()
    if body.get("format") != "grace_offline_predictor_v1" or sha256_mapping(body) != match.group(1):
        raise ValueError("offline predictor content hash or format mismatch")
    return body


def attach_predictor(state, cfg, run):
    """A resume uses the checkpoint's local dependency, never an obsolete donor path."""
    if state.offline_predictor is not None:
        check_feature_protocol(state.offline_predictor['protocol'], cfg)
        state.u.setflags(write=False)
        state.offline_predictor['basis_sha'] = sha256_array(state.u)
        return
    if not cfg.get("offline_predictor"):
        return
    if not state.spec.use_predictor:
        raise ValueError("offline_predictor requires a predictor method")
    path = Path(str(cfg["offline_predictor"]).format(seed=state.rng.seed)).resolve()
    body = read_predictor(path)
    check_feature_protocol(body['protocol'], cfg)
    raw, u = body["predictor"], np.asarray(body["u"], dtype=np.float64)
    if (u.shape != (state.layout.dim, state.predictor.k)
            or int(raw["in_dim"]) != state.predictor.in_dim
            or body.get("layout_names") not in (None, state.layout.names())):
        raise ValueError("offline predictor feature/LoRA layout does not match the consumer")
    from grace_gc.predictor.heads import predictor_from_spec
    import torch
    with torch.random.fork_rng(devices=[]):
        predictor = predictor_from_spec(int(raw["in_dim"]), int(raw["k"]), raw)
        predictor.load_state_dict(raw)
    state.predictor, state.u = predictor, u
    state.basis_id = int(body["basis_id"])
    state.predictor_synced_basis_id = int(body["predictor_synced_basis_id"])
    state.reservoir.items.clear()
    state.reservoir.fixed_basis_id = None
    state.u.setflags(write=False)
    state.offline_predictor = {"path": str(path), "sha256": sha256_file(path),
                               "basis_sha": sha256_array(state.u), "protocol": body["protocol"]}
    run.write_json("offline_predictor.json", {**state.offline_predictor, "source": body["source"],
        "frozen": True, "offline_cost": "source training/export costs are additional; not included in consumer training wall"})


def copy_predictor(reference, directory):
    from grace_gc.trainer.checkpoint import copy_checkpoint
    source = Path(reference["path"])
    target = Path(directory)/source.name
    if source.resolve() != target.resolve() and not target.exists():
        copy_checkpoint(source, target)
    # The source was parsed/validated on load. Verify copied bytes without
    # decompressing and allocating the full fixed U on every checkpoint save.
    if sha256_file(target) != reference["sha256"]:
        raise ValueError("offline predictor file hash mismatch")
    return {"path": target.name, "sha256": reference["sha256"]}
