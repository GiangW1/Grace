"""Reuse a shared actor without inheriting another method's training state."""

from grace_gc.trainer.checkpoint import load_checkpoint
from grace_gc.trainer.state_io import check_snapshot_identity, load_numpy_module_state
from grace_gc.versions import sha256_file, sha256_named


def initialize_actor(actor, cfg, run=None):
    path = cfg.get("init_checkpoint")
    if not path:
        return None
    if cfg.get("resume"):
        raise ValueError("init_checkpoint and resume describe different starting states")
    payload = load_checkpoint(path)
    check_snapshot_identity(payload, cfg)
    if hasattr(actor, "named_all_params"):
        if not payload.get("actor_full"):
            raise ValueError("CPU shared initialization needs the frozen base too")
        load_numpy_module_state(actor.named_all_params(), payload["actor_full"])
        named = actor.named_lora_params()
    else:
        from grace_gc.backends.hf_actor import named_lora_params

        named = named_lora_params(actor)
        load_numpy_module_state(named, payload.get("actor") or {})
    info = {"checkpoint": str(path), "checkpoint_sha256": sha256_file(path),
            "actor_sha256": sha256_named(named), "source_step": payload.get("step"),
            "loaded": "actor only; fresh optimizer, baseline, predictor and RNG"}
    if run is not None:
        run.write_json("initial_actor.json", info)
    return info
