"""Counterfactual one-step AdamW updates from the same saved optimizer state."""

from __future__ import annotations

import numpy as np


def adamw_update_direction(actor_state, layout, optimizer_state, optim_cfg,
                           ascent_gradient, reference_gradient, clip: float):
    """Run the actual PyTorch AdamW step and return its reference-direction gain.

    `ascent_gradient` is already averaged over the fixed N starts. The optimizer
    receives its negative; clipping follows actor_update.apply_correction_clip_step.
    Each invocation constructs the same starting weights and optimizer moments.
    """
    import torch

    from grace_gc.audit.batch_audit import OptimizerReplay

    g = np.asarray(ascent_gradient, dtype=np.float64).reshape(-1)
    ref = np.asarray(reference_gradient, dtype=np.float64).reshape(-1)
    if g.shape != (layout.dim,) or ref.shape != g.shape:
        raise ValueError("gradient/reference dimension disagrees with checkpoint layout")
    if not np.all(np.isfinite(g)) or not np.all(np.isfinite(ref)):
        raise ValueError("counterfactual gradients must be finite")
    if not optimizer_state or not optimizer_state.get("state"):
        raise ValueError("checkpoint lacks advanced AdamW moment state")
    named = []
    for entry in layout.entries:
        if entry.name not in actor_state:
            raise ValueError(f"checkpoint lacks LoRA parameter {entry.name}")
        array = np.asarray(actor_state[entry.name])
        if array.shape != entry.shape:
            raise ValueError(f"checkpoint shape mismatch for {entry.name}")
        named.append((entry.name, torch.as_tensor(array.copy())))
    # Reuse the training audit's deepcopy, parameter-group mapping, CPU flag
    # handling and exact dtype/writeback/clip order. as_tensor alone aliases
    # NumPy optimizer moments and would mutate the checkpoint on every probe.
    replay = OptimizerReplay(named, layout, optimizer_state, {**optim_cfg, "grad_clip": clip})
    if replay.kind != "AdamW":
        raise ValueError("expected-gain probe requires an AdamW checkpoint")
    delta, stats = replay.step(g)
    return {"reference_dot_delta": float(ref @ delta),
            "delta_norm": stats["parameter_update_norm"],
            "preclip_grad_norm": stats["grad_norm_preclip"],
            "clip_triggered": stats["clip_triggered"]}
