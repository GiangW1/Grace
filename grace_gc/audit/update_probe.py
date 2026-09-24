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

    from grace_gc.trainer.state_io import load_optimizer_state

    g = np.asarray(ascent_gradient, dtype=np.float64).reshape(-1)
    ref = np.asarray(reference_gradient, dtype=np.float64).reshape(-1)
    if g.shape != (layout.dim,) or ref.shape != g.shape:
        raise ValueError("gradient/reference dimension disagrees with checkpoint layout")
    if not np.all(np.isfinite(g)) or not np.all(np.isfinite(ref)):
        raise ValueError("counterfactual gradients must be finite")
    if not optimizer_state or not optimizer_state.get("state"):
        raise ValueError("checkpoint lacks advanced AdamW moment state")
    params = []
    starts = []
    for entry in layout.entries:
        if entry.name not in actor_state:
            raise ValueError(f"checkpoint lacks LoRA parameter {entry.name}")
        array = np.asarray(actor_state[entry.name])
        if array.shape != entry.shape:
            raise ValueError(f"checkpoint shape mismatch for {entry.name}")
        tensor = torch.nn.Parameter(torch.as_tensor(array.copy()))
        params.append(tensor)
        starts.append(tensor.detach().clone())
    optimizer = torch.optim.AdamW(params, lr=float(optim_cfg.get("lr", 1e-4)),
                                  betas=tuple(optim_cfg.get("betas", (0.9, 0.99))),
                                  weight_decay=float(optim_cfg.get("weight_decay", 0.)))
    load_optimizer_state(optimizer, optimizer_state)
    norm = float(np.linalg.norm(g))
    factor = 1. if clip <= 0. or norm <= clip else float(clip) / norm
    for param, entry in zip(params, layout.entries):
        chunk = -factor * g[entry.offset:entry.offset + entry.numel].reshape(entry.shape)
        param.grad = torch.as_tensor(chunk, dtype=param.dtype, device=param.device)
    optimizer.step()
    gain = 0.
    delta_sq = 0.
    for param, start, entry in zip(params, starts, layout.entries):
        delta = (param.detach().to(torch.float64) - start.to(torch.float64)).reshape(-1)
        direction = torch.as_tensor(ref[entry.offset:entry.offset + entry.numel],
                                    dtype=torch.float64, device=delta.device)
        gain += float(torch.dot(delta, direction))
        delta_sq += float(torch.dot(delta, delta))
    return {"reference_dot_delta": gain, "delta_norm": float(np.sqrt(delta_sq)),
            "preclip_grad_norm": norm, "clip_triggered": bool(factor < 1.)}
