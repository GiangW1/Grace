"""Fixed nonzero L/R, trainable C: W + (alpha/r) L C R.

SVD directions are a LoRA-XS-style experimental parameterization, not a new
adapter claim. Frozen bases are checkpointed, not reconstructed on restore.
"""
from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class SmallCoreLinear(nn.Module):
    def __init__(self, base, rank, alpha, *, basis="svd", seed=17):
        super().__init__()
        if not isinstance(base, nn.Linear) or not 0 < rank <= min(base.weight.shape):
            raise ValueError("small core needs a Linear and rank <= min(weight.shape)")
        if not math.isfinite(float(alpha)) or not float(alpha) > 0:
            raise ValueError("small-core alpha must be positive")
        self.base_layer = base
        base.requires_grad_(False)
        self.rank, self.scaling = int(rank), float(alpha) / rank
        # FP32 masters and bases; BF16 compute is controlled by actor autocast.
        weight = base.weight.detach().float()
        if basis == "svd":
            left, singular, right = torch.linalg.svd(weight, full_matrices=False)
            # Orthonormal directions; singular values deliberately NOT absorbed
            # into the coordinates. Report this scaling, not exact LoRA-XS.
            left, right = left[:, :rank].contiguous(), right[:rank].contiguous()
        elif basis == "random":
            generator = torch.Generator(device="cpu").manual_seed(int(seed))
            left = torch.linalg.qr(torch.randn(weight.shape[0], rank, generator=generator)).Q
            right = torch.linalg.qr(torch.randn(weight.shape[1], rank, generator=generator)).Q.T
        else:
            raise ValueError("small-core basis must be svd or random")
        self.register_buffer("fixed_left", left.to(weight.device))
        self.register_buffer("fixed_right", right.to(weight.device))
        self.lora_core = nn.Parameter(torch.zeros(rank, rank, device=weight.device, dtype=torch.float32))
        self.stamp_geometry()

    def stamp_geometry(self):
        # Existing snapshot hashes enumerate trainables only. Zero C under
        # different frozen directions must not masquerade as the same geometry.
        digest = hashlib.sha256(str(self.scaling).encode())
        for value in (self.fixed_left, self.fixed_right):
            digest.update(value.detach().float().cpu().contiguous().numpy().tobytes())
        self.lora_core._grace_fixed_geometry_sha = digest.hexdigest()

    def forward(self, x):
        base = self.base_layer(x)
        # Also works outside autocast when the frozen base uses BF16.
        update = F.linear(F.linear(F.linear(x.float(), self.fixed_right),
                                   self.lora_core), self.fixed_left)
        return base + (self.scaling * update).to(base.dtype)


def attach_small_core(model, cfg):
    targets = tuple(cfg.get("targets", ["q_proj", "v_proj"]))
    if set(targets) != {"q_proj", "v_proj"} or float(cfg.get("dropout", 0)) != 0:
        raise ValueError("small-core experiment supports q/v targets and zero dropout")
    selected = [(name, module) for name, module in model.named_modules()
                if name.split(".")[-1] in targets]
    if not selected:
        raise ValueError("no q/v modules found for small-core adapter")
    model.requires_grad_(False)
    for index, (name, module) in enumerate(selected):
        parent, _, child = name.rpartition(".")
        replacement = SmallCoreLinear(module, int(cfg.get("rank", 8)),
                                     float(cfg.get("alpha", 8)),
                                     basis=cfg.get("basis", "svd"),
                                     seed=int(cfg.get("basis_seed", 17)) + index)
        setattr(model.get_submodule(parent) if parent else model, child, replacement)
    model._grace_small_core_config = dict(cfg)
    return model


def fixed_basis_state(model):
    if not hasattr(model, "_grace_small_core_config"):
        return {}
    return {name: value.detach().float().cpu().numpy().copy()
            for name, value in model.named_buffers()
            if name.endswith((".fixed_left", ".fixed_right"))}


def restore_fixed_bases(model, payload):
    if not hasattr(model, "_grace_small_core_config"):
        if payload.get("small_core_bases"):
            raise ValueError("small-core checkpoint cannot initialize ordinary LoRA")
        return
    from grace_gc.trainer.state_io import load_numpy_module_state
    named = [(name, value) for name, value in model.named_buffers()
             if name.endswith((".fixed_left", ".fixed_right"))]
    stored = payload.get("small_core_bases") or {}
    if set(stored) != {name for name, _ in named}:
        raise ValueError("small-core checkpoint requires all frozen bases")
    load_numpy_module_state(named, stored)
    for module in model.modules():
        if isinstance(module, SmallCoreLinear):
            module.stamp_geometry()


def export_small_core_adapter(model, path):
    """Export exactly A=R, B=L@C for standard PEFT/vLLM LoRA loading."""
    from safetensors.torch import save_file
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    tensors = {}
    for name, module in model.named_modules():
        if isinstance(module, SmallCoreLinear):
            key = f"base_model.model.{name}"
            tensors[key + ".lora_A.weight"] = module.fixed_right.detach().float().cpu().contiguous()
            tensors[key + ".lora_B.weight"] = (
                module.fixed_left.float() @ module.lora_core.detach().float()).cpu().contiguous()
    if not tensors:
        raise ValueError("no small-core modules to export")
    cfg = model._grace_small_core_config
    config = {"peft_type": "LORA", "task_type": "CAUSAL_LM", "inference_mode": True,
              "r": int(cfg.get("rank", 8)), "lora_alpha": float(cfg.get("alpha", 8)),
              "lora_dropout": 0.0, "bias": "none", "fan_in_fan_out": False,
              "target_modules": ["q_proj", "v_proj"],
              "base_model_name_or_path": getattr(model.config, "_name_or_path", "")}
    (path / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n")
    save_file(tensors, str(path / "adapter_model.safetensors"))
    return path
