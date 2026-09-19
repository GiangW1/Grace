"""Hugging Face + PEFT actor: logprob, hidden features, named LoRA params."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import numpy as np

from grace_gc.backends.fsdp_actor import token_sum_logprob
from grace_gc.predictor.features import pool_last_hidden, prefix_features


def _cutoff_len(seq, eos_id, prompt_len: int) -> int:
    from grace_gc.data.tokenize import as_stop_ids

    plen = int(prompt_len)
    stops = set(as_stop_ids(eos_id))
    for j in range(plen, len(seq)):
        if int(seq[j]) in stops:
            return j + 1
    return len(seq)


def _torch():
    import torch

    return torch


def pad_token_ids(seqs: list[list[int]], pad_id: int):
    torch = _torch()
    width = max(len(s) for s in seqs)
    ids = np.full((len(seqs), width), pad_id, dtype=np.int64)
    mask = np.zeros((len(seqs), width), dtype=np.int64)
    for i, seq in enumerate(seqs):
        ids[i, : len(seq)] = np.asarray(seq, dtype=np.int64)
        mask[i, : len(seq)] = 1
    return torch.as_tensor(ids), torch.as_tensor(mask)


def named_lora_params(model):
    out = []
    for name, param in model.named_parameters():
        if param.requires_grad and ("lora_A" in name or "lora_B" in name or "lora_a" in name or "lora_b" in name):
            if "q_proj" in name or "v_proj" in name:
                out.append((name, param))
    if not out:
        raise ValueError("no q/v LoRA A/B parameters found on the HF actor")
    return out


def trainable_params(model):
    return [p for _, p in named_lora_params(model)]


def actor_compute_context(model):
    """BF16 compute with FP32 trainable masters; never cast optimizer state."""
    dtype = getattr(model, "_grace_compute_dtype", "native")
    if dtype == "native":
        return nullcontext()
    if dtype != "bfloat16":
        raise ValueError(f"unsupported actor compute dtype: {dtype}")
    torch = _torch()
    return torch.autocast(next(model.parameters()).device.type, dtype=torch.bfloat16)


def actor_numerics(model):
    return {
        "compute_dtype": getattr(model, "_grace_compute_dtype", "native"),
        "lora_parameter_dtypes": {name: str(p.dtype) for name, p in named_lora_params(model)},
        "base_dtype": str(next(model.parameters()).dtype),
        "use_cache": False,
        "training": bool(model.training),
    }


def actor_forward(model, token_ids, pad_id: int, output_hidden_states: bool = False):
    torch = _torch()
    ids, mask = pad_token_ids(token_ids, pad_id)
    device = next(model.parameters()).device
    ids = ids.to(device)
    mask = mask.to(device)
    with actor_compute_context(model):
        out = model(input_ids=ids, attention_mask=mask,
                    output_hidden_states=output_hidden_states, use_cache=False)
    return out, ids, mask


def logprob_sums(model, full_ids: list[list[int] | None], prompt_lens, chosen, pad_id: int, eos_id: int | None = None):
    torch = _torch()
    chosen = np.asarray(chosen, dtype=bool)
    index = [i for i, keep in enumerate(chosen) if keep and full_ids[i] is not None]
    if not index:
        return torch.zeros(len(full_ids))
    seqs = [full_ids[i] for i in index]
    out, ids, _mask = actor_forward(model, seqs, pad_id, output_hidden_states=False)
    plens = [int(prompt_lens[i]) for i in index]
    valid_lens = [len(seqs[j]) for j in range(len(seqs))]
    sums = token_sum_logprob(out.logits, ids, plens, valid_lens=valid_lens, eos_id=eos_id)
    packed = sums.new_zeros(len(full_ids))
    return packed.index_copy(0, torch.as_tensor(index, device=sums.device), sums)


def logprob_one(model, token_ids: list[int], prompt_len: int, pad_id: int, eos_id: int | None = None):
    out, ids, _mask = actor_forward(model, [token_ids], pad_id, output_hidden_states=False)
    return token_sum_logprob(out.logits, ids, [prompt_len], valid_lens=[len(token_ids)], eos_id=eos_id)[0]


def _hidden_entropy(out, chunk_size: int = 64):
    hiddens = out.hidden_states
    # Bound the FP32 softmax temporary, including the decision-position logit.
    chunks = []
    for start in range(0, out.logits.shape[1], chunk_size):
        logp = out.logits[:, start:start + chunk_size].float().log_softmax(-1)
        chunks.append((-(logp.exp() * logp).sum(dim=-1)).detach().cpu().numpy())
    token_ent = np.concatenate(chunks, axis=1)
    last_seq = hiddens[-1].detach().float().cpu().numpy()
    mid_seq = hiddens[len(hiddens) // 2].detach().float().cpu().numpy()
    return last_seq, mid_seq, token_ent


def _features_from_hidden(prefix, last_row, mid_row, token_ent, prompt_len: int, baseline: float, eos_id, feature_mode="legacy") -> tuple[np.ndarray, list[float], np.ndarray]:
    cutoff = _cutoff_len(prefix, eos_id, int(prompt_len))
    end = max(min(cutoff, last_row.shape[0]), 1)
    pos = end - 1
    pooled = pool_last_hidden(last_row, end)
    if feature_mode == "legacy":
        ent = token_ent[: max(end - 1, 1)]
        length = cutoff
    elif feature_mode == "response":
        ent = token_ent[max(int(prompt_len) - 1, 0):end - 1]
        length = max(cutoff - int(prompt_len), 0)
    elif feature_mode == "decision":
        ent = token_ent[pos:pos + 1]
        length = max(cutoff - int(prompt_len), 0)
    else:
        raise ValueError(f"unknown feature_mode: {feature_mode}")
    feat = prefix_features(last_row[pos], mid_row[pos], pooled, ent, float(length), float(baseline))
    plen = max(int(prompt_len), 1)
    ppos = min(max(plen - 1, 0), last_row.shape[0] - 1)
    prompt_feat = prefix_features(
        last_row[ppos],
        mid_row[ppos],
        pool_last_hidden(last_row, plen),
        token_ent[: max(plen - 1, 1)],
        float(plen),
        float(baseline),
    )
    return feat, [float(cutoff), float(prompt_len), 1.0], prompt_feat


def prefix_feature_bundle(model, prefixes: list[list[int]], prompt_lens, baselines, pad_id: int, eos_id: int | None = None, feature_mode="legacy", batch_size: int = 1) -> dict[str, Any]:
    """One prefix at a time. A 512×4B logits tensor does not fit beside vLLM."""
    torch = _torch()
    stop = pad_id if eos_id is None else eos_id
    feats = []
    cost_feat = []
    prompt_feats = []
    with torch.no_grad():
        for start in range(0, len(prefixes), max(1, int(batch_size))):
            batch = prefixes[start:start + max(1, int(batch_size))]
            out, _ids, _mask = actor_forward(model, batch, pad_id, output_hidden_states=True)
            last_seq, mid_seq, token_ent = _hidden_entropy(out)
            for j, prefix in enumerate(batch):
                i = start + j
                feat, cost, prompt_feat = _features_from_hidden(
                    prefix, last_seq[j], mid_seq[j], token_ent[j], int(prompt_lens[i]), float(baselines[i]), stop,
                    feature_mode=feature_mode,
                )
                feats.append(feat)
                cost_feat.append(cost)
                prompt_feats.append(prompt_feat)
    return {
        "features": np.stack(feats, axis=0),
        "cost_feat": np.asarray(cost_feat, dtype=np.float64),
        "prompt_features": np.stack(prompt_feats, axis=0),
    }
