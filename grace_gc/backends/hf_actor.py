"""Hugging Face + PEFT actor: logprob, hidden features, named LoRA params."""

from __future__ import annotations

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


def actor_forward(model, token_ids, pad_id: int, output_hidden_states: bool = False):
    torch = _torch()
    ids, mask = pad_token_ids(token_ids, pad_id)
    device = next(model.parameters()).device
    ids = ids.to(device)
    mask = mask.to(device)
    out = model(input_ids=ids, attention_mask=mask, output_hidden_states=output_hidden_states)
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


def _hidden_entropy(out):
    hiddens = out.hidden_states
    logp = out.logits[:, :-1].float().log_softmax(-1)
    token_ent = (-(logp.exp() * logp).sum(dim=-1)).detach().cpu().numpy()
    last_seq = hiddens[-1].detach().float().cpu().numpy()
    mid_seq = hiddens[len(hiddens) // 2].detach().float().cpu().numpy()
    return last_seq, mid_seq, token_ent


def _features_from_hidden(prefix, last_row, mid_row, token_ent, prompt_len: int, baseline: float, eos_id) -> tuple[np.ndarray, list[float], np.ndarray]:
    cutoff = _cutoff_len(prefix, eos_id, int(prompt_len))
    end = max(min(cutoff, last_row.shape[0]), 1)
    pos = end - 1
    pooled = pool_last_hidden(last_row, end)
    ent = token_ent[: max(end - 1, 1)]
    feat = prefix_features(last_row[pos], mid_row[pos], pooled, ent, float(cutoff), float(baseline))
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


def prefix_feature_bundle(model, prefixes: list[list[int]], prompt_lens, baselines, pad_id: int, eos_id: int | None = None) -> dict[str, Any]:
    """One prefix at a time. A 512×4B logits tensor does not fit beside vLLM."""
    torch = _torch()
    stop = pad_id if eos_id is None else eos_id
    feats = []
    cost_feat = []
    prompt_feats = []
    with torch.no_grad():
        for i, prefix in enumerate(prefixes):
            out, _ids, _mask = actor_forward(model, [prefix], pad_id, output_hidden_states=True)
            last_seq, mid_seq, token_ent = _hidden_entropy(out)
            feat, cost, prompt_feat = _features_from_hidden(
                prefix, last_seq[0], mid_seq[0], token_ent[0], int(prompt_lens[i]), float(baselines[i]), stop
            )
            feats.append(feat)
            cost_feat.append(cost)
            prompt_feats.append(prompt_feat)
    return {
        "features": np.stack(feats, axis=0),
        "cost_feat": np.asarray(cost_feat, dtype=np.float64),
        "prompt_features": np.stack(prompt_feats, axis=0),
    }
