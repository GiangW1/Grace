"""Tiny LoRA engine that implements the Algorithm-1 callbacks."""

from __future__ import annotations

import numpy as np

from grace_gc.core.rng import IsolatedRNG
from grace_gc.data.reward import rule_reward
from grace_gc.data.tokenize import tiny_decode, tiny_encode
from grace_gc.predictor.features import pool_last_hidden, prefix_features, prompt_slice_features
from grace_gc.trainer.cpu_tiny import TinyLoRAActor, _cutoff_len, _sample_tokens, _torch
from grace_gc.trainer.algorithm import StepEngines


def _to_tensor(seqs: list[list[int]], pad: int):
    torch, _nn, _F = _torch()
    width = max(len(s) for s in seqs)
    arr = np.full((len(seqs), width), pad, dtype=np.int64)
    for i, seq in enumerate(seqs):
        arr[i, : len(seq)] = np.asarray(seq, dtype=np.int64)
    return torch.as_tensor(arr)


def make_tiny_engines(actor: TinyLoRAActor, vocab: int) -> StepEngines:
    torch, _nn, _F = _torch()
    pad = actor.eos_id

    def generate_prefix(prompt_ids, max_new, rng: IsolatedRNG, stream: str):
        out = []
        finished = []
        for ids in prompt_ids:
            tok = torch.as_tensor([list(ids)], dtype=torch.long)
            tokens, fin = _sample_tokens(actor, tok, int(max_new), rng, stream)
            out.append(tokens[0].tolist())
            finished.append(bool(fin[0]))
        return out, np.asarray(finished, dtype=bool)

    def continue_selected(prefixes, selected, max_new, rng: IsolatedRNG):
        selected = np.asarray(selected, dtype=bool)
        full: list[list[int] | None] = [None] * len(prefixes)
        if int(max_new) <= 0:
            for i, keep in enumerate(selected):
                if keep:
                    full[i] = list(prefixes[i])
            return full
        for i, keep in enumerate(selected):
            if not keep:
                continue
            pref = list(prefixes[i])
            if pref and pref[-1] == actor.eos_id:
                full[i] = pref
                continue
            tokens, _fin = _sample_tokens(actor, torch.as_tensor([pref], dtype=torch.long), int(max_new), rng, "continuation")
            full[i] = tokens[0].tolist()
        return full

    def prefix_features_fn(prefixes, prompt_lens, baselines):
        toks = _to_tensor(prefixes, pad)
        logits, hiddens = actor.forward(toks)
        logp = _F.log_softmax(logits[:, :-1], dim=-1)
        token_ent = (-(logp.exp() * logp).sum(dim=-1)).detach().cpu().numpy()
        feats = []
        cost_feat = []
        last_seq = hiddens[-1].detach().cpu().numpy()
        mid_seq = hiddens[len(hiddens) // 2].detach().cpu().numpy()
        for i in range(len(prefixes)):
            cutoff = _cutoff_len(prefixes[i], pad, int(prompt_lens[i]))
            pos = min(max(cutoff - 1, 0), last_seq.shape[1] - 1)
            pooled = pool_last_hidden(last_seq[i], cutoff)
            ent = token_ent[i, : max(cutoff - 1, 1)]
            feats.append(
                prefix_features(last_seq[i, pos], mid_seq[i, pos], pooled, ent, float(cutoff), float(baselines[i]))
            )
            cost_feat.append([float(cutoff), float(prompt_lens[i]), 1.0])
        return {
            "features": np.stack(feats, axis=0),
            "cost_feat": np.asarray(cost_feat, dtype=np.float64),
            "prompt_features": prompt_slice_features(last_seq, mid_seq, token_ent, prompt_lens, baselines),
        }

    def logprob_sums(full_ids, prompt_lens, chosen):
        chosen = np.asarray(chosen, dtype=bool)
        acc = []
        index = []
        for i, keep in enumerate(chosen):
            if not keep or full_ids[i] is None:
                continue
            seq = torch.as_tensor(full_ids[i], dtype=torch.long).unsqueeze(0)
            lp, _tlp, _h = actor.token_logprob_sum(seq, int(prompt_lens[i]))
            acc.append(lp[0])
            index.append(i)
        if not acc:
            return torch.zeros(len(full_ids))
        stacked = torch.stack(acc)
        out = stacked.new_zeros(len(full_ids))
        for loc, i in enumerate(index):
            out = out + 0
            out[i] = stacked[loc]
        # Keep graph: reconstruct with index_copy
        out = stacked.new_zeros(len(full_ids))
        idx = torch.as_tensor(index, dtype=torch.long)
        return out.index_copy(0, idx, stacked)

    def logprob_one(token_ids, prompt_len: int):
        seq = torch.as_tensor(token_ids, dtype=torch.long).unsqueeze(0)
        lp, _tlp, _h = actor.token_logprob_sum(seq, int(prompt_len))
        return lp[0]

    def reward_fn(token_ids, gold, truncated=False, text=""):
        if gold:
            scored = rule_reward(text or "", gold, truncated=truncated)
            return 0.0 if scored is None else float(scored)
        body = [t for t in token_ids if t != actor.eos_id]
        last = body[-1] if body else 0
        return 1.0 if last % 2 == 0 else 0.0

    def decode(token_ids):
        return tiny_decode(token_ids, actor.eos_id)

    return StepEngines(
        generate_prefix=generate_prefix,
        continue_selected=continue_selected,
        prefix_features=prefix_features_fn,
        logprob_sums=logprob_sums,
        logprob_one=logprob_one,
        named_lora=actor.named_lora_params,
        trainable_params=actor.trainable_params,
        reward_fn=reward_fn,
        decode=decode,
        eos_id=actor.eos_id,
    )
