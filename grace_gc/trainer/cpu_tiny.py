"""CPU tiny LoRA actor used for mathematical and leakage tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from grace_gc.core.layout import collect_lora_layout
from grace_gc.core.rng import IsolatedRNG
from grace_gc.predictor.features import pool_last_hidden, prefix_features
from grace_gc.trainer.actor_update import (
    apply_correction_clip_step,
    audit_one,
    real_stream_backward,
    restore_grads,
    snapshot_grads,
)
from grace_gc.trainer.advantages import advantages_for_method
from grace_gc.trainer.baseline import HistoricalBaseline
from grace_gc.trainer.grace_step import StartRecord, decide_continuation
from grace_gc.trainer.methods import method_spec


def _torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    return torch, nn, F


class LoRALinear:
    def __init__(self, linear, rank: int, alpha: float):
        torch, nn, _F = _torch()
        self.linear = linear
        d_out, d_in = linear.weight.shape
        self.lora_A = nn.Parameter(torch.zeros(rank, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        self.scale = alpha / rank

    def __call__(self, x):
        return self.linear(x) + self.scale * (x @ self.lora_A.T @ self.lora_B.T)


class TinyBlock:
    def __init__(self, dim: int, n_heads: int, n_kv: int, rank: int, alpha: float):
        torch, nn, _F = _torch()
        self.n_heads = n_heads
        self.n_kv = n_kv
        self.head_dim = dim // n_heads
        self.q = LoRALinear(nn.Linear(dim, dim, bias=False), rank, alpha)
        self.k = nn.Linear(dim, n_kv * self.head_dim, bias=False)
        self.v = LoRALinear(nn.Linear(dim, n_kv * self.head_dim, bias=False), rank, alpha)
        self.o = nn.Linear(dim, dim, bias=False)
        self.ln = nn.LayerNorm(dim)

    def parameters(self):
        yield self.q.lora_A
        yield self.q.lora_B
        yield self.v.lora_A
        yield self.v.lora_B
        yield from self.k.parameters()
        yield from self.o.parameters()
        yield from self.ln.parameters()

    def named_lora(self, prefix: str):
        return [
            (f"{prefix}.q_proj.lora_A", self.q.lora_A),
            (f"{prefix}.q_proj.lora_B", self.q.lora_B),
            (f"{prefix}.v_proj.lora_A", self.v.lora_A),
            (f"{prefix}.v_proj.lora_B", self.v.lora_B),
        ]

    def forward(self, x):
        torch, _nn, F = _torch()
        h = self.ln(x)
        b, t, d = h.shape
        q = self.q(h).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(h).view(b, t, self.n_kv, self.head_dim).transpose(1, 2)
        v = self.v(h).view(b, t, self.n_kv, self.head_dim).transpose(1, 2)
        if self.n_heads != self.n_kv:
            rep = self.n_heads // self.n_kv
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        mask = torch.triu(torch.ones(t, t, dtype=torch.bool, device=h.device), 1)
        attn = attn.masked_fill(mask, float("-inf"))
        w = torch.softmax(attn, dim=-1)
        out = torch.matmul(w, v).transpose(1, 2).reshape(b, t, d)
        return x + self.o(out)


class TinyLoRAActor:
    def __init__(self, vocab: int = 32, dim: int = 16, n_layers: int = 2, n_heads: int = 4, n_kv: int = 2, rank: int = 2, alpha: float = 4.0):
        torch, nn, _F = _torch()
        self.vocab = vocab
        self.dim = dim
        self.n_layers = n_layers
        self.eos_id = vocab - 1
        self.embed = nn.Embedding(vocab, dim)
        self.blocks = [TinyBlock(dim, n_heads, n_kv, rank, alpha) for _ in range(n_layers)]
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        self._freeze_base()

    def _freeze_base(self) -> None:
        for p in self.embed.parameters():
            p.requires_grad = False
        for block in self.blocks:
            for p in list(block.k.parameters()) + list(block.o.parameters()) + list(block.ln.parameters()):
                p.requires_grad = False
            block.q.linear.weight.requires_grad = False
            block.v.linear.weight.requires_grad = False
        for p in self.ln.parameters():
            p.requires_grad = False
        self.head.weight.requires_grad = False

    def named_lora_params(self):
        out = []
        for i, block in enumerate(self.blocks):
            out.extend(block.named_lora(f"layers.{i}"))
        return out

    def trainable_params(self):
        return [p for _, p in self.named_lora_params()]

    def named_all_params(self):
        torch, _nn, _F = _torch()
        out = [
            ("embed.weight", self.embed.weight),
            ("ln.weight", self.ln.weight),
            ("ln.bias", self.ln.bias),
            ("head.weight", self.head.weight),
        ]
        for i, block in enumerate(self.blocks):
            out.append((f"layers.{i}.ln.weight", block.ln.weight))
            out.append((f"layers.{i}.ln.bias", block.ln.bias))
            out.append((f"layers.{i}.k.weight", block.k.weight))
            out.append((f"layers.{i}.o.weight", block.o.weight))
            out.append((f"layers.{i}.q.linear.weight", block.q.linear.weight))
            out.append((f"layers.{i}.v.linear.weight", block.v.linear.weight))
            out.extend(block.named_lora(f"layers.{i}"))
        _ = torch
        return out

    def forward(self, tokens):
        torch, _nn, _F = _torch()
        x = self.embed(tokens)
        hiddens = []
        for block in self.blocks:
            x = block.forward(x)
            hiddens.append(x)
        x = self.ln(x)
        logits = self.head(x)
        return logits, hiddens

    def token_logprob_sum(self, tokens, prompt_len: int, valid_lens=None):
        torch, _nn, F = _torch()
        logits, hiddens = self.forward(tokens)
        logp = F.log_softmax(logits[:, :-1], dim=-1)
        target = tokens[:, 1:]
        token_lp = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        mask = torch.zeros_like(token_lp)
        width = int(token_lp.shape[1])
        for i in range(tokens.shape[0]):
            seq = tokens[i].tolist()
            plen = int(prompt_len[i]) if hasattr(prompt_len, "__len__") else int(prompt_len)
            if valid_lens is not None:
                cutoff = int(valid_lens[i])
            else:
                cutoff = _cutoff_len(seq, self.eos_id, plen)
            end = min(max(cutoff - 1, 0), width)
            start = max(plen - 1, 0)
            if start < end:
                mask[i, start:end] = 1.0
        summed = (token_lp * mask).sum(dim=-1)
        return summed, token_lp, hiddens


@dataclass
class TinyTrainConfig:
    n: int = 8
    decision_tokens: int = 4
    max_new: int = 8
    p_min: float = 0.2
    beta: float = 0.5
    warmup: bool = False
    leak_from_suffix: bool = False
    method: str = "grace"
    seed: int = 0


def _cutoff_len(seq, eos_id: int, prompt_len: int) -> int:
    """Exclusive end index of real tokens; first EOS after the prompt is included."""
    seq = list(seq)
    plen = int(prompt_len)
    for j in range(plen, len(seq)):
        if seq[j] == eos_id:
            return j + 1
    return len(seq)


def _sample_tokens(actor: TinyLoRAActor, prompt, max_new: int, rng: IsolatedRNG, stream: str):
    torch, _nn, F = _torch()
    tokens = prompt.clone()
    finished = torch.zeros(tokens.shape[0], dtype=torch.bool)
    for i in range(tokens.shape[0]):
        if int(tokens[i, -1].item()) == actor.eos_id:
            finished[i] = True
    if bool(finished.all()):
        return tokens, finished.cpu().numpy()
    for _ in range(max_new):
        logits, _ = actor.forward(tokens)
        probs = F.softmax(logits[:, -1], dim=-1).detach().cpu().numpy()
        nxt = []
        for i in range(tokens.shape[0]):
            if finished[i]:
                nxt.append(actor.eos_id)
                continue
            draw = int(rng.integers(stream, 0, actor.vocab))
            u = float(rng.random(stream))
            cdf = np.cumsum(probs[i])
            tok = int(np.searchsorted(cdf, u, side="right"))
            tok = min(max(tok, 0), actor.vocab - 1)
            nxt.append(tok)
            _ = draw
            if tok == actor.eos_id:
                finished[i] = True
        nxt_t = torch.as_tensor(nxt, dtype=tokens.dtype).unsqueeze(1)
        tokens = torch.cat([tokens, nxt_t], dim=1)
        if bool(finished.all()):
            break
    return tokens, finished.cpu().numpy()


def run_tiny_batch(cfg: TinyTrainConfig) -> dict:
    torch, nn, _F = _torch()
    from grace_gc.core.rng import seed_all

    seed_all(int(cfg.seed))
    actor = TinyLoRAActor()
    layout = collect_lora_layout(actor.named_lora_params())
    rng = IsolatedRNG.create(cfg.seed)
    spec = method_spec(cfg.method)
    baseline = HistoricalBaseline()
    prompts = torch.randint(0, actor.vocab - 1, (cfg.n, 3))
    prefixes, fin_prefix = _sample_tokens(actor, prompts, cfg.decision_tokens, rng, "token")
    prompt_len = prompts.shape[1]
    logits, hiddens = actor.forward(prefixes)
    last_seq = hiddens[-1].detach().cpu().numpy()
    mid_seq = hiddens[len(hiddens) // 2].detach().cpu().numpy()
    logp = _F.log_softmax(logits[:, :-1], dim=-1)
    token_ent = (-(logp.exp() * logp).sum(dim=-1)).detach().cpu().numpy()

    feats = []
    for i in range(cfg.n):
        seq = prefixes[i].tolist()
        cutoff = _cutoff_len(seq, actor.eos_id, prompt_len)
        pos = min(max(cutoff - 1, 0), last_seq.shape[1] - 1)
        feats.append(
            prefix_features(
                last_seq[i, pos],
                mid_seq[i, pos],
                pool_last_hidden(last_seq[i], cutoff),
                token_ent[i, : max(cutoff - 1, 1)],
                float(cutoff),
                baseline.get(str(i % 2)),
            )
        )
    feat = np.stack(feats, axis=0)
    k = 2
    u = np.eye(layout.dim, k, dtype=np.float64)
    f = feat[:, :k] if feat.shape[1] >= k else np.pad(feat, ((0, 0), (0, k - feat.shape[1])))
    r_hat_prefix = np.full(cfg.n, 1.0)
    c_hat = np.full(cfg.n, 1.0)
    finished = fin_prefix.astype(bool)
    leaked_cont = None
    if cfg.leak_from_suffix:
        leaked_cont, _ = _sample_tokens(actor, prefixes, cfg.max_new, rng, "continuation")
        first_suffix = leaked_cont[:, prefixes.shape[1]]
        leak_bits = first_suffix.detach().cpu().numpy() / max(actor.vocab - 1, 1)
        r_hat = 0.1 + 2.0 * leak_bits.astype(np.float64)
        leak_flag = True
    else:
        r_hat = r_hat_prefix
        leak_flag = False
    p_prefix, _ = decide_continuation(spec, r_hat_prefix, c_hat, finished, cfg.beta, cfg.p_min, cfg.warmup)
    p, dev = decide_continuation(spec, r_hat, c_hat, finished, cfg.beta, cfg.p_min, cfg.warmup)
    z = np.ones(cfg.n) if (cfg.warmup or spec.name == "full_pg") else rng.bernoulli("selection", p)
    z[finished] = 1.0
    p[finished] = 1.0
    p_prefix[finished] = 1.0
    f[finished] = 0.0

    chosen = z >= 1.0
    if leaked_cont is not None:
        full_tokens = []
        rewards = []
        for i in range(cfg.n):
            if not chosen[i]:
                full_tokens.append(prefixes[i])
                rewards.append(None)
                continue
            seq = leaked_cont[i]
            full_tokens.append(seq)
            last_tok = int(seq[seq != actor.eos_id][-1]) if (seq != actor.eos_id).any() else 0
            rewards.append(1.0 if last_tok % 2 == 0 else 0.0)
    elif chosen.any():
        cont, fin_all = _sample_tokens(actor, prefixes[torch.as_tensor(chosen)], cfg.max_new, rng, "continuation")
        full_tokens = []
        rewards = []
        j = 0
        for i in range(cfg.n):
            if not chosen[i]:
                full_tokens.append(prefixes[i])
                rewards.append(None)
                continue
            seq = cont[j]
            full_tokens.append(seq)
            ended = bool(fin_all[j] or seq[-1].item() == actor.eos_id)
            last_tok = int(seq[seq != actor.eos_id][-1]) if (seq != actor.eos_id).any() else 0
            rewards.append(1.0 if ended and last_tok % 2 == 0 else 0.0)
            j += 1
    else:
        full_tokens = [prefixes[i] for i in range(cfg.n)]
        rewards = [None] * cfg.n

    p_again, _ = decide_continuation(spec, r_hat_prefix, c_hat, finished, cfg.beta, cfg.p_min, cfg.warmup)
    p_again[finished] = 1.0

    n = cfg.n
    pids = [str(i % 2) for i in range(n)]
    adv = advantages_for_method(spec.objective, rewards, pids, baseline)
    records = []
    for i in range(n):
        records.append(
            StartRecord(
                problem_id=pids[i],
                finished=bool(finished[i]),
                p=float(p[i]),
                z=float(z[i]),
                f=f[i],
                r_hat=float(r_hat[i]),
                c_hat=float(c_hat[i]),
                reward=rewards[i],
                advantage=float(adv[i]) if rewards[i] is not None else None,
                g=None,
                audited=False,
                leak_flag=leak_flag,
            )
        )

    opt = torch.optim.SGD(actor.trainable_params(), lr=0.05)
    opt.zero_grad()
    lp_acc = []
    lp_idx = []
    for i in range(n):
        if z[i] < 1.0:
            continue
        lp_sum, _tlp, _h = actor.token_logprob_sum(full_tokens[i].unsqueeze(0), prompt_len)
        lp_acc.append(lp_sum[0])
        lp_idx.append(i)
    if lp_acc:
        stacked = torch.stack(lp_acc)
        lp_sums = stacked.new_zeros(n)
        lp_sums = lp_sums.index_copy(0, torch.as_tensor(lp_idx, dtype=torch.long), stacked)
    else:
        lp_sums = torch.zeros(n)
    loss = real_stream_backward(lp_sums, adv, p, z, n)
    saved = snapshot_grads(actor.trainable_params())
    audit_sel = rng.bernoulli("audit", np.full(n, 0.5))
    names = [nm for nm, _param in actor.named_lora_params()]
    params = actor.trainable_params()
    for i, rec in enumerate(records):
        if rec.z < 1.0 or rec.reward is None or audit_sel[i] < 1.0:
            continue
        lp_sum, _tlp, _h = actor.token_logprob_sum(full_tokens[i].unsqueeze(0), prompt_len)
        rec.g = audit_one(adv[i] * lp_sum[0], params, names, layout)
        rec.audited = True
    restore_grads(params, saved)
    before = {name: param.detach().clone() for name, param in actor.named_lora_params()}
    apply_correction_clip_step(
        actor.named_lora_params(),
        layout,
        u,
        f,
        z,
        p,
        n,
        opt,
        clip=1.0,
        use_correction=spec.use_ht_correction and spec.use_predictor and not cfg.warmup,
    )
    moved = any(not torch.allclose(before[name], param.detach()) for name, param in actor.named_lora_params())
    return {
        "layout_dim": layout.dim,
        "n": n,
        "p": p,
        "p_prefix": p_prefix,
        "p_prefix_again": p_again,
        "z": z,
        "records": records,
        "real_loss": float(loss.detach()) if hasattr(loss, "detach") else float(loss),
        "n_completed": int(z.sum()),
        "n_zero_survivors": int((z < 1.0).sum()),
        "leak_warned": leak_flag,
        "actor_moved": moved,
        "selection_counter": rng.counters["selection"],
        "token_counter": rng.counters["token"],
        "audit_counter": rng.counters["audit"],
        "deviation": dev,
        "natural_finish": finished,
    }
