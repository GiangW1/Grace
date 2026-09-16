"""Shared format SFT on q/v LoRA. This is not predictor warmup_steps."""

from __future__ import annotations

from typing import Any

from grace_gc.data.format_prompt import format_sft_lead
from grace_gc.data.math_data import MathRecord
from grace_gc.logging_util.ledger import Timer


def format_warmup_cfg(cfg: dict[str, Any] | None) -> dict[str, Any]:
    block = (cfg or {}).get("format_warmup") or {}
    lr = block.get("lr")
    if lr is None:
        lr = ((cfg or {}).get("optim") or {}).get("lr", 1e-4)
    return {
        "steps": int(block.get("steps", 0) or 0),
        "batch_size": max(1, int(block.get("batch_size", 4) or 4)),
        "lr": float(lr),
    }


def format_warmup_steps(cfg: dict[str, Any] | None) -> int:
    return int(format_warmup_cfg(cfg)["steps"])


def _sft_summary(steps: int, n_examples: int, losses: list[float], skipped: bool = False) -> dict[str, Any]:
    return {
        "steps": steps,
        "n_examples": n_examples,
        "skipped": skipped,
        "last_loss": None if not losses else float(losses[-1]),
        "mean_loss": None if not losses else float(sum(losses) / len(losses)),
        "trained": "reasoning lead",
        "gold_in_loss": False,
        "eos_in_loss": False,
    }


def run_format_warmup_tiny(actor, records: list[MathRecord], vocab: int, cfg: dict[str, Any], ledger=None) -> dict[str, Any]:
    fw = format_warmup_cfg(cfg)
    steps = fw["steps"]
    if steps <= 0 or not records:
        return _sft_summary(0, len(records or []), [], skipped=True)
    from grace_gc.data.tokenize import tiny_encode
    from grace_gc.trainer.cpu_tiny import _torch

    torch, _nn, F = _torch()
    opt = torch.optim.SGD(actor.trainable_params(), lr=fw["lr"])
    bs = min(fw["batch_size"], len(records))
    max_prompt = int(cfg.get("prompt_max_tokens", 64))
    losses: list[float] = []
    timer = Timer()
    n = len(records)
    for step in range(steps):
        start = (step * bs) % n
        batch = [records[(start + i) % n] for i in range(bs)]
        seqs = []
        prompt_lens = []
        for rec in batch:
            prompt = tiny_encode(rec.prompt, vocab, max_prompt)
            lead = tiny_encode(format_sft_lead(), vocab, 32)
            seqs.append(prompt + lead)
            prompt_lens.append((len(prompt), len(prompt) + len(lead)))
        width = max(len(seq) for seq in seqs)
        ids = torch.full((len(seqs), width), int(actor.eos_id), dtype=torch.long)
        for i, seq in enumerate(seqs):
            ids[i, : len(seq)] = torch.as_tensor(seq, dtype=torch.long)
        logits, _h = actor.forward(ids)
        loss = None
        ntok = 0
        for i, (p0, p1) in enumerate(prompt_lens):
            if p1 <= p0:
                continue
            start_logit = max(p0 - 1, 0)
            part = F.cross_entropy(logits[i, start_logit : p1 - 1], ids[i, start_logit + 1 : p1], reduction="sum")
            loss = part if loss is None else loss + part
            ntok += p1 - p0
        if loss is None or ntok <= 0:
            continue
        loss = loss / ntok
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))
    summary = _sft_summary(steps, n, losses)
    if ledger is not None:
        ledger.add("format_warmup", timer.elapsed(), **{k: v for k, v in summary.items() if k != "skipped"})
    return summary


def _encode_sft_ids(tokenizer, text: str) -> list[int]:
    from grace_gc.data.tokenize import _as_id_list

    return list(_as_id_list(tokenizer(text, add_special_tokens=False)))


def _encode_sft_pair(tokenizer, rec: MathRecord, max_prompt: int) -> tuple[list[int], list[int], int]:
    from grace_gc.data.tokenize import encode_prompt_hf_detail

    prompt_ids, _meta = encode_prompt_hf_detail(tokenizer, rec.prompt, max_prompt, rec.messages)
    resp_ids = _encode_sft_ids(tokenizer, format_sft_lead())
    if not resp_ids:
        raise ValueError("format warmup produced an empty response")
    return list(prompt_ids), resp_ids, len(resp_ids)


def run_format_warmup_hf(actor, tokenizer, records: list[MathRecord], cfg: dict[str, Any], ledger=None) -> dict[str, Any]:
    fw = format_warmup_cfg(cfg)
    steps = fw["steps"]
    if steps <= 0 or not records:
        return _sft_summary(0, len(records or []), [], skipped=True)
    import torch
    import torch.nn.functional as F

    from grace_gc.backends.hf_actor import trainable_params

    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", 0)
    pad_id = int(pad_id)
    opt = torch.optim.AdamW(trainable_params(actor), lr=fw["lr"])
    bs = min(fw["batch_size"], len(records))
    max_prompt = int(cfg.get("prompt_max_tokens", 1024))
    device = next(actor.parameters()).device
    losses: list[float] = []
    timer = Timer()
    n = len(records)
    for step in range(steps):
        start = (step * bs) % n
        batch = [records[(start + i) % n] for i in range(bs)]
        pairs = [_encode_sft_pair(tokenizer, rec, max_prompt) for rec in batch]
        width = max(len(prompt) + len(resp) for prompt, resp, _n_train in pairs)
        ids = torch.full((bs, width), pad_id, dtype=torch.long)
        labels = torch.full((bs, width), -100, dtype=torch.long)
        attn = torch.zeros((bs, width), dtype=torch.long)
        for i, (prompt, resp, n_train) in enumerate(pairs):
            seq = prompt + resp
            ids[i, : len(seq)] = torch.as_tensor(seq, dtype=torch.long)
            trained = resp[:n_train]
            if trained:
                labels[i, len(prompt) : len(prompt) + n_train] = torch.as_tensor(trained, dtype=torch.long)
            attn[i, : len(seq)] = 1
        ids = ids.to(device)
        labels = labels.to(device)
        attn = attn.to(device)
        out = actor(input_ids=ids, attention_mask=attn)
        logits = out.logits[:, :-1].contiguous()
        tgt = labels[:, 1:].contiguous()
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))
    summary = _sft_summary(steps, n, losses)
    if ledger is not None:
        ledger.add("format_warmup", timer.elapsed(), **{k: v for k, v in summary.items() if k != "skipped"})
    return summary
