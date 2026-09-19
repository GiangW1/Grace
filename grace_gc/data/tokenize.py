"""Encode prompts. Tiny vocab on CPU; Hugging Face tokenizer on GPU."""

from __future__ import annotations

from collections.abc import Mapping

from grace_gc.data.math_data import MathRecord


def tiny_encode(text: str, vocab: int, max_len: int) -> list[int]:
    if vocab < 3:
        raise ValueError("tiny vocab must be at least 3")
    ids = [min(ord(ch) % (vocab - 1), vocab - 2) for ch in text[:max_len]]
    return ids or [0]


def tiny_decode(token_ids: list[int], eos_id: int) -> str:
    chars = []
    for tok in token_ids:
        if tok == eos_id:
            break
        chars.append(chr(int(tok) % 128))
    return "".join(chars)


def encode_records_tiny(
    records: list[MathRecord],
    vocab: int,
    max_len: int,
    prompt_meta: list | None = None,
) -> tuple[list[list[int]], list[str], list[str]]:
    prompts = []
    for rec in records:
        text = rec.prompt
        prompts.append(tiny_encode(text, vocab, max_len))
        if prompt_meta is not None:
            prompt_meta.append(
                {
                    "used_chat_template": False,
                    "prompt_truncated": len(text) > int(max_len),
                    "untruncated_len": len(text),
                    "thinking_closed": None,
                }
            )
    return prompts, [rec.problem_id for rec in records], [rec.answer for rec in records]


def as_stop_ids(eos_id) -> list[int]:
    if eos_id is None:
        return []
    if isinstance(eos_id, (list, tuple, set)):
        return [int(x) for x in eos_id]
    return [int(eos_id)]


def is_stop_token(tok, eos_id) -> bool:
    return int(tok) in as_stop_ids(eos_id)


def collect_stop_token_ids(tokenizer) -> list[int]:
    """ChatML + document EOS. Qwen3-Base tokenizer.eos_token_id is only one of them."""
    ids: list[int] = []

    def add(value) -> None:
        if value is None:
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                add(item)
            return
        try:
            tok = int(value)
        except (TypeError, ValueError):
            return
        if tok < 0 or tok in ids:
            return
        ids.append(tok)

    add(getattr(tokenizer, "eos_token_id", None))
    gen = getattr(tokenizer, "generation_config", None)
    if gen is not None:
        add(getattr(gen, "eos_token_id", None))
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    unk = getattr(tokenizer, "unk_token_id", None)
    decode = getattr(tokenizer, "convert_ids_to_tokens", None)
    if convert is not None:
        for name in ("<|im_end|>", "<|endoftext|>"):
            try:
                tid = convert(name)
            except Exception:
                continue
            if tid is None or (unk is not None and int(tid) == int(unk)):
                continue
            if decode is not None:
                try:
                    if str(decode(int(tid))) != name:
                        continue
                except Exception:
                    continue
            add(tid)
    if not ids:
        raise ValueError("tokenizer produced no EOS/stop token ids")
    return ids


def load_hf_tokenizer(model_path: str):
    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        raise ImportError("transformers is required to tokenize GPU prompts") from exc
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def _as_id_list(ids) -> list[int]:
    if isinstance(ids, Mapping) and "input_ids" in ids:
        ids = ids["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return [int(x) for x in ids]


def _render_chat(apply, chat, kwargs) -> list[int]:
    try:
        ids = apply(chat, enable_thinking=False, **kwargs)
    except TypeError:
        ids = apply(chat, **kwargs)
    return _as_id_list(ids)


def _fit_chat_to_max(apply, chat, kwargs, max_len: int, rendered: list[int]) -> list[int]:
    """Keep the assistant header. Shorten the last user message, not the template tail."""
    chat = [dict(item) for item in chat]
    user_idx = None
    for i, item in enumerate(chat):
        if str(item.get("role")) == "user":
            user_idx = i
    if user_idx is None:
        return rendered[-max_len:]
    content = str(chat[user_idx].get("content", ""))
    lo, hi = 0, len(content)
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        trial = [dict(item) for item in chat]
        trial[user_idx] = {**trial[user_idx], "content": content[:mid]}
        trial_ids = _render_chat(apply, trial, kwargs)
        if len(trial_ids) <= max_len:
            best = trial_ids
            lo = mid + 1
        else:
            hi = mid - 1
    return best if best is not None else rendered[-max_len:]


def encode_prompt_hf_detail(tokenizer, text: str, max_len: int, messages=None) -> tuple[list[int], dict]:
    """Same encoding as encode_prompt_hf, plus whether the prompt was shortened."""
    apply = getattr(tokenizer, "apply_chat_template", None)
    meta = {
        "used_chat_template": False,
        "prompt_truncated": False,
        "untruncated_len": None,
        "thinking_closed": None,
    }
    if apply is not None and getattr(tokenizer, "chat_template", None):
        chat = list(messages) if messages else [{"role": "user", "content": text}]
        kwargs = {
            "tokenize": True,
            "add_generation_prompt": True,
        }
        ids = _render_chat(apply, chat, kwargs)
        meta["used_chat_template"] = True
        meta["untruncated_len"] = len(ids)
        if len(ids) > int(max_len):
            meta["prompt_truncated"] = True
            ids = _fit_chat_to_max(apply, chat, kwargs, int(max_len), ids)
        decode = getattr(tokenizer, "decode", None)
        if decode is not None:
            rendered = str(decode(ids))
            if "<think>" in rendered and "</think>" not in rendered:
                raise ValueError(
                    "chat template left thinking open; Qwen3 needs enable_thinking=False"
                )
            meta["thinking_closed"] = "<think>" not in rendered or "</think>" in rendered
        return ids, meta
    raw = tokenizer(text, add_special_tokens=True, truncation=False)
    full = _as_id_list(raw["input_ids"] if isinstance(raw, dict) else raw)
    meta["untruncated_len"] = len(full)
    if len(full) > int(max_len):
        meta["prompt_truncated"] = True
        return full[: int(max_len)], meta
    return full, meta


def encode_prompt_hf(tokenizer, text: str, max_len: int, messages=None) -> list[int]:
    """verl/DAPO: chat template + generation prompt when the tokenizer has one."""
    ids, _meta = encode_prompt_hf_detail(tokenizer, text, max_len, messages)
    return ids


def encode_records_hf(
    records: list[MathRecord],
    tokenizer,
    max_len: int,
    prompt_meta: list | None = None,
) -> tuple[list[list[int]], list[str], list[str]]:
    prompts = []
    for rec in records:
        ids, meta = encode_prompt_hf_detail(tokenizer, rec.prompt, max_len, rec.messages)
        prompts.append(ids)
        if prompt_meta is not None:
            prompt_meta.append(meta)
    return prompts, [rec.problem_id for rec in records], [rec.answer for rec in records]


def tokenizer_inventory(tokenizer) -> dict:
    stop_ids = None
    try:
        stop_ids = collect_stop_token_ids(tokenizer)
    except Exception as exc:
        stop_ids = {"error": str(exc)}
    return {
        "name_or_path": getattr(tokenizer, "name_or_path", None),
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "has_chat_template": bool(getattr(tokenizer, "chat_template", None)),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "unk_token_id": getattr(tokenizer, "unk_token_id", None),
        "stop_token_ids": stop_ids,
    }


def decode_hf(tokenizer, token_ids: list[int]) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=True)
