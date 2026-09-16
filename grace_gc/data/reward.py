"""Rule reward and first parseable answer index."""

from __future__ import annotations

import re


FINAL = re.compile(r"(?:final answer|answer)\s*[:=]\s*([^\n]+)", re.I)


def extract_boxed(text: str) -> str | None:
    key = "\\boxed{"
    start = text.rfind(key)
    if start < 0:
        return None
    i = start + len(key)
    depth = 1
    out: list[str] = []
    while i < len(text) and depth:
        ch = text[i]
        if ch == "{":
            depth += 1
            out.append(ch)
        elif ch == "}":
            depth -= 1
            if depth:
                out.append(ch)
        else:
            out.append(ch)
        i += 1
    if depth != 0:
        return None
    return "".join(out).strip() or None


def extract_answer(text: str) -> str | None:
    if text is None:
        return None
    boxed = extract_boxed(text)
    if boxed:
        return boxed
    final = FINAL.findall(text)
    if final:
        return final[-1].strip()
    return None


def normalize_answer(text: str) -> str:
    return re.sub(r"\s+", "", text.strip().lower())


def first_parseable_index(tokens_text: list[str]) -> int | None:
    acc = ""
    for i, piece in enumerate(tokens_text):
        acc += piece
        if extract_answer(acc) is not None:
            return i
    return None


def math_verify_fns():
    try:
        from math_verify import parse, verify
    except ImportError:
        return None
    return parse, verify


def require_math_verify() -> None:
    """GPU train/eval/audit need symbolic verify. Missing it changes R, not just metadata."""
    if math_verify_fns() is None:
        raise ImportError(
            "math-verify is required for GPU train/eval/audit; exact-match-only would change rewards"
        )


def rule_reward(text: str | None, gold: str, truncated: bool = False) -> float | None:
    if text is None:
        return None
    pred = extract_answer(text)
    if pred is None:
        _ = truncated
        return 0.0
    if normalize_answer(pred) == normalize_answer(gold):
        return 1.0
    fns = math_verify_fns()
    if fns is None:
        return 0.0
    parse, verify = fns
    try:
        # Keep boxed/source text so math-verify can use its own extractor.
        return 1.0 if verify(parse(str(gold)), parse(str(text))) else 0.0
    except Exception as exc:
        raise ValueError("math-verify failed; this is not a scored miss") from exc
