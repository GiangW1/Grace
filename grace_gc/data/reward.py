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


_LEFT_RIGHT = re.compile(r"\\(?:left|right|bigl|bigr|Bigl|Bigr|biggl|biggr|Biggl|Biggr)\s*")


def normalize_answer(text: str) -> str:
    text = str(text).replace("\u03c0", r"\pi")
    text = _LEFT_RIGHT.sub("", text)
    text = text.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    text = text.replace("\\,", "").replace("\\;", "").replace("\\!", "").replace("\\:", "")
    text = text.replace("$", "")
    text = re.sub(r"\\frac\{(\\pi)\}\{([0-9]+)\}", r"\1/\2", text)
    return re.sub(r"\s+", "", text.strip().lower())


_TEXT_GOLD = re.compile(r"\\text\s*\{([^{}]+)\}")


def text_gold_in_response(text: str, gold: str) -> bool:
    """MATH-500 name answers are stored as \\text{Evelyn}, not a boxed expression."""
    match = _TEXT_GOLD.search(str(gold))
    if not match:
        return False
    name = match.group(1).strip()
    if len(name) < 2:
        return False
    return re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", str(text), re.I) is not None


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
    _ = truncated
    pred = extract_answer(text)
    gold_s = str(gold)
    if pred is not None and normalize_answer(pred) == normalize_answer(gold_s):
        return 1.0
    if text_gold_in_response(text, gold_s):
        return 1.0
    fns = math_verify_fns()
    if fns is None:
        return 0.0
    parse, verify = fns
    if pred is not None:
        try:
            if verify(parse(gold_s), parse(str(pred))):
                return 1.0
        except Exception as exc:
            raise ValueError("math-verify failed; this is not a scored miss") from exc
    try:
        if verify(parse(gold_s), parse(str(text))):
            return 1.0
    except Exception:
        return 0.0
    return 0.0
