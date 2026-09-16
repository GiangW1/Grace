"""Rule reward and first parseable answer index."""

from __future__ import annotations

import re


FINAL = re.compile(r"(?:final\s+answer|answer)\s*(?:is\s+|[:=]\s*)(?:\n\s*)?([^\n]+)", re.I)


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


def _boxed_span_end(text: str) -> int | None:
    key = "\\boxed{"
    start = text.rfind(key)
    if start < 0:
        return None
    i = start + len(key)
    depth = 1
    while i < len(text) and depth:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    if depth != 0:
        return None
    return i


def answer_already_emitted(text: str | None) -> bool:
    """True only when the prefix already committed a final-line answer."""
    if not text:
        return False
    s = str(text)
    finals = list(FINAL.finditer(s))
    if finals and not s[finals[-1].end() :].strip():
        return True
    end = _boxed_span_end(s)
    return end is not None and not s[end:].strip()


def extract_answer(text: str) -> str | None:
    if text is None:
        return None
    boxed = extract_boxed(text)
    finals = list(FINAL.finditer(text))
    last_final = finals[-1].group(1).strip() if finals else ""
    if not last_final:
        last_final = None
    if boxed and last_final:
        boxed_pos = text.rfind("\\boxed{")
        return boxed if boxed_pos > finals[-1].start() else last_final
    return boxed or last_final


_LEFT_RIGHT = re.compile(r"\\(?:left|right|bigl|bigr|Bigl|Bigr|biggl|biggr|Biggl|Biggr)\s*")


def normalize_answer(text: str) -> str:
    text = str(text).replace("\u03c0", r"\pi")
    text = _LEFT_RIGHT.sub("", text)
    text = text.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    text = text.replace("\\,", "").replace("\\;", "").replace("\\!", "").replace("\\:", "")
    text = text.replace("$", "")
    text = re.sub(r"\^?(?:\{\\circ\}|\\circ|°)", "", text)
    text = re.sub(r"degrees?", "", text, flags=re.I)
    text = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\frac\{(-?[0-9]+)\}\{(-?[0-9]+)\}", r"\1/\2", text)
    text = re.sub(r"\\frac\{(\\pi)\}\{([0-9]+)\}", r"\1/\2", text)
    return re.sub(r"\s+", "", text.strip().lower())


def _compat_number_unit(pred: str, gold: str) -> bool:
    """5 vs 5cm. Does not equate 5cm with 5mm."""
    if not pred or not gold or pred == gold:
        return pred == gold
    short, long = (pred, gold) if len(pred) <= len(gold) else (gold, pred)
    extra = long[len(short) :]
    return long.startswith(short) and extra.isalpha()


_TEXT_GOLD = re.compile(r"\\text\s*\{([^{}]+)\}")


def text_gold_in_response(text: str, gold: str) -> bool:
    """MATH-500 name answers are stored as \\text{Evelyn}, not a boxed expression."""
    match = _TEXT_GOLD.fullmatch(str(gold).strip())
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
    if pred is not None:
        pred_n = normalize_answer(pred)
        gold_n = normalize_answer(gold_s)
        if pred_n == gold_n or _compat_number_unit(pred_n, gold_n):
            return 1.0
    if text_gold_in_response(pred if pred is not None else text, gold_s):
        return 1.0
    fns = math_verify_fns()
    if fns is None:
        return 0.0
    parse, verify = fns
    # Golds and extracted answers are expressions, not prose to scan for numbers.
    gold_expr = r"\boxed{" + gold_s.replace("\u03c0", r"\pi") + "}"
    if pred is None:
        return 0.0
    pred_expr = str(pred).replace("\u03c0", r"\pi")
    # Plain-text pi denotes the constant when the gold explicitly contains it.
    # Otherwise leave variable products such as p*i to the symbolic parser.
    if re.search(r"\\pi\b", gold_expr):
        pred_expr = re.sub(r"(?<![\\\w])pi(?!\w)", lambda _m: r"\pi", pred_expr)
    try:
        return float(verify(parse(gold_expr), parse(r"\boxed{" + pred_expr + "}")))
    except Exception as exc:
        raise ValueError("math-verify failed; this is not a scored miss") from exc
