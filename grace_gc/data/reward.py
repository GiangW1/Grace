"""Rule reward and first parseable answer index."""

from __future__ import annotations

import json
from functools import lru_cache
import os
import re
import subprocess
import sys


FINAL = re.compile(r"(?:final\s+answer|answer)\s*(?:is\s+|[:=]\s*)(?:\n\s*)?([^\n]+)", re.I)
REWARD_PROTOCOL_VERSION = 2


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
# Digit-prefixed 2pi / 3pi/2 count; pine / api / \pi do not.
_PLAIN_PI = re.compile(r"(?<!\\)(?<![A-Za-z_])pi(?![A-Za-z_])", re.I)


def _gold_has_pi_constant(text: str) -> bool:
    return bool(re.search(r"\\pi\b", str(text).replace("\u03c0", r"\pi")))


def _plain_pi_to_latex(text: str) -> str:
    return _PLAIN_PI.sub(r"\\pi", str(text))


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
    return re.sub(r"\s+", "", text.strip())


_NUMERIC_CORE = re.compile(r"^-?\d+(?:/\d+)?$")
_UNIT_SUFFIXES = {"mm", "cm", "km", "kg", "mg", "mph", "feet", "foot", "inches", "inch",
                  "meters", "metres", "centimeters", "centimetres", "seconds", "minutes", "hours"}


def _compat_number_unit(pred: str, gold: str) -> bool:
    """Accept known unit suffixes, never arbitrary algebraic variable suffixes."""
    if not pred or not gold or pred == gold:
        return pred == gold
    short, long = (pred, gold) if len(pred) <= len(gold) else (gold, pred)
    extra = long[len(short) :]
    return bool(_NUMERIC_CORE.fullmatch(short)) and long.startswith(short) and extra.lower() in _UNIT_SUFFIXES


_TEXT_GOLD = re.compile(r"\\text\s*\{([^{}]+)\}")


def text_gold_in_response(text: str, gold: str) -> bool:
    """MATH-500 name answers are stored as \\text{Evelyn}, not a boxed expression."""
    match = _TEXT_GOLD.fullmatch(str(gold).strip())
    if not match:
        return False
    name = match.group(1).strip()
    if len(name) < 2:
        return False
    # Accept the answer itself or a single explicit concluding assertion, not
    # mentions of the gold name anywhere in a longer reasoning paragraph.
    return re.fullmatch(
        r"\s*(?:the\s+[\w -]+\s+is\s+)?" + re.escape(name) + r"[.!]?\s*",
        str(text), re.I,
    ) is not None


def first_parseable_index(tokens_text: list[str]) -> int | None:
    acc = ""
    for i, piece in enumerate(tokens_text):
        acc += piece
        if extract_answer(acc) is not None:
            return i
    return None


_VERIFY_WORKER = """import json, sys
from math_verify import parse, verify
gold, pred = json.load(sys.stdin)
g = parse(gold, parsing_timeout=None)
p = parse(pred, parsing_timeout=None)
print(json.dumps(bool(verify(g, p, timeout_seconds=None))))
"""


@lru_cache(maxsize=4096)
def _windows_verify(gold: str, pred: str) -> bool:
    """Bound the whole symbolic operation without math-verify's broken Win32 IPC."""
    result = subprocess.run(
        [sys.executable, "-c", _VERIFY_WORKER], input=json.dumps([gold, pred]),
        text=True, encoding="utf-8", capture_output=True, timeout=15, check=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    return bool(json.loads(result.stdout.strip().splitlines()[-1]))


def math_verify_fns():
    try:
        from math_verify import parse, verify
    except ImportError:
        return None
    if os.name == "nt":
        # One bounded process for both parses and verify; no callable pickling.
        return str, _windows_verify
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
    gold_s = str(gold).replace("\u03c0", r"\pi")
    if pred is not None:
        pred_n = normalize_answer(pred)
        gold_n = normalize_answer(gold_s)
        if _gold_has_pi_constant(gold_s):
            pred_n = _plain_pi_to_latex(pred_n)
        if pred_n == gold_n or _compat_number_unit(pred_n, gold_n):
            return 1.0
    if text_gold_in_response(pred if pred is not None else text, gold_s):
        return 1.0
    fns = math_verify_fns()
    if fns is None:
        return 0.0
    parse, verify = fns
    # Golds and extracted answers are expressions, not prose to scan for numbers.
    gold_expr = r"\boxed{" + gold_s + "}"
    if pred is None:
        return 0.0
    pred_expr = str(pred).replace("\u03c0", r"\pi")
    if _gold_has_pi_constant(gold_expr):
        pred_expr = _plain_pi_to_latex(pred_expr)
    try:
        return float(verify(parse(gold_expr), parse(r"\boxed{" + pred_expr + "}")))
    except Exception as exc:
        raise ValueError("math-verify failed; this is not a scored miss") from exc
