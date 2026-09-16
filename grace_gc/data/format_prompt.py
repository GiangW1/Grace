"""DAPO / MATH-500 shared solve instruction and SFT target text."""

from __future__ import annotations

from dataclasses import replace

from grace_gc.data.math_data import MathRecord


DAPO_SOLVE_PREFIX = (
    "Solve the following math problem step by step. "
    "The last line of your response should be of the form Answer: $Answer "
    "(without quotes) where $Answer is the answer to the problem.\n\n"
)
DAPO_SOLVE_SUFFIX = '\n\nRemember to put your answer on its own line after "Answer:".'


def has_solve_instruction(text: str) -> bool:
    low = str(text).lower()
    if "the last line of your response should be" in low:
        return True
    if "answer: $answer" in low:
        return True
    if "remember to put your answer" in low:
        return True
    return low.lstrip().startswith("solve the following math problem")


def ensure_solve_instruction(text: str) -> str:
    text = str(text)
    if has_solve_instruction(text):
        return text
    return f"{DAPO_SOLVE_PREFIX}{text}{DAPO_SOLVE_SUFFIX}"


def wrap_messages(messages: list | None) -> list | None:
    if not messages:
        return None
    out = [dict(item) for item in messages]
    for i in range(len(out) - 1, -1, -1):
        if str(out[i].get("role")) == "user":
            out[i]["content"] = ensure_solve_instruction(str(out[i].get("content", "")))
            break
    return out


def apply_solve_instruction(records: list[MathRecord]) -> list[MathRecord]:
    out: list[MathRecord] = []
    for rec in records:
        prompt = ensure_solve_instruction(rec.prompt)
        messages = wrap_messages(rec.messages) if rec.messages else None
        if messages is None and prompt != rec.prompt:
            messages = [{"role": "user", "content": prompt}]
        if prompt == rec.prompt and messages == rec.messages:
            out.append(rec)
        else:
            out.append(replace(rec, prompt=prompt, messages=messages))
    return out


# First tokens of the SFT target. CE trains these so generation does not start at Answer:.
FORMAT_SFT_LEAD = "I will solve this step by step.\n\n"
FORMAT_SFT_ANSWER_PREFIX = "Answer: "


def format_sft_lead() -> str:
    return FORMAT_SFT_LEAD


def format_sft_answer_prefix() -> str:
    return FORMAT_SFT_ANSWER_PREFIX


def format_sft_response(gold: str) -> str:
    """Last line. Matches the DAPO 'Answer: $Answer' instruction."""
    return f"{FORMAT_SFT_ANSWER_PREFIX}{str(gold).strip()}"


def format_sft_text(gold: str) -> str:
    return f"{FORMAT_SFT_LEAD}{format_sft_response(gold)}"
