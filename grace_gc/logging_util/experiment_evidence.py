"""Shared checks on recorded experiment facts; no training decisions or file I/O."""

from collections import Counter
import math


EVALUATION_FIELDS = ("ordered_records_sha256", "reward_protocol_version", "samples_per_problem",
                     "temperature", "top_p", "max_new_tokens", "sample_batch_size", "sample_seed_start")


def training_seed_issues(seed, *actual_seeds):
    expected = str(seed).removeprefix("seed-")
    return ["actual_training_seed_missing_or_mismatched"] if any(
        value is None or str(value) != expected for value in actual_seeds) else []


def evaluation_issues(left, right):
    return [f"evaluation_{field}_missing_or_mismatched" for field in EVALUATION_FIELDS
            if left.get(field) in (None, "") or left.get(field) != right.get(field)]


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def response_tokens(row, main=False):
    for key in (("generated_response_tokens", "total_generated_tokens") if main else ("response_tokens",)):
        if is_number(row.get(key)):
            return row[key]
    if main and all(is_number(row.get(key)) for key in ("prefix_tokens", "suffix_tokens")):
        return row["prefix_tokens"] + row["suffix_tokens"]
    full, prompt = row.get("full_token_ids"), row.get("prompt_token_ids")
    if full is None and main:
        full = row.get("prefix_token_ids")
    length = len(prompt) if prompt is not None else row.get("prompt_len")
    return len(full) - length if full is not None and is_number(length) else None


def auxiliary_count_evidence(kind, rows, step, prescan_samples):
    """True: counts verified; False: contradicted; None: legacy facts missing.

    Callers may retain a legacy observed sum, but must not call it verified.
    Token availability and whole-run step coverage are checked by the caller.
    """
    checks = []
    if kind == "prescan":
        count = step.get("n_prescan")  # Problems, not answers.
        if not is_number(count):
            return None
        if count == 0:
            return not rows
        have_ids = all(row.get("problem_id") is not None for row in rows)
        if is_number(prescan_samples):
            checks.append(len(rows) == count * prescan_samples)
        if have_ids:
            groups = Counter(row["problem_id"] for row in rows)
            checks.append(len(groups) == count)
            if is_number(prescan_samples):
                checks.append(all(n == prescan_samples for n in groups.values()))
        known = have_ids and is_number(prescan_samples)
    else:
        fact = ((step.get("predictor") or {}).get("fresh_supervision") or {})
        known = is_number(fact.get("n"))
        if known:
            checks.append(len(rows) == fact["n"])
        tokens = [response_tokens(row) for row in rows]
        if is_number(fact.get("generated_tokens")) and all(is_number(n) for n in tokens):
            checks.append(sum(tokens) == fact["generated_tokens"])
    if not all(checks):
        return False
    return True if known else None
