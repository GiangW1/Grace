"""Math records, dedup, and problem-level splits."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from grace_gc.data.reward import extract_answer


@dataclass
class MathRecord:
    problem_id: str
    prompt: str
    answer: str
    source: str | None = None
    split: str | None = None
    messages: list | None = None


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _hash_text(text: str) -> str:
    return hashlib.sha256(_normalize(text).encode("utf-8")).hexdigest()


def _as_rows(value):
    if value is None or isinstance(value, (str, dict, bytes)):
        return value
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value


def _maybe_structured(value):
    """Parquet/pandas sometimes stores chat lists and reward_model as JSON text."""
    value = _as_rows(value)
    if isinstance(value, str):
        text = value.strip()
        if text[:1] in "{[":
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return value
    return value


def _chat_messages(value):
    """Keep DAPO role/content lists for apply_chat_template."""
    value = _maybe_structured(value)
    if not isinstance(value, (list, tuple)) or not value:
        return None
    out = []
    for item in value:
        if not isinstance(item, dict):
            return None
        role = item.get("role")
        content = item.get("content") if item.get("content") is not None else item.get("text")
        if role is None or content is None:
            return None
        out.append({"role": str(role), "content": str(content)})
    return out


def _prompt_text(value) -> str:
    """Flatten DAPO chat lists. `str(list)` is not a prompt."""
    value = _maybe_structured(value)
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return _prompt_text(value.get("content") if value.get("content") is not None else value.get("text"))
    if isinstance(value, (list, tuple)):
        return "\n".join(part for part in (_prompt_text(item) for item in value) if part)
    return ""


def _answer_text(raw: dict) -> str:
    if "answer" in raw and raw["answer"] is not None:
        text = str(raw["answer"])
    elif "gt" in raw and raw["gt"] is not None:
        text = str(raw["gt"])
    else:
        reward = _maybe_structured(raw.get("reward_model"))
        if isinstance(reward, dict) and reward.get("ground_truth") is not None:
            text = str(reward["ground_truth"])
        elif "solution" in raw and raw["solution"] is not None:
            text = str(raw["solution"])
        else:
            return ""
    extracted = extract_answer(text)
    return extracted if extracted is not None else text.strip()


def _experiment_split(raw: dict):
    """Only the top-level experiment split. DAPO extra_info.split=train is the source corpus."""
    split = raw.get("split")
    if split is None:
        return None
    return str(split)


def _first_present(*values, fallback: str) -> str:
    """Keep integer 0. `or` would drop verl DAPO extra_info.index=0 and reshuffle splits."""
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return str(value)
    return fallback


def _problem_id(raw: dict, fallback: str) -> str:
    extra = _maybe_structured(raw.get("extra_info"))
    extra = extra if isinstance(extra, dict) else {}
    return _first_present(
        raw.get("problem_id"),
        raw.get("id"),
        extra.get("index"),
        extra.get("id"),
        fallback=fallback,
    )


def _read_parquet_rows(path: Path) -> list[dict]:
    try:
        import pyarrow.parquet as pq

        return pq.read_table(path).to_pylist()
    except Exception:
        try:
            import pandas as pd
        except Exception as exc:
            raise ImportError(
                f"reading {path} needs pyarrow or pandas; export JSONL or install one of them"
            ) from exc
        return pd.read_parquet(path).to_dict(orient="records")


def _raw_rows(path: Path):
    if path.suffix.lower() == ".parquet":
        for i, raw in enumerate(_read_parquet_rows(path), start=1):
            if not isinstance(raw, dict):
                raise ValueError(f"invalid parquet row at {path}:{i}")
            yield i, raw
        return
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_no}") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"invalid JSONL row at {path}:{line_no}")
        yield line_no, raw


def load_math_records(path: str | Path) -> list[MathRecord]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"data file is not readable: {path}")
    seen: dict[str, MathRecord] = {}
    out: list[MathRecord] = []
    known_splits = {"train", "calib", "audit", "eval"}
    for loc, raw in _raw_rows(path):
        prompt = _prompt_text(raw.get("prompt") if raw.get("prompt") is not None else raw.get("problem") or raw.get("question"))
        answer = _answer_text(raw)
        if not prompt:
            raise ValueError(f"missing prompt at {path}:{loc}")
        if answer == "":
            raise ValueError(f"missing answer/ground_truth at {path}:{loc}")
        split = _experiment_split(raw)
        if split is not None and split not in known_splits:
            raise ValueError(f"unknown split {split!r} at {path}:{loc}")
        key = _hash_text(prompt)
        rec = MathRecord(
            problem_id=_problem_id(raw, key[:16]),
            prompt=prompt,
            answer=answer,
            source=raw.get("source") or raw.get("data_source"),
            split=split,
            messages=_chat_messages(raw.get("prompt")),
        )
        if key in seen:
            prev = seen[key]
            if prev.answer != rec.answer or (
                prev.split is not None and rec.split is not None and prev.split != rec.split
            ):
                raise ValueError(f"duplicate prompt conflict at {path}:{loc}")
            if prev.split is None and rec.split is not None:
                prev.split = rec.split
            continue
        seen[key] = rec
        out.append(rec)
    return out


PAPER_N_CALIB = 256
PAPER_N_AUDIT = 240
# MATH-500 / AIME stay dedicated files. DAPO-Math-17k is above this.
PAPER_CORPUS_MIN = 2000


def split_records(
    records: list[MathRecord],
    train_frac: float | None = None,
    calib_frac: float | None = None,
    audit_frac: float | None = None,
    seed: int = 17,
    n_calib: int = PAPER_N_CALIB,
    n_audit: int = PAPER_N_AUDIT,
    corpus_min: int = PAPER_CORPUS_MIN,
) -> dict[str, list[MathRecord]]:
    """Honor marked splits. Unmarked DAPO-scale files lose paper calib/audit; the rest is train.

    Eval is MATH-500 (a separate file), not a random slice of DAPO.
    Explicit fractions keep the older 70/10/10/10 helper for small tests.
    """
    use_fractions = train_frac is not None or calib_frac is not None or audit_frac is not None
    if use_fractions:
        train_frac = 0.7 if train_frac is None else float(train_frac)
        calib_frac = 0.1 if calib_frac is None else float(calib_frac)
        audit_frac = 0.1 if audit_frac is None else float(audit_frac)
        if train_frac <= 0 or calib_frac < 0 or audit_frac < 0:
            raise ValueError("split fractions must be non-negative and train_frac > 0")
    buckets = {"train": [], "calib": [], "audit": [], "eval": []}
    marked_by_pid: dict[str, str] = {}
    for rec in records:
        if rec.split not in buckets:
            continue
        prev = marked_by_pid.get(rec.problem_id)
        if prev is not None and prev != rec.split:
            raise ValueError(f"conflict split for problem_id {rec.problem_id}: {prev} vs {rec.split}")
        marked_by_pid[rec.problem_id] = rec.split
    unmarked = []
    for rec in records:
        if rec.split in buckets:
            buckets[rec.split].append(rec)
        elif rec.problem_id in marked_by_pid:
            buckets[marked_by_pid[rec.problem_id]].append(rec)
        else:
            unmarked.append(rec)
    if not unmarked:
        return buckets
    ids = sorted({r.problem_id for r in unmarked})
    rng = __import__("numpy").random.default_rng(seed)
    rng.shuffle(ids)
    n = len(ids)
    if use_fractions or marked_by_pid:
        n_train = max(1, int(n * float(train_frac if train_frac is not None else 0.7))) if ids else 0
        n_cal = int(n * float(calib_frac if calib_frac is not None else 0.1))
        n_aud = int(n * float(audit_frac if audit_frac is not None else 0.1))
        train_ids = set(ids[:n_train])
        calib_ids = set(ids[n_train : n_train + n_cal])
        audit_ids = set(ids[n_train + n_cal : n_train + n_cal + n_aud])
        eval_ids = set(ids[n_train + n_cal + n_aud :])
    elif n >= int(corpus_min):
        n_cal = min(int(n_calib), n)
        n_aud = min(int(n_audit), max(0, n - n_cal))
        calib_ids = set(ids[:n_cal])
        audit_ids = set(ids[n_cal : n_cal + n_aud])
        train_ids = set(ids[n_cal + n_aud :])
        eval_ids = set()
    else:
        train_ids = set(ids)
        calib_ids = set()
        audit_ids = set()
        eval_ids = set()
    for rec in unmarked:
        if rec.problem_id in train_ids:
            buckets["train"].append(rec)
        elif rec.problem_id in calib_ids:
            buckets["calib"].append(rec)
        elif rec.problem_id in audit_ids:
            buckets["audit"].append(rec)
        elif rec.problem_id in eval_ids:
            buckets["eval"].append(rec)
        else:
            buckets["train"].append(rec)
    return buckets


def records_for_split(
    records: list[MathRecord],
    name: str,
    seed: int = 17,
    **kwargs,
) -> list[MathRecord]:
    """Eval/audit entry: marked bucket, reserved DAPO audit slice, or a dedicated file."""
    if name not in {"train", "calib", "audit", "eval"}:
        raise ValueError(f"unknown split {name!r}")
    buckets = split_records(records, seed=seed, **kwargs)
    chosen = buckets[name]
    if chosen:
        return chosen
    if any(rec.split is not None for rec in records):
        return []
    n = len({rec.problem_id for rec in records})
    corpus_min = int(kwargs.get("corpus_min", PAPER_CORPUS_MIN))
    if n >= corpus_min and name == "eval":
        raise ValueError(
            "eval split is empty on a train-sized unmarked corpus; "
            "pass MATH-500 or mark split=eval"
        )
    return list(records)
