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
    # Mathematical variables are case-sensitive (A and a can denote different values).
    return re.sub(r"\s+", " ", text.strip())


def _hash_text(text: str) -> str:
    return hashlib.sha256(_normalize(text).encode("utf-8")).hexdigest()


def select_records(records: list[MathRecord], n: int, selection: str = "first", seed: int = 17) -> list[MathRecord]:
    """A common, method-independent subset; seeded selection ignores input order."""
    if n < 0:
        raise ValueError("n_problems must be non-negative")
    if selection not in {"first", "seeded"}:
        raise ValueError(f"unknown data selection {selection!r}")
    ordered = list(records)
    if selection == "seeded":
        import numpy as np

        ordered.sort(key=lambda rec: rec.problem_id)
        order = np.random.default_rng(int(seed)).permutation(len(ordered))
        ordered = [ordered[int(i)] for i in order]
    return ordered[:n] if n else ordered


def selection_manifest(records: list[MathRecord], selection: str, seed: int) -> dict:
    rows = [{"problem_id": r.problem_id, "prompt_sha256": _hash_text(r.prompt),
             "gold_sha256": hashlib.sha256(str(r.answer).encode("utf-8")).hexdigest()}
            for r in records]
    return {"selection": selection, "selection_seed": int(seed), "n_problems": len(rows),
            "prompt_normalization": "whitespace_v2_case_sensitive", "problems": rows,
            "ordered_records_sha256": hashlib.sha256(
                json.dumps(rows, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()}


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


def _read_parquet_rows(path: Path):
    try:
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(path)
    except Exception:
        pf = None
    if pf is not None:
        for batch in pf.iter_batches(batch_size=8192):
            for raw in batch.to_pylist():
                yield raw
        return
    try:
        import pandas as pd
    except Exception as exc:
        raise ImportError(
            f"reading {path} needs pyarrow or pandas; export JSONL or install one of them"
        ) from exc
    for raw in pd.read_parquet(path).to_dict(orient="records"):
        yield raw


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


def last_load_report() -> dict | None:
    report = getattr(load_math_records, "last", None)
    return None if report is None else dict(report)


def looks_like_math500(records=None, path=None, n_source=None) -> bool:
    hint = str(path or "").replace("\\", "/").lower()
    if "math-500" in hint or "math500" in hint or "math_500" in hint:
        return True
    if n_source is not None and int(n_source) == 500:
        return True
    recs = list(records or [])
    if n_source is None and len(recs) == 500:
        return True
    for rec in recs[:16]:
        src = str(getattr(rec, "source", "") or "").lower()
        if "math-500" in src or "math500" in src or "math_500" in src:
            return True
    return False


def load_math_records(path: str | Path) -> list[MathRecord]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"data file is not readable: {path}")
    seen: dict[str, MathRecord] = {}
    order: list[str] = []
    conflict_keys: dict[str, dict] = {}
    first_loc: dict[str, int] = {}
    n_raw = 0
    n_same = 0
    known_splits = {"train", "calib", "audit", "eval"}
    for loc, raw in _raw_rows(path):
        n_raw += 1
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
            if prev.split is not None and rec.split is not None and prev.split != rec.split:
                raise ValueError(f"duplicate prompt split conflict at {path}:{loc}")
            if prev.answer != rec.answer:
                info = conflict_keys.setdefault(
                    key,
                    {
                        "prompt_hash": key,
                        "problem_id": prev.problem_id,
                        "answers": {prev.answer},
                        "first_loc": first_loc.get(key, loc),
                    },
                )
                info["answers"].add(rec.answer)
                continue
            n_same += 1
            if prev.split is None and rec.split is not None:
                prev.split = rec.split
            continue
        seen[key] = rec
        order.append(key)
        first_loc[key] = loc
    conflicts = []
    out: list[MathRecord] = []
    for key in order:
        if key in conflict_keys:
            info = conflict_keys[key]
            conflicts.append(
                {
                    "prompt_hash": info["prompt_hash"],
                    "problem_id": info["problem_id"],
                    "answers": sorted(info["answers"]),
                    "first_loc": info["first_loc"],
                }
            )
            continue
        out.append(seen[key])
    report = {
        "prompt_normalization": "whitespace_v2_case_sensitive",
        "path": str(path),
        "n_raw": n_raw,
        "n_kept": len(out),
        "n_same_answer_dedup": n_same,
        "n_conflict_groups": len(conflicts),
        "conflicts": conflicts,
    }
    load_math_records.last = report
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


def load_training_data(data_path, eval_data_path=None, seed: int = 17):
    """Split first, then exclude external evaluation problems before any training use.

    Match problem text, never gold or source-specific IDs. This detects exact text
    after whitespace/known DAPO wrapper normalization, not semantic near-duplicates.
    """
    from grace_gc.data.format_prompt import apply_solve_instruction, DAPO_SOLVE_PREFIX, DAPO_SOLVE_SUFFIX
    from grace_gc.versions import sha256_file

    records = load_math_records(data_path)
    report = last_load_report()
    buckets = split_records(apply_solve_instruction(records), seed=seed)
    exclusion = {"status": "not_provided", "eval_source": None, "removed": [],
                 "normalization": "whitespace_and_exact_dapo_wrapper_case_sensitive",
                 "scope": "train/calib/audit pools before SFT and sampling; all external eval problems",
                 "note": "No semantic near-duplicate detection; gold does not affect exclusion. This does not certify the exposure history of a reused actor/checkpoint."}
    if eval_data_path:
        evaluation = load_math_records(eval_data_path)

        def key(rec):
            # Compare user content when DAPO supplies chat messages.
            users = [m["content"] for m in (rec.messages or []) if m["role"] == "user"]
            text = _normalize(users[-1] if users else rec.prompt)
            prefix, suffix = _normalize(DAPO_SOLVE_PREFIX), _normalize(DAPO_SOLVE_SUFFIX)
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
            if text.endswith(suffix):
                text = text[:-len(suffix)].strip()
            return _hash_text(text)

        heldout = {}
        for rec in evaluation:
            heldout.setdefault(key(rec), []).append(rec.problem_id)
        exclusion.update(status="applied", eval_source={"path": str(eval_data_path),
                         "sha256": sha256_file(eval_data_path), "n_problems": len(evaluation)})
        for split in ("train", "calib", "audit"):
            kept = []
            for rec in buckets[split]:
                digest = key(rec)
                if digest in heldout:
                    exclusion["removed"].append({"split": split, "problem_id": rec.problem_id,
                        "prompt_sha256": digest, "eval_problem_ids": heldout[digest]})
                else:
                    kept.append(rec)
            buckets[split] = kept
    report["eval_exclusion"] = exclusion
    # Existing callers of last_load_report must still see the training source.
    load_math_records.last = report
    return buckets, report
