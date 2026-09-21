"""Summarize observed training signals; never re-score answers or impute missing G."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from grace_gc.logging_util.experiment_evidence import (
    auxiliary_count_evidence, is_number as _number, response_tokens as _tokens,
)


def _read(path, rows=False):
    if not path.is_file():
        return [] if rows else {}
    text = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()] if rows else json.loads(text)


def _hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stats(values):
    values = list(values)
    known = [v for v in values if _number(v)]
    return {"observed": len(known), "missing": len(values) - len(known),
            "sum": sum(known) if len(known) == len(values) else None,
            "observed_sum": sum(known), "mean": statistics.fmean(known) if known else None,
            "min": min(known) if known else None, "max": max(known) if known else None,
            "zero": sum(v == 0 for v in known), "positive": sum(v > 0 for v in known),
            "negative": sum(v < 0 for v in known)}


def _booleans(values):
    values = list(values)
    known = [v for v in values if isinstance(v, bool)]
    return {"observed": len(known), "missing": len(values) - len(known), "true": sum(known),
            "rate_among_observed": sum(known) / len(known) if known else None}


def _groups(rows, main=True):
    by_group = defaultdict(list)
    for row in rows:
        by_group[(row.get("step"), row.get("problem_id"))].append(row)
    groups = []
    for (step, pid), members in by_group.items():
        completed = [r for r in members if not main or r.get("z_continue") == 1]
        rewards = [r.get("reward") for r in completed]
        known = [r for r in rewards if _number(r)]
        adv = [r.get("advantage") for r in completed]
        complete_adv = bool(adv) and all(_number(a) for a in adv)
        groups.append({"step": step, "problem_id": pid, "starts": len(members), "completed": len(completed),
                       "reward_observed": len(known), "reward_missing": len(rewards) - len(known),
                       "reward_mean": statistics.fmean(known) if known else None,
                       "reward_sample_variance": statistics.variance(known) if len(known) >= 2 else None,
                       "distinct_observed_rewards": len(set(known)),
                       "all_observed_reward_zero": all(r == 0 for r in known) if known else None,
                       "all_observed_reward_one": all(r == 1 for r in known) if known else None,
                       "all_completed_adv_zero": all(abs(a) < 1e-12 for a in adv) if complete_adv else None,
                       "baseline_values": sorted({r["baseline_b"] for r in members if _number(r.get("baseline_b"))}),
                       "complete_rewards": len(known) == len(completed) and bool(completed)})
    return groups


def _group_counts(groups):
    return {"observed": len(groups), "singleton_completed_groups": sum(g["completed"] == 1 for g in groups),
            "unique_problems": len({g["problem_id"] for g in groups}),
            "repeated_problem_batches": len(groups) - len({g["problem_id"] for g in groups}),
            "no_completed_groups": sum(g["completed"] == 0 for g in groups),
            "mixed_reward_groups": sum(g["distinct_observed_rewards"] > 1 for g in groups),
            "all_observed_reward_zero": sum(g["all_observed_reward_zero"] is True for g in groups),
            "all_observed_reward_one": sum(g["all_observed_reward_one"] is True for g in groups),
            "all_completed_adv_zero": sum(g["all_completed_adv_zero"] is True for g in groups),
            "unknown_all_completed_adv_zero": sum(g["all_completed_adv_zero"] is None for g in groups)}


def _auxiliary(kind, rows, steps, exists, prescan_samples):
    ids = {s["step"] for s in steps}
    selected = [r for r in rows if r.get("step") in ids]
    groups = _groups(selected, main=False)
    if kind == "prescan":
        zero = all(s.get("n_prescan") == 0 for s in steps)
    else:
        facts = [(s.get("predictor") or {}).get("fresh_supervision") or {} for s in steps]
        zero = all(f.get("n") == 0 and f.get("generated_tokens") == 0 for f in facts)
    complete = all(auxiliary_count_evidence(kind,
        [row for row in selected if row.get("step") == step["step"]], step, prescan_samples) is True
        for step in steps)
    complete = complete and (exists or zero) and (bool(steps) or exists)
    tokens = _stats(_tokens(r) for r in selected)
    complete = complete and tokens["missing"] == 0
    generated = tokens["sum"] if complete else None
    return {"log_exists": exists, "complete": complete, "zero_proven_by_step_facts": zero and not selected and bool(steps),
            "rows_observed": len(selected), "generated_tokens": generated,
            "tokens_observed": tokens, "reward": _stats(r.get("reward") for r in selected),
            "advantage": _stats(r.get("advantage") for r in selected),
            "gradient_norm_sq": _stats(r.get("gradient_norm_sq") for r in selected),
            "truncated": _booleans(r.get("truncated") for r in selected),
            "baseline": _stats(r.get("baseline_after_prescan", r.get("baseline_b")) for r in selected),
            "group_counts": _group_counts(groups)}


def summarize_training(train):
    train = Path(train).resolve()
    filenames = ("steps.jsonl", "trajectories.jsonl", "prescan.jsonl", "fresh_supervision.jsonl",
                 "cost_control.jsonl", "health.json", "summary.json", "effective_config.yaml", "config.yaml")
    sources = [{"path": str(train / name), "exists": (train / name).is_file(),
                "sha256": _hash(train / name) if (train / name).is_file() else None} for name in filenames]
    steps, main, prescan, fresh, costs = [_read(train / name, True) for name in filenames[:5]]
    health, summary = [_read(train / name) for name in filenames[5:7]]
    cfg_path = train / ("effective_config.yaml" if (train / "effective_config.yaml").is_file() else "config.yaml")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {} if cfg_path.is_file() else {}
    method = cfg.get("method") or next((r["method"] for r in main if r.get("method")), train.parent.name)
    objective = "group_mean_advantage" if method in ("grpo", "grpo-short", "grpo_short") else "R_minus_b"
    step_ids = [s["step"] for s in steps]
    endpoint = summary.get("step", (summary.get("summary") or {}).get("step"))
    complete_range = bool(steps) and _number(endpoint) and len(set(step_ids)) == len(steps) and set(step_ids) == set(range(1, int(endpoint) + 1))
    main_counts = Counter(r.get("step") for r in main)
    groups = _groups(main)
    integrity = {"complete_step_range": bool(complete_range), "final_step": endpoint,
                 "main_rows_match_recorded_n": main_counts == {s["step"]: s.get("n") for s in steps},
                 "duplicate_steps": [s for s, count in Counter(step_ids).items() if count > 1],
                 "stopped_with_reward": [], "advantage_mismatches": [], "advantage_unverifiable": 0,
                 "health_last_step_mismatches": [],
                 "auxiliary_steps_without_step_record": sorted({r.get("step") for r in prescan + fresh if r.get("step") not in step_ids}, key=str)}
    grouped_rows = defaultdict(list)
    for index, row in enumerate(main, 1):
        grouped_rows[(row.get("step"), row.get("problem_id"))].append((index, row))
    for members in grouped_rows.values():
        complete = [r for _, r in members if r.get("z_continue") == 1]
        rewards = [r.get("reward") for r in complete]
        group_mean = statistics.fmean(rewards) if rewards and all(_number(r) for r in rewards) else None
        for line, row in members:
            ref = {"line": line, "step": row.get("step"), "problem_id": row.get("problem_id")}
            if row.get("z_continue") == 0:
                if row.get("reward") is not None:
                    integrity["stopped_with_reward"].append(ref)
                continue
            baseline = group_mean if objective == "group_mean_advantage" else row.get("baseline_b")
            if row.get("z_continue") != 1 or not all(_number(x) for x in (row.get("reward"), row.get("advantage"), baseline)):
                integrity["advantage_unverifiable"] += 1
            elif abs(row["advantage"] - (row["reward"] - baseline)) > 1e-12:
                integrity["advantage_mismatches"].append({**ref, "recorded": row["advantage"], "expected": row["reward"] - baseline})
    last = next((s for s in steps if s["step"] == health.get("step")), {})
    for key in ("grad_norm_preclip", "grad_norm", "parameter_update_norm", "n_zero_adv", "all_completed_adv_zero", "n_truncated"):
        if key in health and key in last and health[key] != last[key]:
            integrity["health_last_step_mismatches"].append({"field": key, "health": health[key], "step": last[key]})

    def phase(selected):
        ids = {s["step"] for s in selected}
        rows = [r for r in main if r.get("step") in ids]
        completed = [r for r in rows if r.get("z_continue") == 1]
        these_groups = [g for g in groups if g["step"] in ids]
        token_stats = _stats(_tokens(r, True) for r in rows)
        main_stats = {"starts": len(rows), "unique_problems": len({r.get("problem_id") for r in rows}),
                      "completed": len(completed), "stopped": sum(r.get("z_continue") == 0 for r in rows),
                      "unknown_completion": sum(r.get("z_continue") not in (0, 1) for r in rows),
                      "reward": _stats(r.get("reward") for r in completed),
                      "advantage": _stats(r.get("advantage") for r in completed),
                      "baseline": {**_stats(r.get("baseline_b") for r in rows), "used_by_objective": objective != "group_mean_advantage"},
                      "pg_zero_advantage_causes": {"reward0_baseline0": sum(r.get("reward") == 0 and r.get("baseline_b") == 0 for r in completed),
                                                   "reward1_baseline1": sum(r.get("reward") == 1 and r.get("baseline_b") == 1 for r in completed)}
                                                   if objective == "R_minus_b" else None,
                      "recorded_gradient_norm_sq": _stats(r.get("true_grad_norm_sq") for r in completed),
                      "zero_raw_G_implied_by_zero_advantage": sum(r.get("advantage") == 0 for r in completed),
                      "generated_tokens": token_stats,
                      "completed_response_tokens": _stats(_tokens(r, True) for r in completed),
                      "truncated_completed": _booleans(r.get("truncated") for r in completed),
                      "p_continue": _stats(r.get("p_continue") for r in rows),
                      "nonzero_prediction_rows": sum(bool(r.get("prediction_coords")) and any(v != 0 for v in r["prediction_coords"]) for r in rows),
                      "prediction_missing_rows": sum(r.get("prediction_coords") is None for r in rows)}
        auxiliary = {name: _auxiliary(name, records, selected, (train / filename).is_file(), (cfg.get("baseline") or {}).get("prescan"))
                     for name, records, filename in (("prescan", prescan, "prescan.jsonl"), ("fresh", fresh, "fresh_supervision.jsonl"))}
        totals = [token_stats["sum"]] + [a["generated_tokens"] for a in auxiliary.values()]
        complete = bool(complete_range) and integrity["main_rows_match_recorded_n"]
        boundary = [r for r in costs if r.get("event") == "batch_complete" and r.get("step") in ids]
        wall = sum(r["last_batch_wall_seconds"] for r in boundary) if (
            complete and Counter(r["step"] for r in boundary) == Counter(ids) and
            all(_number(r.get("last_batch_wall_seconds")) for r in boundary)) else None
        updates = {key: _stats(s.get(key) for s in selected) for key in
                   ("grad_norm_preclip", "grad_norm", "parameter_update_norm", "update_ascent_cosine")}
        updates.update(clip=_booleans(s.get("clip_triggered") for s in selected),
                       all_completed_adv_zero=_booleans(s.get("all_completed_adv_zero") for s in selected),
                       zero_gradient_nonzero_parameter_update_steps=[s["step"] for s in selected if s.get("grad_norm_preclip") == 0 and
                                                                     _number(s.get("parameter_update_norm")) and s["parameter_update_norm"] > 0])
        return {"recorded_steps": sorted(ids), "n_steps": len(selected), "complete_step_and_main_evidence": complete,
                "starts_per_step": [s.get("n") for s in selected], "main": main_stats, "group_counts": _group_counts(these_groups),
                "auxiliary": auxiliary, "total_generation_tokens": sum(totals) if complete and all(_number(v) for v in totals) else None,
                "updates": updates, "complete_batch_wall_seconds": wall,
                "phase_wall_seconds": {key: _stats((s.get("phases") or {}).get(key) for s in selected)
                                       for key in sorted({k for s in selected for k in s.get("phases", {})})}}

    phases = {"all": phase(steps), "warmup": phase([s for s in steps if s.get("warmup") is True]),
              "post_warmup": phase([s for s in steps if s.get("warmup") is False]),
              "unknown_phase": phase([s for s in steps if not isinstance(s.get("warmup"), bool)])}
    step_detail = []
    for s in steps:
        one = phase([s])
        step_detail.append({"step": s["step"], "warmup": s.get("warmup"), "n": s.get("n"),
                    "main": one["main"], "updates": one["updates"],
                    "group_counts": one["group_counts"], "auxiliary": one["auxiliary"],
                    "predictor_supervision": (s.get("predictor") or {}).get("supervision"),
                    "freshness": (s.get("predictor") or {}).get("freshness"),
                    "frozen_new_problems": (s.get("predictor") or {}).get("frozen_new_problems"),
                    "m_shrink": s.get("m_shrink"), "basis_rank": s.get("basis_rank"),
                    "allocation_ready": s.get("allocation_ready"), "control_variate_used": s.get("control_variate_used")})
    return {"train": str(train), "method": method, "objective": objective, "sources": sources, "integrity": integrity,
            "phases": phases, "groups": groups, "steps": step_detail,
            "scope": "Observed log rows only; sample summaries are not independent estimates across training seeds. Missing fields are counted, never filled with zero. Reward statistics exclude stopped/unknown-completion rows. GRPO uses per-step problem-group mean; PG uses frozen recorded R-b. Singleton reward variance is unknown, not evidence of population homogeneity. Zero raw G from zero advantage does not imply zero HT/CV batch gradient or zero Adam update. Token totals exclude SFT and require complete step/main and auxiliary evidence; observed phase timers are not full run wall cost.",
            "prescan_samples_per_problem_from_config": (cfg.get("baseline") or {}).get("prescan"),
            "health_step": health.get("step"), "training_wall_seconds": summary.get("cumulative_wall_seconds", (summary.get("summary") or {}).get("cost_control", {}).get("global_wall_seconds"))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training_dirs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = [summarize_training(path) for path in args.training_dirs]
    result = {"script_sha256": _hash(Path(__file__)), "runs": reports,
              "note": "Descriptive recorded evidence, not a causal diagnosis or new GPU result. Answers are not re-scored."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "runs": len(reports)}))


if __name__ == "__main__":
    main()
