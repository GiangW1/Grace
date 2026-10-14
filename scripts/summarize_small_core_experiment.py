#!/usr/bin/env python3
"""Collect measured learning curves, paired evaluation deltas and online predictor health."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from grace_gc.logging_util.run_dir import atomic_write_text


def read_json(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def paired_delta(actual, reference, *, bootstrap=10000):
    if not actual or not reference:
        return {"status": "unavailable", "reason": "missing evaluation"}
    manifest = actual.get("evaluation_manifest")
    if not manifest or manifest != reference.get("evaluation_manifest"):
        return {"status": "unavailable", "reason": "evaluation manifests differ or are missing"}
    a = {str(r["problem_id"]): r["avg"] for r in actual.get("per_problem", [])}
    b = {str(r["problem_id"]): r["avg"] for r in reference.get("per_problem", [])}
    if not a or a.keys() != b.keys():
        return {"status": "unavailable", "reason": "per-problem evaluations differ or are missing"}
    differences = np.asarray([a[k] - b[k] for k in sorted(a)], float)
    interval = None
    if len(differences) > 1:
        if np.all(differences == differences[0]):
            interval = [float(differences[0])] * 2
        else:
            rng = np.random.default_rng(17)
            means = np.asarray([rng.choice(differences, len(differences), replace=True).mean()
                                for _ in range(bootstrap)])
            interval = np.quantile(means, [0.025, 0.975]).tolist()
    return {"status": "available", "delta_avg": float(differences.mean()),
            "paired_problem_bootstrap_95": interval, "n_problems": len(a),
            "note": "Conditional on these checkpoints and answers; not across-training-seed uncertainty or an equivalence test."}


def step_health(train_dir, through_step=None):
    path = Path(train_dir) / "steps.jsonl"
    if not path.is_file():
        return {"status": "unavailable"}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = [r for r in rows if through_step is None or int(r["step"]) <= through_step]
    post = [r for r in rows if r.get("warmup") is False]
    total_n = sum(r.get("n") or 0 for r in post)
    shrink = [r["m_shrink"] for r in post if r.get("m_shrink") is not None]
    unseen = [{"step": r["step"], **r["predictor"]["frozen_new_problems"]} for r in post
              if (r.get("predictor") or {}).get("frozen_new_problems") is not None]
    phases = sorted({k for r in post for k in (r.get("phases") or {})})
    return {"status": "available", "steps": len(rows), "post_warmup_steps": len(post),
            "stopped_fraction_post_warmup": sum(r.get("n_stopped") or 0 for r in post) / total_n if total_n else None,
            "control_variate_steps": sum(r.get("control_variate_used") is True for r in post),
            "m_shrink_mean": float(np.mean(shrink)) if shrink else None,
            "m_shrink_last": shrink[-1] if shrink else None,
            "frozen_unseen_problem_diagnostics": unseen,
            "phase_seconds_post_warmup": {k: sum((r.get("phases") or {}).get(k, 0) for r in post) for k in phases},
            "note": "m_shrink is logged after predictor updates; not a held-out gradient accuracy metric. Token-cost proxies are not wall-time savings."}


def job_dir(job):
    if job.get("run_dir"):
        return Path(job["run_dir"])
    command = job["command"]
    return Path(command[command.index("--run-dir") + 1])


def summarize_suite(path):
    manifest = read_json(path)
    if manifest is None:
        raise FileNotFoundError(path)
    jobs = manifest["jobs"]
    completed = [j for j in jobs if j.get("status") == "complete"]
    eval_jobs = [j for j in completed if j["phase"] in {"eval", "eval_snapshot", "initial_eval"}]
    summaries = [(j, read_json(job_dir(j) / "eval_summary.json")) for j in eval_jobs]
    initial = next((s for j, s in summaries if j["phase"] == "initial_eval"), None)
    train = {(j["arm"], j["seed"]): j for j in completed if j["phase"] == "train"}
    curves = []
    for job, result in summaries:
        if job["phase"] == "initial_eval":
            continue
        if result is None:
            curves.append({"arm": job["arm"], "seed": job["seed"], "status": "unavailable", "reason": "missing eval_summary.json"})
            continue
        training = train.get((job["arm"], job["seed"]))
        directory = job_dir(training) if training else Path(job.get("train_dir", "."))
        step = result.get("checkpoint_step")
        checkpoints = read_json(directory / "checkpoints.json") or {}
        checkpoint = next((r for r in checkpoints.get("steps", []) if r["step"] == step), {})
        train_summary = read_json(directory / "summary.json") or {}
        hash_matches = bool(checkpoint.get("sha256") and checkpoint["sha256"] == result.get("checkpoint_sha256"))
        baseline_arm = job["arm"].split("_", 1)[0] + "_full_pg"
        reference = next((s for j, s in summaries if s and j["arm"] == baseline_arm
                          and j["seed"] == job["seed"] and j["phase"] == job["phase"]
                          and s.get("checkpoint_step") == step), None)
        curves.append({"arm": job["arm"], "seed": job["seed"], "phase": job["phase"], "status": "available",
                       "step": step, "avg": result.get("avg"), "pass_at_k": result.get("pass_at_k"),
                       "truncate_rate": result.get("truncate_rate"), "mean_response_tokens": result.get("mean_response_tokens"),
                       "checkpoint_sha256": result.get("checkpoint_sha256"),
                       "checkpoint_command_seconds": checkpoint.get("available_command_wall_seconds") if hash_matches else None,
                       "checkpoint_timing_status": "verified" if hash_matches else "unavailable_or_hash_mismatch",
                       "training_command_seconds": training.get("command_wall_seconds") if training and job["phase"] == "eval" else None,
                       "training_stop_reason": train_summary.get("stop_reason"),
                       "cost_control": train_summary.get("cost_control"),
                       "vs_initial": paired_delta(result, initial), "vs_same_step_full_pg": paired_delta(result, reference),
                       "training_health": step_health(directory, step)})
    by_arm = []
    for arm in sorted({r["arm"] for r in curves}):
        final = [r for r in curves if r["arm"] == arm and r.get("phase") == "eval" and r.get("avg") is not None]
        values = [r["avg"] for r in final]
        by_arm.append({"arm": arm, "n_training_seeds": len(values), "seeds": [r["seed"] for r in final],
                       "avg_mean": float(np.mean(values)) if values else None,
                       "avg_seed_std": float(np.std(values, ddof=1)) if len(values) > 1 else None})
    return {"suite_status": manifest["status"], "initial_evaluation": initial, "curves": curves, "final_by_arm": by_arm,
            "incomplete_jobs": [{k: j.get(k) for k in ("arm", "seed", "phase", "status", "reason")}
                                for j in jobs if j.get("status") != "complete"],
            "prediction_reports": [{"arm": j["arm"], "seed": j["seed"],
                                    "report": read_json(job_dir(j) / "direct_prediction_summary.json")}
                                   for j in completed if j["phase"] == "prediction_report"],
            "note": "Time curves use checkpoint publication clocks only when checkpoint hashes match. Eval/audit time is excluded from training but remains in suite.json. No interpolation, time-to-target claim, or automatic go/no-go decision. Different terminal steps under wall budgets are not same-step comparisons."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, help="path to suite.json")
    args = parser.parse_args(argv)
    result = summarize_suite(args.suite)
    dest = Path(args.suite).parent / "comparison_summary.json"
    atomic_write_text(dest, json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
