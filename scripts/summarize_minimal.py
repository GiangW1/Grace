"""Summarize real small-GPU runs without treating paper targets as execution gates."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np


def read_json(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []


def finite(value):
    return value is not None and math.isfinite(float(value))


def number(value):
    return f"{value:.5g}" if finite(value) else "not available"


def paired_delta(left, right):
    l = {r["problem_id"]: r["avg"] for r in left.get("per_problem", [])}
    r = {r["problem_id"]: r["avg"] for r in right.get("per_problem", [])}
    if not l or set(l) != set(r):
        return {"note": "matching evaluated problems unavailable"}
    delta = np.asarray([l[key] - r[key] for key in sorted(l)])
    rng = np.random.default_rng(17)
    means = delta[rng.integers(0, len(delta), size=(5000, len(delta)))].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {
        "effect": float(delta.mean()), "ci95": [float(lo), float(hi)],
        "n_problems": len(delta), "resampling_unit": "problem", "n_seeds": 1,
        "note": "Exploratory matched-step comparison; not equal-compute or multi-seed evidence.",
    }


def summarize(root):
    rows, evaluations, audits = [], {}, {}
    for method in ("full_pg", "grace", "uniform_cv", "grpo", "uniform_ht", "reward_cv", "prompt_cv", "grpo_short"):
        folder = root / method
        if not folder.is_dir():
            continue
        summary = read_json(folder / "train/summary.json")
        health = read_json(folder / "train/health.json")
        steps = read_rows(folder / "train/steps.jsonl")
        initial = read_json(folder / "eval-0/eval_summary.json")
        midpoint = read_json(folder / "eval-20/eval_summary.json")
        final = read_json(folder / "eval-40/eval_summary.json")
        audit_path = folder / "audit/audit_summary.json"
        audit = read_json(audit_path)
        completed_audits = []
        for candidate in folder.glob("audit*/audit_summary.json"):
            result = read_json(candidate)
            if result.get("finished") and "n_bundles" in result:
                completed_audits.append((result, candidate))
        if completed_audits:
            audit, audit_path = max(completed_audits, key=lambda item: item[0]["finished"])
        evaluations[method], audits[method] = final, audit
        hours = None
        attempts = [read_json(path) for path in sorted((folder / "train/attempts").glob("*/summary.json"))]
        attempts.append(summary)
        completed_intervals = [item for item in attempts if item.get("started") and item.get("finished")]
        if completed_intervals:
            hours = sum((datetime.fromisoformat(item["finished"]) - datetime.fromisoformat(item["started"])).total_seconds()
                        for item in completed_intervals) / 3600
        active = [step for step in steps if not step.get("warmup")]
        ratio = (audit.get("variance_cost") or {}).get("ratio")
        gain = final["avg"] - initial["avg"] if finite(final.get("avg")) and finite(initial.get("avg")) else None
        rows.append({
            "method": method, "train_status": summary.get("run_status", "running_or_not_started"),
            "steps": len(steps), "initial_avg4": initial.get("avg"), "final_avg4": final.get("avg"),
            "midpoint_avg4": midpoint.get("avg"), "final_minus_initial_avg4": gain,
            "final_pass4": final.get("pass_at_k"), "eval_n_problems": final.get("n_problems"),
            "final_parse_rate": final.get("parse_rate"), "final_truncate_rate": final.get("truncate_rate"),
            "train_a100_hours": hours, "train_attempts": len(attempts), "basis_id": health.get("basis_id"),
            "post_warmup_starts": sum(s["n"] for s in active),
            "post_warmup_stopped": sum(s.get("n_stopped", 0) for s in active),
            "reservoir_n": health.get("reservoir_n"), "audit_bundles": audit.get("n_bundles"),
            "audit_run_dir": str(audit_path.parent) if audit else None,
            "audit_selected": audit.get("n_gated"), "variance_cost_ratio": ratio if finite(ratio) else None,
        })
    comparison = paired_delta(evaluations.get("grace", {}), evaluations.get("grpo", {}))
    mechanism = paired_delta(evaluations.get("grace", {}), evaluations.get("uniform_cv", {}))
    targets = []
    for method, audit in audits.items():
        times = audit.get("times", [])
        for key, target, direction in (("rho_l_curve", 0.5, "<="), ("rho_a_curve", 0.8, ">=")):
            values = audit.get(key, [])
            value = values[times.index(512)] if 512 in times and len(values) == len(times) else None
            met = None if not finite(value) else (value <= target if direction == "<=" else value >= target)
            targets.append({"method": method, "metric": key + "@512", "observed": value if finite(value) else None,
                            "target": target, "direction": direction, "target_met_observed": met,
                            "statistically_supported": None})
    payload = {"methods": rows, "grace_minus_grpo": comparison, "grace_minus_uniform_cv": mechanism,
               "paper_targets": targets, "scope": "Single seed, small batches, MATH-500 subset, small audit.",
               "unmeasured": ["equal-compute multi-benchmark +2 percentage points", "0.65x time-to-target",
                              "hard-problem discoveries +30%", "medium-stratum ELF >=55%", "method overhead <=8%",
                              "multi-seed statistical support", "all attribution controls"],
               "cost_note": "Training hours sum start/finish intervals on one A100, including archived failed attempts. Recovery idle time is excluded. Do not sum overlapping ledger rows. See RECOVERY.md for runtime memory changes."}
    (root / "comparison.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    if rows:
        with (root / "comparison.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = ["# Minimal GPU experiment", "", payload["scope"], "",
             "| Method | Steps | Initial avg@4 | Step 20 avg@4 | Final avg@4 | Final - initial | pass@4 | Truncate rate | Train A100-hours | Audit bundles |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append("| " + " | ".join([row["method"], str(row["steps"]), number(row["initial_avg4"]),
                     number(row["midpoint_avg4"]), number(row["final_avg4"]), number(row["final_minus_initial_avg4"]),
                     number(row["final_pass4"]), number(row["final_truncate_rate"]), number(row["train_a100_hours"]),
                     str(row["audit_bundles"])]) + " |")
    lines += ["", "GRACE minus GRPO (matched steps): " + json.dumps(comparison),
              "", "GRACE minus Uniform-CV: " + json.dumps(mechanism),
              "", "## Paper targets", "", "| Method | Metric | Observed | Target | Observed target met |",
              "|---|---|---:|---|---|"]
    for target in targets:
        lines.append(f"| {target['method']} | {target['metric']} | {number(target['observed'])} | {target['direction']} {target['target']} | {target['target_met_observed']} |")
    lines += ["", "Undefined selected-subset metrics stay unavailable; all observations remain in each audit summary.",
              "", payload["cost_note"], "", "Not established by this experiment: " + "; ".join(payload["unmeasured"]) + "."]
    (root / "report.md").write_text("\n".join(lines) + "\n")
    print(root / "report.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    summarize(parser.parse_args().root)
