"""Aggregate completed seed-level comparisons without pooling dependent answers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.summarize_minimal import finite, read_rows, stage_path
from grace_gc.logging_util.experiment_evidence import evaluation_issues, training_seed_issues


def pairing_issues(seed, a, b):
    issues = []
    if any(any(issue.get("reason") == "checkpoint_source_unverified" and issue.get("stage") in {"eval-final", "eval-40"}
               for issue in row.get("source_issues", [])) for row in (a, b)):
        issues.append("final_checkpoint_source_unverified")
    issues.extend(training_seed_issues(seed, a.get("actual_seed"), b.get("actual_seed")))
    initial = [row.get("shared_initial_actor_hash") for row in (a, b)]
    if not initial[0] or initial[0] != initial[1]:
        issues.append("shared_initial_actor_missing_or_mismatched")
    ma, mb = a.get("evaluation_manifest") or {}, b.get("evaluation_manifest") or {}
    issues.extend(evaluation_issues(ma, mb))
    return issues


def seed_interval(values):
    """Student-t interval for the mean of paired training-seed differences."""
    n = len(values)
    result = {"n_seeds": n, "seed_ci95": None, "seed_ci95_df": n - 1 if n else None,
              "seed_ci95_method": "Student-t mean interval: mean +/- t(0.975, n-1) * seed_sd / sqrt(n)",
              "seed_ci95_unavailable_reason": None}
    if n < 2:
        result["seed_ci95_unavailable_reason"] = "fewer_than_two_paired_training_seeds"
        return result
    try:
        from scipy.stats import t
    except (ImportError, OSError):
        result["seed_ci95_unavailable_reason"] = "scipy_unavailable_install_analysis_extra"
        return result
    mean = float(np.mean(values))
    half = float(t.ppf(.975, n - 1) * np.std(values, ddof=1) / np.sqrt(n))
    result["seed_ci95"] = [mean - half, mean + half]
    return result


def all_starts(seed_root, row):
    """Keep missing or truncated step evidence unknown, rather than count zero."""
    count = row.get("all_starts")
    if finite(count) and count >= 0 and int(count) == count:
        return int(count), "comparison_row"
    train = (row.get("stage_paths") or {}).get("train")
    endpoint = row.get("final_training_step")
    if train and finite(endpoint) and endpoint >= 0 and int(endpoint) == endpoint:
        path = stage_path(seed_root, train) / "steps.jsonl"
        steps = read_rows(path)
        ids = [step.get("step") for step in steps]
        if (path.is_file() and len(ids) == len(set(ids)) and set(ids) == set(range(1, int(endpoint) + 1))
                and all(finite(step.get("n")) and step["n"] >= 0 and int(step["n"]) == step["n"] for step in steps)):
            return sum(int(step["n"]) for step in steps), "complete_training_steps"
    return None, "missing_or_incomplete_training_steps"


def summarize(root: Path):
    runs = []
    for path in sorted(root.glob("seed-*/comparison.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        runs.append((path.parent.name, {r["method"]: r for r in raw.get("methods", [])}))
    methods = sorted({method for _, rows in runs for method in rows})
    output = {"unit": "training seed",
              "seeds_found": [seed for seed, _ in runs], "methods": [], "paired": []}
    requested = root / "experiment.json"
    if requested.exists():
        output["requested"] = json.loads(requested.read_text(encoding="utf-8"))
        output["missing_seed_reports"] = [seed for seed in output["requested"].get("requested_seeds", [])
                                          if f"seed-{seed}" not in output["seeds_found"]]
    output["comparison_mode"] = output.get("requested", {}).get("comparison_mode", "legacy_snapshot")
    output["cost_note"] = "Inspect actual final steps and costs per seed; wall budgets end at batch boundaries and can overshoot."
    for method in methods:
        rows = [(seed, rows[method]) for seed, rows in runs if method in rows]
        per_seed = []
        for seed, row in rows:
            count, source = all_starts(root / seed, row)
            per_seed.append({**row, "seed": seed, "all_starts": count, "all_starts_source": source,
                             "aggregation_issues": training_seed_issues(seed, row.get("actual_seed"))})
        eligible = [row for row in per_seed if not row["aggregation_issues"]]
        values = [r["final_avg4"] for r in eligible if finite(r.get("final_avg4"))]
        passes = [r["final_pass4"] for r in eligible if finite(r.get("final_pass4"))]
        starts = [row["all_starts"] for row in eligible if row["all_starts"] is not None]
        output["methods"].append({
            "method": method, "n_seeds_with_final_eval": len(values),
            "final_avg4_mean": float(np.mean(values)) if values else None,
            "final_avg4_seed_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            "n_seeds_with_final_pass4": len(passes),
            "final_pass4_mean": float(np.mean(passes)) if passes else None,
            "final_pass4_seed_sd": float(np.std(passes, ddof=1)) if len(passes) > 1 else None,
            "n_seeds_with_all_starts": len(starts),
            "all_starts_mean": float(np.mean(starts)) if starts else None,
            "all_starts_seed_sd": float(np.std(starts, ddof=1)) if len(starts) > 1 else None,
            "per_seed": per_seed,
        })
    for baseline in ("full_pg", "uniform_cv", "grpo"):
        pairs, incomparable = [], []
        for seed, rows in runs:
            a, b = rows.get("grace", {}), rows.get(baseline, {})
            if finite(a.get("final_avg4")) and finite(b.get("final_avg4")):
                issues = pairing_issues(seed, a, b)
                if issues:
                    incomparable.append({"seed": seed, "issues": issues})
                else:
                    pairs.append({"seed": seed, "delta": a["final_avg4"] - b["final_avg4"],
                                  "grace_step": a.get("final_training_step"), "baseline_step": b.get("final_training_step"),
                                  "grace_cost": a.get("cost_control"), "baseline_cost": b.get("cost_control"),
                                  "grace_wall_hours": a.get("train_wall_hours"), "baseline_wall_hours": b.get("train_wall_hours")})
            else:
                incomparable.append({"seed": seed, "issues": [f"{name}_final_evaluation_missing_or_nonfinite"
                    for name, row in (("grace", a), (baseline, b)) if not finite(row.get("final_avg4"))]})
        values = [p["delta"] for p in pairs]
        output["paired"].append({
            "comparison": f"grace minus {baseline}", "pairs": pairs,
            "incomparable": incomparable,
            "mean_delta": float(np.mean(values)) if values else None,
            "seed_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            **seed_interval(values),
            "note": "The interval assumes approximately normal, independent and identically distributed paired training-seed differences; it is unstable with few seeds, not a problem bootstrap and not adjusted for multiple comparisons. Scores and differences use accuracy fractions, not percentage points.",
        })
    # Shared SFT is paid once per seed; keep it separate from each method's RL cost.
    output["shared_initialization"] = []
    for seed, _ in runs:
        stages_path = root / seed / "stages.json"
        if not stages_path.exists():
            continue
        stages = json.loads(stages_path.read_text(encoding="utf-8"))
        init = stages.get("shared", {}).get("init")
        if init:
            init_path = stage_path(root / seed, init)
            summary = init_path / "summary.json"
            output["shared_initialization"].append({"seed": seed, "run_dir": str(init_path),
                "summary": json.loads(summary.read_text(encoding="utf-8")) if summary.exists() else None})
    (root / "seeds_comparison.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    summarize(parser.parse_args().root)
