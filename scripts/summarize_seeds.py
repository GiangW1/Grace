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
from scripts.summarize_minimal import stage_path


_PROTOCOL_FIELDS = ("ordered_records_sha256", "reward_protocol_version", "samples_per_problem",
                    "temperature", "top_p", "max_new_tokens", "sample_batch_size")


def pairing_issues(seed, a, b):
    issues = []
    if any(any(issue.get("reason") == "checkpoint_source_unverified" and issue.get("stage") in {"eval-final", "eval-40"}
               for issue in row.get("source_issues", [])) for row in (a, b)):
        issues.append("final_checkpoint_source_unverified")
    expected_seed = seed.removeprefix("seed-")
    if any(str(row.get("actual_seed")) != expected_seed for row in (a, b)):
        issues.append("actual_training_seed_missing_or_mismatched")
    initial = [row.get("shared_initial_actor_hash") for row in (a, b)]
    if not initial[0] or initial[0] != initial[1]:
        issues.append("shared_initial_actor_missing_or_mismatched")
    ma, mb = a.get("evaluation_manifest") or {}, b.get("evaluation_manifest") or {}
    for field in (*_PROTOCOL_FIELDS, "sample_seed_start"):
        if ma.get(field) is None or ma.get(field) != mb.get(field):
            issues.append(f"evaluation_{field}_missing_or_mismatched")
    return issues


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
        values = [r["final_avg4"] for _, r in rows if r.get("final_avg4") is not None]
        output["methods"].append({
            "method": method, "n_seeds_with_final_eval": len(values),
            "final_avg4_mean": float(np.mean(values)) if values else None,
            "final_avg4_seed_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            "per_seed": [{"seed": seed, **row} for seed, row in rows],
        })
    for baseline in ("full_pg", "uniform_cv", "grpo"):
        pairs, incomparable = [], []
        for seed, rows in runs:
            a, b = rows.get("grace", {}), rows.get(baseline, {})
            if a.get("final_avg4") is not None and b.get("final_avg4") is not None:
                issues = pairing_issues(seed, a, b)
                if issues:
                    incomparable.append({"seed": seed, "issues": issues})
                else:
                    pairs.append({"seed": seed, "delta": a["final_avg4"] - b["final_avg4"],
                                  "grace_step": a.get("final_training_step"), "baseline_step": b.get("final_training_step"),
                                  "grace_cost": a.get("cost_control"), "baseline_cost": b.get("cost_control"),
                                  "grace_wall_hours": a.get("train_wall_hours"), "baseline_wall_hours": b.get("train_wall_hours")})
        values = [p["delta"] for p in pairs]
        output["paired"].append({
            "comparison": f"grace minus {baseline}", "pairs": pairs,
            "incomparable": incomparable,
            "mean_delta": float(np.mean(values)) if values else None,
            "seed_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            "note": "Descriptive seed dispersion; no small-sample significance claim.",
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
