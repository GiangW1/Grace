"""Summarize real small-GPU runs without treating paper targets as execution gates."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from grace_gc.logging_util.experiment_evidence import evaluation_issues


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.is_file() else []


def stage_path(root, value):
    root = Path(root)
    parts = PurePosixPath(str(value).replace("\\", "/")).parts
    if root.name in parts:
        anchor = len(parts) - 1 - list(reversed(parts)).index(root.name)
        return root.joinpath(*parts[anchor + 1:])
    path = Path(value)
    if path.is_absolute() or PureWindowsPath(str(value)).is_absolute():
        return path
    return root / path


def checkpoint_matches(checkpoint, train, source_train=None):
    if not checkpoint:
        return None
    parts = str(checkpoint).replace("\\", "/").split("/")
    parent = parts[:-2] if len(parts) >= 2 and parts[-2] == "checkpoints" else parts[:-1]
    if len(parent) < 3:
        return None
    # Keep experiment/seed identity, while allowing the package's mount point to
    # change. Original run metadata (or an absolute stage manifest) survives moves.
    origin = read_json(Path(train) / "run_meta.json").get("run_dir")
    if not origin and source_train and (PurePosixPath(str(source_train).replace("\\", "/")).is_absolute()
                                       or PureWindowsPath(str(source_train)).is_absolute()):
        origin = source_train
    expected = str(origin or Path(train).resolve()).replace("\\", "/").rstrip("/").split("/")
    source_seed = next((x for x in reversed(parent) if re.fullmatch(r"seed-\d+", x)), None)
    target_seed = next((x for x in reversed(expected) if re.fullmatch(r"seed-\d+", x)), None)
    if source_seed != target_seed:
        return False
    width = 4 if source_seed is not None else 3
    return parent[-width:] == expected[-width:]


def select_stages(root, method, manifest):
    folder, explicit = root / method, manifest.get(method, {})
    train_candidates = [path for path in folder.glob("train*") if path.is_dir() and
                        ((path / "summary.json").is_file() or (path / "run_meta.json").is_file())]
    def started(path):
        return (read_json(path / "run_meta.json").get("started") or read_json(path / "summary.json").get("started") or "")
    train = stage_path(root, explicit["train"]) if explicit.get("train") else max(train_candidates, key=started, default=folder / "train")
    chosen, issues = {"train": train}, []
    for stage in ("eval-0", "eval-20", "eval-40", "eval-final", "audit", "audit-training", "batch-audit"):
        name = ("batch_audit_summary.json" if stage == "batch-audit" else
                "audit_summary.json" if stage.startswith("audit") else "eval_summary.json")
        candidates = [stage_path(root, explicit[stage])] if explicit.get(stage) else list(folder.glob(stage + "*"))
        if stage == "audit":
            candidates = [path for path in candidates if not path.name.startswith("audit-training")]
        matched = []
        for path in candidates:
            if not path.is_dir():
                continue
            result = read_json(path / name)
            if not result:
                continue
            checkpoint = result.get("checkpoint")
            if not checkpoint:
                import yaml
                config_path = path / "config.yaml"
                cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
                checkpoint = (cfg or {}).get("checkpoint")
            match = checkpoint_matches(checkpoint, train, explicit.get("train"))
            if match is False or (match is None and len(train_candidates) > 1 and not explicit.get(stage)):
                issues.append({"stage": stage, "path": str(path), "reason": "checkpoint_source_mismatch_or_unknown"})
                continue
            if match is None:
                issues.append({"stage": stage, "path": str(path), "reason": "checkpoint_source_unverified"})
            matched.append((result.get("finished") or result.get("started") or "", path))
        chosen[stage] = max(matched, key=lambda item: item[0])[1] if matched else None
    return chosen, issues


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
        "effect": float(delta.mean()),
        "ci95": [float(lo), float(hi)],
        "n_problems": len(delta),
        "resampling_unit": "problem",
        "n_seeds": 1,
        "note": "Exploratory snapshot comparison; inspect actual steps, measured costs and budget overshoot separately.",
    }


def summarize(root):
    root = Path(root)
    manifest = read_json(root / "stages.json")
    experiment = read_json(root.parent / "experiment.json")
    wall_reference = read_json(root / "wall_budget.json")
    rows, evaluations, audits = [], {}, {}
    for method in ("full_pg", "grace", "uniform_cv", "grpo", "uniform_ht", "reward_cv", "prompt_cv", "grpo_short"):
        folder = root / method
        if not folder.is_dir() and method not in manifest:
            continue
        stages, source_issues = select_stages(root, method, manifest)
        train = stages["train"]
        train_ledger = read_json(train / "compute_ledger.json")
        summary = read_json(train / "summary.json")
        health = read_json(train / "health.json")
        steps = read_rows(train / "steps.jsonl")
        initial = read_json(stages["eval-0"] / "eval_summary.json") if stages["eval-0"] else {}
        midpoint = read_json(stages["eval-20"] / "eval_summary.json") if stages["eval-20"] else {}
        actual_step = summary.get("step", (summary.get("summary") or {}).get("step"))
        final_path = stages["eval-final"] or (stages["eval-40"] if actual_step in (None, 40) else None)
        final = read_json(final_path / "eval_summary.json") if final_path else {}
        final_source = read_json(final_path / "actor_source.json") if final_path else {}
        evaluated_step = final.get("checkpoint_step", final_source.get("checkpoint_step"))
        if evaluated_step is None:
            match = re.search(r"step_(\d+)\.npz$", str(final.get("checkpoint", "")))
            if match:
                evaluated_step = int(match.group(1))
        if actual_step is not None and evaluated_step is not None and actual_step != evaluated_step:
            source_issues.append({"stage": "eval-final", "reason": "checkpoint_step_is_not_training_endpoint",
                                  "training_step": actual_step, "evaluation_step": evaluated_step})
            final, final_path = {}, None
        audit_path = stages["audit"] / "audit_summary.json" if stages["audit"] else None
        audit = read_json(audit_path) if audit_path else {}
        training_audit = read_json(stages["audit-training"] / "audit_summary.json") if stages["audit-training"] else {}
        training_ratio = (training_audit.get("variance_cost") or {}).get("ratio")
        batch_audit = read_json(stages["batch-audit"] / "batch_audit_summary.json") if stages["batch-audit"] else {}
        initial_actor = read_json(train / "initial_actor.json")
        import yaml
        config_path = train / "effective_config.yaml"
        if not config_path.is_file():
            config_path = train / "config.yaml"
        actual_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
        evaluation_manifest = final.get("evaluation_manifest") or (
            read_json(final_path / "evaluation_manifest.json") if final_path else {}
        )
        evaluations[method], audits[method] = final, audit
        hours = None
        attempts = [read_json(path) for path in sorted((train / "attempts").glob("*/summary.json"))]
        attempts.append(summary)
        completed_intervals = [item for item in attempts if item.get("started") and item.get("finished")]
        if finite(summary.get("cumulative_wall_seconds")):
            hours = summary["cumulative_wall_seconds"] / 3600
        elif completed_intervals:
            hours = sum(
                (datetime.fromisoformat(item["finished"]) - datetime.fromisoformat(item["started"])).total_seconds()
                for item in completed_intervals
            ) / 3600
        active = [step for step in steps if not step.get("warmup")]
        ratio = (audit.get("variance_cost") or {}).get("ratio")
        gain = final["avg"] - initial["avg"] if finite(final.get("avg")) and finite(initial.get("avg")) else None
        rows.append({
            "method": method,
            "experiment_variant": (actual_config or {}).get("experiment_variant"),
            "estimator_configuration": {
                "baseline": (actual_config or {}).get("baseline"),
                "allocation": (actual_config or {}).get("allocation"),
                "control_variate": ((actual_config or {}).get("predictor") or {}).get("control_variate", True),
            },
            "actual_seed": (actual_config or {}).get("seed"),
            "evaluation_manifest": evaluation_manifest,
            "stage_paths": {key: None if value is None else str(value) for key, value in stages.items()},
            "source_issues": source_issues,
            "shared_initial_actor_hash": initial_actor.get("actor_sha256"),
            "train_status": summary.get("run_status", "running_or_not_started"),
            "steps": len(steps),
            "final_training_step": actual_step,
            "final_evaluated_step": evaluated_step if final else None,
            "final_evaluation_path": str(final_path) if final_path else None,
            "cost_control": summary.get("cost_control"),
            "comparison_mode": experiment.get("comparison_mode", "legacy_snapshot"),
            "wall_reference": wall_reference or None,
            "allocation_ready_steps_this_session": (summary.get("summary") or {}).get("allocation_ready_steps_this_session"),
            "initial_avg4": initial.get("avg"),
            "final_avg4": final.get("avg"),
            "midpoint_avg4": midpoint.get("avg"),
            "final_minus_initial_avg4": gain,
            "final_pass4": final.get("pass_at_k"),
            "eval_n_problems": final.get("n_problems"),
            "final_parse_rate": final.get("parse_rate"),
            "final_truncate_rate": final.get("truncate_rate"),
            "train_a100_hours": hours if not train_ledger.get("hardware") or "a100" in train_ledger["hardware"].lower() else None,
            "train_wall_hours": hours,
            "train_hardware": train_ledger.get("hardware"),
            "train_gpu_reserved_hours": train_ledger.get("gpu_reserved_seconds", 0) / 3600 if "gpu_reserved_seconds" in train_ledger else None,
            "train_attempts": len(attempts),
            "basis_id": health.get("basis_id"),
            "post_warmup_starts": sum(s["n"] for s in active),
            "post_warmup_stopped": sum(s.get("n_stopped", 0) for s in active),
            "reservoir_n": health.get("reservoir_n"),
            "audit_bundles": audit.get("n_bundles"),
            "audit_run_dir": str(audit_path.parent) if audit else None,
            "audit_selected": audit.get("n_gated"),
            "variance_cost_ratio": ratio if finite(ratio) else None,
            "variance_cost_uncertainty": audit.get("variance_cost_uncertainty"),
            "audit_independent_report": audit.get("independent_report"),
            "training_matched_audit_variance_cost_ratio": training_ratio if finite(training_ratio) else None,
            "training_matched_audit_variance_cost_uncertainty": training_audit.get("variance_cost_uncertainty"),
            "training_matched_audit_independent_report": training_audit.get("independent_report"),
            "training_matched_audit_bundles": training_audit.get("n_bundles"),
            "fixed_batch_audit": batch_audit or None,
        })
    by_method = {row["method"]: row for row in rows}

    def comparison_with_evidence(baseline):
        left, right = by_method.get("grace", {}), by_method.get(baseline, {})
        issues = []
        for row in (left, right):
            if any(issue.get("reason") == "checkpoint_source_unverified" and issue.get("stage") in {"eval-final", "eval-40"}
                   for issue in row.get("source_issues", [])):
                issues.append("final_checkpoint_source_unverified")
        for field in ("shared_initial_actor_hash", "actual_seed"):
            if left.get(field) is None or left.get(field) == "" or left.get(field) != right.get(field):
                issues.append(field + "_missing_or_mismatched")
        lm, rm = left.get("evaluation_manifest") or {}, right.get("evaluation_manifest") or {}
        issues.extend(evaluation_issues(lm, rm))
        if issues:
            return {"available": False, "issues": issues,
                    "note": "Paired comparison unavailable: common initialization, training seed or evaluation protocol is unverified. Per-method observations are preserved."}
        result = paired_delta(evaluations.get("grace", {}), evaluations.get(baseline, {}))
        available = "effect" in result
        return {**result, "available": available,
                "issues": [] if available else ["evaluated_problem_ids_missing_or_mismatched"]}

    comparison = comparison_with_evidence("grpo")
    mechanism = comparison_with_evidence("uniform_cv")
    targets = []
    for method, audit in audits.items():
        times = audit.get("times", [])
        for key, target, direction in (("rho_l_curve", 0.5, "<="), ("rho_a_curve", 0.8, ">=")):
            values = audit.get(key, [])
            value = values[times.index(512)] if 512 in times and len(values) == len(times) else None
            met = None if not finite(value) else (value <= target if direction == "<=" else value >= target)
            targets.append({
                "method": method,
                "metric": key + "@512",
                "observed": value if finite(value) else None,
                "target": target,
                "direction": direction,
                "target_met_observed": met,
                "statistically_supported": None,
            })
    payload = {
        "methods": rows,
        "shared_initial_actor_consistency": {
            "by_method": {row["method"]: row["shared_initial_actor_hash"] for row in rows},
            "consistent": (len({row["shared_initial_actor_hash"] for row in rows}) == 1)
            if rows and all(row["shared_initial_actor_hash"] for row in rows) else None,
            "note": "Unavailable hashes remain unknown; this is evidence, not an execution gate.",
        },
        "grace_minus_grpo": comparison,
        "grace_minus_uniform_cv": mechanism,
        "grace_minus_full_pg": comparison_with_evidence("full_pg"),
        "comparison_mode": experiment.get("comparison_mode", "legacy_snapshot"),
        "wall_reference": wall_reference or None,
        "paper_targets": targets,
        "scope": "Single seed, small batches, MATH-500 subset, small audit.",
        "unmeasured": [
            "equal-compute multi-benchmark +2 percentage points",
            "0.65x time-to-target",
            "hard-problem discoveries +30%",
            "medium-stratum ELF >=55%",
            "method overhead <=8%",
            "multi-seed statistical support",
            "all attribution controls",
        ],
        "cost_note": (
            "Training hours use cumulative session envelopes, or legacy start/finish intervals including archived failed attempts. "
            "Recovery idle time is excluded. Do not sum overlapping ledger rows. "
            "See RECOVERY.md for runtime memory changes."
        ),
    }
    (root / "comparison.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    if rows:
        with (root / "comparison.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = [
        "# Minimal GPU experiment",
        "",
        payload["scope"],
        "",
        "| Method | Steps | Initial avg@4 | Step 20 avg@4 | Final avg@4 | Final - initial | pass@4 | Truncate rate | Train wall-hours | Audit bundles |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["method"],
                    str(row["steps"]),
                    number(row["initial_avg4"]),
                    number(row["midpoint_avg4"]),
                    number(row["final_avg4"]),
                    number(row["final_minus_initial_avg4"]),
                    number(row["final_pass4"]),
                    number(row["final_truncate_rate"]),
                    number(row["train_wall_hours"]),
                    str(row["audit_bundles"]),
                ]
            )
            + " |"
        )
    lines += [
        "",
        "GRACE minus GRPO (reported final snapshots): " + json.dumps(comparison),
        "",
        "GRACE minus Uniform-CV: " + json.dumps(mechanism),
        "",
        "Budget mode: " + str(payload["comparison_mode"]) + ". Actual costs and boundary overshoot are retained per method; equal requested budgets do not mean identical elapsed time.",
        "",
        "## Paper targets",
        "",
        "| Method | Metric | Observed | Target | Observed target met |",
        "|---|---|---:|---|---|",
    ]
    for target in targets:
        lines.append(
            f"| {target['method']} | {target['metric']} | {number(target['observed'])} | "
            f"{target['direction']} {target['target']} | {target['target_met_observed']} |"
        )
    lines += [
        "",
        "Undefined selected-subset metrics stay unavailable; all observations remain in each audit summary.",
        "",
        payload["cost_note"],
        "",
        "Not established by this experiment: " + "; ".join(payload["unmeasured"]) + ".",
    ]
    (root / "report.md").write_text("\n".join(lines) + "\n")
    print(root / "report.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    summarize(parser.parse_args().root)
