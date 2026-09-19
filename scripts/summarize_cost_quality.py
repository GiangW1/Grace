"""Checkpoint quality against recorded availability cost; never interpolate scores."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import statistics
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.summarize_minimal import checkpoint_matches, finite, read_json, read_rows, stage_path
from scripts.summarize_seeds import pairing_issues, seed_interval


PROTOCOL_FIELDS = ("ordered_records_sha256", "reward_protocol_version", "samples_per_problem",
                   "temperature", "top_p", "max_new_tokens", "sample_batch_size", "sample_seed_start")
CLOCK_FIELDS = {"command": "available_command_wall_seconds", "training_entry": "available_global_wall_seconds"}


def training_recipe_hash(config):
    recipe = {k: copy.deepcopy(v) for k, v in config.items()
              if not k.startswith("_") and k not in {"seed", "init_checkpoint", "resume", "run_dir"}}
    if isinstance(recipe.get("vllm"), dict):
        recipe["vllm"].pop("seed", None)  # Engine seed follows the training replicate.
    return hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()


def aggregation_protocol(method, point):
    return {"hardware": method["hardware"], "training_recipe_sha256": method["training_recipe_sha256"],
            "requested_k": point["requested_k"],
            "evaluation": {k: point["evaluation_manifest"].get(k) for k in PROTOCOL_FIELDS if k != "sample_seed_start"}}


def protocol_conflicts(protocols):
    return ["cross_seed_protocol_mismatch"] if len({json.dumps(p, sort_keys=True) for p in protocols}) > 1 else []


def evaluation_steps(train, selection="0 20 40", budget=None):
    published = read_json(Path(train)/"checkpoints.json").get("steps", [])
    if selection == "all":
        steps = {int(r["step"]) for r in published}
    else:
        steps = {int(x) for x in selection.split()}
    if any(step < 0 for step in steps):
        raise ValueError("evaluation steps must be nonnegative")
    if budget is not None:
        available = [r for r in published if finite(r.get("available_command_wall_seconds"))
                     and 0 <= r["available_command_wall_seconds"] <= budget]
        if available:
            steps.add(int(max(available, key=lambda r: (r["available_command_wall_seconds"], r["step"]))["step"]))
    return sorted(steps)


def collect_curve(train, evaluations, clock="command"):
    train = Path(train)
    field = CLOCK_FIELDS[clock]
    index = {row["step"]: row for row in read_json(train/"checkpoints.json").get("steps", [])}
    points = []
    training_steps = read_rows(train/"steps.jsonl")
    seen = set()
    for folder in evaluations:
        folder = Path(folder)
        if folder.resolve() in seen:
            continue  # eval-final often aliases eval-40
        seen.add(folder.resolve())
        result = read_json(folder/"eval_summary.json")
        if not result:
            continue
        step = result.get("checkpoint_step")
        saved = index.get(step, {})
        protocol = result.get("evaluation_manifest") or read_json(folder/"evaluation_manifest.json")
        issues = []
        if checkpoint_matches(result.get("checkpoint"), train) is not True:
            issues.append("checkpoint_source_unverified")
        if not saved.get("sha256") or saved["sha256"] != result.get("checkpoint_sha256"):
            issues.append("checkpoint_hash_missing_or_mismatched")
        seconds = saved.get(field)
        if not finite(seconds) or seconds < 0:
            seconds = None
            issues.append("checkpoint_availability_clock_missing")
        if any(protocol.get(key) is None for key in PROTOCOL_FIELDS) or result.get("requested_k") is None:
            issues.append("evaluation_protocol_incomplete")
        prefix = [r for r in training_steps if isinstance(r.get("step"), int) and step is not None and r["step"] <= step]
        counts_complete = (isinstance(step, int) and step >= 0 and len(prefix) == step
                           and {r["step"] for r in prefix} == set(range(1, step + 1))
                           and all(finite(r.get("n")) and r["n"] >= 0 and int(r["n"]) == r["n"] for r in prefix))
        points.append({"step": step, "wall_seconds": seconds, "clock": clock,
                       "main_starts": sum(int(r["n"]) for r in prefix) if counts_complete else None,
                       "avg": result.get("avg"), "pass_at_k": result.get("pass_at_k"),
                       "requested_k": result.get("requested_k"), "evaluation_manifest": protocol,
                       "evaluation_path": str(folder), "checkpoint_sha256": result.get("checkpoint_sha256"),
                       "available_at": saved.get("available_at"), "issues": issues})
    points.sort(key=lambda r: (r["step"] is None, r["step"] if r["step"] is not None else 0, r["evaluation_path"]))
    for row in points:
        if sum(p["step"] == row["step"] for p in points) > 1:
            row["issues"].append("repeated_checkpoint_evaluation")
    reference = next((r for r in points if "evaluation_protocol_incomplete" not in r["issues"]), None)
    if reference:
        for row in points:
            if (any(row["evaluation_manifest"].get(k) != reference["evaluation_manifest"].get(k) for k in PROTOCOL_FIELDS)
                    or row["requested_k"] != reference["requested_k"]):
                row["issues"].append("evaluation_protocol_changed_within_curve")
    return points


def select_at_budget(points, seconds):
    available = [p for p in points if not p["issues"] and finite(p["wall_seconds"]) and p["wall_seconds"] <= seconds]
    return max(available, key=lambda p: (p["wall_seconds"], p["step"]), default=None)


def first_observed_target(points, target):
    available = [p for p in points if not p["issues"] and finite(p["wall_seconds"]) and finite(p["avg"]) and p["avg"] >= target]
    return min(available, key=lambda p: p["wall_seconds"], default=None)


def summarize(root, budgets=(), target=None, clock="command"):
    import yaml

    root = Path(root)
    is_seed = (root/"stages.json").is_file()
    requested = read_json((root.parent if is_seed else root)/"experiment.json")
    seed_roots = [root] if is_seed else sorted(set(root.glob("seed-*")) |
        {root/f"seed-{s}" for s in requested.get("requested_seeds", [])})
    output = {"clock": clock, "seeds": {}, "paired_at_budget": [],
              "requested": requested,
              "missing_seed_reports": [p.name for p in seed_roots if not (p/"stages.json").is_file()],
              "missing_methods": {}, "method_at_budget": [],
              "scope": "Latest evaluated checkpoint published within each cutoff, never the best score or an interpolated model. This is retrospective availability; actual full job spend/overshoot is retained, not a claim that the process stopped exactly at the cutoff. Shared SFT, evaluation and audits are separately charged.",
              "target_note": "First observed target crossing on saved/evaluated checkpoints only; not an exact crossing time or persistent attainment."}
    pairs_by_budget = {}
    for seed_root in seed_roots:
        manifest = read_json(seed_root/"stages.json")
        commands = read_rows(seed_root.parent/"command_timing.jsonl")
        def command_seconds(stage):
            matches = [r for r in commands if r.get("stage") == f"{seed_root.name}/{stage}"]
            return sum(r["wall_seconds"] for r in matches) if matches else None
        shared = command_seconds("shared-init")
        query = list(budgets)
        if not query:
            value = read_json(seed_root/"wall_budget.json").get("run_wall_seconds")
            query = [value] if finite(value) else []
        methods = {}
        for method, stages in manifest.items():
            if not stages.get("train"):
                continue
            train = stage_path(seed_root, stages["train"])
            evals = [stage_path(seed_root, path) for name, path in stages.items() if name.startswith("eval-")]
            points = collect_curve(train, evals, clock)
            config_path = train/"config.yaml"
            config = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
            config = config or {}
            actual = command_seconds(f"{method}/train")
            methods[method] = {"points": points, "actual_command_wall_seconds": actual,
                "shared_initialization_command_seconds": shared,
                "current_chain_initialization_plus_training_seconds": shared + actual if shared is not None and actual is not None else None,
                "initialization_cost_scope": "Current chain initialization command only. If SHARED_INIT_CHAIN imported an existing SFT actor, its original training cost is not recovered here and must be reported separately; this sum is not the full deployment cost in that case.",
                "training_entry_wall_seconds": read_json(train/"summary.json").get("cumulative_wall_seconds"),
                "hardware": read_json(train/"compute_ledger.json").get("hardware"),
                "training_recipe_sha256": training_recipe_hash(config),
                "actual_seed": config.get("seed"),
                "shared_initial_actor_hash": read_json(train/"initial_actor.json").get("actor_sha256"),
                "at_budget": {str(b): select_at_budget(points, b) for b in query},
                "first_observed_target": first_observed_target(points, target) if target is not None else None,
                "target_avg": target}
        output["seeds"][seed_root.name] = methods
        output["missing_methods"][seed_root.name] = sorted(set(requested.get("requested_methods", [])) - set(methods))
        for budget in query:
            for baseline in ("full_pg", "uniform_cv", "grpo", "uniform_ht"):
                expected = set(methods) | set(requested.get("requested_methods", []))
                if "grace" not in expected or baseline not in expected:
                    continue
                entries = [methods.get(name, {}) for name in ("grace", baseline)]
                selected = [m.get("at_budget", {}).get(str(budget)) for m in entries]
                issues = []
                if any(p is None or not finite(p["avg"]) for p in selected):
                    issues.append("no_verified_evaluation_at_or_before_cutoff")
                else:
                    rows = [{**m, "evaluation_manifest": p["evaluation_manifest"]} for m, p in zip(entries, selected)]
                    issues.extend(pairing_issues(seed_root.name, *rows))
                    if selected[0]["requested_k"] != selected[1]["requested_k"]:
                        issues.append("evaluation_k_mismatch")
                    if not entries[0]["hardware"] or entries[0]["hardware"] != entries[1]["hardware"]:
                        issues.append("hardware_missing_or_mismatched")
                pair = {"seed": seed_root.name, "issues": issues, "selected": selected,
                        "aggregation_protocol": [aggregation_protocol(m, p) for m, p in zip(entries, selected)] if not issues else None,
                        "delta": selected[0]["avg"] - selected[1]["avg"] if not issues else None}
                pairs_by_budget.setdefault((float(budget), baseline), []).append(pair)
    for (budget, baseline), pairs in pairs_by_budget.items():
        eligible = [p for p in pairs if p["delta"] is not None]
        conflicts = protocol_conflicts([p["aggregation_protocol"] for p in eligible])
        values = [p["delta"] for p in eligible] if not conflicts else []
        output["paired_at_budget"].append({"budget_seconds": budget, "comparison": f"grace minus {baseline}",
            "aggregation_issues": conflicts, "n_eligible_pairs": len(eligible),
            "pairs": pairs, "mean_delta": sum(values)/len(values) if values else None, **seed_interval(values),
            "note": "Pairing checks recorded seed/actor/protocol/hardware, not physical device exclusivity. Seed-t assumes independent, approximately normal paired training-seed differences; a few seeds cannot verify that assumption. Sparse checkpoints underspend differently. Include actual full job spend, shared SFT and diagnostics separately."})
    groups = {}
    for seed, methods in output["seeds"].items():
        for method, record in methods.items():
            for budget, point in record["at_budget"].items():
                issues = [] if str(record["actual_seed"]) == seed.removeprefix("seed-") else ["actual_training_seed_missing_or_mismatched"]
                if point is None:
                    issues.append("no_verified_evaluation_at_or_before_cutoff")
                groups.setdefault((budget, method), []).append({"seed": seed, "point": point, "issues": issues,
                    "aggregation_protocol": aggregation_protocol(record, point) if point else None})
    for (budget, method), rows in groups.items():
        eligible = [r for r in rows if not r["issues"]]
        conflicts = protocol_conflicts([r["aggregation_protocol"] for r in eligible])
        entry = {"budget_seconds": float(budget), "method": method, "per_seed": rows,
                 "aggregation_issues": conflicts, "n_eligible_seeds": len(eligible)}
        for field in ("avg", "pass_at_k", "step", "main_starts", "wall_seconds"):
            values = [r["point"][field] for r in eligible if finite(r["point"].get(field))] if not conflicts else []
            entry[field] = {"n_seeds": len(values), "mean": statistics.mean(values) if values else None,
                            "seed_sd": statistics.stdev(values) if len(values) > 1 else None}
        output["method_at_budget"].append(entry)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--budgets", nargs="*", type=float, default=[])
    parser.add_argument("--target-avg", type=float)
    parser.add_argument("--clock", choices=CLOCK_FIELDS, default="command")
    args = parser.parse_args()
    if any(not finite(b) or b < 0 for b in args.budgets):
        parser.error("budgets must be finite and nonnegative")
    if args.target_avg is not None and (not finite(args.target_avg) or not 0 <= args.target_avg <= 1):
        parser.error("target avg must be finite and in [0,1]")
    result = summarize(args.root, args.budgets, args.target_avg, args.clock)
    path = args.root/"cost_quality.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("cost_quality", path)
