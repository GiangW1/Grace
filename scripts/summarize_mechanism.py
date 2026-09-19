"""Check observed pairing across four fixed-N training arms, without execution gates."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.summarize_minimal import checkpoint_matches, finite, paired_delta, read_json, read_rows, stage_path


def _config(path):
    return (yaml.safe_load(path.read_text(encoding="utf-8")) or {}) if path.is_file() else {}


def _response_tokens(row, main):
    if main:
        for key in ("generated_response_tokens", "total_generated_tokens"):
            if row.get(key) is not None:
                return int(row[key])
        if row.get("prefix_tokens") is not None and row.get("suffix_tokens") is not None:
            return int(row["prefix_tokens"])+int(row["suffix_tokens"])
    elif row.get("response_tokens") is not None:
        return int(row["response_tokens"])
    full, prompt = row.get("full_token_ids"), row.get("prompt_token_ids")
    if full is None and main:
        full = row.get("prefix_token_ids")
    length = len(prompt) if prompt is not None else row.get("prompt_len")
    return len(full)-int(length) if full is not None and length is not None else None


def training_costs(train, steps, summary, main_rows=None):
    """Observed generated responses; absent auxiliary logs are unknown, not zero."""
    train = Path(train)
    all_cost, active_cost, issues = {}, {}, []
    known_warmup = bool(steps) and all(isinstance(row.get("warmup"), bool) for row in steps)
    active = {row["step"] for row in steps if row.get("warmup") is False}
    step_ids = {row["step"] for row in steps}
    endpoint = summary.get("step", (summary.get("summary") or {}).get("step"))
    all_steps_recorded = bool(steps) and endpoint is not None and len(step_ids) == len(steps) and step_ids == set(range(1, int(endpoint)+1))
    zero_evidence = []
    config_path = train/"effective_config.yaml"
    config = _config(config_path if config_path.is_file() else train/"config.yaml")
    prescan_samples = (config.get("baseline") or {}).get("prescan")

    def auxiliary_matches_step_facts(key, rows, selected_steps):
        by_step = {}
        for row in rows:
            by_step.setdefault(row.get("step"), []).append(row)
        for step in selected_steps:
            recorded = by_step.get(step["step"], [])
            if key == "prescan_tokens":
                count = step.get("n_prescan")  # Number of unseen problems, not answers.
                if not finite(count):
                    continue  # Legacy logs without this fact retain their observed sums.
                if count == 0 and recorded:
                    return False
                if finite(prescan_samples) and len(recorded) != count * prescan_samples:
                    return False
                if all(row.get("problem_id") is not None for row in recorded):
                    groups = Counter(row["problem_id"] for row in recorded)
                    if len(groups) != count or (finite(prescan_samples) and any(n != prescan_samples for n in groups.values())):
                        return False
            else:
                fact = ((step.get("predictor") or {}).get("fresh_supervision") or {})
                if finite(fact.get("n")) and len(recorded) != fact["n"]:
                    return False
                tokens = [_response_tokens(row, False) for row in recorded]
                if finite(fact.get("generated_tokens")) and all(finite(n) for n in tokens) and sum(tokens) != fact["generated_tokens"]:
                    return False
        return True

    def proves_zero(key, selected_steps):
        if key == "prescan_tokens":
            return all(row.get("n_prescan") == 0 for row in selected_steps)
        if key == "fresh_tokens":
            facts = [((row.get("predictor") or {}).get("fresh_supervision") or {}) for row in selected_steps]
            return all(fact.get("n") == 0 and fact.get("generated_tokens") == 0 for fact in facts)
        return False
    for key, filename in (("main_tokens", "trajectories.jsonl"), ("prescan_tokens", "prescan.jsonl"),
                          ("fresh_tokens", "fresh_supervision.jsonl")):
        path = train/filename
        rows = main_rows if key == "main_tokens" and main_rows is not None else read_rows(path)
        values = [_response_tokens(row, key == "main_tokens") for row in rows]
        complete = path.is_file() and all(value is not None and value >= 0 for value in values)
        if key == "main_tokens" and steps:
            complete = complete and Counter(row.get("step") for row in rows) == {row["step"]: row["n"] for row in steps}
        all_cost[key] = sum(values) if complete else None
        active_cost[key] = sum(value for row, value in zip(rows, values) if row.get("step") in active) if (
            complete and known_warmup and all(row.get("step") in step_ids for row in rows)) else None
        if all_steps_recorded and key != "main_tokens":
            for scope, target, selected_steps in (("all_training", all_cost, steps),
                    ("post_warmup", active_cost, [row for row in steps if row["step"] in active])):
                if scope == "post_warmup" and not known_warmup:
                    continue
                no_rows = not any(row.get("step") in {s["step"] for s in selected_steps} for row in rows)
                if no_rows:
                    # An absent/empty log with a positive count is incomplete.
                    if proves_zero(key, selected_steps):
                        target[key] = 0
                        zero_evidence.append(scope+"."+key)
                    else:
                        target[key] = None
                if not auxiliary_matches_step_facts(key, rows, selected_steps):
                    target[key] = None
                    issues.append(scope+"."+filename+"_inconsistent_with_step_facts")
        if not complete:
            issues.append(filename+"_missing_or_incomplete_token_evidence")
    if not all_steps_recorded:
        issues.append("training_step_range_missing_or_incomplete")
        for value in (all_cost, active_cost):
            for key in value:
                value[key] = None
    for value in (all_cost, active_cost):
        value["total_generation_tokens"] = sum(value.values()) if all(v is not None for v in value.values()) else None
    cumulative = summary.get("cumulative_wall_seconds")
    ledger = read_json(train/"compute_ledger.json")
    all_cost["wall_seconds"] = float(cumulative) if finite(cumulative) else ledger.get("wall_seconds")
    boundary_path = train/"cost_control.jsonl"
    boundaries = [row for row in read_rows(boundary_path) if row.get("event") == "batch_complete" and row.get("step") in active]
    active_cost["wall_seconds"] = sum(row["last_batch_wall_seconds"] for row in boundaries) if (
        all_steps_recorded and boundary_path.is_file() and known_warmup and {row.get("step") for row in boundaries} == active and
        all(finite(row.get("last_batch_wall_seconds")) for row in boundaries)) else None
    return {"all_training": all_cost, "post_warmup": active_cost, "issues": issues,
            "zero_proven_by_complete_step_facts": zero_evidence,
            "scope": "Generated response tokens from main/prescan/fresh logs; excludes shared or in-entry SFT tokens. Absent auxiliary rows become zero only when every completed step in that range records zero generation (n_prescan=0; fresh n=0 and generated_tokens=0), never from configuration alone; otherwise missing evidence remains null. Present auxiliary rows are checked against recorded problem/sample counts and fresh token totals when available; prescan sample counts also use the saved configuration. Legacy logs lacking these count fields retain observed sums, without claiming independent completeness certification. Total training wall is the cumulative run envelope (ledger fallback), including setup, auxiliary computation and persistence, not a sum of token proxies. Post-warmup wall sums recorded complete batch boundaries including persistence; unassigned setup/failure tails are not allocated to it."}


def _evaluation_evidence(train, source_train, final_path, final, summary, config, steps):
    source = read_json(final_path/"actor_source.json") if final_path else {}
    eval_config = _config(final_path/"config.yaml") if final_path else {}
    protocol = final.get("evaluation_manifest") or (read_json(final_path/"evaluation_manifest.json") if final_path else {})
    checkpoint = final.get("checkpoint") or source.get("checkpoint") or eval_config.get("checkpoint")
    match = checkpoint_matches(checkpoint, train, source_train)
    issues = []
    if match is not True:
        issues.append("final_checkpoint_source_mismatched" if match is False else "final_checkpoint_source_unverified")
    train_step = summary.get("step", (summary.get("summary") or {}).get("step"))
    eval_step = final.get("checkpoint_step", source.get("checkpoint_step"))
    if train_step is None or eval_step is None or train_step != eval_step:
        issues.append("final_checkpoint_step_missing_or_not_training_endpoint")
    endpoint = next((row for row in reversed(steps) if row.get("step") == train_step), {})
    if source.get("actor_sha256") and endpoint.get("post_update_snapshot_sha") and source["actor_sha256"] != endpoint["post_update_snapshot_sha"]:
        issues.append("final_actor_hash_mismatched_to_training_endpoint")
    if final.get("seed") is not None and protocol.get("sample_seed_start") is not None and final["seed"] != protocol["sample_seed_start"]:
        issues.append("evaluation_seed_inconsistent_with_manifest")
    return {"actual_seed": config.get("seed"), "evaluation_manifest": protocol,
            "final_training_step": train_step, "final_checkpoint_step": eval_step,
            "final_evaluation_path": str(final_path) if final_path else None,
            "final_actor_sha256": source.get("actor_sha256"), "source_issues": issues}


def _paired_with_evidence(left, right, left_eval, right_eval):
    issues = [f"{label}:{issue}" for label, arm in (("grace", left), ("arm", right)) for issue in arm.get("source_issues", [])]
    for field in ("initial_actor_sha256", "actual_seed"):
        if left.get(field) is None or left.get(field) == "" or left.get(field) != right.get(field):
            issues.append(field+"_missing_or_mismatched")
    if not all(arm.get("input_step_range_complete") and arm.get("input_rows_match_all_recorded_steps") and arm.get("input_tokens_recorded") and arm.get("input_sequence_sha256")
               for arm in (left, right)) or left.get("input_sequence_sha256") != right.get("input_sequence_sha256"):
        issues.append("complete_problem_and_prompt_token_sequence_missing_or_mismatched")
    if not all(arm.get("starts_per_step") and all(n == 16 for n in arm["starts_per_step"]) for arm in (left, right)):
        issues.append("fixed_n16_missing_or_mismatched")
    lm, rm = left.get("evaluation_manifest") or {}, right.get("evaluation_manifest") or {}
    for field in ("ordered_records_sha256", "reward_protocol_version", "samples_per_problem", "temperature",
                  "top_p", "max_new_tokens", "sample_batch_size", "sample_seed_start"):
        if lm.get(field) is None or lm.get(field) == "" or lm.get(field) != rm.get(field):
            issues.append("evaluation_"+field+"_missing_or_mismatched")
    if issues:
        return {"available": False, "issues": issues, "note": "Mechanism paired inference withheld because initialization, training seed, complete fixed-N inputs, endpoint source or evaluation protocol is unverified; raw per-arm scores remain visible."}
    result = paired_delta(left_eval, right_eval)
    available = "effect" in result
    return {**result, "available": available, "issues": [] if available else ["evaluated_problem_ids_missing_or_mismatched"]}


def summarize(root):
    root = Path(root)
    plan = read_json(root/"mechanism.json")
    chains = {name: stage_path(root, value) for name, value in plan.get("arms", {}).items()}
    seeds = sorted({path.name for chain in chains.values() for path in chain.glob("seed-*") if path.is_dir()})
    result = {"design": plan.get("design"), "seeds": {},
              "scope": "Recorded shared actor and fixed input design, not identical future tokens, equal optimizer states after learning, equal cost, or training-seed confidence intervals. Missing evidence is reported, not an execution gate."}
    for seed in seeds:
        arms, evaluations = {}, {}
        for name, chain in chains.items():
            folder = chain/seed
            stages = read_json(folder/"stages.json").get("grace", {})
            if not stages.get("train"):
                arms[name] = {"status": "missing_train_stage"}; continue
            train = stage_path(folder, stages["train"])
            summary = read_json(train/"summary.json")
            config_path = train/"effective_config.yaml"
            config = _config(config_path if config_path.is_file() else train/"config.yaml")
            rows, steps = read_rows(train/"trajectories.jsonl"), read_rows(train/"steps.jsonl")
            inputs = [[row.get("step"), row.get("problem_id"), row.get("prompt_token_ids")] for row in rows]
            counts = Counter(row.get("step") for row in rows)
            complete_rows = bool(steps) and counts == {row["step"]: row["n"] for row in steps}
            endpoint = summary.get("step", (summary.get("summary") or {}).get("step"))
            step_ids = {row["step"] for row in steps}
            complete_range = (len(step_ids) == len(steps) and step_ids == set(range(1, int(endpoint)+1))) if endpoint is not None else None
            final_path = stage_path(folder, stages["eval-final"]) if stages.get("eval-final") else None
            final = read_json(final_path/"eval_summary.json") if final_path else {}
            evaluations[name] = final
            costs = training_costs(train, steps, summary, rows)
            arms[name] = {"status": summary.get("run_status", "recorded"),
                "train": str(train), "initial_actor_sha256": read_json(train/"initial_actor.json").get("actor_sha256"),
                "n_steps": len(steps), "starts_per_step": [row.get("n") for row in steps],
                "input_rows": len(rows), "input_tokens_recorded": bool(rows) and all(row.get("prompt_token_ids") is not None for row in rows),
                "input_rows_match_all_recorded_steps": complete_rows,
                "input_step_range_complete": complete_range,
                "input_sequence_sha256": hashlib.sha256(json.dumps(inputs, separators=(",", ":")).encode()).hexdigest() if rows else None,
                "final_avg": final.get("avg"), "final_pass_at_k": final.get("pass_at_k"),
                "training_wall_seconds": costs["all_training"]["wall_seconds"], "training_costs": costs,
                **_evaluation_evidence(train, stages["train"], final_path, final, summary, config, steps)}
        hashes = [row.get("initial_actor_sha256") for row in arms.values()]
        input_hashes = [row.get("input_sequence_sha256") for row in arms.values()]
        complete = len(arms) == 4 and all(row.get("n_steps", 0) and row.get("input_step_range_complete") for row in arms.values())
        match = {"shared_initial_actor": len(set(hashes)) == 1 if len(hashes) == 4 and all(hashes) else None,
                 "same_problem_and_prompt_token_sequence": len(set(input_hashes)) == 1 if complete and all(row.get("input_tokens_recorded") and row.get("input_rows_match_all_recorded_steps") for row in arms.values()) else None,
                 "fixed_n16_all_steps": all(all(n == 16 for n in row["starts_per_step"]) for row in arms.values()) if complete else None}
        contrasts = {name: _paired_with_evidence(arms.get("grace", {}), arms[name], evaluations.get("grace", {}), final)
                     for name, final in evaluations.items() if name != "grace"}
        result["seeds"][seed] = {"arms": arms, "pairing": match, "final_grace_minus_arm": contrasts}
    result["batch_audits"] = {path.name: {key: read_json(path/"batch_audit_summary.json").get(key)
        for key in ("status", "fixed_n", "replicates", "interventions", "intervention_scope", "manifest")}
        for path in root.glob("batch-audit-seed-*") if path.is_dir()}
    (root/"mechanism_summary.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    args = parser.parse_args()
    print(json.dumps(summarize(args.root), indent=2))
