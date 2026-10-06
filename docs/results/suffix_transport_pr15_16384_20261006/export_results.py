"""Export compact evidence from the completed PR15 frozen audits."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def token_hash(tokens):
    return hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()


def rows(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_rows(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        for value in values:
            stream.write(json.dumps(value, allow_nan=False) + "\n")


def describe(values):
    return {"min": min(values), "median": statistics.median(values),
            "mean": statistics.mean(values), "max": max(values)}


def export_job(source, target):
    summary = json.loads((source / "summary.json").read_text())
    groups = list(rows(source / "groups.jsonl"))
    draws = []
    for row in rows(source / "transport_draws.jsonl"):
        row["suffix_tokens"] = len(row["suffix_token_ids"])
        row["suffix_token_ids_sha256"] = token_hash(row.pop("suffix_token_ids"))
        donor, alpha = row["donor"], row["alpha"]
        likelihoods = row["log_likelihoods"]
        maximum = max(likelihoods)
        weights = [math.exp(value - maximum) for value in likelihoods]
        posterior = [value / sum(weights) for value in weights]
        if any(a != b for a, b in zip(alpha, posterior)):
            if max(abs(a - b) for a, b in zip(alpha, posterior)) > 1e-12:
                raise ValueError("saved posterior disagrees with sequence likelihoods")
        row.update(donor_posterior=alpha[donor],
                   non_donor_mass=sum(value for i, value in enumerate(alpha) if i != donor),
                   ess=1 / sum(value * value for value in alpha),
                   donor_loglik_gap=likelihoods[donor] - max(
                       value for i, value in enumerate(likelihoods) if i != donor))
        draws.append(row)
    write_rows(target / "transport_draws_compact.jsonl", draws)

    compact_groups = []
    for row in groups:
        prefixes = row.pop("prefix_token_ids")
        row.update(prefix_lengths=[len(tokens) for tokens in prefixes],
                   prefix_token_ids_sha256=[token_hash(tokens) for tokens in prefixes],
                   unique_prefixes=len({tuple(tokens) for tokens in prefixes}))
        compact_groups.append(row)
    write_rows(target / "groups_compact.jsonl", compact_groups)

    compact_trajectories = []
    for row in rows(source / "trajectories.jsonl"):
        tokens = row.pop("token_ids")
        response = row.pop("response")
        row.update(response_tokens=len(tokens) - row["prompt_len"],
                   token_ids_sha256=token_hash(tokens),
                   response_sha256=hashlib.sha256(response.encode()).hexdigest())
        compact_trajectories.append(row)
    write_rows(target / "trajectories_compact.jsonl", compact_trajectories)

    n = len(compact_trajectories)
    successes = sum(row["reward"] == 1 for row in compact_trajectories)
    truncated = sum(row["truncated"] for row in compact_trajectories)
    if (len(groups) != summary["n_groups"] or n != summary["actual_completions"]
            or successes != summary["actual_successes"]
            or len({row["problem_id"] for row in groups}) != summary["n_problems"]
            or len(draws) != sum(row["moments"]["n_draws"] for row in groups)):
        raise ValueError("raw evidence counts disagree with the saved audit summary")
    conditional = summary["methods"]
    population = summary["population"]["methods"]
    return {
        "source_identity": json.loads((source / "source_identity.json").read_text()),
        "n_problems": summary["n_problems"], "n_groups": len(groups), "n_draws": len(draws),
        "actual_completions": n, "actual_successes": successes,
        "sample_success_fraction": successes / n,
        "truncated": truncated, "truncated_fraction": truncated / n,
        "response_tokens": describe([row["response_tokens"] for row in compact_trajectories]),
        "suffix_tokens": describe([row["suffix_tokens"] for row in draws]),
        "donor_posterior": describe([row["donor_posterior"] for row in draws]),
        "non_donor_mass": describe([row["non_donor_mass"] for row in draws]),
        "ess": describe([row["ess"] for row in draws]),
        "donor_loglik_gap": describe([row["donor_loglik_gap"] for row in draws]),
        "donor_is_posterior_argmax": sum(
            row["alpha"].index(max(row["alpha"])) == row["donor"] for row in draws),
        "correction_mean_norm_squared": describe([
            row["moments"]["correction_mean_norm_squared"] for row in groups]),
        "correction_trace_variance": describe([
            row["moments"]["centered_donor_correction_gram"][1][1] for row in groups]),
        "conditional_st_over_full_pg": (
            conditional["st_lambda_1"]["mean_conditional_trace_variance"]
            / conditional["full_pg"]["mean_conditional_trace_variance"]),
        "population_st_over_full_pg": (
            population["st_lambda_1"]["population_trace_variance"]
            / population["full_pg"]["population_trace_variance"]),
        "contrasts": summary["contrasts"],
        "actual_wall_seconds": summary["actual_wall_seconds"],
        "reserved_gpu_seconds": summary["reserved_gpu_seconds"],
    }


def export_results(root, output):
    root, output = root.resolve(), output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    copied = []
    sources = list(root.glob("*.json"))
    sources += list((root / "actor-profile").glob("*.json"))
    sources += list((root / "logs").glob("*.log"))
    sources += [root / "input" / "audit.jsonl"]
    for job in (root / "jobs").iterdir():
        if job.is_dir():
            sources += list(job.glob("*.json")) + list(job.glob("*.yaml"))
            if not job.name.startswith("audit-"):
                sources += list(job.glob("*.jsonl"))
    for source in sorted(sources):
        relative = source.relative_to(root)
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append({"path": relative.as_posix(), "bytes": source.stat().st_size,
                       "sha256": sha256(source), "operation": "byte_identical_copy"})
    manifest = {"source_run": str(root), "copied_files": copied, "compacted_files": [],
                "note": "Raw text/tokens are retained in the separate filtered archive. "
                        "Dense moments, model/adapter weights and checkpoints are excluded "
                        "from this export. Source files have not been deleted."}
    analysis = {"requested_commit": "1e4e496333b6ea1b1f2c369e5baa1b19079ce185",
                "source_run": str(root), "jobs": {},
                "scope": "Frozen-actor variance audit, not trained accuracy or equal-cost "
                         "training efficiency. Population variances are point estimates; "
                         "conditional contrasts have paired problem-bootstrap intervals."}
    for m in (2, 4):
        name = f"audit-m{m}-t512"
        analysis["jobs"][name] = export_job(root / "jobs" / name, output / "jobs" / name)
        for original in ("groups.jsonl", "transport_draws.jsonl", "trajectories.jsonl"):
            source = root / "jobs" / name / original
            target = output / "jobs" / name / original.replace(".jsonl", "_compact.jsonl")
            manifest["compacted_files"].append({
                "source": source.relative_to(root).as_posix(),
                "source_sha256": sha256(source), "path": target.relative_to(output).as_posix(),
                "bytes": target.stat().st_size, "sha256": sha256(target),
                "operation": "remove_generated_text_and_token_arrays; retain_lengths_and_hashes"})
    write_json(output / "analysis.json", analysis)
    write_json(output / "export_manifest.json", manifest)
    return analysis


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    report = export_results(args.run_dir, args.output)
    print(json.dumps({name: {key: value[key] for key in
                            ("n_groups", "actual_completions", "truncated_fraction",
                             "conditional_st_over_full_pg")}
                      for name, value in report["jobs"].items()}, indent=2))
