#!/usr/bin/env python3
"""Summarize saved thinking outputs, keeping original and rescored rewards separate."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grace_gc.audit.learning_stats import _strict_pre_answer
from grace_gc.data.reward import rule_reward, score_prefilled_answer


def summarize(root):
    rows, by_position, probes = [], {}, []
    for path in sorted((root / "collection/problems").glob("*/path-*/bundle.json")):
        row = json.loads(path.read_text())
        rows.append(row)
        counts = by_position.setdefault(str(row["t"]), Counter())
        counts["prefixes"] += 1
        counts["answer_emitted_prefixes"] += bool(row["answer_emitted"])
        counts["open_thinking_prefixes"] += "<think>" in row["prefix_text"] and "</think>" not in row["prefix_text"]
        for record in row["continuation_records"]:
            counts["continuations"] += 1
            counts["published_correct"] += int(record["reward"])
            counts["truncated"] += bool(record["truncated"])
            counts["published_correct_truncated"] += int(record["reward"]) if record["truncated"] else 0
            corrected_reward = float(rule_reward(
                record["text"], row["gold"], truncated=bool(record["truncated"])) or 0.0)
            counts["corrected_correct"] += int(corrected_reward)
            counts["corrected_correct_truncated"] += int(corrected_reward) if record["truncated"] else 0
            if not record["truncated"] and "</think>" in record["text"]:
                counts["completed_post_think_correct"] += int(corrected_reward)
        observations = json.loads((path.parent / "functional_probes.json").read_text())
        # Answer: was supplied in the prompt, so scoring only the new text
        # incorrectly treats a bare correct expression as missing an answer.
        corrected = [float(score_prefilled_answer(probe["text"], row["gold"]) or 0)
                     for probe in observations]
        mean = sum(corrected) / len(corrected)
        probe_row = {"problem_id": row["problem_id"], "path_id": row["path_id"], "t": row["t"],
                     "original_rewards": [probe["reward"] for probe in observations],
                     "rescored_rewards": corrected, "qualification_mean_reward": mean,
                     "functional_recoverable": mean >= .5,
                     "normal_q_mean_reward": row.get(
                         "normal_q_mean_reward", (row.get("half_mean_reward") or [None])[0]),
                     "answer_emitted": row["answer_emitted"], "finished": row["finished"]}
        probe_row["strict_pre_answer_undecided"] = _strict_pre_answer(probe_row)
        probes.append(probe_row)
    totals = Counter()
    for counts in by_position.values():
        totals.update(counts)
    benefit = json.loads((root / "collection/benefit/predictor_benefit_summary.json").read_text())
    comparisons = []
    for row in benefit["lopo"]:
        if row["selection"] != "all":
            continue
        product = next(value for value in row["variance_cost"] if value["p"] == .5)
        comparisons.append({"model": row["model"], "residual_ratio_to_zero": row["residual_ratio_to_zero"],
                            "variance_cost_ratio": product["variance_cost_ratio"],
                            "variance_cost_ci95": product["ratio_ci95"],
                            "token_cost_ratio": product["token_cost_ratio"]})
    report = {"checked_utc": datetime.now(timezone.utc).isoformat(),
              "model": "Qwen/Qwen3-4B", "enable_thinking": True,
              "n_problems": len({row["problem_id"] for row in rows}),
              "totals": dict(totals), "by_position": {key: dict(value) for key, value in by_position.items()},
              "published_correct_rate": totals["published_correct"] / totals["continuations"],
              "corrected_correct_rate": totals["corrected_correct"] / totals["continuations"],
              "truncation_rate": totals["truncated"] / totals["continuations"],
              "completed_post_think_correct_rate": totals["completed_post_think_correct"] / totals["continuations"],
              "probe_rescore": {"n": sum(len(row["rescored_rewards"]) for row in probes),
                                "original_correct": sum(sum(row["original_rewards"]) for row in probes),
                                "rescored_correct": sum(sum(row["rescored_rewards"]) for row in probes),
                                "recoverable_prefixes": sum(row["functional_recoverable"] for row in probes),
                                "strict_pre_answer_undecided_prefixes": sum(row["strict_pre_answer_undecided"] for row in probes)},
              "gradient_statistics": {
                  "valid_for_reward_protocol": False,
                  "reason": "saved gradients were collected before reward protocol v3 and were not replayed",
              },
              "original_reward_comparisons_p05": comparisons,
              "limitations": ["Published PR13 predictor and gradient statistics were collected before reward protocol v3 and must be recomputed.",
                              "Published PR13 functional probe files were collected before the prefilled Answer: scoring fix and are retained only as legacy evidence.",
                              "Strict-subset labels now use ordinary continuation q and the corrected functional probe separately.",
                              "Fresh zero-B LoRA has no Adam optimizer metric; Euclidean statistics only.",
                              "Token-cost estimates exclude GPU, scheduling and feature overhead."]}
    return report, probes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report, probes = summarize(args.run_dir)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "checked_summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (args.output / "functional_probe_rescores.jsonl").write_text(
        "".join(json.dumps(row, allow_nan=False) + "\n" for row in probes))
    print(json.dumps({"output": str(args.output), "totals": report["totals"], "probe_rescore": report["probe_rescore"]}))


if __name__ == "__main__":
    main()
