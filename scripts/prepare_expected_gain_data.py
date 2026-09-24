#!/usr/bin/env python3
"""Reserve disjoint training-source problems for the global benefit experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grace_gc.data.math_data import load_training_data
from grace_gc.audit.expected_gain import problem_split, start_experiment_run
from grace_gc.versions import sha256_file


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--eval-data-path", required=True,
                        help="external evaluation set to exclude by normalized problem text")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--predictor-problems", type=int, default=256)
    parser.add_argument("--reference-problems", type=int, default=64)
    parser.add_argument("--reference-check-problems", type=int, default=64)
    parser.add_argument("--prior-train-ids", default=None,
                        help="JSON list of previously inspected problem IDs")
    parser.add_argument("--omit-prior-from-predictor-file", action="store_true",
                        help="replay matching old audit separately and merge before the suite")
    args = parser.parse_args(argv)
    pools, report = load_training_data(args.data_path, args.eval_data_path, seed=args.seed)
    available = pools["train"] + pools["calib"] + pools["audit"]
    by_id = {}
    for rec in available:
        if rec.problem_id in by_id and by_id[rec.problem_id].prompt != rec.prompt:
            raise ValueError(f"problem ID {rec.problem_id} maps to multiple prompt texts")
        by_id[rec.problem_id] = rec
    prior = set() if args.prior_train_ids is None else set(map(
        str, json.loads(Path(args.prior_train_ids).read_text(encoding="utf-8"))))
    if not prior <= set(by_id):
        raise ValueError("prior training IDs include problems outside the permitted source")
    if len(prior) > args.predictor_problems:
        raise ValueError("prior training set exceeds predictor-problems")
    rest = sorted(set(by_id) - prior)
    rest = list(np.random.default_rng(args.seed).permutation(rest))
    requested = args.predictor_problems + args.reference_problems + args.reference_check_problems
    if len(by_id) < requested:
        raise ValueError(f"need {requested} distinct permitted problems, found {len(by_id)}")
    selected = list(sorted(prior)) + rest
    cut1 = args.predictor_problems
    cut2 = cut1 + args.reference_problems
    roles = {"predictor": selected[:cut1],
             "reference": selected[cut1:cut2],
             "reference_check": selected[cut2:cut2 + args.reference_check_problems]}
    if min(args.predictor_problems, args.reference_problems, args.reference_check_problems) <= 0:
        raise ValueError("problem counts must be positive")
    out = start_experiment_run(args.output_dir, "expected_gain_data", vars(args))
    split = problem_split([{"problem_id": pid} for pid in roles["predictor"]],
                           seed=args.seed, prior_train=prior)
    (out / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    # A small unconditional training-side pool for the optimizer background.
    # These IDs are fixed before observing which 512-token prefixes survive.
    roles["background"] = list(np.random.default_rng(args.seed + 1).permutation(split["train"]))[:16]
    for role, ids in roles.items():
        with (out / f"{role}.jsonl").open("w", encoding="utf-8") as handle:
            for pid in ids:
                if role == "predictor" and args.omit_prior_from_predictor_file and pid in prior:
                    continue
                rec = by_id[pid]
                raw = {"problem_id": pid, "prompt": rec.messages or rec.prompt,
                       "answer": rec.answer, "source": rec.source, "split": "audit"}
                handle.write(json.dumps(raw, ensure_ascii=False) + "\n")
    manifest = {"seed": args.seed, "source_sha256": sha256_file(args.data_path),
                "external_eval_sha256": sha256_file(args.eval_data_path),
                "prior_train_ids": sorted(prior),
                "roles": {key: value for key, value in roles.items() if key != "background"},
                "background_train_ids": roles["background"],
                "predictor_file_omits_prior": args.omit_prior_from_predictor_file,
                "n_excluded_eval": len(report["eval_exclusion"]["removed"]),
                "prompt_digest_by_id": {pid: hashlib.sha256(by_id[pid].prompt.encode()).hexdigest()
                                        for ids in roles.values() for pid in ids}}
    (out / "expected_gain_data_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "prior_train_ids.json").write_text(
        json.dumps(sorted(prior), indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(out),
                      "counts": {key: len(value) for key, value in roles.items()}},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
