"""Prepare a fresh audit selection and an immutable shared starting actor."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grace_gc.data.format_prompt import DAPO_SOLVE_PREFIX, DAPO_SOLVE_SUFFIX
from grace_gc.data.math_data import load_math_records, load_training_data, select_records, selection_manifest
from grace_gc.versions import sha256_file


def prompt_key(record):
    users = [message["content"] for message in (record.messages or []) if message["role"] == "user"]
    value = " ".join((users[-1] if users else record.prompt).split())
    prefix, suffix = " ".join(DAPO_SOLVE_PREFIX.split()), " ".join(DAPO_SOLVE_SUFFIX.split())
    if value.startswith(prefix):
        value = value[len(prefix):].strip()
    if value.endswith(suffix):
        value = value[:-len(suffix)].strip()
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--eval-data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--previous-input", type=Path, action="append", default=[])
    args = parser.parse_args()
    destination = args.run_dir / "input"
    destination.mkdir(parents=True, exist_ok=True)
    buckets, report = load_training_data(args.data, args.eval_data, seed=17)
    previous = [record for path in args.previous_input for record in load_math_records(path)]
    previous_keys, previous_ids = {prompt_key(row) for row in previous}, {row.problem_id for row in previous}
    candidates = [row for row in buckets["audit"]
                  if row.problem_id not in previous_ids and prompt_key(row) not in previous_keys]
    selected = select_records(candidates, 32, "seeded", 17)
    selected_keys = {prompt_key(row) for row in selected}
    training_keys = {prompt_key(row) for row in buckets["train"]}
    evaluation_keys = {prompt_key(row) for row in load_math_records(args.eval_data)}
    assert not selected_keys & (training_keys | evaluation_keys | previous_keys)
    for name, records in (("audit", selected), ("train", buckets["train"])):
        path = destination / (name + ".jsonl")
        if path.exists():
            raise FileExistsError(path)
        path.write_text("".join(json.dumps(asdict(replace(row, split=name)), ensure_ascii=True) + "\n"
                                for row in records))
    shared = args.run_dir / "shared-initial-actor.npz"
    if shared.exists():
        raise FileExistsError(shared)
    shutil.copy2(args.checkpoint, shared)
    metadata = {"source": str(args.data), "source_sha256": sha256_file(args.data),
                "eval_source": str(args.eval_data), "eval_sha256": sha256_file(args.eval_data),
                "shared_actor": str(shared), "shared_actor_sha256": sha256_file(shared),
                "previous_inputs": [{"path": str(path), "sha256": sha256_file(path)} for path in args.previous_input],
                "audit_candidate_count": len(candidates), "audit": selection_manifest(selected, "seeded", 17),
                "train_count": len(buckets["train"]), "audit_previous_prompt_overlap": 0,
                "audit_train_prompt_overlap": 0, "audit_eval_prompt_overlap": 0,
                "normalization": "whitespace and exact DAPO wrapper; no semantic near-duplicate check",
                "data_conflicts_and_eval_exclusion": report}
    (args.run_dir / "input_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"audit": len(selected), "train": len(buckets["train"]), "prior_prompt_overlap": 0}))


if __name__ == "__main__":
    main()
