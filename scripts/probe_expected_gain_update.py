#!/usr/bin/env python3
"""Validate local gain labels with real saved-state AdamW on independent suffixes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grace_gc.audit.benefit_replay import audit_rows
from grace_gc.audit.expected_gain import load_replay, reference_gradient
from grace_gc.audit.update_probe import adamw_update_direction
from grace_gc.logging_util.run_dir import resolve_run_dir
from grace_gc.trainer.loop import build_run_config


def _key(row):
    return (str(row["problem_id"]), str(row["path_id"]), int(row["t"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundles", required=True,
                        help="original predictor audit_bundles.jsonl with token IDs")
    parser.add_argument("--replay-dir", required=True)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--split-manifest", required=True,
                        help="split.json from expected_gain_suite.py")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--config", action="append", default=[])
    parser.add_argument("--n-start", type=int, required=True)
    parser.add_argument("--n-prefixes", type=int, default=16)
    parser.add_argument("--suffixes-per-prefix", type=int, default=4)
    parser.add_argument("--backgrounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)
    if min(args.n_start, args.n_prefixes, args.suffixes_per_prefix,
           args.backgrounds) <= 0:
        raise ValueError("counts must be positive")
    cfg = build_run_config(args.config, {"checkpoint": args.checkpoint,
                                         "model_path": args.model_path})
    from functools import partial
    from types import SimpleNamespace

    from grace_gc.audit.run import _policy_grad_vec
    from grace_gc.backends.hf_actor import logprob_one, named_lora_params, trainable_params
    from grace_gc.backends.verl_trainer import load_lora_actor
    from grace_gc.core.layout import collect_lora_layout
    from grace_gc.data.tokenize import load_hf_tokenizer
    from grace_gc.trainer.checkpoint import load_checkpoint
    from grace_gc.trainer.state_io import check_snapshot_identity, load_numpy_module_state
    from grace_gc.versions import sha256_named

    payload = load_checkpoint(args.checkpoint)
    check_snapshot_identity(payload, cfg)
    if int(payload.get("n_ref", args.n_start)) != args.n_start:
        raise ValueError("n-start must equal the saved training step's fixed N")
    if not payload.get("optimizer") or not payload["optimizer"].get("state"):
        raise ValueError("optimizer probe needs the saved AdamW moments and step")
    actor = load_lora_actor(args.model_path, cfg.get("lora") or {})
    named = named_lora_params(actor)
    load_numpy_module_state(named, payload.get("actor") or {})
    layout = collect_lora_layout(named)
    if payload.get("layout_names") != layout.names() or int(payload.get("layout_dim", -1)) != layout.dim:
        raise ValueError("checkpoint and actor LoRA layout differ")
    actor_sha = sha256_named(named)
    for folder in (args.replay_dir, args.reference_dir):
        provenance = json.loads((Path(folder) / "replay_provenance.json").read_text(encoding="utf-8"))
        if provenance.get("actor_sha256") != actor_sha:
            raise ValueError(f"replay actor differs from optimizer checkpoint: {folder}")
    tok = load_hf_tokenizer(args.model_path)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    if pad is None:
        raise ValueError("tokenizer needs pad or EOS ID")
    engines = SimpleNamespace(
        logprob_one=partial(logprob_one, actor, pad_id=int(pad), eos_id=tok.eos_token_id),
        trainable_params=partial(trainable_params, actor),
        named_lora=partial(named_lora_params, actor),
    )
    gradient = partial(_policy_grad_vec, engines, layout)
    metadata, _ = load_replay(args.replay_dir)
    ref_rows, ref_means = load_replay(args.reference_dir)
    reference = reference_gradient(ref_rows, ref_means)
    split = json.loads(Path(args.split_manifest).read_text(encoding="utf-8"))
    train_ids, validation_ids = set(split["train"]), set(split["validation"])
    train = [row for row in metadata if row["problem_id"] in train_ids]
    validation = [row for row in metadata if row["problem_id"] in validation_ids]
    if not train or not validation:
        raise ValueError("split needs live training and validation prefixes")
    source = {_key(row): row for row in audit_rows(args.bundles, 512)
              if not bool(row.get("finished", False))}
    if any(_key(row) not in source for row in train + validation):
        raise ValueError("audit bundles and replay prefixes differ")
    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(len(validation), size=min(args.n_prefixes, len(validation)), replace=False)
    chosen_rows = [validation[int(i)] for i in chosen]
    # Backgrounds are independent training-side continuations, normalized as
    # the other N-1 starts of the same fixed training step.
    backgrounds = []
    for _ in range(args.backgrounds):
        indices = rng.choice(len(train), size=args.n_start - 1,
                             replace=len(train) < args.n_start - 1)
        total = np.zeros(layout.dim, dtype=np.float64)
        for index in indices:
            bundle = source[_key(train[int(index)])]
            record = bundle["continuation_records"][int(rng.integers(len(bundle["continuation_records"])))]
            total += gradient(record["token_ids"], int(record["prompt_len"]),
                              float(record["reward"]), float(record["baseline"]))
        backgrounds.append(total / args.n_start)
    clip = float((cfg.get("optim") or {}).get("grad_clip", 1.))
    baseline = [adamw_update_direction(payload["actor"], layout, payload["optimizer"],
                                       cfg.get("optim") or {}, b, reference, clip)
                for b in backgrounds]
    report = []
    for row in chosen_rows:
        records = source[_key(row)]["continuation_records"]
        indices = row["continuation_indices"][:args.suffixes_per_prefix]
        for suffix_index in indices:
            record = records[int(suffix_index)]
            g = gradient(record["token_ids"], int(record["prompt_len"]),
                         float(record["reward"]), float(record["baseline"]))
            linear = float(reference @ g / args.n_start)
            for background_index, b in enumerate(backgrounds):
                update = adamw_update_direction(
                    payload["actor"], layout, payload["optimizer"],
                    cfg.get("optim") or {}, b + g / args.n_start, reference, clip)
                report.append({"problem_id": row["problem_id"], "path_id": row["path_id"],
                               "suffix_index": int(suffix_index), "background": background_index,
                               "sgd_directional_label_without_lr": linear,
                               "adamw_marginal_reference_dot_delta":
                                   update["reference_dot_delta"] -
                                   baseline[background_index]["reference_dot_delta"],
                               "clip_triggered_with": update["clip_triggered"],
                               "clip_triggered_background": baseline[background_index]["clip_triggered"]})
        print(f"optimizer_probe problem={row['problem_id']} path={row['path_id']}", flush=True)
    out = resolve_run_dir(args.run_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "optimizer_probe.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in report), encoding="utf-8")
    summary = {"n_prefixes": len(chosen_rows), "n_rows": len(report),
               "n_start": args.n_start, "backgrounds": len(backgrounds),
               "clip": clip, "actor_sha256": actor_sha,
               "label_scope": "fixed-actor reference-gradient local proxy; not observed reward change"}
    (out / "optimizer_probe_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({**summary, "run_dir": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
