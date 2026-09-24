#!/usr/bin/env python3
"""Replay an existing frozen audit into exact full-space prefix means.

Use a separate, disjoint audit for the reference task pool. Neither the
stored 8-D coordinates nor a JL sketch is substituted for a true gradient.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grace_gc.audit.benefit_replay import audit_rows, replay_means
from grace_gc.logging_util.run_dir import resolve_run_dir
from grace_gc.trainer.loop import build_run_config


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundles", required=True)
    parser.add_argument("--reference-dir", default=None,
                        help="disjoint reference replay; records two independent half-suffix gain means")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--config", action="append", default=[])
    parser.add_argument("--decision-tokens", type=int, default=512)
    parser.add_argument("--max-continuations", type=int, default=16)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--skip-prompt-features", action="store_true",
                        help="omit the prompt-only forward baseline to save replay time")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)
    cfg = build_run_config(args.config, {"checkpoint": args.checkpoint,
                                         "model_path": args.model_path})

    from functools import partial
    from types import SimpleNamespace

    from grace_gc.audit.run import _policy_grad_vec
    from grace_gc.backends.hf_actor import (logprob_one, named_lora_params,
                                            prefix_feature_bundle, trainable_params)
    from grace_gc.backends.verl_trainer import load_lora_actor
    from grace_gc.core.layout import collect_lora_layout
    from grace_gc.data.tokenize import load_hf_tokenizer
    from grace_gc.trainer.checkpoint import load_checkpoint
    from grace_gc.trainer.state_io import check_snapshot_identity, load_numpy_module_state
    from grace_gc.versions import sha256_file, sha256_named

    payload = load_checkpoint(args.checkpoint)
    check_snapshot_identity(payload, cfg)
    actor = load_lora_actor(args.model_path, cfg.get("lora") or {})
    named = named_lora_params(actor)
    load_numpy_module_state(named, payload.get("actor") or {})
    layout = collect_lora_layout(named)
    if payload.get("layout_dim") is not None and int(payload["layout_dim"]) != layout.dim:
        raise ValueError("checkpoint LoRA layout dimension does not match actor")
    if payload.get("layout_names") is not None and list(payload["layout_names"]) != layout.names():
        raise ValueError("checkpoint LoRA parameter order does not match actor")
    tok = load_hf_tokenizer(args.model_path)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    if pad is None:
        raise ValueError("tokenizer needs a pad or EOS ID")
    engines = SimpleNamespace(
        logprob_one=partial(logprob_one, actor, pad_id=int(pad), eos_id=tok.eos_token_id),
        trainable_params=partial(trainable_params, actor),
        named_lora=partial(named_lora_params, actor),
    )
    destination = resolve_run_dir(args.run_dir)
    reference = None
    if args.reference_dir:
        from grace_gc.audit.expected_gain import load_replay, reference_gradient
        ref_provenance = json.loads((Path(args.reference_dir) / "replay_provenance.json")
                                    .read_text(encoding="utf-8"))
        if ref_provenance.get("actor_sha256") != sha256_named(named):
            raise ValueError("reference replay uses a different actor")
        reference_rows, reference_means = load_replay(args.reference_dir)
        if reference_means.shape[1] != layout.dim:
            raise ValueError("reference replay LoRA dimension differs")
        reference = reference_gradient(reference_rows, reference_means)
    def prompt_features(row):
        tokens = row.get("prompt_token_ids")
        if not tokens:
            raise ValueError("audit lacks prompt token IDs for the prompt-only baseline")
        feature_baseline = (float(row["features"][-1]) if row.get("features") is not None
                            else float(row.get("baseline") or 0.))
        return prefix_feature_bundle(actor, [tokens], [len(tokens)],
                                     [feature_baseline],
                                     int(pad), eos_id=tok.eos_token_id)["prompt_features"][0]

    result = replay_means(audit_rows(args.bundles, args.decision_tokens),
                          partial(_policy_grad_vec, engines, layout), layout.dim,
                          destination, args.max_continuations, args.seed,
                          None if args.skip_prompt_features else prompt_features,
                          reference)
    provenance = {"checkpoint": str(Path(args.checkpoint).resolve()),
                  "checkpoint_sha256": sha256_file(args.checkpoint),
                  "actor_sha256": sha256_named(named),
                  "source_bundles": str(Path(args.bundles).resolve()),
                  "decision_tokens": args.decision_tokens,
                  "max_continuations": args.max_continuations,
                  "seed": args.seed,
                  "prompt_features_replayed": not args.skip_prompt_features,
                  "reference_dir": args.reference_dir,
                  "layout_names": layout.names(), "layout_dim": layout.dim,
                  "model_path": args.model_path}
    (destination / "replay_provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({**result, "run_dir": str(destination)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
