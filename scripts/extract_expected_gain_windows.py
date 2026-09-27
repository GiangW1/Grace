#!/usr/bin/env python3
"""Replay frozen prefix token IDs through the actor for parameter-free window features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grace_gc.audit.expected_gain import start_experiment_run
from grace_gc.predictor.features import multiwindow_hidden_features


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--window-tokens", type=int, default=128)
    args = parser.parse_args(argv)
    if args.window_tokens <= 0:
        raise ValueError("window-tokens must be positive")

    import torch
    from grace_gc.audit.benefit_replay import replay_config
    from grace_gc.backends.hf_actor import actor_forward, named_lora_params
    from grace_gc.backends.verl_trainer import load_lora_actor
    from grace_gc.core.layout import collect_lora_layout
    from grace_gc.data.tokenize import load_hf_tokenizer
    from grace_gc.trainer.checkpoint import load_checkpoint
    from grace_gc.trainer.state_io import check_snapshot_identity, load_numpy_module_state
    from grace_gc.versions import sha256_file, sha256_named

    replay = Path(args.replay_dir)
    rows = [json.loads(line) for line in (replay / "prefixes.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    provenance = json.loads((replay / "replay_provenance.json").read_text(encoding="utf-8"))
    payload = load_checkpoint(args.checkpoint)
    cfg = replay_config(payload, args.model_path)
    if provenance.get("lora") is not None and cfg.get("lora") != provenance["lora"]:
        raise ValueError("replay and checkpoint LoRA configurations differ")
    check_snapshot_identity(payload, cfg)
    actor = load_lora_actor(args.model_path, cfg.get("lora") or {})
    named = named_lora_params(actor)
    load_numpy_module_state(named, payload.get("actor") or {})
    layout = collect_lora_layout(named)
    if (provenance.get("checkpoint_sha256") != sha256_file(args.checkpoint) or
            provenance.get("actor_sha256") != sha256_named(named) or
            provenance.get("layout_names") != layout.names() or
            int(provenance.get("layout_dim", -1)) != layout.dim or
            provenance.get("model_path") != args.model_path):
        raise ValueError("replay and loaded frozen actor/layout differ")
    tokenizer = load_hf_tokenizer(args.model_path)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad is None:
        raise ValueError("tokenizer needs a pad or EOS ID")
    destination = start_experiment_run(args.run_dir, "expected_gain_multiwindow_features", vars(args))
    started = perf_counter()
    matrix = None
    with torch.no_grad():
        for i, row in enumerate(rows):
            tokens, prompt = row.get("prefix_token_ids"), row.get("prompt_token_ids")
            if not tokens or not prompt or tokens[:len(prompt)] != prompt or len(tokens) <= len(prompt):
                raise ValueError(f"replay prefix {i} has invalid prompt/prefix token IDs")
            out, _, _ = actor_forward(actor, [tokens], int(pad), output_hidden_states=True)
            hidden = out.hidden_states[-1][0].detach().float().cpu().numpy()
            feature = multiwindow_hidden_features(hidden, len(prompt), args.window_tokens)
            if matrix is None:
                matrix = np.lib.format.open_memmap(destination / "multiwindow_features.npy", mode="w+",
                                                   dtype=np.float64, shape=(len(rows), len(feature)))
            matrix[i] = feature
            print(f"window_features={i + 1}/{len(rows)}", flush=True)
    if matrix is None:
        raise ValueError("replay has no prefixes")
    matrix.flush()
    entries = [{"name": entry.name, "offset": entry.offset, "numel": entry.numel}
               for entry in layout.entries]
    (destination / "layout_entries.json").write_text(json.dumps(entries, indent=2), encoding="utf-8")
    feature_meta = {"replay_prefix_sha256": sha256_file(replay / "prefixes.jsonl"),
                    "checkpoint_sha256": provenance["checkpoint_sha256"],
                    "actor_sha256": provenance["actor_sha256"],
                    "lora": provenance.get("lora"),
                    "layout_names": layout.names(), "layout_dim": layout.dim,
                    "model_path": args.model_path, "window_tokens": args.window_tokens,
                    "n_prefixes": len(rows), "feature_dim": matrix.shape[1],
                    "wall_seconds": perf_counter() - started}
    (destination / "feature_provenance.json").write_text(
        json.dumps(feature_meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"run_dir": str(destination), "n_prefixes": len(rows),
                      "wall_seconds": feature_meta["wall_seconds"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
