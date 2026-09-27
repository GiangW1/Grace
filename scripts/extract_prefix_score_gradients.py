#!/usr/bin/env python3
"""Extract exact prefix score gradients for stages A/B.

This is the only GPU-side step in the dynamic score experiment.  It uses the
same frozen actor and LoRA layout as the replay bundle, and stores one
``[layout_dim, 1]`` score-gradient basis column per live prefix.
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grace_gc.audit.benefit_replay import audit_file, replay_config
from grace_gc.audit.expected_gain import start_experiment_run
from grace_gc.core.layout import collect_lora_layout, pack_grads
from grace_gc.versions import sha256_array, sha256_file, sha256_named


def _prefix_score_gradient(engines, layout, token_ids, prompt_len):
    torch = __import__("torch")
    logprob = engines.logprob_one(token_ids, int(prompt_len))
    grads = torch.autograd.grad(logprob, engines.trainable_params(), allow_unused=True)
    packed = []
    for (name, param), grad in zip(engines.named_lora(), grads):
        packed.append((name, np.zeros(tuple(param.shape), dtype=np.float64)
                       if grad is None else grad.detach().float().cpu().numpy()))
    return pack_grads(packed, layout)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--config", action="append", default=[])
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)

    from grace_gc.backends.hf_actor import logprob_one, named_lora_params, trainable_params
    from grace_gc.backends.verl_trainer import load_lora_actor
    from grace_gc.data.tokenize import collect_stop_token_ids, load_hf_tokenizer
    from grace_gc.trainer.checkpoint import load_checkpoint
    from grace_gc.trainer.state_io import check_snapshot_identity, load_numpy_module_state

    rows_path = audit_file(args.replay_dir).parent / "prefixes.jsonl"
    if not rows_path.is_file():
        raise FileNotFoundError(f"replay prefixes are not readable: {rows_path}")
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    payload = load_checkpoint(args.checkpoint)
    cfg = replay_config(payload, args.model_path, args.config, args.replay_dir)
    check_snapshot_identity(payload, cfg)
    actor = load_lora_actor(args.model_path, cfg.get("lora") or {})
    named = named_lora_params(actor)
    load_numpy_module_state(named, payload.get("actor") or {})
    replay_meta_path = Path(args.replay_dir) / "replay_provenance.json"
    replay_meta = json.loads(replay_meta_path.read_text(encoding="utf-8")) if replay_meta_path.is_file() else {}
    actor_hash = sha256_named(named)
    if replay_meta.get("actor_sha256") and replay_meta["actor_sha256"] != actor_hash:
        raise ValueError("replay and checkpoint use different actor weights")
    layout = collect_lora_layout(named)
    if replay_meta.get("layout_dim") is not None and int(replay_meta["layout_dim"]) != layout.dim:
        raise ValueError("replay and checkpoint LoRA dimensions differ")
    if replay_meta.get("layout_names") is not None and list(replay_meta["layout_names"]) != layout.names():
        raise ValueError("replay and checkpoint LoRA parameter order differs")
    tok = load_hf_tokenizer(args.model_path)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    if pad is None:
        raise ValueError("tokenizer needs a pad or EOS ID")
    stop_ids = collect_stop_token_ids(tok)
    engines = type("Engines", (), {
        "logprob_one": partial(logprob_one, actor, pad_id=int(pad), eos_id=stop_ids),
        "trainable_params": partial(trainable_params, actor),
        "named_lora": partial(named_lora_params, actor),
    })()

    destination = start_experiment_run(args.run_dir, "expected_gain_prefix_score_gradients", vars(args))
    output = np.lib.format.open_memmap(destination / "score_gradients.npy", mode="w+",
                                       dtype=np.float64, shape=(len(rows), layout.dim, 1))
    for index, row in enumerate(rows):
        prompt = row.get("prompt_token_ids")
        prefix = row.get("prefix_token_ids")
        if not prompt or not prefix or list(prefix[:len(prompt)]) != list(prompt):
            raise ValueError(f"prefix row {index} lacks aligned prompt/prefix token IDs")
        output[index, :, 0] = _prefix_score_gradient(engines, layout, list(map(int, prefix)), len(prompt))
        output.flush()
        print(f"score_gradient={index + 1}/{len(rows)} problem={row['problem_id']}", flush=True)
    del output
    provenance = {
        "replay_dir": str(Path(args.replay_dir).resolve()),
        "replay_prefixes_sha256": sha256_file(rows_path),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "actor_sha256": actor_hash,
        "layout_names": layout.names(), "layout_dim": layout.dim,
        "shape": [len(rows), layout.dim, 1],
        "basis_kind": "prefix_score_gradient",
        "score_gradients_sha256": sha256_array(np.load(destination / "score_gradients.npy", mmap_mode="r")),
    }
    (destination / "score_gradient_provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"run_dir": str(destination), **provenance}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
