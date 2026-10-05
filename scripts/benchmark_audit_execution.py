"""Measure exact frozen-actor score gradients with and without activation checkpointing."""

import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from audit_separated_thinking import LORA, write_json, utc
from grace_gc.audit.run import _score_grad_vec
from grace_gc.backends.hf_actor import logprob_one, named_lora_params
from grace_gc.backends.verl_trainer import load_lora_actor
from grace_gc.core.layout import collect_lora_layout
from grace_gc.data.tokenize import collect_stop_token_ids, load_hf_tokenizer
from grace_gc.trainer.checkpoint import load_checkpoint
from grace_gc.trainer.state_io import load_numpy_module_state
from grace_gc.versions import sha256_named


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    row = json.loads(Path(args.bundle).read_text())
    sample = max(row["continuation_records"], key=lambda item: len(item["token_ids"]))
    actor = load_lora_actor(args.model_path, LORA)
    named = named_lora_params(actor)
    load_numpy_module_state(named, load_checkpoint(args.checkpoint)["actor"])
    layout = collect_lora_layout(named)
    tokenizer = load_hf_tokenizer(args.model_path)
    pad = int(tokenizer.pad_token_id or 0)
    eos = collect_stop_token_ids(tokenizer)
    engines = SimpleNamespace(
        logprob_one=lambda ids, plen: logprob_one(actor, ids, plen, pad, eos_id=eos),
        named_lora=lambda: named, trainable_params=lambda: [p for _, p in named])
    result = {"started_utc": utc(), "actor_sha256": sha256_named(named),
              "bundle": str(Path(args.bundle).resolve()), "sequence_tokens": len(sample["token_ids"]),
              "prompt_len": sample["prompt_len"], "dimension": layout.dim, "modes": []}
    reference = None
    _score_grad_vec(engines, layout, sample["token_ids"][:1024], min(sample["prompt_len"], 512))
    for checkpointing in (True, False):
        if checkpointing:
            actor.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        else:
            actor.gradient_checkpointing_disable()
        actor._grace_gradient_checkpointing = checkpointing
        timings, peaks = [], []
        mode = {"gradient_checkpointing": checkpointing}
        try:
            for _ in range(args.repeats):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                started = time.perf_counter()
                gradient = _score_grad_vec(engines, layout, sample["token_ids"], sample["prompt_len"])
                torch.cuda.synchronize()
                timings.append(time.perf_counter() - started)
                peaks.append(torch.cuda.max_memory_allocated() / 2**30)
            if reference is None:
                reference = gradient.copy()
            mode.update(status="completed", seconds=timings, mean_seconds=float(np.mean(timings)),
                        peak_allocated_gib=max(peaks), all_finite=bool(np.isfinite(gradient).all()),
                        relative_l2_difference=float(np.linalg.norm(gradient - reference) /
                                                     max(np.linalg.norm(reference), 1e-30)),
                        max_absolute_difference=float(np.max(np.abs(gradient - reference))))
        except torch.cuda.OutOfMemoryError:
            mode.update(status="out_of_memory")
            torch.cuda.empty_cache()
        result["modes"].append(mode)
        write_json(args.output, result)
        print(json.dumps(mode), flush=True)
    result["finished_utc"] = utc()
    write_json(args.output, result)


if __name__ == "__main__":
    main()

