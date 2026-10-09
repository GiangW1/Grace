"""Profile ST actor execution on marked teacher-forcing stress fixtures."""

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    import numpy as np
    import torch
    from grace_gc.backends.hf_actor import named_lora_params
    from grace_gc.backends.suffix_transport import (configure_checkpoint_stride, gradient_vector,
                                                    range_logprob, range_logprob_batch, split_logprob)
    from grace_gc.backends.verl_trainer import load_lora_actor
    from grace_gc.config import load_config
    from grace_gc.logging_util.run_dir import RunDirectory
    from grace_gc.trainer.initialization import initialize_actor
    from grace_gc.versions import sha256_named

    cfg = load_config(args.source_run / "config.yaml")
    actor = load_lora_actor(cfg["model_path"], cfg["lora"])
    for module in actor.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.
        if isinstance(getattr(module, "attention_dropout", None), (float, int)):
            module.attention_dropout = 0.
    named = named_lora_params(actor)
    for _, parameter in named:
        parameter.data = parameter.data.float()
    actor._grace_compute_dtype = "bfloat16"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    initialize_actor(actor, cfg, RunDirectory(args.output.parent))
    initial_sha = sha256_named(named)
    rows = []
    with (args.source_run / "trajectories.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("truncated"):
                rows.append(row)
            if len(rows) == 2:
                break
    if len(rows) != 2:
        raise ValueError("two truncated source trajectories are required")
    pad_id = int(actor.config.pad_token_id or 0)

    def measure(action):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        tick = time.perf_counter()
        value = action()
        torch.cuda.synchronize()
        return value, {"seconds": time.perf_counter() - tick,
                       "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                       "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}

    report = {"scope": "Execution profile only; 16384 fixtures repeat real token IDs, not new model rollouts or scientific results.",
              "source_run": str(args.source_run), "profiles": {}, "actor_sha256": initial_sha}
    for length in (8192, 16384):
        fixtures = []
        for row in rows:
            plen = row["prompt_len"]
            response = row["token_ids"][plen:]
            fixtures.append((row["token_ids"][:plen] + (response * 2)[:length], plen, plen + 512))
        profiles = {}
        reference = None
        for label, stride, fused in (("separate_all_checkpointed", 1, False),
                                     ("fused_all_checkpointed", 1, True),
                                     ("fused_half_checkpointed", 2, True)):
            configure_checkpoint_stride(actor, stride)

            def evaluate():
                values = []
                for tokens, start, split in fixtures:
                    if fused:
                        prefix_lp, suffix_lp = split_logprob(actor, tokens, start, split, pad_id)
                        score = float(suffix_lp.detach().cpu())
                        gradient = gradient_vector(prefix_lp + suffix_lp, named)
                    else:
                        gradient = gradient_vector(range_logprob(actor, tokens, start, pad_id), named)
                        with torch.no_grad():
                            score = float(range_logprob(actor, tokens, split, pad_id).cpu())
                    values.append((gradient, score))
                return values

            try:
                values, metrics = measure(evaluate)
            except torch.cuda.OutOfMemoryError:
                profiles[label] = {"status": "oom"}
                gc.collect()
                torch.cuda.empty_cache()
                continue
            if reference is None:
                reference = values
            metrics["max_gradient_relative_l2"] = max(float(np.linalg.norm(a[0] - b[0]) / max(np.linalg.norm(b[0]), 1e-12))
                                                       for a, b in zip(values, reference))
            metrics["max_suffix_logprob_abs_error"] = max(abs(a[1] - b[1]) for a, b in zip(values, reference))
            if metrics["max_gradient_relative_l2"] > .03 or not all(np.isfinite(v[0]).all() for v in values):
                raise AssertionError(f"invalid optimized gradient: {metrics}")
            profiles[label] = metrics
            print(json.dumps({"response_tokens": length, "profile": label, **metrics}), flush=True)
        sequences = [fixture[0] for fixture in fixtures]
        starts = [fixture[2] for fixture in fixtures]
        with torch.no_grad():
            serial, serial_metrics = measure(lambda: torch.stack([range_logprob(actor, tokens, start, pad_id)
                                                                  for tokens, start in zip(sequences, starts)]).cpu())
            batched, batched_metrics = measure(lambda: range_logprob_batch(actor, sequences, starts, pad_id).cpu())
        profiles["posterior_serial"] = serial_metrics
        profiles["posterior_batch_2"] = batched_metrics
        profiles["posterior_batch_2"]["max_logprob_abs_error"] = float((serial - batched).abs().max())
        report["profiles"][str(length)] = profiles
    if sha256_named(named) != initial_sha:
        raise AssertionError("profiling changed the actor")
    report["actor_unchanged"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(str(args.output), flush=True)


if __name__ == "__main__":
    main()
