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

from grace_gc.audit.benefit_replay import audit_rows, audit_file, continuation_digest, replay_config
from grace_gc.audit.expected_gain import load_replay, reference_gradient, start_experiment_run
from grace_gc.audit.update_probe import adamw_update_direction, adamw_update_vector


def _key(row):
    return (str(row["problem_id"]), str(row["path_id"]), int(row["t"]))


def _load_sources(paths, decision_tokens=None):
    source = {}
    for path in paths:
        for row in audit_rows(path, decision_tokens):
            key = _key(row)
            if key in source:
                raise ValueError(f"duplicate audit prefix across inputs: {key}")
            source[key] = row
    return source


def _checked_gradient(gradient, bundle, index, replay_row=None):
    record = bundle["continuation_records"][int(index)]
    digest = None if replay_row is None else (replay_row.get("continuation_sha256") or {}).get(str(index))
    if digest and continuation_digest(record) != digest:
        raise ValueError("source continuation differs from the trajectory used for replay labels")
    g = gradient(record["token_ids"], int(record["prompt_len"]),
                 float(record["reward"]), float(record["baseline"]))
    norms = bundle.get("true_grad_norm_sq")
    if norms is not None and not np.isclose(float(g @ g), float(norms[int(index)]), rtol=5e-3, atol=1e-5):
        raise ValueError("optimizer probe failed the source trajectory gradient-norm check")
    return g


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundles", required=True, action="append",
                        help="original predictor audit_bundles.jsonl with token IDs")
    parser.add_argument("--background-bundles", required=True, action="append",
                        help="training-side t=0 audit, independent of both reference pools")
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
    parser.add_argument("--decision-tokens", type=int, default=512,
                        help="predictor decision token used by the fixed-N source audit")
    parser.add_argument("--grad-clip", type=float, default=None,
                        help="explicit override; otherwise use checkpoint run_config.optim.grad_clip")
    parser.add_argument("--export-direction", default=None,
                        help="optional .npy path for a counterfactual AdamW parameter delta")
    parser.add_argument("--export-background", type=int, default=0,
                        help="background index used with --export-direction (default: 0)")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)
    if min(args.n_start, args.n_prefixes, args.suffixes_per_prefix,
           args.backgrounds) <= 0:
        raise ValueError("counts must be positive")
    if args.export_background < 0:
        raise ValueError("export-background must be non-negative")
    from functools import partial
    from types import SimpleNamespace

    from grace_gc.audit.run import _policy_grad_vec
    from grace_gc.backends.hf_actor import logprob_one, named_lora_params, trainable_params
    from grace_gc.backends.verl_trainer import load_lora_actor
    from grace_gc.core.layout import collect_lora_layout
    from grace_gc.data.tokenize import load_hf_tokenizer
    from grace_gc.trainer.checkpoint import load_checkpoint
    from grace_gc.trainer.state_io import check_snapshot_identity, load_numpy_module_state
    from grace_gc.versions import sha256_array, sha256_file, sha256_named

    payload = load_checkpoint(args.checkpoint)
    cfg = replay_config(payload, args.model_path, args.config)
    stored_clip = ((payload.get("run_config") or {}).get("optim") or {}).get("grad_clip")
    if args.grad_clip is None and stored_clip is None:
        raise ValueError("checkpoint lacks grad_clip; supply the original training value with --grad-clip")
    clip = float(stored_clip if args.grad_clip is None else args.grad_clip)
    if not np.isfinite(clip):
        raise ValueError("grad-clip must be finite")
    out = start_experiment_run(args.run_dir, "expected_gain_update_probe", vars(args))
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
    for path in args.bundles + args.background_bundles:
        source_meta = audit_file(path).parent / "actor_source.json"
        if source_meta.is_file():
            source_sha = json.loads(source_meta.read_text(encoding="utf-8")).get("actor_sha256")
            if source_sha and source_sha != actor_sha:
                raise ValueError(f"audit actor differs from optimizer checkpoint: {path}")
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
    train_ids, validation_ids = set(map(str, split["train"])), set(map(str, split["validation"]))
    if train_ids & validation_ids:
        raise ValueError("training and validation problem IDs overlap")
    if {str(row["problem_id"]) for row in ref_rows} & (train_ids | validation_ids):
        raise ValueError("reference problems overlap the optimizer training/validation pool")
    train = [row for row in metadata if row["problem_id"] in train_ids]
    validation = [row for row in metadata if row["problem_id"] in validation_ids]
    if not train or not validation:
        raise ValueError("split needs live training and validation prefixes")
    source = _load_sources(args.bundles, args.decision_tokens)
    if any(_key(row) not in source for row in validation):
        raise ValueError("audit bundles and replay prefixes differ")
    background_source = _load_sources(args.background_bundles, 0)
    background_pool = [row for row in background_source.values() if str(row["problem_id"]) in train_ids]
    if not background_pool:
        raise ValueError("background audit has no unconditional t=0 trajectories on training problems")
    background_sources = []
    for path in args.background_bundles:
        source_file = audit_file(path)
        background_sources.append({"path": str(source_file.resolve()),
                                   "sha256": sha256_file(source_file)})
    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(len(validation), size=min(args.n_prefixes, len(validation)), replace=False)
    chosen_rows = [validation[int(i)] for i in chosen]
    # Backgrounds are independent training-side continuations, normalized as
    # the other N-1 starts of the same fixed training step.
    backgrounds, background_traces = [], []
    for background_index in range(args.backgrounds):
        indices = rng.choice(len(background_pool), size=args.n_start - 1, replace=True)
        total = np.zeros(layout.dim, dtype=np.float64)
        trace = []
        for index in indices:
            bundle = background_pool[int(index)]
            suffix_index = int(rng.integers(len(bundle["continuation_records"])))
            total += _checked_gradient(gradient, bundle, suffix_index)
            record = bundle["continuation_records"][suffix_index]
            trace.append({"prefix": _key(bundle), "suffix_index": suffix_index,
                          "continuation_sha256": continuation_digest(record)})
        backgrounds.append(total / args.n_start)
        background_traces.append(trace)
    baseline = [adamw_update_direction(payload["actor"], layout, payload["optimizer"],
                                       cfg.get("optim") or {}, b, reference, clip)
                for b in backgrounds]
    exported_direction = None
    if args.export_direction:
        if args.export_background >= len(backgrounds):
            raise ValueError("export-background is outside the sampled background range")
        delta, direction_stats = adamw_update_vector(
            payload["actor"], layout, payload["optimizer"], cfg.get("optim") or {},
            backgrounds[args.export_background], clip)
        # AdamW receives the negative ascent gradient, so its parameter delta
        # is already aligned with the ascent convention used by replay labels.
        direction = np.asarray(delta, dtype=np.float64)
        direction_path = Path(args.export_direction)
        if direction_path.suffix.lower() != ".npy":
            raise ValueError("export-direction must end in .npy")
        direction_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(direction_path, direction, allow_pickle=False)
        exported_direction = {
            "path": str(direction_path.resolve()),
            "direction_kind": "counterfactual_adamw_parameter_delta",
            "convention": "parameter_delta_from_negative_ascent_gradient",
            "actor_sha256": actor_sha,
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "checkpoint_step": payload.get("step"),
            "model_path": args.model_path,
            "layout_names": layout.names(),
            "layout_dim": layout.dim,
            "optim": cfg.get("optim") or {},
            "n_start": int(args.n_start),
            "seed": int(args.seed),
            "background_index": int(args.export_background),
            "background_sources": background_sources,
            "background_draws": background_traces,
            "background_gradient_sha256": sha256_array(backgrounds[args.export_background]),
            "all_background_gradient_sha256": [sha256_array(value) for value in backgrounds],
            "shape": list(direction.shape),
            "norm": float(np.linalg.norm(direction)),
            "direction_sha256": sha256_array(direction),
            "parameter_delta_norm": float(direction_stats["parameter_update_norm"]),
            "clip_triggered": bool(direction_stats["clip_triggered"]),
        }
        sidecars = [out / "exported_direction.json", direction_path.with_suffix(".json")]
        for sidecar in dict.fromkeys(path.resolve() for path in sidecars):
            sidecar.write_text(json.dumps(exported_direction, indent=2), encoding="utf-8")
    report = []
    for row in chosen_rows:
        records = source[_key(row)]["continuation_records"]
        indices = row["continuation_indices"][:args.suffixes_per_prefix]
        for suffix_index in indices:
            g = _checked_gradient(gradient, source[_key(row)], int(suffix_index), row)
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
    (out / "optimizer_probe.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in report), encoding="utf-8")
    summary = {"n_prefixes": len(chosen_rows), "n_rows": len(report),
               "n_start": args.n_start, "backgrounds": len(backgrounds),
               "clip": clip, "actor_sha256": actor_sha,
               "optimizer_execution": "CPU PyTorch using saved groups/moments and shared training clip path; not bitwise CUDA validation",
               "background_pool": "training-only t=0 full trajectories, sampled with replacement",
               "background_sources": background_sources,
               "label_scope": "fixed-actor counterfactual AdamW direction proxy; not historical update reconstruction or observed reward change",
               "exported_direction": exported_direction}
    (out / "optimizer_probe_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({**summary, "run_dir": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
