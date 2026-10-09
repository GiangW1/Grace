#!/usr/bin/env python3
"""Frozen thinking audit with separate rollout/gradient GPUs and resumable moments."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from contextlib import contextmanager
try:
    import fcntl
except ImportError:  # pragma: no cover - exercised on Windows workers
    fcntl = None
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import urllib.request

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from grace_gc.audit.dynamic_score import load_diagonal_metric
from grace_gc.audit.prefix_audit import bundle_from_dict, bundle_to_dict
from grace_gc.audit.run import _bundles_from_engines
from grace_gc.audit.qualification import FORCED_ANSWER_PREFIX
from grace_gc.core.rng import IsolatedRNG
from grace_gc.data.format_prompt import apply_solve_instruction
from grace_gc.data.math_data import (
    load_math_records,
    select_records_difficulty,
    selection_manifest,
)
from grace_gc.data.tokenize import encode_records_hf, load_hf_tokenizer
from grace_gc.data.reward import REWARD_PROTOCOL_VERSION
from grace_gc.trainer.checkpoint import load_checkpoint, save_checkpoint
from grace_gc.trainer.state_io import load_numpy_module_state, numpy_module_state
from grace_gc.versions import sha256_array, sha256_file, sha256_named

REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
LORA = {"rank": 16, "alpha": 32, "targets": ["q_proj", "v_proj"],
        "dropout": 0.0, "compute_dtype": "bfloat16", "gradient_checkpointing": True}


@contextmanager
def problem_lock(path):
    """Use an advisory lock on POSIX and a one-byte msvcrt lock on Windows."""
    with Path(path).open("a+", encoding="utf-8") as handle:
        if fcntl is not None:
            fcntl.flock(handle, fcntl.LOCK_EX)
        else:  # Windows has no fcntl, but workers still need an atomic guard.
            import msvcrt
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write("0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        try:
            yield handle
        finally:
            if fcntl is not None:
                fcntl.flock(handle, fcntl.LOCK_UN)
            else:
                import msvcrt
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save_array(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    temporary.replace(path)


def utc():
    return datetime.now(timezone.utc).isoformat()


def problem_directory(root, pid):
    return root / "problems" / hashlib.sha256(str(pid).encode()).hexdigest()[:24]


def prefix_directory(directory, path_id, t):
    return directory / f"path-{str(path_id).rsplit(':', 1)[1]}-t{t}"


class RemoteLLM:
    def __init__(self, url, model, cache, request_concurrency=8):
        self.url = url.rstrip("/") + "/v1/completions"
        self.model = model
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.request_concurrency = int(request_concurrency)
        if self.request_concurrency < 1:
            raise ValueError("request concurrency must be positive")

    def generate(self, prompts, sampling_params, **_kwargs):
        params = (list(sampling_params) if isinstance(sampling_params, (list, tuple))
                  else [sampling_params] * len(prompts))

        def one(item):
            prompt, param = item
            if isinstance(prompt, dict):
                prompt = prompt.get("prompt_token_ids")
            prompt = list(map(int, prompt))
            payload = {"model": self.model, "prompt": prompt,
                       "max_tokens": int(param.max_tokens), "temperature": float(param.temperature),
                       "top_p": float(param.top_p), "top_k": int(param.top_k),
                       "min_p": float(param.min_p), "repetition_penalty": float(param.repetition_penalty),
                       "seed": int(param.seed), "n": 1, "stop": [],
                       "stop_token_ids": list(map(int, param.stop_token_ids or [])),
                       "return_token_ids": True, "add_special_tokens": False}
            encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
            cached = self.cache / (hashlib.sha256(encoded).hexdigest() + ".json")
            if cached.exists():
                body = json.loads(cached.read_text())
            else:
                request = urllib.request.Request(self.url, data=encoded,
                                                 headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(request, timeout=3600) as response:
                    body = json.loads(response.read())
                choices = body.get("choices") or []
                if not choices or choices[0].get("token_ids") is None:
                    raise RuntimeError("vLLM response is missing token IDs")
                if choices[0].get("prompt_token_ids") != prompt:
                    raise RuntimeError("vLLM prompt token IDs differ")
                write_json(cached, body)
            choice = body["choices"][0]
            if choice.get("prompt_token_ids") != prompt:
                raise RuntimeError("cached vLLM prompt token IDs differ")
            return SimpleNamespace(prompt_token_ids=prompt, num_cached_tokens=None,
                                   outputs=[SimpleNamespace(token_ids=choice["token_ids"],
                                   finish_reason=choice.get("finish_reason"),
                                   stop_reason=choice.get("stop_reason"), logprobs=None)])

        with ThreadPoolExecutor(max_workers=min(self.request_concurrency, max(len(prompts), 1))) as pool:
            return list(pool.map(one, zip(prompts, params)))


def reduce_bundle(bundle, rng, directory, seed, learning_completion=False, provenance=None,
                  metric=None, metric_name="euclidean"):
    target = prefix_directory(directory, bundle.path_id, bundle.t)
    target.mkdir(parents=True, exist_ok=True)
    gradients = np.asarray(bundle.trajectory_grads)
    weights = (np.ones(gradients.shape[1], dtype=np.float64) if metric is None else
               np.asarray(metric, dtype=np.float64).reshape(-1))
    if (weights.shape != (gradients.shape[1],) or
            not np.all(np.isfinite(weights)) or np.any(weights < 0.0) or
            not np.any(weights > 0.0)):
        raise ValueError("metric weights must be finite, nonnegative, and match trajectory dimension")
    order_seed = int.from_bytes(hashlib.sha256(
        f"{seed}:{bundle.problem_id}:{bundle.path_id}:{bundle.t}".encode()).digest()[:4], "little")
    order = np.random.default_rng(order_seed).permutation(len(gradients))
    halves = (order[:len(order) // 2], order[len(order) // 2:])
    if any(len(part) == 0 for part in halves):
        raise ValueError("A/B continuation statistics require at least two samples")
    raw = bundle_to_dict(bundle)
    for half, indices in zip(("a", "b"), halves):
        mean = np.zeros(gradients.shape[1], dtype=np.float64)
        for index in indices:
            mean += gradients[index] / len(indices)
        if learning_completion:
            save_array(target / f"half_mean_full_grads_{half}.npy", mean[None, :])
        else:
            save_array(target / f"mean_{half}.npy", mean)
    if learning_completion:
        save_array(target / "prefix_score_gradients.npy", np.asarray(bundle.prefix_score_grad)[None, :, None])
    else:
        save_array(target / "prefix_score.npy", bundle.prefix_score_grad)
    half_metric_norm_sq = [
        float(np.mean(np.sum(gradients[indices] * gradients[indices] * weights[None, :], axis=1)))
        for indices in halves
    ]
    raw.update(enable_thinking=True,
               require_complete_answers=True,
               normal_q_mean_reward=float(np.mean(bundle.rewards[halves[0]])),
               half_counts=list(map(len, halves)),
               half_indices=[part.tolist() for part in halves],
               half_mean_reward=[float(np.mean(bundle.rewards[part])) for part in halves],
               half_mean_advantage=[float(np.mean(bundle.rewards[part]) - bundle.baseline) for part in halves],
               half_mean_suffix_cost=[float(np.mean(bundle.suffix_cost[part])) for part in halves],
               half_mean_full_norm_sq=[float(np.mean(np.asarray(bundle.true_grad_norm_sq)[part]))
                                       for part in halves],
               half_full_metric_norm_sq=half_metric_norm_sq,
               metric_name=str(metric_name),
               metric_weights_sha256=sha256_array(weights),
               reward_protocol_version=REWARD_PROTOCOL_VERSION,
               n_continuations=len(gradients), rng_after=rng.state_dict(),
               sufficient_statistics="Full-space FP64 A/B means, second moments and prefix score; JL only for display")
    if learning_completion:
        row = {key: raw.get(key) for key in (
            "problem_id", "path_id", "t", "half_counts", "half_mean_reward", "baseline",
            "answer_emitted", "finished", "functional_recoverable", "reward_protocol_version",
            "require_complete_answers")}
        prefixes = target / "prefixes.jsonl"
        temporary = prefixes.with_suffix(".jsonl.tmp")
        temporary.write_text(json.dumps(row, allow_nan=False) + "\n")
        temporary.replace(prefixes)
        write_json(target / "score_gradient_provenance.json", {
            **(provenance or {}), "replay_prefixes_sha256": sha256_file(prefixes),
            "gradient_label": "full_trajectory", "storage": "one prefix; independent A/B full-space means"})
    # Publishing metadata last makes each completed prefix the resume unit.
    write_json(target / "bundle.json", raw)
    bundle.trajectory_grads = None
    bundle.prefix_score_grad = None


def validate_metric_record(record, metric_hash, euclidean, *, prefix=False, allow_legacy=False):
    recorded_hash = record.get("metric_weights_sha256")
    if recorded_hash is None:
        name = record.get("metric_name", record.get("metric"))
        if not (allow_legacy and euclidean and name in (None, "euclidean")):
            raise ValueError("saved statistics lack metric provenance; use a new run directory")
    elif recorded_hash != metric_hash:
        raise ValueError("saved metric weights differ; use a new run directory")
    if prefix and not euclidean:
        norms = np.asarray(record.get("half_full_metric_norm_sq"), dtype=np.float64)
        if norms.shape != (2,) or not np.all(np.isfinite(norms)) or np.any(norms < 0):
            raise ValueError("prefix lacks valid A/B second moments for the saved metric")


def prepare_metric(root, metric, *, allow_legacy=False):
    metric_path = root / "metric_weights.npy"
    metric_hash, euclidean = sha256_array(metric), bool(np.all(metric == 1.0))
    # Workers share this file; validate before any worker can replace it.
    with problem_lock(root / ".metric.lock"):
        if metric_path.is_file():
            saved = load_diagonal_metric(metric_path, len(metric))
            if sha256_array(saved) != metric_hash:
                raise ValueError("saved metric weights differ; use a new run directory")
        for path in root.glob("worker-*-provenance.json"):
            validate_metric_record(json.loads(path.read_text()), metric_hash, euclidean,
                                   allow_legacy=allow_legacy)
        for path in root.glob("problems/*/path-*/bundle.json"):
            validate_metric_record(json.loads(path.read_text()), metric_hash, euclidean,
                                   prefix=True, allow_legacy=allow_legacy)
        if not metric_path.is_file():
            save_array(metric_path, metric)


def prepare_collection_scope(root, n_continuations, *, smoke=False, n_prefixes=2,
                             n_baseline=16, max_new_tokens=8192, decision_grid=None,
                             learning_completion=False):
    scope = {"n_prefixes": 1 if smoke else int(n_prefixes), "n_continuations": int(n_continuations),
             "n_baseline": 2 if smoke else int(n_baseline), "max_new_tokens": int(max_new_tokens),
             "decision_grid": [64, 128] if smoke else list(
                 [1024, 2048] if decision_grid is None else decision_grid)}
    path = root / "collection_scope.json"
    with problem_lock(root / ".collection-scope.lock"):
        if path.is_file():
            if json.loads(path.read_text()) != scope:
                raise ValueError("saved collection parameters differ; use a new run directory")
        else:
            for saved in root.glob("worker-*-provenance.json"):
                prior = json.loads(saved.read_text())
                count = prior.get("n_continuations", 2 if smoke else (8 if learning_completion else 64))
                if count != n_continuations:
                    raise ValueError("saved continuation count differs; use a new run directory")
                if any(key in prior and prior[key] != value for key, value in scope.items()):
                    raise ValueError("saved collection parameters differ; use a new run directory")
            for saved in root.glob("problems/*/path-*/bundle.json"):
                if json.loads(saved.read_text()).get("n_continuations") != n_continuations:
                    raise ValueError("saved continuation count differs; use a new run directory")
            write_json(path, scope)
    return scope


def collect(args):
    import torch
    from grace_gc.backends.gpu_engine import make_gpu_engines
    from grace_gc.backends.hf_actor import actor_numerics, named_lora_params
    from grace_gc.backends.verl_trainer import load_lora_actor
    from grace_gc.core.layout import collect_lora_layout, layout_hash
    from grace_gc.audit.qualification import classify_functional_recovery, score_probe_sample
    from grace_gc.trainer.methods import method_spec

    root = Path(args.run_dir)
    root.mkdir(parents=True, exist_ok=True)
    for path in (*root.glob("worker-*-provenance.json"), *root.glob("problems/*/summary.json")):
        previous = json.loads(path.read_text())
        if (previous.get("reward_protocol_version") != REWARD_PROTOCOL_VERSION
                or not previous.get("require_complete_answers", previous.get("enable_thinking", False))):
            raise ValueError("run uses a different reward protocol; use a new run directory")
    collection_scope = prepare_collection_scope(
        root, 2 if args.smoke else args.n_continuations, smoke=args.smoke,
        n_prefixes=args.n_prefixes, n_baseline=args.n_baseline,
        max_new_tokens=args.max_new_tokens, decision_grid=args.decision_tokens,
        learning_completion=args.learning_completion)
    records = []
    for path in sorted(Path(args.data_dir).glob("shard-*.jsonl")):
        records.extend(load_math_records(path))
    records = apply_solve_instruction(records)
    difficulty_manifest_path = None
    difficulty_manifest_hash = None
    requested_difficulty_counts = None
    actual_difficulty_counts = None
    difficulty_mapping = None
    if args.difficulty_manifest:
        counts = (None if args.difficulty_counts is None
                  else json.loads(args.difficulty_counts))
        if counts is not None and not isinstance(counts, dict):
            raise ValueError("--difficulty-counts must be a JSON object")
        difficulty_manifest_path = str(Path(args.difficulty_manifest).resolve())
        difficulty_manifest_hash = sha256_file(args.difficulty_manifest)
        requested_difficulty_counts = None if counts is None else {
            label: int(counts[label]) for label in ("easy", "medium", "hard")
        }
        records = select_records_difficulty(
            records, args.n_problems, args.difficulty_manifest, counts, seed=args.seed
        )
        payload = json.loads(Path(args.difficulty_manifest).read_text(encoding="utf-8"))
        difficulty_mapping = payload.get("difficulty", payload) if isinstance(payload, dict) else None
        if not isinstance(difficulty_mapping, dict):
            raise ValueError("difficulty manifest must map problem IDs to easy/medium/hard")
        actual_difficulty_counts = {
            label: sum(1 for record in records
                       if str(difficulty_mapping.get(str(record.problem_id), "")).lower() == label)
            for label in ("easy", "medium", "hard")
        }
        selection = "difficulty"
    else:
        if len(records) != args.n_problems or len({rec.problem_id for rec in records}) != args.n_problems:
            raise ValueError(f"expected {args.n_problems} distinct problems")
        selection = "provided"
    selected_records = list(records)
    if args.smoke:
        records = selected_records[:1]
    else:
        records = selected_records[args.worker_index::args.workers]
    # Record the global selection before worker sharding; smoke mode is the
    # only intentional exception and should advertise its one-problem scope.
    if args.smoke or args.worker_index == 0:
        manifest_records = records if args.smoke else selected_records
        manifest_difficulty_counts = (None if difficulty_mapping is None else {
            label: sum(1 for record in manifest_records
                       if str(difficulty_mapping.get(str(record.problem_id), "")).lower() == label)
            for label in ("easy", "medium", "hard")
        })
        selection_payload = selection_manifest(manifest_records, selection, args.seed)
        selection_payload.update(
            difficulty_manifest=difficulty_manifest_path,
            difficulty_manifest_sha256=difficulty_manifest_hash,
            difficulty_counts=requested_difficulty_counts,
            actual_difficulty_counts=manifest_difficulty_counts,
            reward_protocol_version=REWARD_PROTOCOL_VERSION,
        )
        write_json(root / "selection.json", selection_payload)
    write_json(root / f"worker-{args.worker_index}-status.json", {
        "status": "loading_actor", "started_utc": utc(), "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "problem_ids": [rec.problem_id for rec in records]})
    torch.manual_seed(args.seed)
    actor = load_lora_actor(args.model_path, LORA)
    if not args.gradient_checkpointing:
        actor.gradient_checkpointing_disable()
        actor._grace_gradient_checkpointing = False
    named = named_lora_params(actor)
    layout = collect_lora_layout(named)
    metric_file = None if args.metric_file is None else Path(args.metric_file)
    if metric_file is None:
        metric = np.ones(layout.dim, dtype=np.float64)
        metric_name = str(args.metric_name or "euclidean")
    else:
        metric = load_diagonal_metric(metric_file, layout.dim)
        sidecar = metric_file.with_suffix(".json")
        sidecar_name = None
        if sidecar.is_file():
            sidecar_payload = json.loads(sidecar.read_text(encoding="utf-8"))
            sidecar_name = sidecar_payload.get("metric_name")
        metric_name = str(args.metric_name or sidecar_name or "diagonal_file")
    checkpoint = Path(args.checkpoint)
    if checkpoint.exists():
        payload = load_checkpoint(checkpoint)
        if payload["model_revision"] != REVISION or payload["lora"] != LORA:
            raise ValueError("frozen actor snapshot differs from this thinking test")
        load_numpy_module_state(named, payload["actor"])
    else:
        if not args.smoke:
            raise FileNotFoundError("initialize the new frozen LoRA in the smoke test first")
        if any(torch.count_nonzero(param).item() for name, param in named if "lora_B" in name):
            raise ValueError("the initial LoRA must leave the thinking model policy unchanged")
        save_checkpoint(checkpoint, {"actor": numpy_module_state(named), "lora": LORA,
                                    "model_revision": REVISION, "spec": "full_pg", "step": 0,
                                    "layout_dim": layout.dim, "layout_names": layout.names(),
                                    "origin": "Fresh zero-B LoRA on post-trained Qwen3-4B; no optimizer history"})
    prepare_metric(root, metric, allow_legacy=args.learning_completion)
    provenance = {"model": "Qwen/Qwen3-4B", "model_revision": REVISION,
                  "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                  "audit_source_sha256": sha256_file(REPO / "grace_gc/audit/run.py"),
                  "runner_sha256": sha256_file(__file__), "actor_sha256": sha256_named(named),
                  "checkpoint_sha256": sha256_file(checkpoint), "checkpoint": str(checkpoint),
                  "layout_dim": layout.dim, "layout_names": layout.names(), "layout_sha256": layout_hash(layout),
                  "actor_numerics": actor_numerics(actor), "enable_thinking": True,
                  "temperature": 1.0, "metric": metric_name,
                  "request_concurrency": args.request_concurrency,
                  "batch_continuations": args.batch_continuations,
                  "feature_batch_size": args.feature_batch_size,
                  **collection_scope,
                  "metric_file": (None if metric_file is None else str(metric_file.resolve())),
                  "metric_weights_sha256": sha256_array(metric),
                  "adam_metric": ("not_requested" if metric_name == "euclidean"
                                  else "fixed_diagonal_metric"),
                  "difficulty_manifest": difficulty_manifest_path,
                  "difficulty_manifest_sha256": difficulty_manifest_hash,
                  "difficulty_counts": requested_difficulty_counts,
                  "actual_difficulty_counts": actual_difficulty_counts,
                  "reward_protocol_version": REWARD_PROTOCOL_VERSION,
                  "require_complete_answers": True}
    write_json(root / f"worker-{args.worker_index}-provenance.json", provenance)
    tokenizer = load_hf_tokenizer(args.model_path)
    cfg = {"temperature": 1.0, "predictor": {
        "feature_batch_size": args.feature_batch_size, "feature_mode": "legacy"}}
    remote = RemoteLLM(args.server_url, args.server_model, root / "rollout-cache",
                       request_concurrency=args.request_concurrency)
    engines, _ = make_gpu_engines(actor, remote, tokenizer, cfg, root / "unused-adapter")

    def repeated(prefix, count, max_new, rng):
        fulls = engines.continue_selected([list(prefix) for _ in range(count)], np.ones(count, dtype=bool), max_new, rng)
        rollout = engines.last_rollout
        reasons = rollout["continue_finish_reasons"]
        return fulls, [reasons.get(i) for i in range(count)], rollout["sampling"]["request_seeds"]

    engines.continue_repeated = repeated

    def encode(rec, count):
        meta = []
        result = encode_records_hf([rec] * count, tokenizer, 1024, prompt_meta=meta, enable_thinking=True)
        encode.last_meta = meta
        text = tokenizer.decode(result[0][0], skip_special_tokens=False)
        assistant_tail = text.rsplit("<|im_start|>assistant\n", 1)[-1]
        if "</think>" in assistant_tail or not all(item["enable_thinking"] for item in meta):
            raise ValueError("thinking prompt contains a preclosed think section")
        if args.smoke:
            write_json(root / "prompt_validation.json", {
                "enable_thinking": True, "assistant_tail": assistant_tail,
                "prompt_text": text, "prompt_meta": meta[0],
                "template_behavior": "Thinking tokens are generated by Qwen3, not prefilled"})
        return result

    forced = tokenizer.encode(FORCED_ANSWER_PREFIX, add_special_tokens=False)
    probe_records = {}

    def qualify(prefix, prompt_len, gold, pid, t, path_index):
        probe = list(prefix) + forced
        rng = IsolatedRNG.create(args.seed + path_index * 100003 + int(t) * 1009 + len(prefix))
        fulls, reasons, seeds = repeated(probe, 4, 32, rng)
        texts = [engines.decode(full[len(probe):]) for full in fulls]
        probe_records[(pid, f"{pid}:{path_index}", t)] = [
            {**score_probe_sample(full, len(probe), text, gold, 32,
                                  eos_id=engines.eos_id, finish_reason=reason),
             "seed": seed, "scored_text": "Answer:" + text}
            for text, seed, reason, full in zip(texts, seeds, reasons, fulls)]
        rewards = [row["reward"] for row in probe_records[(pid, f"{pid}:{path_index}", t)]]
        return classify_functional_recovery(rewards, 0.5)

    started = time.monotonic()
    for problem_index, rec in enumerate(records):
        directory = problem_directory(root, rec.problem_id)
        directory.mkdir(parents=True, exist_ok=True)
        with problem_lock(directory / ".lock"):
            if (directory / "summary.json").exists():
                continue
            seed = (args.seed + int.from_bytes(hashlib.sha256(rec.problem_id.encode()).digest()[:4], "little")) % (2**32)

            def sink(bundle, rng):
                key = (bundle.problem_id, bundle.path_id, bundle.t)
                target = prefix_directory(directory, bundle.path_id, bundle.t)
                if key in probe_records:
                    write_json(target / "functional_probes.json", probe_records[key])
                reduce_bundle(bundle, rng, directory, args.seed,
                              learning_completion=args.learning_completion, provenance=provenance,
                              metric=metric, metric_name=metric_name)
                write_json(root / f"worker-{args.worker_index}-status.json", {
                    "status": "collecting", "updated_utc": utc(), "problem_id": rec.problem_id,
                    "problem_index": problem_index, "n_assigned_problems": len(records),
                    "last_completed_path": bundle.path_id, "last_completed_t": bundle.t,
                    "elapsed_seconds": time.monotonic() - started})
                torch.cuda.empty_cache()

            def resume(pid, path_id, t, rng):
                path = prefix_directory(directory, path_id, t) / "bundle.json"
                if not path.exists():
                    return None
                raw = json.loads(path.read_text())
                if (raw.get("reward_protocol_version") != REWARD_PROTOCOL_VERSION
                        or not raw.get("require_complete_answers", raw.get("enable_thinking", False))):
                    raise ValueError("prefix uses a different reward protocol; use a new run directory")
                rng.load_state_dict(raw["rng_after"])
                print(f"phase=resume_prefix pid={pid} path={path_id} t={t}", flush=True)
                return bundle_from_dict(raw)

            bundles = _bundles_from_engines(
                [rec], engines, layout, collection_scope["n_prefixes"],
                collection_scope["n_continuations"],
                collection_scope["decision_grid"], collection_scope["max_new_tokens"], seed, encode,
                spec=method_spec("full_pg"), jl_dim=256, jl_seed=args.seed,
                n_baseline=2 if args.smoke else args.n_baseline, store_features=not args.learning_completion,
                store_trajectory_gradients=True, qualification_fn=qualify,
                bundle_sink=sink, bundle_resume=resume, require_complete_answers=True,
                batch_continuations=args.batch_continuations)
            write_json(directory / "raw_problem.json", _bundles_from_engines.last[0])
            write_json(directory / "summary.json", {"status": "completed", "problem_id": rec.problem_id,
                       "n_bundles": len(bundles), "seed": seed, "finished_utc": utc(),
                       "enable_thinking": True, "reward_protocol_version": REWARD_PROTOCOL_VERSION,
                       "require_complete_answers": True})
            print(f"phase=problem_completed worker={args.worker_index} problem={problem_index+1}/{len(records)}", flush=True)
    write_json(root / f"worker-{args.worker_index}-status.json", {
        "status": "completed", "finished_utc": utc(), "n_problems": len(records),
        "elapsed_seconds": time.monotonic() - started})


def merge(args):
    from grace_gc.audit.benefit_replay import prefix_keys_sha256

    root = Path(args.run_dir)
    directories = sorted((root / "problems").glob("*/path-*/bundle.json"))
    summaries = list((root / "problems").glob("*/summary.json"))
    selection_path = root / "selection.json"
    allow_legacy = bool(getattr(args, "allow_legacy", False))
    if selection_path.is_file():
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        expected_problems = int(selection["n_problems"])
        if (selection.get("reward_protocol_version") != REWARD_PROTOCOL_VERSION
                and not allow_legacy):
            raise ValueError(
                f"selection manifest is not reward protocol v{REWARD_PROTOCOL_VERSION}; rerun collection "
                "or pass --allow-legacy for explicitly unverified diagnostics"
            )
        expected_problem_ids = {str(row["problem_id"])
                                for row in selection.get("problems", [])}
        if len(expected_problem_ids) != expected_problems:
            raise ValueError("selection manifest has duplicate or missing problem IDs")
    else:
        # PR13 runs predate the frozen selection manifest; preserve their
        # analysis path only when the caller explicitly opts into legacy data.
        if not allow_legacy:
            raise ValueError(
                "run lacks a frozen selection/reward manifest; rerun collection "
                "before analysis or pass --allow-legacy for diagnostics"
            )
        expected_problems = len(summaries)
        selection = {"selection": "legacy_unrecorded", "n_problems": expected_problems}
        expected_problem_ids = None
    if len(summaries) != expected_problems:
        raise ValueError(f"only {len(summaries)} of {expected_problems} selected problems are complete")
    summary_payloads = [json.loads(path.read_text(encoding="utf-8")) for path in summaries]
    if expected_problem_ids is not None:
        summary_problem_ids = {str(row.get("problem_id")) for row in summary_payloads}
        if summary_problem_ids != expected_problem_ids:
            raise ValueError(
                "completed problem IDs differ from selection manifest: "
                f"missing={sorted(expected_problem_ids - summary_problem_ids)} "
                f"extra={sorted(summary_problem_ids - expected_problem_ids)}"
            )
    rows = [json.loads(path.read_text()) for path in directories]
    bundle_counts = Counter(str(row["problem_id"]) for row in rows)
    if expected_problem_ids is not None:
        extra = set(bundle_counts) - expected_problem_ids
        if extra:
            raise ValueError(
                "bundle problem IDs differ from selection manifest: "
                f"extra={sorted(extra)}"
            )
        expected_protocol = selection.get("reward_protocol_version")
        if rows and expected_protocol is not None:
            observed = {row.get("reward_protocol_version") for row in rows}
            if observed != {expected_protocol}:
                raise ValueError(
                    f"bundles use reward protocols {sorted(observed, key=str)}, "
                    f"expected {expected_protocol}; rerun collection before analysis"
                )
    for summary in summary_payloads:
        count = bundle_counts[str(summary.get("problem_id"))]
        expected = summary.get("n_bundles")
        if expected is not None and count != int(expected):
            raise ValueError(f"saved bundle count differs from completed summary for {summary['problem_id']}")
        if expected_problem_ids is not None and count == 0 and expected != 0:
            raise ValueError(f"missing bundles without a zero-prefix summary for {summary.get('problem_id')}")
    no_prefix_ids = sorted(str(summary["problem_id"]) for summary in summary_payloads
                           if summary.get("n_bundles") == 0)
    if not rows:
        result = {"status": "completed", "finished_utc": utc(), "n_original_problems": expected_problems,
                  "n_observed_problems": 0, "n_prefixes": 0, "lopo": [], "predictor_benefit": None,
                  "no_surviving_prefix_problem_ids": no_prefix_ids,
                  "reason": "No paths survived the decision points"}
        write_json(root / "benefit/predictor_benefit_summary.json", result)
        write_json(root / "analysis_status.json", result)
        return
    provenance = json.loads((root / "worker-0-provenance.json").read_text())
    dimension = provenance["layout_dim"]
    metric_path = root / "metric_weights.npy"
    metric = (load_diagonal_metric(metric_path, dimension)
              if metric_path.is_file() else np.ones(dimension, dtype=np.float64))
    metric_hash, euclidean = sha256_array(metric), bool(np.all(metric == 1.0))
    for path in root.glob("worker-*-provenance.json"):
        validate_metric_record(json.loads(path.read_text()), metric_hash, euclidean, allow_legacy=allow_legacy)
    for row in rows:
        validate_metric_record(row, metric_hash, euclidean, prefix=True, allow_legacy=allow_legacy)
    replay = root / "replay"
    replay.mkdir(exist_ok=True)
    for name, filename in (("a", "half_mean_full_grads_a.npy"), ("b", "half_mean_full_grads_b.npy"),
                           ("prefix", "prefix_score_gradients.npy")):
        shape = (len(rows), dimension, 1) if name == "prefix" else (len(rows), dimension)
        matrix = np.lib.format.open_memmap(replay / filename, mode="w+", dtype=np.float64, shape=shape)
        for index, path in enumerate(directories):
            source = path.parent / ("prefix_score.npy" if name == "prefix" else f"mean_{name}.npy")
            if name == "prefix":
                matrix[index, :, 0] = np.load(source, mmap_mode="r")
            else:
                matrix[index] = np.load(source, mmap_mode="r")
        matrix.flush()
        del matrix
    metric_name = str(provenance.get("metric") or "euclidean")
    save_array(replay / "metric_weights.npy", metric)
    for index, half in enumerate(("a", "b")):
        full_norm = np.asarray([row["half_mean_full_norm_sq"][index] for row in rows])
        metric_norm = np.asarray([
            (row.get("half_full_metric_norm_sq") or row["half_mean_full_norm_sq"])[index]
            for row in rows
        ])
        save_array(replay / f"half_mean_full_norm_sq_{half}.npy", full_norm)
        save_array(replay / f"half_full_metric_norm_sq_{half}.npy", metric_norm)
    compact = [{key: value for key, value in row.items() if key not in
                {"rng_after", "grads", "suffix_texts", "continuation_records", "baseline_samples", "prefix_text"}}
               for row in rows]
    (replay / "prefixes.jsonl").write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in compact))
    replay_provenance = {**provenance, "metric_name": metric_name,
                         "metric_weights_sha256": sha256_array(metric),
                         "reward_protocol_version": provenance.get(
                             "reward_protocol_version", selection.get("reward_protocol_version"))}
    write_json(replay / "replay_provenance.json", replay_provenance)
    basis = np.load(replay / "prefix_score_gradients.npy", mmap_mode="r")
    write_json(replay / "score_gradient_provenance.json", {
        **replay_provenance, "shape": list(basis.shape), "score_gradients_sha256": sha256_array(basis),
        "replay_prefixes_sha256": sha256_file(replay / "prefixes.jsonl"),
        "prefix_keys_sha256": prefix_keys_sha256(compact)})
    write_json(replay / "preparation_summary.json", {
        "status": "completed", "n_original_problems": expected_problems, "n_prefixes": len(rows),
        "n_observed_problems": len(bundle_counts),
        "no_surviving_prefix_problem_ids": no_prefix_ids,
        "n_continuations": sum(row["n_continuations"] for row in rows),
        "enable_thinking": True, "functional_qualification_available": True,
        "metric_name": metric_name, "metric_weights_sha256": sha256_array(metric),
        "difficulty_manifest": selection.get("difficulty_manifest"),
        "difficulty_manifest_sha256": selection.get("difficulty_manifest_sha256"),
        "reward_protocol_version": replay_provenance.get("reward_protocol_version"),
        "verification": "Collected features, prefix score and gradient moments from the same frozen actor"})
    analysis = root / "analysis"
    analysis.mkdir(exist_ok=True)
    reports = [("oracle", "expected_gain_dynamic_oracle.py", []),
               ("predictor-legacy", "expected_gain_dynamic_predictor.py", ["--features", "legacy", "--device", "cpu",
                 "--models", "zero,constant,ridge,mlp64,mlp256"]),
               ("predictor-cheap", "expected_gain_dynamic_predictor.py", ["--features", "cheap", "--device", "cpu",
                 "--models", "zero,constant,ridge,mlp64,mlp256"])]
    for name, script, extra in reports:
        command = [sys.executable, str(REPO / "scripts" / script), "--replay-dir", str(replay),
                   "--gradient-target", "trajectory", "--metric-file", str(replay / "metric_weights.npy"),
                   "--run-dir", str(analysis / name), *extra]
        if selection.get("difficulty_manifest"):
            command.extend(["--difficulty-manifest", str(selection["difficulty_manifest"])])
        with (analysis / f"{name}.log").open("w") as log:
            subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
    with (analysis / "mechanism.log").open("w") as log:
        command = [sys.executable, str(REPO / "scripts/analyze_learning_mechanism.py"),
                   "--replay-dir", str(replay), "--metric-file", str(replay / "metric_weights.npy"),
                   "--metric-name", metric_name, "--bootstrap", "1000",
                   "--run-dir", str(analysis / "mechanism")]
        if selection.get("difficulty_manifest"):
            command.extend(["--difficulty-manifest", str(selection["difficulty_manifest"])])
        subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
    from thinking_predictor_benefit import report_benefit
    report_benefit(replay, root / "benefit", args.seed,
                   metric_file=replay / "metric_weights.npy", metric_name=metric_name,
                   difficulty_manifest=selection.get("difficulty_manifest"))
    write_json(root / "analysis_status.json", {"status": "completed", "finished_utc": utc(),
               "n_original_problems": expected_problems, "n_prefixes": len(rows), "enable_thinking": True,
               "n_observed_problems": len(bundle_counts), "no_surviving_prefix_problem_ids": no_prefix_ids,
               "metric_name": metric_name, "difficulty_manifest": selection.get("difficulty_manifest"),
               "summary": str(root / "benefit/predictor_benefit_summary.json")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--model-path", default="/SSD/00/wja/GRACE/data/Qwen3-4B")
    parser.add_argument("--data-dir", default="/SSD/00/wja/grace-pr13-20260929/input-shards-32")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--metric-file", default=None,
                        help="fixed diagonal metric weights; saved into replay before analysis")
    parser.add_argument("--metric-name", default=None,
                        help="name recorded for --metric-file (otherwise read its JSON sidecar)")
    parser.add_argument("--server-url", default="http://127.0.0.1:18013")
    parser.add_argument("--server-model", default="grace-thinking")
    parser.add_argument("--request-concurrency", type=int, default=8)
    parser.add_argument("--batch-continuations", action="store_true")
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--feature-batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--n-problems", type=int, default=32)
    parser.add_argument("--n-prefixes", type=int, default=2)
    parser.add_argument("--n-continuations", type=int, default=64)
    parser.add_argument("--n-baseline", type=int, default=16)
    parser.add_argument("--decision-tokens", type=int, nargs="+", default=[1024, 2048])
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--learning-completion", action="store_true",
                        help="write one-row trajectory replay shards for Experiment A")
    parser.add_argument("--difficulty-manifest", default=None)
    parser.add_argument("--difficulty-counts", default=None,
                        help="JSON object such as {\"easy\":8,\"medium\":16,\"hard\":8}")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--allow-legacy", action="store_true",
                        help="analyze pre-v3 runs only as explicitly unverified diagnostics")
    args = parser.parse_args()
    if (args.request_concurrency < 1 or args.feature_batch_size < 1
            or args.n_problems < 1 or args.n_prefixes < 1 or args.n_continuations < 2
            or args.n_baseline < 1 or args.workers < 1 or not 0 <= args.worker_index < args.workers
            or any(t < 1 or t >= args.max_new_tokens for t in args.decision_tokens)):
        parser.error("invalid collection dimensions or decision positions")
    if args.analyze_only and args.learning_completion:
        parser.error("use analyze_learning_completion.py for Experiment A shards")
    if args.analyze_only:
        merge(args)
    else:
        collect(args)


if __name__ == "__main__":
    main()
