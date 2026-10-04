#!/usr/bin/env python3
"""Frozen thinking audit with separate rollout/gradient GPUs and resumable moments."""

from __future__ import annotations

import argparse
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

from grace_gc.audit.prefix_audit import bundle_from_dict, bundle_to_dict
from grace_gc.audit.run import _bundles_from_engines
from grace_gc.core.rng import IsolatedRNG
from grace_gc.data.format_prompt import apply_solve_instruction
from grace_gc.data.math_data import (
    load_math_records,
    select_records_difficulty,
    selection_manifest,
)
from grace_gc.data.tokenize import encode_records_hf, load_hf_tokenizer
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
    def __init__(self, url, model, cache):
        self.url = url.rstrip("/") + "/v1/completions"
        self.model = model
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)

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

        with ThreadPoolExecutor(max_workers=min(8, max(len(prompts), 1))) as pool:
            return list(pool.map(one, zip(prompts, params)))


def reduce_bundle(bundle, rng, directory, seed):
    target = prefix_directory(directory, bundle.path_id, bundle.t)
    target.mkdir(parents=True, exist_ok=True)
    gradients = np.asarray(bundle.trajectory_grads)
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
        save_array(target / f"mean_{half}.npy", mean)
    save_array(target / "prefix_score.npy", bundle.prefix_score_grad)
    raw.update(enable_thinking=True,
               normal_q_mean_reward=float(np.mean(bundle.rewards[halves[0]])),
               half_counts=list(map(len, halves)),
               half_indices=[part.tolist() for part in halves],
               half_mean_reward=[float(np.mean(bundle.rewards[part])) for part in halves],
               half_mean_advantage=[float(np.mean(bundle.rewards[part]) - bundle.baseline) for part in halves],
               half_mean_suffix_cost=[float(np.mean(bundle.suffix_cost[part])) for part in halves],
               half_mean_full_norm_sq=[float(np.mean(np.asarray(bundle.true_grad_norm_sq)[part]))
                                       for part in halves],
               n_continuations=len(gradients), rng_after=rng.state_dict(),
               sufficient_statistics="Full-space FP64 A/B means, second moments and prefix score; JL only for display")
    # Publishing metadata last makes each completed prefix the resume unit.
    write_json(target / "bundle.json", raw)
    bundle.trajectory_grads = None
    bundle.prefix_score_grad = None


def collect(args):
    import torch
    from grace_gc.backends.gpu_engine import make_gpu_engines
    from grace_gc.backends.hf_actor import actor_numerics, named_lora_params
    from grace_gc.backends.verl_trainer import load_lora_actor
    from grace_gc.core.layout import collect_lora_layout, layout_hash
    from grace_gc.audit.qualification import classify_functional_recovery
    from grace_gc.data.reward import score_prefilled_answer
    from grace_gc.trainer.methods import method_spec

    root = Path(args.run_dir)
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for path in sorted(Path(args.data_dir).glob("shard-*.jsonl")):
        records.extend(load_math_records(path))
    records = apply_solve_instruction(records)
    if args.difficulty_manifest:
        counts = (None if args.difficulty_counts is None
                  else json.loads(args.difficulty_counts))
        if counts is not None and not isinstance(counts, dict):
            raise ValueError("--difficulty-counts must be a JSON object")
        records = select_records_difficulty(
            records, args.n_problems, args.difficulty_manifest, counts, seed=args.seed
        )
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
        write_json(root / "selection.json", selection_manifest(
            manifest_records, selection, args.seed))
    write_json(root / f"worker-{args.worker_index}-status.json", {
        "status": "loading_actor", "started_utc": utc(), "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "problem_ids": [rec.problem_id for rec in records]})
    torch.manual_seed(args.seed)
    actor = load_lora_actor(args.model_path, LORA)
    named = named_lora_params(actor)
    layout = collect_lora_layout(named)
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
    provenance = {"model": "Qwen/Qwen3-4B", "model_revision": REVISION,
                  "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
                  "audit_source_sha256": sha256_file(REPO / "grace_gc/audit/run.py"),
                  "runner_sha256": sha256_file(__file__), "actor_sha256": sha256_named(named),
                  "checkpoint_sha256": sha256_file(checkpoint), "checkpoint": str(checkpoint),
                  "layout_dim": layout.dim, "layout_names": layout.names(), "layout_sha256": layout_hash(layout),
                  "actor_numerics": actor_numerics(actor), "enable_thinking": True,
                  "temperature": 1.0, "metric": "euclidean",
                  "adam_metric": "unavailable: new snapshot has no optimizer moments"}
    write_json(root / f"worker-{args.worker_index}-provenance.json", provenance)
    tokenizer = load_hf_tokenizer(args.model_path)
    cfg = {"temperature": 1.0, "predictor": {"feature_batch_size": 1, "feature_mode": "legacy"}}
    remote = RemoteLLM(args.server_url, args.server_model, root / "rollout-cache")
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

    forced = tokenizer.encode("</think>\nAnswer:", add_special_tokens=False)
    probe_records = {}

    def qualify(prefix, prompt_len, gold, pid, t, path_index):
        probe = list(prefix) + forced
        rng = IsolatedRNG.create(args.seed + path_index * 100003 + int(t) * 1009 + len(prefix))
        fulls, reasons, seeds = repeated(probe, 4, 32, rng)
        texts = [engines.decode(full[len(probe):]) for full in fulls]
        rewards = [float(score_prefilled_answer(text, gold, truncated=False) or 0.0)
                   for text in texts]
        probe_records[(pid, f"{pid}:{path_index}", t)] = [
            {"text": text, "reward": reward, "seed": seed, "finish_reason": reason,
             "token_ids": full[len(probe):]} for text, reward, seed, reason, full
            in zip(texts, rewards, seeds, reasons, fulls)]
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
                reduce_bundle(bundle, rng, directory, args.seed)
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
                rng.load_state_dict(raw["rng_after"])
                print(f"phase=resume_prefix pid={pid} path={path_id} t={t}", flush=True)
                return bundle_from_dict(raw)

            bundles = _bundles_from_engines(
                [rec], engines, layout, 1 if args.smoke else 2, 2 if args.smoke else 64,
                [64, 128] if args.smoke else [1024, 2048], 8192, seed, encode,
                spec=method_spec("full_pg"), jl_dim=256, jl_seed=args.seed,
                n_baseline=2 if args.smoke else 16, store_features=True,
                store_trajectory_gradients=True, qualification_fn=qualify,
                bundle_sink=sink, bundle_resume=resume)
            write_json(directory / "raw_problem.json", _bundles_from_engines.last[0])
            write_json(directory / "summary.json", {"status": "completed", "problem_id": rec.problem_id,
                       "n_bundles": len(bundles), "seed": seed, "finished_utc": utc(),
                       "enable_thinking": True})
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
    if selection_path.is_file():
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        expected_problems = int(selection["n_problems"])
    else:
        # PR13 runs predate the frozen selection manifest; preserve their
        # analysis path while making the missing provenance explicit.
        expected_problems = len(summaries)
        selection = {"selection": "legacy_unrecorded", "n_problems": expected_problems}
    if len(summaries) != expected_problems:
        raise ValueError(f"only {len(summaries)} of {expected_problems} selected problems are complete")
    replay = root / "replay"
    replay.mkdir(exist_ok=True)
    rows = [json.loads(path.read_text()) for path in directories]
    provenance = json.loads((root / "worker-0-provenance.json").read_text())
    dimension = provenance["layout_dim"]
    if not rows:
        write_json(root / "analysis_status.json", {"status": "completed", "n_original_problems": expected_problems,
                   "n_prefixes": 0, "predictor_benefit": None, "reason": "No paths survived the decision points"})
        return
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
    for index, half in enumerate(("a", "b")):
        save_array(replay / f"half_mean_full_norm_sq_{half}.npy",
                   np.asarray([row["half_mean_full_norm_sq"][index] for row in rows]))
    compact = [{key: value for key, value in row.items() if key not in
                {"rng_after", "grads", "suffix_texts", "continuation_records", "baseline_samples", "prefix_text"}}
               for row in rows]
    (replay / "prefixes.jsonl").write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in compact))
    save_array(replay / "euclidean_weights.npy", np.ones(dimension))
    write_json(replay / "replay_provenance.json", provenance)
    basis = np.load(replay / "prefix_score_gradients.npy", mmap_mode="r")
    write_json(replay / "score_gradient_provenance.json", {
        **provenance, "shape": list(basis.shape), "score_gradients_sha256": sha256_array(basis),
        "replay_prefixes_sha256": sha256_file(replay / "prefixes.jsonl"),
        "prefix_keys_sha256": prefix_keys_sha256(compact)})
    write_json(replay / "preparation_summary.json", {
        "status": "completed", "n_original_problems": expected_problems, "n_prefixes": len(rows),
        "n_continuations": sum(row["n_continuations"] for row in rows),
        "enable_thinking": True, "functional_qualification_available": True,
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
                   "--gradient-target", "trajectory", "--metric-file", str(replay / "euclidean_weights.npy"),
                   "--run-dir", str(analysis / name), *extra]
        with (analysis / f"{name}.log").open("w") as log:
            subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
    with (analysis / "mechanism.log").open("w") as log:
        subprocess.run([sys.executable, str(REPO / "scripts/analyze_learning_mechanism.py"),
                        "--replay-dir", str(replay), "--metric-name", "euclidean", "--bootstrap", "1000",
                        "--run-dir", str(analysis / "mechanism")], cwd=REPO,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    from thinking_predictor_benefit import report_benefit
    report_benefit(replay, root / "benefit", args.seed)
    write_json(root / "analysis_status.json", {"status": "completed", "finished_utc": utc(),
               "n_original_problems": expected_problems, "n_prefixes": len(rows), "enable_thinking": True,
               "summary": str(root / "benefit/predictor_benefit_summary.json")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--model-path", default="/SSD/00/wja/GRACE/data/Qwen3-4B")
    parser.add_argument("--data-dir", default="/SSD/00/wja/grace-pr13-20260929/input-shards-32")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--server-url", default="http://127.0.0.1:18013")
    parser.add_argument("--server-model", default="grace-thinking")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--n-problems", type=int, default=32)
    parser.add_argument("--difficulty-manifest", default=None)
    parser.add_argument("--difficulty-counts", default=None,
                        help="JSON object such as {\"easy\":8,\"medium\":16,\"hard\":8}")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    if args.analyze_only:
        merge(args)
    else:
        collect(args)


if __name__ == "__main__":
    main()
