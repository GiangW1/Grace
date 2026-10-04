#!/usr/bin/env python3
"""Run Experiment A with one or two frozen gradient workers and one vLLM GPU."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
import signal
import time

import yaml
import numpy as np

import supervise_thinking_audit as runtime
from audit_separated_thinking import REVISION, write_json
from grace_gc.data.math_data import load_math_records, records_for_split, select_records, selection_manifest
from grace_gc.data.reward import REWARD_PROTOCOL_VERSION
from grace_gc.data.reward import rule_reward
from grace_gc.audit.qualification import score_functional_probe
from grace_gc.trainer.loop import build_run_config


def prepare_inputs(root, data_path, config):
    audit = config["audit"]
    seed = int(config.get("seed", 17))
    records = select_records(records_for_split(load_math_records(data_path), "audit", seed=seed),
                             int(audit["n_problems"]), "seeded", seed)
    directory = root / "input"
    directory.mkdir(exist_ok=True)
    target = directory / "shard-0.jsonl"
    content = "".join(json.dumps(asdict(record), ensure_ascii=True) + "\n" for record in records)
    if target.exists() and target.read_text() != content:
        raise ValueError("selected inputs changed; use a new run directory")
    target.write_text(content)
    write_json(root / "input_manifest.json", {
        **selection_manifest(records, "seeded", seed), "data_path": str(Path(data_path).resolve()),
        "split": "audit", "difficulty_manifest": None})
    (root / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    return directory, len(records)


def analyze_collection(root, seed):
    directories = sorted(path.parent for path in (root / "collection/problems").glob("*/path-*/bundle.json"))
    if not directories:
        write_json(root / "analysis/learning_completion_summary.json", {
            "status": "unavailable", "reason": "no live prefixes were collected"})
        return
    command = [runtime.PYTHON, "-u", runtime.REPO / "scripts/analyze_learning_completion.py",
               "--replay-dir", *directories, "--gram-cache", root / "gram-cache",
               "--block-mb", "256", "--bootstrap", "1000", "--seed", str(seed),
               "--run-dir", root / "analysis"]
    runtime.ENVIRONMENT.update(OPENBLAS_NUM_THREADS="8", OMP_NUM_THREADS="8")
    if runtime.launch(command, root / "logs/analysis.log").wait():
        raise RuntimeError("Experiment A analysis failed; see logs/analysis.log")


def verify_smoke(root):
    directories = sorted(path.parent for path in (root / "smoke/problems").glob("*/path-*/bundle.json"))
    if not directories:
        raise RuntimeError("smoke produced no replay prefixes")
    for directory in directories:
        row = json.loads((directory / "bundle.json").read_text())
        if row["reward_protocol_version"] != REWARD_PROTOCOL_VERSION or not row["require_complete_answers"]:
            raise ValueError("smoke reward protocol differs")
        for record in row["continuation_records"]:
            if record["reward"] != rule_reward(record["text"], record["gold"],
                                               truncated=record["truncated"], require_complete=True):
                raise ValueError("smoke rollout reward differs from completed-answer scoring")
        for probe in json.loads((directory / "functional_probes.json").read_text()):
            if probe["reward"] != score_functional_probe(probe["text"], row["gold"], probe["truncated"]):
                raise ValueError("smoke probe score omits prefill or truncation")
        a = np.load(directory / "half_mean_full_grads_a.npy", mmap_mode="r")
        b = np.load(directory / "half_mean_full_grads_b.npy", mmap_mode="r")
        g = np.load(directory / "prefix_score_gradients.npy", mmap_mode="r")
        if a.shape != b.shape or a.ndim != 2 or a.shape[0] != 1 or g.shape != (*a.shape, 1):
            raise ValueError("smoke gradient shard shapes differ")
    write_json(root / "smoke_validation.json", {"status": "passed", "prefixes": len(directories),
                "reward_protocol_version": REWARD_PROTOCOL_VERSION,
                "checked": ["completed rollout scoring", "prefilled probe scoring", "trajectory shard shapes"]})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--model-path", default=str(runtime.MODEL))
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--gpus", type=int, nargs="+", default=[1, 2, 3],
                        help="one or two gradient GPUs followed by the rollout GPU")
    parser.add_argument("--port", type=int, default=18014)
    args = parser.parse_args(argv)
    if len(args.gpus) not in (2, 3) or len(set(args.gpus)) != len(args.gpus):
        parser.error("two or three distinct GPUs are required; the last GPU generates rollouts")
    gradient_gpus, rollout_gpu = args.gpus[:-1], args.gpus[-1]
    config = build_run_config(args.config, {"backend": "gpu_verl", "method": "full_pg", "enable_thinking": True})
    audit = config["audit"]
    if not audit.get("require_complete_answers"):
        parser.error("Experiment A must use completed-answer rewards")
    root = Path(args.run_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    runtime.ROOT, runtime.MODEL = root, Path(args.model_path).resolve()
    runtime.CHECKPOINT = root / "frozen-thinking-lora.npz"
    runtime.URL = f"http://127.0.0.1:{args.port}"
    signal.signal(signal.SIGTERM, runtime.cleanup)
    signal.signal(signal.SIGINT, runtime.cleanup)
    try:
        data_dir, count = prepare_inputs(root, args.data_path, config)
        max_new = int(audit.get("max_new_tokens", config["max_new_tokens"]))
        extra = ["--data-dir", str(data_dir), "--n-problems", str(count),
                 "--n-prefixes", str(audit["n_prefixes"]), "--n-continuations", str(audit["n_continuations"]),
                 "--n-baseline", str(audit["n_baseline"]), "--max-new-tokens", str(max_new),
                 "--model-path", str(runtime.MODEL), "--server-url", runtime.URL,
                 "--server-model", "grace-thinking-exp-a", "--workers", str(len(gradient_gpus)),
                 "--learning-completion",
                 "--decision-tokens", *map(str, audit["decision_grid"])]
        write_json(root / "scope.json", {
            "experiment": "PR14 Experiment A: population learning completion vs answer resolution",
            "model": "Qwen/Qwen3-4B", "model_revision": REVISION, "enable_thinking": True,
            "reward_protocol_version": REWARD_PROTOCOL_VERSION, "require_complete_answers": True,
            "n_problems": count, "n_prefixes": audit["n_prefixes"], "n_continuations": audit["n_continuations"],
            "n_baseline": audit["n_baseline"], "decision_grid": audit["decision_grid"],
            "max_new_tokens": max_new, "primary_metric": "euclidean", "crossing_thresholds": None,
            "gpu_roles": {**{str(gpu): f"gradient worker {index}"
                            for index, gpu in enumerate(gradient_gpus)},
                          str(rollout_gpu): "vLLM rollout server"}})
        runtime.status("waiting_model_download")
        marker = runtime.MODEL / "thinking_download_verified.txt"
        while not marker.exists():
            time.sleep(10)
        if marker.read_text().strip() != REVISION:
            raise ValueError("downloaded thinking model revision differs")
        runtime.wait_for_card(rollout_gpu)
        runtime.status("starting_vllm", gpu=rollout_gpu)
        command = [runtime.VLLM, "serve", runtime.MODEL, "--host", "127.0.0.1", "--port", str(args.port),
                   "--served-model-name", "grace-thinking-exp-a", "--dtype", "bfloat16",
                   "--generation-config", "vllm", "--max-model-len", str(max_new + 1088),
                   "--gpu-memory-utilization", "0.40", "--max-num-seqs", "16", "--enforce-eager",
                   "--return-tokens-as-token-ids"]
        server = runtime.launch(command, root / "logs/vllm.log", rollout_gpu)
        while not runtime.ready():
            if server.poll() is not None:
                raise RuntimeError("vLLM exited during startup; see logs/vllm.log")
            time.sleep(5)
        runtime.status("smoke_running", gpu=args.gpus[0])
        runtime.worker(0, args.gpus[0], smoke=True, extra_args=extra)
        verify_smoke(root)
        runtime.status("collecting", n_problems=count, gradient_gpus=gradient_gpus, rollout_gpu=rollout_gpu)
        with ThreadPoolExecutor(max_workers=len(gradient_gpus)) as pool:
            futures = [pool.submit(runtime.worker, i, gpu, extra_args=extra) for i, gpu in enumerate(gradient_gpus)]
            for future in futures:
                future.result()
        runtime.cleanup()
        runtime.status("offline_learning_completion_analysis")
        analyze_collection(root, int(config.get("seed", 17)))
        runtime.status("completed", summary=str(root / "analysis/learning_completion_summary.json"))
    except BaseException as exc:
        if not isinstance(exc, SystemExit):
            runtime.status("failed", error=str(exc))
        raise
    finally:
        runtime.cleanup()


if __name__ == "__main__":
    main()
