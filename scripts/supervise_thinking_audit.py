#!/usr/bin/env python3
"""Persistently collect the original 32 problems and run the offline analysis."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

from audit_separated_thinking import REVISION, utc, write_json
from grace_gc.data.reward import REWARD_PROTOCOL_VERSION

REPO = Path(__file__).resolve().parents[1]
ROOT = Path("/SSD/01/wja/grace-pr13-thinking-20261001")
MODEL = Path("/SSD/00/wja/GRACE/data/Qwen3-4B")
PYTHON = Path("/SSD/00/wja/GRACE/runs/conda/envs/grace/bin/python")
VLLM = PYTHON.with_name("vllm")
CHECKPOINT = ROOT / "frozen-thinking-lora.npz"
URL = "http://127.0.0.1:18013"
METRIC_FILE = os.environ.get("GRACE_METRIC_FILE")
METRIC_NAME = os.environ.get("GRACE_METRIC_NAME")
DIFFICULTY_MANIFEST = os.environ.get("GRACE_DIFFICULTY_MANIFEST")
DIFFICULTY_COUNTS = os.environ.get("GRACE_DIFFICULTY_COUNTS")
ENVIRONMENT = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
               "TOKENIZERS_PARALLELISM": "false", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
children = []


def status(phase, **extra):
    write_json(ROOT / "supervisor_status.json", {"status": phase, "updated_utc": utc(), **extra})
    print(f"{utc()} phase={phase} {extra}", flush=True)


def free_memory(gpu):
    try:
        result = subprocess.check_output(["nvidia-smi", f"--id={gpu}",
                     "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True, timeout=20)
        return int(result.strip())
    except (subprocess.SubprocessError, ValueError):
        return 0


def wait_for_card(gpu, needed=64000):
    while free_memory(gpu) < needed:
        print(f"{utc()} waiting_for_gpu={gpu} free_MiB={free_memory(gpu)}", flush=True)
        time.sleep(30)


def ready():
    try:
        with urllib.request.urlopen(URL + "/health", timeout=5) as response:
            return response.status == 200
    except Exception:
        return False


def launch(command, log, gpu=None):
    environment = ENVIRONMENT if gpu is None else {**ENVIRONMENT, "CUDA_VISIBLE_DEVICES": str(gpu)}
    handle = log.open("a")
    process = subprocess.Popen(list(map(str, command)), cwd=REPO, env=environment,
                               stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
    handle.close()
    children.append(process)
    return process


def audit_flags():
    flags = []
    if METRIC_FILE:
        flags += ["--metric-file", METRIC_FILE]
    if METRIC_NAME:
        flags += ["--metric-name", METRIC_NAME]
    if DIFFICULTY_MANIFEST:
        flags += ["--difficulty-manifest", DIFFICULTY_MANIFEST]
    if DIFFICULTY_COUNTS:
        flags += ["--difficulty-counts", DIFFICULTY_COUNTS]
    return flags


def cleanup(*_args):
    for process in children:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
    for process in children:
        if process.poll() is None:
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    if _args:
        raise SystemExit(0)


def worker(index, gpu, smoke=False, extra_args=()):
    directory = ROOT / ("smoke" if smoke else "collection")
    name = "smoke" if smoke else f"worker-{index}"
    log = ROOT / "logs" / (name + ".log")
    command = [PYTHON, "-u", REPO / "scripts/audit_separated_thinking.py", "--run-dir", directory,
               "--checkpoint", CHECKPOINT, "--worker-index", str(index), *extra_args]
    command.extend(audit_flags())
    if smoke:
        command.append("--smoke")
    while True:
        completion = directory / f"worker-{index}-status.json"
        if completion.exists() and json.loads(completion.read_text()).get("status") == "completed":
            return
        wait_for_card(gpu)
        print(f"{utc()} launch={name} gpu={gpu}", flush=True)
        position = log.stat().st_size if log.exists() else 0
        code = launch(command, log, gpu).wait()
        if code == 0:
            return
        with log.open("rb") as stream:
            stream.seek(position)
            failure = stream.read().decode(errors="replace")
        if "out of memory" in failure.lower() or "cuda error" in failure.lower():
            print(f"{utc()} retry_oom={name}; completed prefixes retained", flush=True)
            time.sleep(30)
            continue
        raise RuntimeError(f"{name} exited with {code}; see {log}")


def verify_smoke():
    paths = list((ROOT / "smoke/problems").glob("*/path-*/bundle.json"))
    rows = [json.loads(path.read_text()) for path in paths]
    thinking = any("<think>" in text for row in rows for text in row["suffix_texts"])
    feature_shapes = sorted({len(row["features"]) for row in rows if row.get("features") is not None})
    probes = sum(row.get("qualification_n") == 4 for row in rows)
    result = {"finished_utc": utc(), "n_prefixes": len(rows),
              "generated_think_tag_observed": thinking, "feature_shapes": feature_shapes,
              "prefixes_with_four_functional_probes": probes,
              "full_space_sidecars": all((path.parent / "prefix_score.npy").exists() and
                                          (path.parent / "mean_a.npy").exists() and
                                          (path.parent / "mean_b.npy").exists() for path in paths)}
    write_json(ROOT / "smoke_validation.json", result)
    if rows and (not thinking or not feature_shapes or not result["full_space_sidecars"]):
        raise RuntimeError("smoke execution is missing requested thinking or gradient artifacts")


def main(argv=None):
    global ROOT, MODEL, CHECKPOINT, URL
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=str(ROOT))
    parser.add_argument("--model-path", default=str(MODEL))
    parser.add_argument("--data-dir", default="/SSD/00/wja/grace-pr13-20260929/input-shards-32")
    parser.add_argument("--checkpoint")
    parser.add_argument("--n-problems", type=int, default=32)
    parser.add_argument("--n-continuations", type=int, default=64)
    parser.add_argument("--request-concurrency", type=int, default=8)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 2, 3],
                        help="one or two gradient GPUs followed by the rollout GPU")
    parser.add_argument("--port", type=int, default=18013)
    args = parser.parse_args(argv)
    if len(args.gpus) not in (2, 3) or len(set(args.gpus)) != len(args.gpus):
        parser.error("two or three distinct GPUs are required; the last GPU generates rollouts")
    if args.n_problems < 1:
        parser.error("n_problems must be positive")
    if args.request_concurrency < 1:
        parser.error("request concurrency must be positive")
    if args.n_continuations < 2:
        parser.error("A/B continuation statistics require at least two samples")
    ROOT, MODEL = Path(args.run_dir).resolve(), Path(args.model_path).resolve()
    CHECKPOINT = Path(args.checkpoint).resolve() if args.checkpoint else ROOT / "frozen-thinking-lora.npz"
    URL = f"http://127.0.0.1:{args.port}"
    gradient_gpus, rollout_gpu = args.gpus[:-1], args.gpus[-1]
    extra = ["--workers", str(len(gradient_gpus)), "--data-dir", str(Path(args.data_dir).resolve()),
             "--n-problems", str(args.n_problems), "--model-path", str(MODEL),
             "--server-url", URL, "--server-model", "grace-thinking",
             "--request-concurrency", str(args.request_concurrency),
             "--n-continuations", str(args.n_continuations)]
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "logs").mkdir(exist_ok=True)
    if (ROOT / "scope.json").is_file():
        saved_scope = json.loads((ROOT / "scope.json").read_text())
        if saved_scope.get("n_continuations", 64) != args.n_continuations:
            raise ValueError("saved continuation count differs; use a new run directory")
    write_json(ROOT / "scope.json", {"n_problems": args.n_problems, "append_32": False,
               "model": "Qwen/Qwen3-4B", "model_revision": REVISION, "enable_thinking": True,
               "n_prefixes": 2, "n_continuations": args.n_continuations, "n_baseline": 16,
               "request_concurrency": args.request_concurrency,
               "max_new_tokens": 8192, "decision_grid": [1024, 2048],
               "functional_probes": {"n": 4, "max_new_tokens": 32, "threshold": .5},
               "gpu_roles": {**{str(gpu): f"gradient worker {index}" for index, gpu in enumerate(gradient_gpus)},
                             str(rollout_gpu): "vLLM generation"},
               "data_dir": str(Path(args.data_dir).resolve()), "checkpoint": str(CHECKPOINT),
               "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
               "reward_protocol_version": REWARD_PROTOCOL_VERSION,
               "primary_metric": METRIC_NAME or "euclidean",
               "metric_file": METRIC_FILE,
               "adam_metric": ("not_requested" if not METRIC_FILE else "fixed_diagonal_metric"),
               "difficulty_manifest": DIFFICULTY_MANIFEST,
               "difficulty_counts": DIFFICULTY_COUNTS})
    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT, cleanup)
    try:
        status("waiting_model_download")
        marker = MODEL / "thinking_download_verified.txt"
        while not marker.exists():
            time.sleep(30)
        if marker.read_text().strip() != REVISION:
            raise ValueError("downloaded model revision differs")
        wait_for_card(rollout_gpu)
        status("starting_vllm", gpu=rollout_gpu)
        command = [VLLM, "serve", MODEL, "--host", "127.0.0.1", "--port", str(args.port),
                   "--served-model-name", "grace-thinking", "--dtype", "bfloat16", "--generation-config", "vllm",
                   "--max-model-len", "9280", "--gpu-memory-utilization", "0.40",
                   "--max-num-seqs", "16", "--enforce-eager", "--return-tokens-as-token-ids"]
        server = launch(command, ROOT / "logs/vllm.log", rollout_gpu)
        while not ready():
            if server.poll() is not None:
                raise RuntimeError("vLLM exited during startup; see logs/vllm.log")
            time.sleep(5)
        status("smoke_running", gpu=gradient_gpus[0])
        worker(0, gradient_gpus[0], smoke=True, extra_args=extra)
        verify_smoke()
        status("collecting", n_problems=args.n_problems, gradient_gpus=gradient_gpus, rollout_gpu=rollout_gpu)
        with ThreadPoolExecutor(max_workers=len(gradient_gpus)) as pool:
            futures = [pool.submit(worker, index, gpu, extra_args=extra)
                       for index, gpu in enumerate(gradient_gpus)]
            for future in futures:
                future.result()
        cleanup()
        status("offline_predictor_analysis")
        command = [PYTHON, "-u", REPO / "scripts/audit_separated_thinking.py", "--run-dir", ROOT / "collection",
                   "--checkpoint", CHECKPOINT, "--analyze-only"]
        command.extend(audit_flags())
        if launch(command, ROOT / "logs/offline-analysis.log").wait():
            raise RuntimeError("offline analysis failed; see logs/offline-analysis.log")
        status("completed", summary=str(ROOT / "collection/benefit/predictor_benefit_summary.json"))
    except BaseException as exc:
        if not isinstance(exc, SystemExit):
            status("failed", error=str(exc))
        raise
    finally:
        cleanup()


if __name__ == "__main__":
    main()
