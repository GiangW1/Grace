#!/usr/bin/env python3
"""Persistently collect the original 32 problems and run the offline analysis."""

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

REPO = Path(__file__).resolve().parents[1]
ROOT = Path("/SSD/01/wja/grace-pr13-thinking-20261001")
MODEL = Path("/SSD/00/wja/GRACE/data/Qwen3-4B")
PYTHON = Path("/SSD/00/wja/GRACE/runs/conda/envs/grace/bin/python")
VLLM = PYTHON.with_name("vllm")
CHECKPOINT = ROOT / "frozen-thinking-lora.npz"
URL = "http://127.0.0.1:18013"
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


def worker(index, gpu, smoke=False):
    directory = ROOT / ("smoke" if smoke else "collection")
    name = "smoke" if smoke else f"worker-{index}"
    log = ROOT / "logs" / (name + ".log")
    command = [PYTHON, "-u", REPO / "scripts/audit_separated_thinking.py", "--run-dir", directory,
               "--checkpoint", CHECKPOINT, "--worker-index", str(index)]
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


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "logs").mkdir(exist_ok=True)
    write_json(ROOT / "scope.json", {"n_problems": 32, "append_32": False,
               "model": "Qwen/Qwen3-4B", "model_revision": REVISION, "enable_thinking": True,
               "n_prefixes": 2, "n_continuations": 64, "n_baseline": 16,
               "max_new_tokens": 8192, "decision_grid": [1024, 2048],
               "functional_probes": {"n": 4, "max_new_tokens": 32, "majority": .5},
               "gpu_roles": {"0": "actor gradients", "2": "actor gradients", "3": "vLLM generation"},
               "primary_metric": "euclidean", "adam_metric": "unavailable: no optimizer history",
               "difficulty_manifest": "unavailable; original 32 problems retained"})
    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT, cleanup)
    try:
        status("waiting_model_download")
        marker = MODEL / "thinking_download_verified.txt"
        while not marker.exists():
            time.sleep(30)
        if marker.read_text().strip() != REVISION:
            raise ValueError("downloaded model revision differs")
        wait_for_card(3)
        status("starting_vllm")
        command = [VLLM, "serve", MODEL, "--host", "127.0.0.1", "--port", "18013",
                   "--served-model-name", "grace-thinking", "--dtype", "bfloat16", "--generation-config", "vllm",
                   "--max-model-len", "9280", "--gpu-memory-utilization", "0.40",
                   "--max-num-seqs", "16", "--enforce-eager", "--return-tokens-as-token-ids"]
        server = launch(command, ROOT / "logs/vllm.log", 3)
        while not ready():
            if server.poll() is not None:
                raise RuntimeError("vLLM exited during startup; see logs/vllm.log")
            time.sleep(5)
        status("smoke_running")
        worker(0, 0, smoke=True)
        verify_smoke()
        status("collecting_32")
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(worker, index, gpu) for index, gpu in ((0, 0), (1, 2))]
            for future in futures:
                future.result()
        os.killpg(server.pid, signal.SIGTERM)
        server.wait(timeout=30)
        status("offline_predictor_analysis")
        command = [PYTHON, "-u", REPO / "scripts/audit_separated_thinking.py", "--run-dir", ROOT / "collection",
                   "--checkpoint", CHECKPOINT, "--analyze-only"]
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
