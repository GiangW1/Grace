"""Persist the two-GPU PR15 experiment queue and retain every execution attempt."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def utc():
    return datetime.now(timezone.utc).isoformat()


def gpu_snapshot(gpus):
    try:
        value = subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits"], text=True, timeout=20)
        result = []
        for row in value.splitlines():
            index, free, utilization = map(int, row.split(","))
            if str(index) in gpus:
                result.append({"gpu": index, "free_mib": free, "utilization": utilization})
        return result
    except (subprocess.SubprocessError, ValueError):
        return []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--gpus", default="0,2")
    parser.add_argument("--model-path", default="/SSD/00/wja/GRACE/data/Qwen3-4B")
    parser.add_argument("--eval-data", default="/SSD/00/wja/GRACE/data/math500/test.jsonl")
    parser.add_argument("--train-seeds", type=int, nargs="+", default=[17, 29, 43])
    parser.add_argument("--budget-seconds", type=int, default=3600)
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    gpus = args.gpus.split(",")
    if len(gpus) != 2 or len(set(gpus)) != 2:
        parser.error("two distinct GPUs are required")
    root = args.run_dir.resolve()
    (root / "logs").mkdir(parents=True, exist_ok=True)
    (root / "jobs").mkdir(exist_ok=True)
    if args.detach:
        command = [sys.executable, "-u", str(Path(__file__).resolve()),
                   *[value for value in sys.argv[1:] if value != "--detach"]]
        with (root / "logs/supervisor.log").open("a") as stream:
            process = subprocess.Popen(command, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, start_new_session=True)
        write_json(root / "supervisor_launch.json", {"pid": process.pid, "started_utc": utc(),
                                                     "command": command})
        print(json.dumps({"supervisor_pid": process.pid, "run_dir": str(root)}), flush=True)
        return
    base = [sys.executable, "-u", str(REPO / "scripts/suffix_transport.py"),
            "--config", "configs/hardware/a100_1.yaml",
            "--config", "configs/experiments/suffix_transport.yaml",
            "--config", "configs/experiments/suffix_transport_execution_a100_2.yaml",
            "--gpus", args.gpus, "--model-path", args.model_path,
            "--eval-data-path", args.eval_data,
            "--init-checkpoint", str(root / "shared-initial-actor.npz")]
    audit_data, train_data = str(root / "input/audit.jsonl"), str(root / "input/train.jsonl")
    batch = ["--config", "configs/experiments/suffix_transport_training_batch.yaml"]
    jobs = [{"name": "cpu-math-smoke", "cpu": True, "command": [sys.executable, "-u",
             str(REPO / "scripts/suffix_transport.py"), "--mode", "smoke", "--cpu"]}]
    jobs += [
        {"name": "full-pg-one-step", "command": base + batch + ["--mode", "train", "--method", "full_pg",
         "--data-path", train_data, "--steps", "1", "--n-problems", "2"]},
        {"name": "gpu-short-smoke", "command": base + ["--mode", "smoke", "--data-path", audit_data]},
        {"name": "st-one-step", "command": base + batch + ["--mode", "train", "--method", "suffix_transport",
         "--data-path", train_data, "--steps", "1", "--n-problems", "2"]},
        {"name": "gpu-long-smoke", "command": base + ["--mode", "smoke", "--data-path", audit_data,
         "--decision-tokens", "512", "--max-new-tokens", "8192"]}]
    for m in (2, 4):
        jobs.append({"name": f"audit-m{m}-t512", "command": base + ["--mode", "audit", "--data-path", audit_data,
            "--group-size", str(m), "--n-problems", "32", "--groups-per-problem", "2", "--audit-draws", "8",
            "--decision-tokens", "512", "--max-new-tokens", "8192"]})
    for seed in args.train_seeds:
        for method in ("full_pg", "donor_only", "suffix_transport"):
            name = f"train-{method}-seed{seed}"
            jobs.append({"name": name, "command": base + batch + ["--mode", "train", "--data-path", train_data,
                "--method", method, "--seed", str(seed), "--steps", "100000", "--n-problems", "0",
                "--budget-seconds", str(args.budget_seconds)]})
            jobs.append({"name": f"eval-{method}-seed{seed}", "training_job": name})
    queue = {"requested_commit": "1e4e496333b6ea1b1f2c369e5baa1b19079ce185",
             "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
             "gpu_roles": {gpus[0]: "HF actor/gradient", gpus[1]: "vLLM rollout"},
             "execution": {"max_num_seqs": 32, "vllm_memory_utilization": .65,
                           "gradient_checkpointing": True, "audit_draw_batch_size": 8,
                           "training_prompts_per_step": 8},
             "train_seeds": args.train_seeds, "training_budget_seconds_per_method": args.budget_seconds,
             "jobs": jobs, "retry_scope": "OOM retries use a new run directory; completed jobs are skipped. No ST optimizer/RNG or partial-audit resume."}
    plan = root / "queue.json"
    if plan.exists() and json.loads(plan.read_text()) != queue:
        raise ValueError("queue changed; use a new run root")
    write_json(plan, queue)
    state_file = root / "supervisor_status.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else {"completed": {}, "attempts": {}}
    child = None

    def status(phase, **extra):
        state.update(status=phase, updated_utc=utc(), supervisor_pid=os.getpid(), **extra)
        write_json(state_file, state)
        print(f"{utc()} phase={phase} job={state.get('current_job')}", flush=True)

    def stop(_signum, _frame):
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGINT)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=20)
        status("stopped")
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": args.gpus,
                   "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
                   "TOKENIZERS_PARALLELISM": "false", "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                   "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    for job in jobs:
        name = job["name"]
        if name in state["completed"]:
            continue
        state["current_job"] = name
        if "training_job" in job:
            trained = Path(state["completed"][job["training_job"]]["run_dir"])
            checkpoint = json.loads((trained / "budget_checkpoint.json").read_text())
            if not checkpoint.get("path"):
                state["completed"][name] = {"status": "unavailable", "reason": "No in-budget actor checkpoint"}
                status("evaluation_unavailable")
                continue
            command = [sys.executable, "-u", str(REPO / "scripts/evaluate.py"), "--generate",
                "--backend", "gpu_verl", "--method", "full_pg", "--config", str(trained / "config.yaml"),
                "--data-path", args.eval_data, "--model-path", args.model_path,
                "--checkpoint", checkpoint["path"], "--k", "1", "--seed", "17"]
        else:
            command = job["command"]
        while True:
            attempt = state["attempts"].get(name, 0)
            directory = root / "jobs" / (name if attempt == 0 else f"{name}-attempt-{attempt}")
            if directory.exists():
                attempt += 1
                state["attempts"][name] = attempt
                continue
            if not job.get("cpu"):
                while True:
                    cards = gpu_snapshot(gpus)
                    if len(cards) == 2 and all(card["free_mib"] >= 64000 for card in cards):
                        break
                    status("waiting_for_gpus", gpu_snapshot=cards)
                    time.sleep(30)
            log = root / "logs" / (directory.name + ".log")
            invocation = [*command, "--run-dir", str(directory)]
            with log.open("a") as stream:
                child = subprocess.Popen(invocation, cwd=REPO, env=environment,
                                         stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            status("running", child_pid=child.pid, run_dir=str(directory), log=str(log), command=invocation)
            while child.poll() is None:
                time.sleep(15)
                status("running", gpu_snapshot=gpu_snapshot(gpus))
            code = child.wait()
            if code == 0:
                state["completed"][name] = {"status": "completed", "run_dir": str(directory), "finished_utc": utc()}
                status("job_completed", exit_code=0)
                break
            failure = log.read_text(errors="replace")
            if "out of memory" in failure.lower() or "cuda error" in failure.lower():
                state["attempts"][name] = attempt + 1
                status("retry_after_cuda_failure", exit_code=code, failed_run_dir=str(directory))
                time.sleep(60)
                continue
            status("failed", exit_code=code, failed_run_dir=str(directory))
            raise RuntimeError(f"{name} failed; see {log}")
    status("completed", current_job=None)


if __name__ == "__main__":
    main()
