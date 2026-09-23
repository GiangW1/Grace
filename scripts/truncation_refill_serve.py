#!/usr/bin/env python3
"""Frozen-actor sparse truncation and refill experiment through vllm serve.

Every arm has the same predetermined starts and request-level random plan.
The server owns GPU scheduling; this client chooses when a prefix, selected
continuation, or new start is submitted.  No actor update occurs in this probe.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import statistics
import sys
from threading import Event, Lock, Thread
from time import perf_counter
from urllib.request import Request, urlopen

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grace_gc.core.rng import IsolatedRNG
from grace_gc.data.math_data import load_math_records, select_records, selection_manifest
from grace_gc.data.tokenize import collect_stop_token_ids, encode_records_hf, load_hf_tokenizer
from grace_gc.logging_util.run_dir import RunDirectory, default_run_dir, resolve_run_dir, utc_now
from grace_gc.trainer.loop import build_run_config
from grace_gc.versions import collect_environment, sha256_file, sha256_mapping
from scripts.scheduler_probe_serve import VLLMServeClient, _decode_response, _request_row


@dataclass(frozen=True)
class Arm:
    name: str
    checkpoints: tuple[int, ...]
    scheduler: str = "stream"
    p_first: float = 1.0
    p_second: float = 1.0
    learned: bool = False


@dataclass
class PendingStart:
    index: int
    problem_id: str
    prompt_ids: list[int]
    token_ids: list[int]
    record: dict
    requests: list[dict]
    settled: bool
    logical_started: float


def make_trial_plan(n_starts: int, seed: int) -> dict:
    """The arm and probability never affect token seeds or selection uniforms."""
    rng = IsolatedRNG.create(int(seed))
    return {
        "seed": int(seed),
        "prefix_seeds": [int(x) for x in rng.integers("token", 0, 2**31 - 1, n_starts)],
        "continuation_seeds": rng.integers("continuation", 0, 2**31 - 1,
                                            (n_starts, 2)).astype(int).tolist(),
        "selection_uniforms": rng.random("selection", (n_starts, 2)).tolist(),
    }


def matched_second_probability(lengths: list[int], first: int, second: int,
                               single_p: float, two_first_p: float) -> dict:
    """Match expected generated tokens using independent full trajectories."""
    if not lengths or not 0 < single_p <= 1 or not 0 < two_first_p <= 1 or first >= second:
        raise ValueError("nonempty lengths, positive probabilities, and first < second are required")
    lengths = np.asarray(lengths, dtype=np.float64)
    middle = float(np.maximum(np.minimum(lengths, second) - first, 0).mean())
    tail = float(np.maximum(lengths - second, 0).mean())
    if tail <= 0:
        raise ValueError("calibration has no answers beyond the second checkpoint")
    p_second = (single_p * (middle + tail) - two_first_p * middle) / (two_first_p * tail)
    if not 0 < p_second <= 1:
        raise ValueError(f"matched p_second={p_second:.4g} is outside (0,1]; change p_two_first")
    return {"p_second": p_second, "middle_tokens": middle, "tail_tokens": tail,
            "n_full": int(len(lengths)), "single_p": single_p, "two_first_p": two_first_p}


def calibration_lengths(path: str) -> list[int]:
    lengths = []
    with Path(path).open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if row.get("arm") not in {None, "full_pg"} or row.get("workload") not in {None, "full_pg"}:
                continue
            full = row.get("full_token_ids")
            prompt = row.get("prompt_token_ids")
            if full is not None and prompt is not None:
                lengths.append(len(full) - len(prompt))
    if not lengths:
        raise ValueError("calibration records contain no full_pg token trajectories")
    return lengths


class FrozenPredictor:
    """The same HF feature and predictor heads as GPU training, serialized on GPU."""

    def __init__(self, model_path: str, actor_checkpoint: str, artifacts: dict[int, str],
                 tokenizer, eos_ids, *, lambdas: dict[int, float], uniform_p: dict[int, float],
                 p_min: float, uniform_shrink: float):
        from grace_gc.backends.hf_actor import named_lora_params
        from grace_gc.backends.verl_trainer import load_lora_actor
        from grace_gc.predictor.heads import predictor_from_spec
        from grace_gc.predictor.offline import read_predictor
        from grace_gc.trainer.baseline import HistoricalBaseline
        from grace_gc.trainer.checkpoint import load_checkpoint
        from grace_gc.trainer.state_io import load_numpy_module_state

        payload = load_checkpoint(actor_checkpoint)
        actor_sha = sha256_mapping(payload.get("actor"))
        self.actor = load_lora_actor(model_path, (payload.get("run_config") or {}).get("lora") or {})
        load_numpy_module_state(named_lora_params(self.actor), payload.get("actor") or {})
        self.actor.eval()
        self.pad_id = int(tokenizer.pad_token_id)
        self.eos_ids = eos_ids
        self.baseline = HistoricalBaseline()
        if payload.get("baseline"):
            self.baseline.load_state_dict(payload["baseline"])
        self.models = {}
        self.feature_modes = {}
        self.artifact_hashes = {}
        self.lambdas = lambdas
        self.uniform_p = uniform_p
        self.p_min = p_min
        self.uniform_shrink = uniform_shrink
        self.lock = Lock()
        for checkpoint, path in artifacts.items():
            body = read_predictor(path)
            protocol = body["protocol"]
            if (int(protocol["decision_tokens"]) != checkpoint
                    or int(protocol["max_new_tokens"]) != 2048):
                raise ValueError(f"predictor {path} was not trained for {checkpoint}/2048")
            if body["source"].get("actor_sha256") != actor_sha:
                raise ValueError(f"predictor {path} and frozen actor checkpoint differ")
            if int(body["basis_id"]) != int(body["predictor_synced_basis_id"]):
                raise ValueError(f"predictor {path} is not synchronized to its basis")
            raw = body["predictor"]
            model = predictor_from_spec(int(raw["in_dim"]), int(raw["k"]), raw)
            model.load_state_dict(raw)
            self.models[checkpoint] = model
            self.feature_modes[checkpoint] = str(protocol["feature_mode"])
            self.artifact_hashes[checkpoint] = sha256_file(path)

    def decide(self, checkpoint: int, token_ids: list[int], prompt_len: int,
               problem_id: str) -> dict:
        from grace_gc.backends.hf_actor import prefix_feature_bundle

        if checkpoint not in self.models:
            raise ValueError(f"no predictor for checkpoint {checkpoint}")
        with self.lock:
            started = perf_counter()
            bundle = prefix_feature_bundle(
                self.actor, [token_ids], [prompt_len], [self.baseline.get(problem_id)],
                self.pad_id, eos_id=self.eos_ids,
                feature_mode=self.feature_modes[checkpoint], batch_size=1,
            )
            feature_seconds = perf_counter() - started
            started = perf_counter()
            model = self.models[checkpoint]
            output = model.forward_numpy(bundle["features"], bundle["cost_feat"])
            risk = float(output.r_hat[0])
            cost = (max(2048 - checkpoint, 1) if model.constant_cost
                    else max(float(output.c_hat[0]), 1e-8))
            if not math.isfinite(risk) or not math.isfinite(cost):
                raise ValueError("predictor returned non-finite risk or cost")
            risk = max(risk, 0.0)
            adaptive = math.sqrt(risk / (self.lambdas[checkpoint] * cost))
            adaptive = min(1.0, max(self.p_min, adaptive))
            p = (1 - self.uniform_shrink) * adaptive + self.uniform_shrink * self.uniform_p[checkpoint]
            p = min(1.0, max(self.p_min, p))
            prediction_seconds = perf_counter() - started
        return {"p": float(p), "risk": risk, "cost": cost,
                "f": output.f[0].tolist(), "feature_seconds": feature_seconds,
                "prediction_seconds": prediction_seconds}


def _stage(client, pending: PendingStart, *, limit: int, stage: int, plan: dict,
           eos_ids, trial_started: float, cache_salt: str) -> None:
    generated = len(pending.token_ids) - len(pending.prompt_ids)
    remaining = int(limit) - generated
    if remaining <= 0:
        raise ValueError("stage limit must exceed previously generated tokens")
    seed = (plan["prefix_seeds"][pending.index] if stage == 0 else
            plan["continuation_seeds"][pending.index][stage - 1])
    request_id = f"grace-trunc-{plan['seed']}-{pending.index}-{stage}"
    response = client.complete(pending.token_ids, remaining, seed, eos_ids,
                               request_id, cache_salt)
    full, ended = _decode_response(response, pending.token_ids, eos_ids)
    row = _request_row(response, f"stage_{stage}", trial_started, len(pending.token_ids))
    row["start_index"] = pending.index
    pending.requests.append(row)
    pending.token_ids = full
    pending.record["stage_tokens"].append(len(full) - len(response.prompt_token_ids))
    pending.record["request_count"] += 1
    pending.record["generated_tokens"] = len(full) - len(pending.prompt_ids)
    pending.record["finish_reason"] = response.finish_reason
    pending.record["stop_reason"] = response.stop_reason
    pending.settled = bool(ended or pending.record["generated_tokens"] >= pending.record["max_new_tokens"])
    if pending.settled:
        pending.record["stage"] = "complete"
        pending.record["natural_finish"] = bool(ended)
        pending.record["truncated"] = not bool(ended)
        pending.record["full_token_ids"] = list(full)
        pending.record["settled_offset_seconds"] = response.returned_at - trial_started


def _begin_start(client, start: dict, index: int, arm: Arm, plan: dict, *, max_new: int,
                 eos_ids, trial_started: float, cache_salt: str) -> PendingStart:
    prompt = [int(x) for x in start["prompt_token_ids"]]
    pending = PendingStart(
        index=index, problem_id=str(start["problem_id"]), prompt_ids=prompt,
        token_ids=list(prompt), record={
            "start_index": index, "problem_id": str(start["problem_id"]),
            "prompt_token_ids": prompt, "prefix_token_ids": None, "full_token_ids": None,
            "stage": "running", "stage_tokens": [], "decisions": [], "request_count": 0,
            "generated_tokens": 0, "natural_finish": False, "truncated": False,
            "max_new_tokens": max_new, "inclusion_probability": 1.0,
        }, requests=[], settled=False, logical_started=perf_counter(),
    )
    first = arm.checkpoints[0] if arm.checkpoints else max_new
    _stage(client, pending, limit=first, stage=0, plan=plan, eos_ids=eos_ids,
           trial_started=trial_started, cache_salt=cache_salt)
    pending.record["prefix_token_ids"] = list(pending.token_ids)
    return pending


def _finish_start(client, pending: PendingStart, arm: Arm, plan: dict, *, max_new: int,
                  eos_ids, trial_started: float, cache_salt: str,
                  predictor: FrozenPredictor | None) -> PendingStart:
    for position, checkpoint in enumerate(arm.checkpoints):
        if pending.settled:
            break
        started = perf_counter()
        if arm.learned:
            if predictor is None:
                raise ValueError("learned arm needs a predictor")
            decision = predictor.decide(checkpoint, pending.token_ids, len(pending.prompt_ids),
                                        pending.problem_id)
            p = float(decision.pop("p"))
        else:
            p = arm.p_first if position == 0 else arm.p_second
            decision = {"feature_seconds": 0.0, "prediction_seconds": 0.0}
        if not 0 < p <= 1 or not math.isfinite(p):
            raise ValueError("continuation probability must be finite and in (0, 1]")
        selected = bool(plan["selection_uniforms"][pending.index][position] < p)
        pending.record["inclusion_probability"] *= p
        pending.record["decisions"].append({
            "checkpoint": checkpoint, "p": p, "selected": selected,
            "uniform": plan["selection_uniforms"][pending.index][position],
            "decision_seconds": perf_counter() - started, **decision,
        })
        if not selected:
            pending.record["stage"] = "stopped"
            break
        next_limit = (arm.checkpoints[position + 1] if position + 1 < len(arm.checkpoints)
                      else max_new)
        _stage(client, pending, limit=next_limit, stage=position + 1, plan=plan,
               eos_ids=eos_ids, trial_started=trial_started, cache_salt=cache_salt)
    if pending.record["stage"] == "running":
        if not pending.settled:
            raise ValueError("start exited without a stop or full completion")
        pending.record["stage"] = "complete"
    pending.record.setdefault("settled_offset_seconds", perf_counter() - trial_started)
    pending.record["logical_seconds"] = (
        pending.record["settled_offset_seconds"] - (pending.logical_started - trial_started)
    )
    return pending


def _sample_server(client, trial_started: float, interval: float,
                   stop: Event, samples: list[dict]) -> None:
    url = client.base_url.removesuffix("/v1").rstrip("/") + "/metrics"
    headers = {"Authorization": f"Bearer {client.api_key}"} if client.api_key else {}
    names = {"vllm:num_requests_running", "vllm:num_requests_waiting",
             "vllm:kv_cache_usage_perc", "vllm:num_preemptions"}
    while not stop.is_set():
        try:
            with urlopen(Request(url, headers=headers), timeout=2) as response:
                raw = response.read().decode("utf-8", errors="replace")
            values = {}
            for line in raw.splitlines():
                if not line or line.startswith("#"):
                    continue
                name = line.split("{", 1)[0].split(" ", 1)[0]
                if name in names:
                    values[name] = values.get(name, 0.0) + float(line.split()[-1])
            samples.append({"offset_seconds": perf_counter() - trial_started, **values})
        except (OSError, ValueError) as exc:
            samples.append({"offset_seconds": perf_counter() - trial_started,
                            "metrics_error": str(exc)[:300]})
            return
        stop.wait(interval)


def run_trial(client, starts: list[dict], arm: Arm, plan: dict, *, capacity: int,
              max_new: int, eos_ids, cache_salt: str,
              predictor: FrozenPredictor | None = None,
              metrics_interval: float = 0.0) -> tuple[dict, list[dict], list[dict], list[dict]]:
    if not starts or capacity <= 0:
        raise ValueError("starts and capacity must be positive")
    trial_started = perf_counter()
    results: dict[int, PendingStart] = {}
    samples: list[dict] = []
    metrics_stop = Event()
    metrics_thread = None
    if metrics_interval > 0 and hasattr(client, "base_url"):
        metrics_thread = Thread(target=_sample_server,
                                args=(client, trial_started, metrics_interval,
                                      metrics_stop, samples), daemon=True)
        metrics_thread.start()

    def begin(index):
        return _begin_start(client, starts[index], index, arm, plan, max_new=max_new,
                            eos_ids=eos_ids, trial_started=trial_started, cache_salt=cache_salt)

    def finish(pending):
        return _finish_start(client, pending, arm, plan, max_new=max_new,
                             eos_ids=eos_ids, trial_started=trial_started,
                             cache_salt=cache_salt, predictor=predictor)

    try:
        with ThreadPoolExecutor(max_workers=min(capacity, len(starts)),
                                thread_name_prefix="grace-trunc") as executor:
            if arm.scheduler == "barrier":
                # Same in-flight capacity as stream, but no decision or suffix is
                # admitted before every predetermined prefix settles.
                prefixes = [executor.submit(begin, i) for i in range(len(starts))]
                pending = [future.result() for future in prefixes]
                futures = [executor.submit(finish, row) for row in pending]
                results = {i: future.result() for i, future in enumerate(futures)}
            elif arm.scheduler == "stream":
                next_index = min(capacity, len(starts))
                active = {executor.submit(lambda i=i: finish(begin(i))): i
                          for i in range(next_index)}
                while active:
                    completed, _ = wait(tuple(active), return_when=FIRST_COMPLETED)
                    for future in sorted(completed, key=lambda f: active[f]):
                        index = active.pop(future)
                        results[index] = future.result()
                        if next_index < len(starts):
                            i = next_index
                            active[executor.submit(lambda i=i: finish(begin(i)))] = i
                            next_index += 1
            else:
                raise ValueError(f"unknown scheduler {arm.scheduler!r}")
    finally:
        metrics_stop.set()
        if metrics_thread is not None:
            metrics_thread.join(timeout=3)
    wall = perf_counter() - trial_started
    ordered = [results[i].record for i in range(len(starts))]
    requests = [row for i in range(len(starts)) for row in results[i].requests]
    settled = sorted(row["settled_offset_seconds"] for row in ordered)
    p90_index = max(math.ceil(0.9 * len(settled)) - 1, 0)
    summary = {
        "arm": arm.name, "scheduler": arm.scheduler, "n_starts": len(starts),
        "client_capacity": min(capacity, len(starts)), "wall_seconds": wall,
        "tail_after_90_percent_seconds": wall - settled[p90_index],
        "generated_tokens": sum(row["generated_tokens"] for row in ordered),
        "n_http_requests": len(requests),
        "stopped": sum(row["stage"] == "stopped" for row in ordered),
        "completed": sum(row["stage"] == "complete" for row in ordered),
        "hit_max": sum(row["truncated"] for row in ordered),
        "n_decisions": sum(len(row["decisions"]) for row in ordered),
        "feature_seconds": sum(d["feature_seconds"] for row in ordered for d in row["decisions"]),
        "prediction_seconds": sum(d["prediction_seconds"] for row in ordered for d in row["decisions"]),
        "n_server_metric_samples": len(samples),
        "scope": "one frozen-actor vllm serve rollout; startup and predictor loading excluded",
    }
    return summary, ordered, requests, samples


def _arms(names: list[str], first: int, second: int, p_single: float,
          p_two_first: float, p_two_second: float) -> list[Arm]:
    catalog = {
        "full_pg": Arm("full_pg", ()),
        "single_keep": Arm("single_keep", (first,)),
        "two_keep": Arm("two_keep", (first, second)),
        "single_barrier": Arm("single_barrier", (first,), "barrier", p_single),
        "single_stream": Arm("single_stream", (first,), "stream", p_single),
        "two_stream": Arm("two_stream", (first, second), "stream", p_two_first, p_two_second),
        "learned_single": Arm("learned_single", (first,), learned=True),
        "learned_two": Arm("learned_two", (first, second), learned=True),
    }
    unknown = set(names) - set(catalog)
    if unknown:
        raise ValueError(f"unknown arms: {sorted(unknown)}")
    return [catalog[name] for name in names]


def summarize_trials(trials: list[dict]) -> dict:
    grouped = {}
    by_repeat = {}
    for row in trials:
        grouped.setdefault(row["arm"], []).append(row)
        by_repeat.setdefault(row["repeat"], {})[row["arm"]] = row
    metrics = ("wall_seconds", "generated_tokens", "tail_after_90_percent_seconds",
               "feature_seconds", "prediction_seconds", "stopped", "hit_max")
    aggregate = {}
    for name, rows in grouped.items():
        aggregate[name] = {"n_repeats": len(rows)}
        for key in metrics:
            values = [float(row[key]) for row in rows]
            aggregate[name][f"{key}_mean"] = statistics.mean(values)
            aggregate[name][f"{key}_sd"] = statistics.stdev(values) if len(values) > 1 else 0.0
    pairs = (("full_pg", "single_stream"), ("full_pg", "two_stream"),
             ("single_barrier", "single_stream"), ("single_stream", "two_stream"),
             ("single_keep", "single_stream"), ("two_keep", "two_stream"),
             ("single_stream", "learned_single"), ("two_stream", "learned_two"))
    comparisons = []
    for repeat, row in sorted(by_repeat.items()):
        for left, right in pairs:
            if left in row and right in row:
                a, b = row[left], row[right]
                comparisons.append({"repeat": repeat, "left": left, "right": right,
                                    "wall_ratio_right_over_left": b["wall_seconds"] / a["wall_seconds"],
                                    "token_ratio_right_over_left": (
                                        b["generated_tokens"] / a["generated_tokens"]
                                        if a["generated_tokens"] else None)})
    return {"by_arm": aggregate, "comparisons": comparisons}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", default=[])
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--server-model", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key-env", default="VLLM_API_KEY")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--n-problems", type=int, default=64)
    parser.add_argument("--starts-per-problem", type=int, default=8)
    parser.add_argument("--selection", choices=("first", "seeded"), default="seeded")
    parser.add_argument("--selection-seed", type=int, default=17)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--capacity", type=int, default=128)
    parser.add_argument("--first", type=int, default=512)
    parser.add_argument("--second", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--metrics-interval", type=float, default=1.0,
                        help="seconds between optional /metrics reads; 0 disables them")
    parser.add_argument("--arms", default="full_pg,single_keep,two_keep,single_barrier,single_stream,two_stream")
    parser.add_argument("--p-single", type=float, default=0.5)
    parser.add_argument("--p-two-first", type=float, default=0.65)
    parser.add_argument("--p-two-second", type=float, default=0.5)
    parser.add_argument("--calibration-records", default=None)
    parser.add_argument("--actor-checkpoint", default=None)
    parser.add_argument("--predictor-first", default=None)
    parser.add_argument("--predictor-second", default=None)
    parser.add_argument("--lambda-first", type=float, default=1.0)
    parser.add_argument("--lambda-second", type=float, default=1.0)
    parser.add_argument("--p-min", type=float, default=0.2)
    parser.add_argument("--uniform-shrink", type=float, default=0.5)
    args = parser.parse_args(argv)
    if (not 0 < args.first < args.second < args.max_new_tokens or args.capacity <= 0
            or args.n_problems <= 0 or args.starts_per_problem <= 0 or args.repeats <= 0
            or args.request_timeout <= 0 or not math.isfinite(args.metrics_interval)
            or args.metrics_interval < 0 or not math.isfinite(args.temperature)
            or not 0 < args.temperature <= 1
            or any(not 0 < p <= 1 for p in (args.p_single, args.p_two_first, args.p_two_second,
                                             args.p_min))
            or any(not math.isfinite(x) or x <= 0 for x in (args.lambda_first, args.lambda_second))
            or not 0 <= args.uniform_shrink <= 1):
        raise ValueError("invalid checkpoints, capacity, sampling, or allocation options")
    names = [part.strip() for part in args.arms.split(",") if part.strip()]
    if not names or len(names) != len(set(names)):
        raise ValueError("--arms must list distinct nonempty names")
    if any(not math.isfinite(p) for p in (args.p_single, args.p_two_first,
                                          args.p_two_second, args.p_min)):
        raise ValueError("continuation probabilities must be finite")
    calibration = None
    if args.calibration_records:
        calibration = matched_second_probability(
            calibration_lengths(args.calibration_records), args.first, args.second,
            args.p_single, args.p_two_first,
        )
        args.p_two_second = float(calibration["p_second"])
        calibration["source"] = str(args.calibration_records)
        calibration["sha256"] = sha256_file(args.calibration_records)
    arms = _arms(names, args.first, args.second, args.p_single,
                 args.p_two_first, args.p_two_second)
    if any(arm.learned for arm in arms) and (not args.actor_checkpoint or not args.predictor_first):
        raise ValueError("learned arms require --actor-checkpoint and --predictor-first")
    if "learned_two" in names and not args.predictor_second:
        raise ValueError("learned_two requires an independently fitted --predictor-second")
    if any(arm.learned for arm in arms) and args.max_new_tokens != 2048:
        raise ValueError("predictor artifacts in this experiment require max_new_tokens=2048")

    cfg = build_run_config(args.config, {
        "backend": "vllm_serve", "model_path": args.model_path, "seed": args.seed,
        "decision_tokens": args.first, "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "hardware": {"name": "a100", "n_gpu": 1, "gpu_memory_gb": 80},
    })
    requested = args.run_dir or str(default_run_dir("truncation-refill-serve"))
    run = RunDirectory(resolve_run_dir(requested))
    run.write_run_meta(kind="truncation_refill_serve", started=utc_now(), requested=requested)
    run.write_yaml("config.yaml", {**cfg, "truncation_refill_serve": {
        **vars(args), "arms": names, "p_two_second_effective": args.p_two_second,
        "cost_match_calibration": calibration,
    }})
    run.write_json("environment.json", collect_environment(cfg))
    records = select_records(load_math_records(args.data_path), args.n_problems,
                             args.selection, args.selection_seed)
    if not records:
        raise ValueError("selected data is empty")
    if len(records) != args.n_problems:
        raise ValueError(f"requested {args.n_problems} problems, found {len(records)}")
    starts = [record for record in records for _ in range(args.starts_per_problem)]
    run.write_json("workload.json", {
        "n_starts": len(starts), "starts_per_problem": args.starts_per_problem,
        **selection_manifest(records, args.selection, args.selection_seed),
    })
    tokenizer = load_hf_tokenizer(args.model_path)
    prompt_ids, problem_ids, _golds = encode_records_hf(
        starts, tokenizer, int(cfg.get("prompt_max_tokens", 1024)),
    )
    starts_payload = [{"problem_id": problem_ids[i], "prompt_token_ids": prompt_ids[i]}
                      for i in range(len(prompt_ids))]
    run.write_jsonl("prompt_manifest.jsonl", [
        {"start_index": i, "problem_id": row["problem_id"],
         "prompt_tokens": len(row["prompt_token_ids"])}
        for i, row in enumerate(starts_payload)
    ])
    eos_ids = collect_stop_token_ids(tokenizer)
    client = VLLMServeClient(args.base_url, args.server_model,
                            api_key=os.environ.get(args.api_key_env),
                            timeout_seconds=args.request_timeout,
                            temperature=args.temperature)
    client.health()
    served_models = client.models()
    if served_models and args.server_model not in served_models:
        raise ValueError(f"server model {args.server_model!r} not in /v1/models: {served_models}")

    predictor = None
    preparation_seconds = 0.0
    if any(arm.learned for arm in arms):
        started = perf_counter()
        artifacts = {args.first: args.predictor_first}
        if args.predictor_second:
            artifacts[args.second] = args.predictor_second
        predictor = FrozenPredictor(
            args.model_path, args.actor_checkpoint, artifacts, tokenizer, eos_ids,
            lambdas={args.first: args.lambda_first, args.second: args.lambda_second},
            uniform_p={args.first: args.p_single, args.second: args.p_two_second},
            p_min=args.p_min, uniform_shrink=args.uniform_shrink,
        )
        preparation_seconds = perf_counter() - started
    run.write_json("preparation.json", {
        "predictor_loading_seconds": preparation_seconds,
        "actor_checkpoint_sha256": sha256_file(args.actor_checkpoint) if args.actor_checkpoint else None,
        "predictor_artifact_hashes": predictor.artifact_hashes if predictor else {},
        "served_models": served_models,
    })

    summaries = []
    for repeat in range(args.repeats):
        plan = make_trial_plan(len(starts_payload), args.seed + 100003 * repeat)
        run.append_jsonl("plans.jsonl", {"repeat": repeat, **plan})
        ordered = arms if repeat % 2 == 0 else list(reversed(arms))
        for arm in ordered:
            salt = f"grace-trunc-{plan['seed']}-{arm.name}"
            summary, rows, requests, samples = run_trial(
                client, starts_payload, arm, plan, capacity=args.capacity,
                max_new=args.max_new_tokens, eos_ids=eos_ids,
                cache_salt=salt, predictor=predictor,
                metrics_interval=args.metrics_interval,
            )
            summary.update({"repeat": repeat, "trial_seed": plan["seed"],
                            "cache_salt": salt, "p_first": arm.p_first,
                            "p_second": arm.p_second, "learned": arm.learned})
            summaries.append(summary)
            run.append_jsonl("trials.jsonl", summary)
            run.write_jsonl("records.jsonl", [
                {**row, "arm": arm.name, "repeat": repeat} for row in rows
            ], append=True)
            run.write_jsonl("requests.jsonl", [
                {**row, "arm": arm.name, "repeat": repeat} for row in requests
            ], append=True)
            run.write_jsonl("server_metrics.jsonl", [
                {**row, "arm": arm.name, "repeat": repeat} for row in samples
            ], append=True)
            print(summary, flush=True)
    run.write_json("summary.json", {"trials": summaries, "n_starts": len(starts_payload),
                                    "cost_match_calibration": calibration,
                                    **summarize_trials(summaries)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
