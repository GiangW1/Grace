#!/usr/bin/env python3
"""Frozen-actor scheduler probe through one vLLM OpenAI-compatible server.

This is an independent serving-path experiment.  It keeps the workload and
per-start random plan fixed, then compares client admission policies while
the same ``vllm serve`` process handles every request:

* ``no_refill`` waits for a client batch of logical starts to settle;
* ``refill`` submits the next logical start as soon as one settles;
* ``all`` submits every logical start to the server-side queue as a production
  baseline.

The prefix and selected suffix of one logical start are always submitted by
the same worker.  A suffix therefore does not consume a new refill slot; only
an early-stopped or fully completed logical start releases a slot.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
import http.client
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grace_gc.backends.vllm_continuous import (
    ContinuousRecord,
    make_request_plan,
    record_to_dict,
)
from grace_gc.backends.vllm_two_phase import (
    _finish_name,
    _is_length_finish,
    _require_known_finish,
    trim_generated_tokens,
)
from grace_gc.data.math_data import load_math_records, select_records
from grace_gc.data.tokenize import collect_stop_token_ids, encode_records_hf, load_hf_tokenizer
from grace_gc.logging_util.run_dir import RunDirectory, default_run_dir, resolve_run_dir
from grace_gc.trainer.loop import build_run_config
from grace_gc.versions import collect_environment


@dataclass
class ServeResponse:
    request_id: str
    prompt_token_ids: list[int]
    token_ids: list[int]
    finish_reason: str | None
    stop_reason: int | str | None
    usage: dict
    wall_seconds: float
    submitted_at: float
    returned_at: float


@dataclass
class StartResult:
    record: ContinuousRecord
    request_rows: list[dict]
    submitted_at: float
    settled_at: float


class VLLMServeClient:
    """Small stdlib HTTP client with one keep-alive connection per worker."""

    def __init__(self, base_url: str, model: str, *, api_key: str | None = None,
                 timeout_seconds: float = 600.0, temperature: float = 0.2):
        parsed = urlsplit(str(base_url).rstrip("/"))
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("base_url must be an http(s) URL")
        self.base_url = str(base_url).rstrip("/")
        self.model = str(model)
        self.api_key = api_key
        self.timeout_seconds = float(timeout_seconds)
        self.temperature = float(temperature)
        self._scheme = parsed.scheme
        self._host = parsed.hostname
        self._port = parsed.port
        prefix = parsed.path.rstrip("/")
        self._prefix = prefix if prefix.endswith("/v1") else f"{prefix}/v1" if prefix else "/v1"
        self._local = __import__("threading").local()

    def _connection(self):
        conn = getattr(self._local, "connection", None)
        if conn is not None:
            return conn
        cls = http.client.HTTPSConnection if self._scheme == "https" else http.client.HTTPConnection
        conn = cls(self._host, self._port, timeout=self.timeout_seconds)
        self._local.connection = conn
        return conn

    def _reset_connection(self):
        conn = getattr(self._local, "connection", None)
        if conn is not None:
            try:
                conn.close()
            except OSError:
                pass
        self._local.connection = None

    def _path(self, endpoint: str) -> str:
        return f"{self._prefix}/{endpoint.lstrip('/')}"

    def _request(self, method: str, endpoint: str, payload: dict | None = None) -> dict:
        body = None if payload is None else json.dumps(payload, separators=(",", ":")).encode()
        headers = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        last_error = None
        for attempt in range(2):
            try:
                conn = self._connection()
                conn.request(method, endpoint, body=body, headers=headers)
                response = conn.getresponse()
                raw = response.read()
                if response.status >= 400:
                    detail = raw.decode("utf-8", errors="replace")[:2000]
                    self._reset_connection()
                    raise RuntimeError(
                        f"vLLM server returned HTTP {response.status} for {endpoint}: {detail}"
                    )
                if not raw:
                    return {}
                return json.loads(raw.decode("utf-8"))
            except (ConnectionError, OSError, http.client.HTTPException) as exc:
                last_error = exc
                self._reset_connection()
                if attempt == 1:
                    raise RuntimeError(f"vLLM request failed for {endpoint}: {exc}") from exc
        raise RuntimeError(f"vLLM request failed for {endpoint}: {last_error}")

    def health(self) -> dict:
        return self._request("GET", "/health")

    def models(self) -> list[str]:
        payload = self._request("GET", self._path("models"))
        return [str(row.get("id")) for row in payload.get("data", []) if row.get("id")]

    def complete(self, prompt_token_ids: list[int], max_tokens: int, seed: int,
                 stop_token_ids: list[int], request_id: str, cache_salt: str) -> ServeResponse:
        started = perf_counter()
        payload = {
            "model": self.model,
            "prompt": [int(x) for x in prompt_token_ids],
            "max_tokens": int(max_tokens),
            "temperature": self.temperature,
            "top_p": 1.0,
            "top_k": -1,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "n": 1,
            "stop": [],
            "stop_token_ids": [int(x) for x in stop_token_ids],
            "seed": int(seed),
            "stream": False,
            "return_token_ids": True,
            # Prompt IDs already include the tokenizer's intended special
            # tokens; do not let the OpenAI renderer add another BOS token.
            "add_special_tokens": False,
            "request_id": str(request_id),
            # Keep one arm's prefix cache reusable while preventing an arm
            # from warming the cache of the next arm.
            "cache_salt": str(cache_salt),
        }
        response = self._request("POST", self._path("completions"), payload)
        returned = perf_counter()
        choices = response.get("choices") or []
        if not choices:
            raise ValueError(f"vLLM response for {request_id} has no choices")
        choice = choices[0]
        returned_prompt = choice.get("prompt_token_ids")
        if returned_prompt is None:
            raise ValueError(
                f"vLLM response for {request_id} omitted prompt_token_ids; "
                "serve probe requires token-order verification"
            )
        returned_prompt = [int(x) for x in returned_prompt]
        expected_prompt = [int(x) for x in prompt_token_ids]
        if returned_prompt != expected_prompt:
            raise ValueError(f"vLLM returned a prompt mismatch for {request_id}")
        token_ids = choice.get("token_ids")
        if token_ids is None:
            raise ValueError(
                f"vLLM response for {request_id} omitted token_ids; "
                "start vLLM with a version supporting return_token_ids"
            )
        return ServeResponse(
            request_id=str(request_id),
            prompt_token_ids=returned_prompt,
            token_ids=[int(x) for x in token_ids],
            finish_reason=_finish_name(choice.get("finish_reason")),
            stop_reason=choice.get("stop_reason"),
            usage=dict(response.get("usage") or {}),
            wall_seconds=float(returned - started),
            submitted_at=float(started),
            returned_at=float(returned),
        )


def _request_row(response: ServeResponse, stage: str, trial_started: float,
                 prompt_len: int) -> dict:
    usage = response.usage
    return {
        "request_id": response.request_id,
        "stage": str(stage),
        "prompt_tokens": int(prompt_len),
        "generated_tokens": len(response.token_ids),
        "finish_reason": response.finish_reason,
        "stop_reason": response.stop_reason,
        "usage": usage,
        "wall_seconds": float(response.wall_seconds),
        "submitted_offset_seconds": float(response.submitted_at - trial_started),
        "returned_offset_seconds": float(response.returned_at - trial_started),
    }


def _decode_response(response: ServeResponse, prompt_token_ids: list[int], eos_id) -> tuple[list[int], bool]:
    _require_known_finish(response.finish_reason)
    generated, ended = trim_generated_tokens(
        response.token_ids, eos_id, response.finish_reason, response.stop_reason,
    )
    return [int(x) for x in prompt_token_ids] + [int(x) for x in generated], bool(ended)


def _run_logical_start(client: VLLMServeClient, start: dict, index: int, *, mode: str,
                       decision_tokens: int, max_new_tokens: int, p: float,
                       request_plan: dict, eos_id, trial_started: float,
                       cache_salt: str) -> StartResult:
    record = ContinuousRecord(
        start_index=int(index), problem_id=str(start["problem_id"]),
        prompt_token_ids=[int(x) for x in start["prompt_token_ids"]],
    )
    request_rows = []
    logical_started = perf_counter()
    prefix_request_id = f"grace-serve-{request_plan['seed']}-{index}-prefix"
    prefix_limit = int(max_new_tokens if mode == "full_pg" else decision_tokens)
    prefix_response = client.complete(
        record.prompt_token_ids, prefix_limit, int(request_plan["prefix_seeds"][index]),
        list(eos_id if isinstance(eos_id, (list, tuple, set)) else [eos_id]),
        prefix_request_id, cache_salt,
    )
    request_rows.append(_request_row(prefix_response, "full" if mode == "full_pg" else "prefix",
                                     trial_started, len(record.prompt_token_ids)))
    full, ended = _decode_response(prefix_response, record.prompt_token_ids, eos_id)
    generated_prefix = max(len(full) - len(record.prompt_token_ids), 0)
    record.prefix_request_id = prefix_request_id
    record.request_count = 1
    record.prefix_token_ids = list(full)
    record.prefix_tokens = int(generated_prefix)
    record.prefix_wall_seconds = float(prefix_response.wall_seconds)

    if mode == "full_pg":
        record.full_token_ids = list(full)
        record.natural_finish = bool(ended)
        record.truncated = bool(_is_length_finish(prefix_response.finish_reason) and not ended)
        record.finish_reason = prefix_response.finish_reason
        record.stop_reason = prefix_response.stop_reason
        record.stage = "complete"
        return StartResult(record, request_rows, logical_started, perf_counter())

    if ended:
        record.full_token_ids = list(full)
        record.natural_finish = True
        record.finish_reason = prefix_response.finish_reason
        record.stop_reason = prefix_response.stop_reason
        record.stage = "complete"
        return StartResult(record, request_rows, logical_started, perf_counter())
    if generated_prefix >= int(max_new_tokens):
        record.full_token_ids = list(full)
        record.truncated = True
        record.finish_reason = prefix_response.finish_reason
        record.stop_reason = prefix_response.stop_reason
        record.stage = "complete"
        return StartResult(record, request_rows, logical_started, perf_counter())
    if not _is_length_finish(prefix_response.finish_reason):
        raise ValueError(f"prefix request {prefix_request_id} ended with {prefix_response.finish_reason!r}")

    selected = bool(request_plan["selected"][index])
    record.selected = selected
    if not selected:
        record.full_token_ids = None
        record.stage = "stopped"
        record.finish_reason = prefix_response.finish_reason
        record.stop_reason = prefix_response.stop_reason
        return StartResult(record, request_rows, logical_started, perf_counter())

    remain = int(max_new_tokens) - int(generated_prefix)
    if remain <= 0:
        record.full_token_ids = list(full)
        record.truncated = True
        record.stage = "complete"
        return StartResult(record, request_rows, logical_started, perf_counter())
    suffix_request_id = f"grace-serve-{request_plan['seed']}-{index}-suffix"
    suffix_response = client.complete(
        full, remain, int(request_plan["suffix_seeds"][index]),
        list(eos_id if isinstance(eos_id, (list, tuple, set)) else [eos_id]),
        suffix_request_id, cache_salt,
    )
    request_rows.append(_request_row(suffix_response, "suffix", trial_started, len(full)))
    suffix_full, suffix_ended = _decode_response(suffix_response, full, eos_id)
    record.suffix_request_id = suffix_request_id
    record.request_count = 2
    record.full_token_ids = list(suffix_full)
    record.suffix_tokens = max(len(suffix_full) - len(full), 0)
    record.suffix_wall_seconds = float(suffix_response.wall_seconds)
    record.natural_finish = bool(suffix_ended)
    record.truncated = bool(_is_length_finish(suffix_response.finish_reason) and not suffix_ended)
    record.finish_reason = suffix_response.finish_reason
    record.stop_reason = suffix_response.stop_reason
    record.stage = "complete"
    return StartResult(record, request_rows, logical_started, perf_counter())


def _occupancy(intervals: list[tuple[float, float]], wall_seconds: float,
               capacity: int) -> tuple[float, float]:
    events = []
    for start, end in intervals:
        events.extend(((float(start), 1), (float(end), -1)))
    events.sort(key=lambda item: (item[0], item[1]))
    active = 0
    previous = 0.0
    area = 0.0
    for timestamp, delta in events:
        if timestamp > previous:
            area += active * (timestamp - previous)
        active += delta
        previous = timestamp
    area = max(0.0, min(area, float(capacity) * float(wall_seconds)))
    return area, max(0.0, float(capacity) * float(wall_seconds) - area)


def run_scheduler(client: VLLMServeClient, starts: list[dict], *, scheduler: str,
                  capacity: int, all_concurrency: int | None, mode: str,
                  decision_tokens: int, max_new_tokens: int, p: float,
                  request_plan: dict, eos_id, cache_salt: str) -> tuple[dict, list[ContinuousRecord], list[dict]]:
    scheduler = str(scheduler)
    if scheduler not in {"all", "no_refill", "refill"}:
        raise ValueError("scheduler must be all, no_refill, or refill")
    if not starts:
        raise ValueError("serve scheduler needs at least one start")
    capacity = max(1, int(capacity))
    trial_started = perf_counter()
    results: dict[int, StartResult] = {}
    request_rows: list[dict] = []
    intervals: list[tuple[float, float]] = []
    worker_capacity = len(starts) if scheduler == "all" else capacity
    if scheduler == "all" and all_concurrency is not None:
        worker_capacity = max(1, min(len(starts), int(all_concurrency)))

    def submit(executor, index):
        return executor.submit(
            _run_logical_start, client, starts[index], index, mode=mode,
            decision_tokens=decision_tokens, max_new_tokens=max_new_tokens, p=p,
            request_plan=request_plan, eos_id=eos_id, trial_started=trial_started,
            cache_salt=cache_salt,
        )

    with ThreadPoolExecutor(max_workers=worker_capacity, thread_name_prefix="grace-serve") as executor:
        if scheduler == "no_refill":
            for batch_start in range(0, len(starts), capacity):
                batch_indices = list(range(batch_start, min(len(starts), batch_start + capacity)))
                futures = [submit(executor, index) for index in batch_indices]
                for index, future in zip(batch_indices, futures):
                    results[index] = future.result()
        elif scheduler == "all":
            futures = {submit(executor, index): index for index in range(len(starts))}
            for future, index in futures.items():
                results[index] = future.result()
        else:
            next_index = 0
            pending = {}
            initial = min(len(starts), worker_capacity)
            for index in range(initial):
                pending[submit(executor, index)] = index
                next_index += 1
            while pending:
                done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
                for future in sorted(done, key=lambda item: pending[item]):
                    index = pending.pop(future)
                    results[index] = future.result()
                    if next_index < len(starts) and scheduler == "refill":
                        pending[submit(executor, next_index)] = next_index
                        next_index += 1

    for index in range(len(starts)):
        result = results[index]
        request_rows.extend(result.request_rows)
        intervals.append((result.submitted_at - trial_started, result.settled_at - trial_started))
    wall_seconds = perf_counter() - trial_started
    client_capacity = len(starts) if scheduler == "all" else capacity
    occupied, idle = _occupancy(intervals, wall_seconds, client_capacity)
    ordered = [results[index].record for index in range(len(starts))]
    summary = {
        "scheduler": scheduler,
        "client_capacity": int(client_capacity),
        "n_starts": len(ordered),
        "p": float(p),
        "decision_tokens": int(decision_tokens),
        "max_new_tokens": int(max_new_tokens),
        "wall_seconds": float(wall_seconds),
        "n_http_requests": int(len(request_rows)),
        "logical_occupancy_seconds": float(occupied),
        "idle_slot_seconds": float(idle),
        "max_active_logical": int(min(client_capacity, len(ordered))),
        "completed": int(sum(row.stage == "complete" for row in ordered)),
        "stopped": int(sum(row.stage == "stopped" for row in ordered)),
        "continued": len(ordered) if mode == "full_pg" else int(sum(row.selected is True for row in ordered)),
        "prefix_tokens": int(sum(row.prefix_tokens for row in ordered)),
        "suffix_tokens": int(sum(row.suffix_tokens for row in ordered)),
        "generated_tokens": int(sum(row.prefix_tokens + row.suffix_tokens for row in ordered)),
        "natural_finished": int(sum(row.natural_finish for row in ordered)),
        "truncated": int(sum(row.truncated for row in ordered)),
        "scope": "external vllm serve wall time; one client admission policy; excludes server startup and model loading",
    }
    return summary, ordered, request_rows


def _comparison(left_name: str, left_summary: dict, left_records: list[ContinuousRecord],
                right_name: str, right_summary: dict, right_records: list[ContinuousRecord]) -> dict:
    fields = (
        "start_index", "problem_id", "prompt_token_ids", "prefix_token_ids", "full_token_ids",
        "stage", "selected", "natural_finish", "truncated", "finish_reason", "stop_reason",
        "prefix_tokens", "suffix_tokens", "request_count",
    )
    exact = []
    for left, right in zip(left_records, right_records):
        lrow, rrow = record_to_dict(left), record_to_dict(right)
        exact.append(all(lrow[key] == rrow[key] for key in fields))
    left_tokens = int(left_summary["generated_tokens"])
    right_tokens = int(right_summary["generated_tokens"])
    return {
        "left_scheduler": left_name,
        "right_scheduler": right_name,
        "n_starts": len(exact),
        "exact_record_matches": int(sum(exact)),
        "all_records_match": bool(all(exact)),
        "left_wall_seconds": float(left_summary["wall_seconds"]),
        "right_wall_seconds": float(right_summary["wall_seconds"]),
        "right_over_left_wall_ratio": float(right_summary["wall_seconds"] / left_summary["wall_seconds"]),
        "left_generated_tokens": left_tokens,
        "right_generated_tokens": right_tokens,
        "generated_token_difference": int(right_tokens - left_tokens),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Frozen-workload scheduler probe through vllm serve")
    parser.add_argument("--config", action="append", default=[])
    parser.add_argument("--model-path", default=None, help="local tokenizer/model path used to encode prompts")
    parser.add_argument("--server-model", default=None, help="model id exposed by /v1/models")
    parser.add_argument("--base-url", default=None, help="vLLM base URL, e.g. http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--selection", choices=("first", "seeded"), default="seeded")
    parser.add_argument("--selection-seed", type=int, default=None)
    parser.add_argument("--n-problems", type=int, default=8)
    parser.add_argument("--starts-per-problem", type=int, default=16)
    parser.add_argument("--capacity", type=int, default=None)
    parser.add_argument("--decision-tokens", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None,
                        help="sampling temperature for all requests (default: 0.2)")
    parser.add_argument("--p", type=float, action="append", default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--mode", choices=("all", "no_refill", "refill", "both"), default=None)
    parser.add_argument("--workload", choices=("full_pg", "fixed_ht", "both"), default=None)
    parser.add_argument("--request-timeout", type=float, default=None)
    parser.add_argument("--all-concurrency", type=int, default=None)
    args = parser.parse_args(argv)

    seed = int(args.seed if args.seed is not None else 17)
    overrides = {
        "backend": "vllm_serve", "model_path": args.model_path, "seed": seed,
        "hardware": {"n_gpu": 1, "name": "a100", "gpu_memory_gb": 80},
    }
    if args.max_new_tokens is not None:
        overrides["max_new_tokens"] = int(args.max_new_tokens)
    if args.decision_tokens is not None:
        overrides["decision_tokens"] = int(args.decision_tokens)
    cfg = build_run_config(args.config, {key: value for key, value in overrides.items() if value is not None})
    probe_cfg = cfg.get("scheduler_serve_probe") or {}
    base_url = str(args.base_url or probe_cfg.get("base_url", "http://127.0.0.1:8000/v1"))
    server_model = args.server_model or probe_cfg.get("server_model")
    model_path = args.model_path or cfg.get("model_path")
    if not model_path or not server_model:
        raise ValueError("--model-path and --server-model (or config values) are required")
    api_key_env = str(args.api_key_env or probe_cfg.get("api_key_env", "VLLM_API_KEY"))
    api_key = os.environ.get(api_key_env)
    capacity = int(args.capacity if args.capacity is not None else probe_cfg.get("capacity", 8))
    decision_tokens = int(args.decision_tokens if args.decision_tokens is not None else probe_cfg.get("decision_tokens", 512))
    max_new_tokens = int(args.max_new_tokens if args.max_new_tokens is not None else probe_cfg.get("max_new_tokens", 2048))
    temperature = float(args.temperature if args.temperature is not None else probe_cfg.get("temperature", 0.2))
    configured_ps = [float(x) for x in (args.p if args.p is not None else probe_cfg.get("continuation_probabilities", [0.0, 0.5, 0.75, 1.0]))]
    repeats = int(args.repeats if args.repeats is not None else probe_cfg.get("repeats", 3))
    configured_modes = [str(x) for x in probe_cfg.get("schedulers", ["all", "no_refill", "refill"])]
    if args.mode == "both":
        modes = ["no_refill", "refill"]
    elif args.mode is not None:
        modes = [args.mode]
    else:
        modes = configured_modes
    configured_workloads = [str(x) for x in probe_cfg.get("workloads", ["full_pg", "fixed_ht"])]
    workload = args.workload or ("both" if set(("full_pg", "fixed_ht")).issubset(configured_workloads) else configured_workloads[0])
    all_concurrency = args.all_concurrency if args.all_concurrency is not None else probe_cfg.get("all_concurrency")
    request_timeout = float(args.request_timeout if args.request_timeout is not None else probe_cfg.get("request_timeout_seconds", 600.0))
    if capacity <= 0 or repeats <= 0 or not modes:
        raise ValueError("capacity, repeats, and schedulers must be positive/non-empty")
    if args.n_problems <= 0 or args.starts_per_problem <= 0:
        raise ValueError("n_problems and starts_per_problem must be positive")
    if all_concurrency is not None and int(all_concurrency) <= 0:
        raise ValueError("all_concurrency must be positive when provided")
    if request_timeout <= 0:
        raise ValueError("request-timeout must be positive")
    if not 0.0 <= temperature <= 2.0:
        raise ValueError("temperature must be in [0, 2]")
    if decision_tokens <= 0 or max_new_tokens < decision_tokens:
        raise ValueError("decision_tokens must be positive and <= max_new_tokens")
    if any(not 0.0 <= p <= 1.0 for p in configured_ps):
        raise ValueError("each --p must be in [0, 1]")
    if any(mode not in {"all", "no_refill", "refill"} for mode in modes):
        raise ValueError("schedulers must be all, no_refill, or refill")
    cfg["model_path"] = model_path
    cfg["max_new_tokens"] = max_new_tokens
    cfg["decision_tokens"] = decision_tokens
    requested = args.run_dir or str(default_run_dir("scheduler-serve-probe"))
    run = RunDirectory(resolve_run_dir(requested))
    run.write_yaml("config.yaml", {**cfg, "scheduler_serve_probe": {
        **probe_cfg, "base_url": base_url, "server_model": str(server_model), "capacity": capacity,
        "decision_tokens": decision_tokens, "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "continuation_probabilities": configured_ps, "repeats": repeats, "schedulers": modes,
        "workload": workload, "request_timeout_seconds": request_timeout,
        "all_concurrency": all_concurrency, "api_key_env": api_key_env,
        "api_key_provided": bool(api_key), "cli": {key: value for key, value in vars(args).items() if key != "api_key_env"},
    }})
    run.write_json("environment.json", collect_environment(cfg))

    records = select_records(load_math_records(args.data_path), int(args.n_problems), args.selection,
                             int(args.selection_seed if args.selection_seed is not None else seed))
    if not records:
        raise ValueError("the selected data set is empty")
    starts = []
    for record in records:
        starts.extend([record] * int(args.starts_per_problem))
    run.write_json("workload.json", {
        "n_problems": len(records), "starts_per_problem": int(args.starts_per_problem),
        "n_starts": len(starts), "problem_ids": [record.problem_id for record in records],
    })

    tokenizer = load_hf_tokenizer(str(model_path))
    prompt_meta = []
    prompt_ids, problem_ids, _golds = encode_records_hf(
        starts, tokenizer, int(cfg.get("prompt_max_tokens", 1024)), prompt_meta=prompt_meta,
    )
    run.write_jsonl("prompt_manifest.jsonl", [
        {"start_index": i, "problem_id": problem_ids[i], "prompt_tokens": len(prompt_ids[i]),
         "prompt_meta": prompt_meta[i] if i < len(prompt_meta) else None}
        for i in range(len(prompt_ids))
    ])
    eos_id = collect_stop_token_ids(tokenizer)
    client = VLLMServeClient(base_url, str(server_model), api_key=api_key,
                             timeout_seconds=request_timeout, temperature=temperature)
    client.health()
    served_models = client.models()
    if served_models and str(server_model) not in served_models:
        raise ValueError(f"server model {server_model!r} is not in /v1/models: {served_models}")

    starts_payload = [
        {"problem_id": problem_ids[i], "prompt_token_ids": prompt_ids[i]}
        for i in range(len(prompt_ids))
    ]
    workloads = ["full_pg", "fixed_ht"] if workload == "both" else [workload]
    summaries = []
    comparisons = []
    for workload_index, workload_name in enumerate(workloads):
        ps = [1.0] if workload_name == "full_pg" else configured_ps
        for p_index, p in enumerate(ps):
            for repeat in range(repeats):
                trial_seed = seed + 100003 * repeat + 1009 * p_index + 10000019 * workload_index
                request_plan = make_request_plan(len(starts_payload), p, trial_seed)
                run.append_jsonl("scheduler_serve_plans.jsonl", {
                    "workload": workload_name, "p": float(p), "repeat": repeat, **request_plan,
                })
                paired = {}
                ordered_modes = modes if repeat % 2 == 0 else list(reversed(modes))
                for scheduler in ordered_modes:
                    cache_salt = f"grace-phase12-serve-{trial_seed}-{workload_name}-{p_index}-{scheduler}"
                    summary, trial_records, request_rows = run_scheduler(
                        client, starts_payload, scheduler=scheduler, capacity=capacity,
                        all_concurrency=all_concurrency, mode=workload_name,
                        decision_tokens=decision_tokens, max_new_tokens=max_new_tokens, p=p,
                        request_plan=request_plan, eos_id=eos_id, cache_salt=cache_salt,
                    )
                    summary.update({"workload": workload_name, "p": float(p), "repeat": repeat,
                                    "trial_seed": trial_seed, "base_url": base_url,
                                    "server_model": str(server_model), "cache_salt": cache_salt})
                    summaries.append(summary)
                    paired[scheduler] = (summary, trial_records)
                    run.append_jsonl("scheduler_serve_trials.jsonl", summary)
                    run.write_jsonl("scheduler_serve_requests.jsonl", [
                        {**row, "workload": workload_name, "scheduler": scheduler,
                         "p": float(p), "repeat": repeat}
                        for row in request_rows
                    ], append=True)
                    run.write_jsonl("scheduler_serve_records.jsonl", [
                        {**record_to_dict(row), "workload": workload_name, "scheduler": scheduler,
                         "p": float(p), "repeat": repeat}
                        for row in trial_records
                    ], append=True)
                    print(summary, flush=True)
                names = list(paired)
                for left_index in range(len(names)):
                    for right_index in range(left_index + 1, len(names)):
                        left_name, right_name = names[left_index], names[right_index]
                        comparison = _comparison(left_name, *paired[left_name], right_name, *paired[right_name])
                        comparison.update({"workload": workload_name, "p": float(p), "repeat": repeat,
                                           "trial_seed": trial_seed})
                        comparisons.append(comparison)
                        run.append_jsonl("scheduler_serve_comparisons.jsonl", comparison)
    run.write_json("scheduler_serve_summary.json", {
        "trials": summaries, "comparisons": comparisons,
        "scope": "Frozen actor scheduler probe through one vllm serve endpoint; compare client admission policies.",
        "server": {"base_url": base_url, "model": str(server_model), "served_models": served_models,
                   "capacity": capacity, "one_gpu_a100_80g": True},
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
