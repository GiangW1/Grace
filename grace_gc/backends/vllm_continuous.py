"""Request-level vLLM scheduling for frozen-actor throughput probes.

This module deliberately does not participate in the training path yet.  It
keeps one actor snapshot fixed, submits a predetermined number of starts, and
refills a bounded request queue when a prefix is stopped or a response
finishes.  The returned records make the scheduling experiment auditable.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from grace_gc.backends.vllm_two_phase import (
    _finish_name,
    _is_length_finish,
    _require_known_finish,
    _vllm_prompts,
    build_sampling_params,
    trim_generated_tokens,
)
from grace_gc.core.rng import IsolatedRNG


@dataclass
class ContinuousRecord:
    start_index: int
    problem_id: str
    prompt_token_ids: list[int]
    prefix_token_ids: list[int] | None = None
    full_token_ids: list[int] | None = None
    stage: str = "prefix"
    selected: bool | None = None
    natural_finish: bool = False
    truncated: bool = False
    finish_reason: str | None = None
    stop_reason: int | str | None = None
    prefix_tokens: int = 0
    suffix_tokens: int = 0
    request_count: int = 0
    prefix_request_id: str | None = None
    suffix_request_id: str | None = None
    prefix_wall_seconds: float = 0.0
    suffix_wall_seconds: float = 0.0


def _engine_from_llm(llm):
    engine = getattr(llm, "llm_engine", None) or getattr(llm, "engine", None)
    if engine is None or not callable(getattr(engine, "add_request", None)):
        raise TypeError(
            "the installed vLLM does not expose LLMEngine.add_request; "
            "continuous scheduling requires the vLLM 0.18 engine API"
        )
    if not callable(getattr(engine, "step", None)):
        raise TypeError("the installed vLLM engine has no step() method")
    return engine


def _completion(output):
    outputs = getattr(output, "outputs", None)
    if not outputs:
        raise ValueError("vLLM returned a request without a completion")
    completion = outputs[0]
    finish_reason = getattr(completion, "finish_reason", None)
    stop_reason = getattr(completion, "stop_reason", None)
    _require_known_finish(finish_reason)
    return completion, finish_reason, stop_reason


def make_request_plan(n_starts: int, p: float, seed: int) -> dict:
    """Preassign RNG by start so queue completion order cannot change samples."""
    n_starts = int(n_starts)
    p = float(p)
    if n_starts <= 0:
        raise ValueError("n_starts must be positive")
    if not 0.0 <= p <= 1.0:
        raise ValueError("continuation probability p must be in [0, 1]")
    rng = IsolatedRNG.create(int(seed))
    return {
        "seed": int(seed),
        "prefix_seeds": [int(rng.integers("token", 0, 2**31 - 1)) for _ in range(n_starts)],
        "suffix_seeds": [int(rng.integers("continuation", 0, 2**31 - 1)) for _ in range(n_starts)],
        "selected": rng.bernoulli("selection", [p] * n_starts).astype(bool).tolist(),
    }


class ContinuousRollout:
    """Run a fixed workload with a bounded request queue.

    ``mode='full_pg'`` generates every start to ``max_new_tokens``.  In
    ``mode='fixed_ht'`` each start first generates ``decision_tokens``; a
    scalar Bernoulli continuation draw either enqueues its suffix or settles
    the start as an early stop.  All starts remain in the returned ledger.
    """

    def __init__(self, llm, eos_id, *, lora_request=None, capacity=8):
        self.llm = llm
        self.engine = _engine_from_llm(llm)
        self.eos_id = eos_id
        self.lora_request = lora_request
        self.capacity = max(1, int(capacity))

    def _submit(self, request_id, token_ids, max_tokens, seed):
        params = build_sampling_params(
            int(max_tokens), 1.0, int(seed), eos_id=self.eos_id,
        )
        prompt = _vllm_prompts([[int(x) for x in token_ids]])[0]
        kwargs = {}
        if self.lora_request is not None:
            kwargs["lora_request"] = self.lora_request
        self.engine.add_request(request_id, prompt, params, **kwargs)

    def run(self, starts, *, mode="fixed_ht", decision_tokens=512,
            max_new_tokens=2048, p=0.5, seed=17, request_plan=None):
        mode = str(mode)
        if mode not in {"full_pg", "fixed_ht"}:
            raise ValueError("mode must be full_pg or fixed_ht")
        decision_tokens = int(decision_tokens)
        max_new_tokens = int(max_new_tokens)
        if decision_tokens <= 0 or max_new_tokens < decision_tokens:
            raise ValueError("decision_tokens must be positive and <= max_new_tokens")
        p = float(p)
        if mode == "full_pg":
            p = 1.0
        if not 0.0 <= p <= 1.0:
            raise ValueError("continuation probability p must be in [0, 1]")

        starts = list(starts)
        if not starts:
            raise ValueError("continuous rollout needs at least one start")
        records = {
            int(i): ContinuousRecord(
                start_index=int(i), problem_id=str(item["problem_id"]),
                prompt_token_ids=[int(x) for x in item["prompt_token_ids"]],
            )
            for i, item in enumerate(starts)
        }
        plan = make_request_plan(len(starts), p, seed) if request_plan is None else dict(request_plan)
        prefix_seeds = [int(x) for x in plan.get("prefix_seeds") or []]
        suffix_seeds = [int(x) for x in plan.get("suffix_seeds") or []]
        selected_plan = [bool(x) for x in plan.get("selected") or []]
        if any(len(values) != len(starts) for values in (prefix_seeds, suffix_seeds, selected_plan)):
            raise ValueError("request_plan arrays must match the number of starts")
        request_meta = {}
        active = set()
        next_start = 0
        request_counter = 0
        step_calls = 0
        max_active = 0
        completed = 0
        started = perf_counter()

        def submit_start(index):
            nonlocal request_counter, next_start
            rec = records[index]
            request_id = f"grace-cont-{int(seed)}-{request_counter}"
            request_counter += 1
            request_meta[request_id] = {
                "start_index": index, "stage": "full" if mode == "full_pg" else "prefix",
                "submitted": perf_counter(), "max_tokens": max_new_tokens,
            }
            self._submit(
                request_id,
                rec.prompt_token_ids,
                max_new_tokens if mode == "full_pg" else decision_tokens,
                prefix_seeds[index],
            )
            rec.prefix_request_id = request_id
            rec.request_count += 1
            active.add(request_id)
            next_start += 1

        def submit_suffix(index, prefix):
            nonlocal request_counter
            rec = records[index]
            remain = max_new_tokens - (len(prefix) - len(rec.prompt_token_ids))
            if remain <= 0:
                rec.full_token_ids = list(prefix)
                rec.truncated = True
                rec.stage = "complete"
                return False
            request_id = f"grace-cont-{int(seed)}-{request_counter}"
            request_counter += 1
            request_meta[request_id] = {
                "start_index": index, "stage": "suffix",
                "submitted": perf_counter(), "max_tokens": remain,
            }
            self._submit(request_id, prefix, remain, suffix_seeds[index])
            rec.suffix_request_id = request_id
            rec.request_count += 1
            active.add(request_id)
            return True

        while next_start < len(starts) and len(active) < self.capacity:
            submit_start(next_start)

        while active:
            step_calls += 1
            max_active = max(max_active, len(active))
            outputs = self.engine.step()
            if outputs is None:
                continue
            if not isinstance(outputs, (list, tuple)):
                outputs = [outputs]
            for output in outputs:
                request_id = str(getattr(output, "request_id", ""))
                if request_id not in active or not bool(getattr(output, "finished", True)):
                    continue
                active.remove(request_id)
                meta = request_meta.pop(request_id)
                rec = records[int(meta["start_index"])]
                completion, raw_finish, raw_stop = _completion(output)
                gen, ended = trim_generated_tokens(
                    completion.token_ids, self.eos_id, raw_finish, raw_stop,
                )
                prompt = rec.prompt_token_ids if meta["stage"] == "full" else rec.prefix_token_ids
                if prompt is None:
                    prompt = rec.prompt_token_ids
                full = list(prompt) + [int(x) for x in gen]
                now = perf_counter()
                elapsed = max(0.0, now - float(meta["submitted"]))
                if meta["stage"] == "full":
                    rec.full_token_ids = full
                    rec.prefix_tokens = max(len(full) - len(rec.prompt_token_ids), 0)
                    rec.natural_finish = bool(ended)
                    rec.truncated = bool(_is_length_finish(raw_finish) and not ended)
                    rec.finish_reason = _finish_name(raw_finish)
                    rec.stop_reason = raw_stop
                    rec.stage = "complete"
                    rec.prefix_wall_seconds = elapsed
                elif meta["stage"] == "prefix":
                    rec.prefix_token_ids = full
                    rec.prefix_tokens = max(len(full) - len(rec.prompt_token_ids), 0)
                    rec.prefix_wall_seconds = elapsed
                    rec.finish_reason = _finish_name(raw_finish)
                    rec.stop_reason = raw_stop
                    if ended:
                        rec.full_token_ids = full
                        rec.natural_finish = True
                        rec.stage = "complete"
                    elif rec.prefix_tokens >= max_new_tokens:
                        rec.full_token_ids = full
                        rec.natural_finish = False
                        rec.truncated = True
                        rec.stage = "complete"
                    elif not _is_length_finish(raw_finish):
                        raise ValueError(f"prefix request {request_id} ended with {raw_finish!r}")
                    else:
                        rec.selected = selected_plan[rec.start_index]
                        if rec.selected:
                            submitted = submit_suffix(rec.start_index, full)
                            if submitted:
                                rec.stage = "suffix"
                            # A prefix can exactly consume max_new_tokens.  It
                            # is already a complete trajectory; do not turn
                            # that edge case into an early-stop record.
                            elif rec.stage == "complete":
                                rec.natural_finish = False
                            else:
                                rec.full_token_ids = None
                                rec.natural_finish = False
                                rec.truncated = False
                                rec.stage = "stopped"
                        else:
                            rec.full_token_ids = None
                            rec.natural_finish = False
                            rec.truncated = False
                            rec.stage = "stopped"
                else:
                    rec.full_token_ids = full
                    rec.suffix_tokens = max(len(full) - len(rec.prefix_token_ids or []), 0)
                    rec.suffix_wall_seconds = elapsed
                    rec.natural_finish = bool(ended)
                    rec.truncated = bool(_is_length_finish(raw_finish) and not ended)
                    rec.finish_reason = _finish_name(raw_finish)
                    rec.stop_reason = raw_stop
                    rec.stage = "complete"
                if rec.stage in {"complete", "stopped"}:
                    completed += 1
                    while next_start < len(starts) and len(active) < self.capacity:
                        submit_start(next_start)

        if completed != len(starts):
            raise RuntimeError(f"continuous rollout settled {completed}/{len(starts)} starts")
        finished = perf_counter() - started
        ordered = [records[i] for i in range(len(starts))]
        summary = {
            "mode": mode,
            "capacity": int(self.capacity),
            "n_starts": len(starts),
            "p": float(p),
            "decision_tokens": decision_tokens,
            "max_new_tokens": max_new_tokens,
            "wall_seconds": float(finished),
            "engine_step_calls": int(step_calls),
            "max_active": int(max_active),
            "completed": int(sum(r.stage == "complete" for r in ordered)),
            "stopped": int(sum(r.stage == "stopped" for r in ordered)),
            "continued": len(ordered) if mode == "full_pg" else int(sum(r.selected is True for r in ordered)),
            "prefix_tokens": int(sum(r.prefix_tokens for r in ordered)),
            "suffix_tokens": int(sum(r.suffix_tokens for r in ordered)),
            "generated_tokens": int(sum(r.prefix_tokens + r.suffix_tokens for r in ordered)),
            "natural_finished": int(sum(r.natural_finish for r in ordered)),
            "truncated": int(sum(r.truncated for r in ordered)),
            "scope": "external request-level wall time, including engine.step and queue refill; excludes actor construction and model loading",
        }
        return summary, ordered


def record_to_dict(record: ContinuousRecord) -> dict:
    return {
        "start_index": record.start_index,
        "problem_id": record.problem_id,
        "prompt_token_ids": record.prompt_token_ids,
        "prefix_token_ids": record.prefix_token_ids,
        "full_token_ids": record.full_token_ids,
        "stage": record.stage,
        "selected": record.selected,
        "natural_finish": record.natural_finish,
        "truncated": record.truncated,
        "finish_reason": record.finish_reason,
        "stop_reason": record.stop_reason,
        "prefix_tokens": record.prefix_tokens,
        "suffix_tokens": record.suffix_tokens,
        "request_count": record.request_count,
        "prefix_request_id": record.prefix_request_id,
        "suffix_request_id": record.suffix_request_id,
        "prefix_wall_seconds": record.prefix_wall_seconds,
        "suffix_wall_seconds": record.suffix_wall_seconds,
    }
