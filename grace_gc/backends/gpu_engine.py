"""GPU Algorithm-1 engines: vLLM generate + HF actor update."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any  # noqa: F401

import numpy as np

from grace_gc.backends.hf_actor import (
    logprob_one,
    logprob_sums,
    named_lora_params,
    prefix_feature_bundle,
    trainable_params,
)
from grace_gc.backends.vllm_two_phase import continue_selected, generate_phase
from grace_gc.backends.weight_sync import apply_lora_request, make_lora_request, reset_vllm_prefix_cache, save_lora_adapter
from grace_gc.data.reward import rule_reward
from grace_gc.data.tokenize import collect_stop_token_ids, decode_hf
from grace_gc.trainer.algorithm import StepEngines


def make_gpu_engines(actor, llm, tokenizer, cfg: dict[str, Any], adapter_dir: Path) -> tuple[StepEngines, dict[str, Any]]:
    pad_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
    eos_id = collect_stop_token_ids(tokenizer)
    temperature = float(cfg.get("temperature", 1.0))
    if abs(temperature - 1.0) > 1e-12:
        raise ValueError(
            "training temperature must be 1 so vLLM sampling matches the HF score"
        )
    extra = {"lora_request": None, "lora_id": 1, "llm": llm}
    box: dict[str, Any] = {}

    def generate_prefix(prompt_ids, max_new, rng, stream: str):
        execution: dict = {}
        engines = box.get("engines")
        if engines is not None:
            # The local dict is updated even when generate raises.
            engines.last_rollout = {"prefix_execution": execution}
        phase = generate_phase(
            llm,
            prompt_ids,
            int(max_new),
            temperature,
            eos_id,
            rng,
            stream,
            lora_request=extra.get("lora_request"),
            execution=execution,
        )
        if engines is not None:
            engines.last_rollout = {
                "prefix_execution": execution,
                "prefix_num_cached_tokens": list(phase.num_cached_tokens or []),
                "prefix_finish_reasons": list(phase.finish_reasons or []),
                "prefix_stop_reasons": list(phase.stop_reasons or []),
                "prefix_logprob_sums": list(phase.logprob_sums or []),
                "prefix_token_logprobs": list(phase.token_logprobs or []),
                "prefix_request_seeds": (phase.sampling or {}).get("request_seeds", []),
                "has_generate_logprobs": True,
                "sampling": phase.sampling,
            }
        return phase.token_ids, phase.natural_finish

    def continue_fn(prefixes, selected, max_new, rng):
        execution: dict = {}
        engines = box.get("engines")
        if engines is not None:
            roll = dict(engines.last_rollout or {})
            roll["continue_execution"] = [*(roll.get("continue_execution") or []), execution]
            engines.last_rollout = roll
        out = continue_selected(
            llm,
            prefixes,
            selected,
            int(max_new),
            temperature,
            eos_id,
            rng,
            lora_request=extra.get("lora_request"),
            execution=execution,
        )
        phase = getattr(continue_selected, "last_phase", None)
        idx = getattr(continue_selected, "last_idx", None) or []
        engines = box.get("engines")
        if engines is not None:
            roll = dict(engines.last_rollout or {})
            finish_map = dict(roll.get("continue_finish_reasons") or {})
            stop_map = dict(roll.get("continue_stop_reasons") or {})
            logprob_map = dict(roll.get("continue_logprob_sums") or {})
            token_lp_map = dict(roll.get("continue_token_logprobs") or {})
            seed_map = dict(roll.get("continue_request_seeds") or {})
            cache_map = dict(roll.get("continue_num_cached_tokens") or {})
            if phase is not None:
                for j, i in enumerate(idx):
                    if phase.finish_reasons:
                        finish_map[i] = phase.finish_reasons[j]
                    if phase.stop_reasons:
                        stop_map[i] = phase.stop_reasons[j]
                    if phase.logprob_sums:
                        logprob_map[i] = phase.logprob_sums[j]
                    if phase.token_logprobs:
                        token_lp_map[i] = phase.token_logprobs[j]
                    if phase.num_cached_tokens is not None:
                        cache_map[i] = phase.num_cached_tokens[j]
                    request_seeds = (phase.sampling or {}).get("request_seeds") or []
                    if j < len(request_seeds):
                        seed_map[i] = request_seeds[j]
                if phase.sampling is not None:
                    roll["sampling"] = phase.sampling
            roll["continue_finish_reasons"] = finish_map
            roll["continue_stop_reasons"] = stop_map
            roll["continue_logprob_sums"] = logprob_map
            roll["continue_token_logprobs"] = token_lp_map
            roll["continue_request_seeds"] = seed_map
            roll["continue_num_cached_tokens"] = cache_map
            roll["has_generate_logprobs"] = True
            engines.last_rollout = roll
        return out

    def features(prefixes, prompt_lens, baselines):
        pcfg = cfg.get("predictor") or {}
        return prefix_feature_bundle(actor, prefixes, prompt_lens, baselines, pad_id, eos_id=eos_id,
                                     feature_mode=pcfg.get("feature_mode", "legacy"),
                                     batch_size=int(pcfg.get("feature_batch_size", 1)))

    def lp_sums(full_ids, prompt_lens, chosen):
        return logprob_sums(actor, full_ids, prompt_lens, chosen, pad_id, eos_id=eos_id)

    def lp_one(token_ids, prompt_len: int):
        return logprob_one(actor, token_ids, prompt_len, pad_id, eos_id=eos_id)

    def reward_fn(token_ids, gold, truncated=False, text=""):
        _ = token_ids
        return rule_reward(text or "", gold, truncated=truncated)

    def decode(token_ids):
        return decode_hf(tokenizer, token_ids)

    engines = StepEngines(
        generate_prefix=generate_prefix,
        continue_selected=continue_fn,
        prefix_features=features,
        logprob_sums=lp_sums,
        logprob_one=lp_one,
        named_lora=lambda: named_lora_params(actor),
        trainable_params=lambda: trainable_params(actor),
        reward_fn=reward_fn,
        decode=decode,
        eos_id=eos_id,
    )
    box["engines"] = engines
    extra["pad_id"] = pad_id
    extra["eos_id"] = eos_id
    extra["sync"] = lambda: _sync(actor, adapter_dir, extra)
    return engines, extra


def _sync(actor, adapter_dir: Path, extra: dict[str, Any]):
    detail = {"timings": {}, "status": "running",
              "scope": "Subtimings already included in sync wall time; do not add to the compute ledger"}
    extra["last_sync"] = detail

    def timed(name, action):
        started = perf_counter()
        try:
            return action()
        except Exception as exc:
            detail.update(status="failed", failed_stage=name, error_type=type(exc).__name__, error=str(exc))
            raise
        finally:
            detail["timings"][name] = perf_counter() - started
            if hasattr(extra.get("llm"), "last_execution") and name != "adapter_save":
                detail.setdefault("workers", {})[name] = extra["llm"].last_execution

    prev_id = int(extra["lora_id"]) if extra.get("lora_request") is not None else None
    extra["lora_id"] = int(extra["lora_id"]) + 1
    dest = Path(adapter_dir) / f"step-{extra['lora_id']}"
    timed("adapter_save", lambda: save_lora_adapter(actor, dest))
    extra["adapter_path"] = dest

    def load_adapter():
        extra["lora_request"] = make_lora_request(dest, extra["lora_id"])
        apply_lora_request(extra.get("llm"), extra["lora_request"], remove_id=prev_id)

    timed("adapter_load", load_adapter)
    timed("prefix_cache_reset", lambda: reset_vllm_prefix_cache(extra.get("llm")))
    detail["status"] = "complete"
    return extra["lora_request"]


def apply_lora_to_generate(generate_kwargs: dict, extra: dict[str, Any]) -> dict:
    if extra.get("lora_request") is not None:
        generate_kwargs = dict(generate_kwargs)
        generate_kwargs["lora_request"] = extra["lora_request"]
    return generate_kwargs
