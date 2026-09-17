"""GPU Algorithm-1 engines: vLLM generate + HF actor update."""

from __future__ import annotations

from pathlib import Path
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
        phase = generate_phase(
            llm,
            prompt_ids,
            int(max_new),
            temperature,
            eos_id,
            rng,
            stream,
            lora_request=extra.get("lora_request"),
        )
        engines = box.get("engines")
        if engines is not None:
            engines.last_rollout = {
                "prefix_finish_reasons": list(phase.finish_reasons or []),
                "prefix_stop_reasons": list(phase.stop_reasons or []),
                "prefix_logprob_sums": list(phase.logprob_sums or []),
                "has_generate_logprobs": True,
                "sampling": phase.sampling,
            }
        return phase.token_ids, phase.natural_finish

    def continue_fn(prefixes, selected, max_new, rng):
        out = continue_selected(
            llm,
            prefixes,
            selected,
            int(max_new),
            temperature,
            eos_id,
            rng,
            lora_request=extra.get("lora_request"),
        )
        phase = getattr(continue_selected, "last_phase", None)
        idx = getattr(continue_selected, "last_idx", None) or []
        engines = box.get("engines")
        if engines is not None:
            roll = dict(engines.last_rollout or {})
            finish_map = dict(roll.get("continue_finish_reasons") or {})
            stop_map = dict(roll.get("continue_stop_reasons") or {})
            logprob_map = dict(roll.get("continue_logprob_sums") or {})
            if phase is not None:
                for j, i in enumerate(idx):
                    if phase.finish_reasons:
                        finish_map[i] = phase.finish_reasons[j]
                    if phase.stop_reasons:
                        stop_map[i] = phase.stop_reasons[j]
                    if phase.logprob_sums:
                        logprob_map[i] = phase.logprob_sums[j]
                if phase.sampling is not None:
                    roll["sampling"] = phase.sampling
            roll["continue_finish_reasons"] = finish_map
            roll["continue_stop_reasons"] = stop_map
            roll["continue_logprob_sums"] = logprob_map
            roll["has_generate_logprobs"] = True
            engines.last_rollout = roll
        return out

    def features(prefixes, prompt_lens, baselines):
        return prefix_feature_bundle(actor, prefixes, prompt_lens, baselines, pad_id, eos_id=eos_id)

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
    prev_id = int(extra["lora_id"]) if extra.get("lora_request") is not None else None
    extra["lora_id"] = int(extra["lora_id"]) + 1
    dest = Path(adapter_dir) / f"step-{extra['lora_id']}"
    save_lora_adapter(actor, dest)
    extra["adapter_path"] = dest
    extra["lora_request"] = make_lora_request(dest, extra["lora_id"])
    apply_lora_request(extra.get("llm"), extra["lora_request"], remove_id=prev_id)
    reset_vllm_prefix_cache(extra.get("llm"))
    return extra["lora_request"]


def apply_lora_to_generate(generate_kwargs: dict, extra: dict[str, Any]) -> dict:
    if extra.get("lora_request") is not None:
        generate_kwargs = dict(generate_kwargs)
        generate_kwargs["lora_request"] = extra["lora_request"]
    return generate_kwargs
