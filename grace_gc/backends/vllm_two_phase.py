"""Two-phase vLLM rollout: same snapshot, original token IDs, selected suffixes only."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from grace_gc.core.rng import IsolatedRNG


@dataclass
class PhaseResult:
    token_ids: list[list[int]]
    natural_finish: np.ndarray
    prompt_lens: np.ndarray
    finish_reasons: list[str | None] | None = None
    stop_reasons: list | None = None
    sampling: dict | None = None


def _require_vllm():
    try:
        from vllm import LLM, SamplingParams
    except Exception as exc:
        raise ImportError(
            "vLLM is not installed. GPU two-phase rollout cannot start. "
            "Install the verl-pinned vLLM version on the server."
        ) from exc
    return LLM, SamplingParams


def build_sampling_params(
    max_tokens: int,
    temperature: float,
    seed: int | None,
    eos_id: int | list[int] | None = None,
    top_p: float = 1.0,
):
    _LLM, SamplingParams = _require_vllm()
    kwargs = {
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
        # Disable nucleus/top-k inherited from generation_config. Training
        # log-prob is the full softmax; paper sampling must match that.
        "top_k": -1,
        "n": 1,
        # Empty stop strings so Instruct generation_config cannot cut a
        # rollout early and then look like a natural EOS.
        "stop": [],
        "repetition_penalty": 1.0,
        "min_p": 0.0,
    }
    if seed is not None:
        kwargs["seed"] = int(seed)
    from grace_gc.data.tokenize import as_stop_ids

    stop_ids = as_stop_ids(eos_id)
    if stop_ids:
        kwargs["stop_token_ids"] = stop_ids
    used, dropped = _fit_sampling_params(SamplingParams, kwargs, seed is not None)
    build_sampling_params.last = {"used": dict(used), "dropped": list(dropped)}
    return SamplingParams(**used)


def _fit_sampling_params(SamplingParams, kwargs: dict, need_seed: bool) -> tuple[dict, list[str]]:
    """Keep required fields. Only min_p / repetition_penalty may be dropped."""
    try:
        SamplingParams(**kwargs)
        return dict(kwargs), []
    except TypeError:
        pass
    optional = ("min_p", "repetition_penalty")
    for drop in optional:
        trial = dict(kwargs)
        if drop not in trial:
            continue
        trial.pop(drop, None)
        try:
            SamplingParams(**trial)
            return trial, [drop]
        except TypeError:
            continue
    dropped = [name for name in optional if name in kwargs]
    trial = dict(kwargs)
    for name in dropped:
        trial.pop(name, None)
    try:
        SamplingParams(**trial)
        return trial, dropped
    except TypeError as exc:
        msg = str(exc).lower()
        if need_seed and "seed" in msg:
            raise ValueError(
                "vLLM SamplingParams rejected seed; rollout RNG would not be isolated"
            ) from exc
        raise ValueError(
            "vLLM SamplingParams rejected a required sampling field; "
            "generation would not match the actor log-prob"
        ) from exc


def _usable_stop_reason(stop_reason, stop_set: set[int]) -> int | None:
    """vLLM stop_reason is the token id or stop string that actually fired."""
    if stop_reason is None:
        return None
    try:
        sid = int(stop_reason)
    except (TypeError, ValueError):
        return None
    return sid if sid in stop_set else None


def _finish_name(finish_reason) -> str | None:
    if finish_reason is None:
        return None
    if isinstance(finish_reason, str):
        return finish_reason.lower()
    name = getattr(finish_reason, "name", None)
    if name is not None:
        return str(name).lower()
    return str(finish_reason).lower()


def _is_stop_finish(finish_reason) -> bool:
    """LLM.generate usually returns 'stop'; EngineCore may leak FinishReason.STOP."""
    return _finish_name(finish_reason) == "stop"


def _is_length_finish(finish_reason) -> bool:
    return _finish_name(finish_reason) == "length"


def _require_known_finish(finish_reason) -> None:
    """Abort/unknown must not look like a natural EOS and then get a reward."""
    if finish_reason is None:
        return
    if _is_stop_finish(finish_reason) or _is_length_finish(finish_reason):
        return
    raise ValueError(
        f"vLLM finish_reason {finish_reason!r} is not stop or length; "
        "the sequence would be scored as a complete trajectory"
    )


def generated_was_truncated(gen_len: int, max_new: int, ended: bool, finish_reason=None) -> bool:
    """Length-cap is truncation even when max_model_len stops short of max_new."""
    if ended:
        return False
    return int(gen_len) >= int(max_new) or _is_length_finish(finish_reason)


def trim_generated_tokens(
    token_ids,
    eos_id: int | list[int] | None,
    finish_reason: str | None = None,
    stop_reason=None,
):
    """Cut at the first stop id. Restore an omitted stop only when vLLM names that id.

    Qwen3-Base tokenizer.eos_token_id is <|endoftext|>, but ChatML stops on
    <|im_end|>. Inventing stops[0] would put the wrong token on the log-prob path.
    """
    from grace_gc.data.tokenize import as_stop_ids

    gen = [int(tok) for tok in list(token_ids)]
    stops = as_stop_ids(eos_id)
    if not stops:
        return gen, _is_stop_finish(finish_reason)
    stop_set = set(stops)
    for j, tok in enumerate(gen):
        if tok in stop_set:
            return gen[: j + 1], True
    if _is_stop_finish(finish_reason):
        extra = _usable_stop_reason(stop_reason, stop_set)
        if extra is not None:
            gen = gen + [extra]
        return gen, True
    return gen, False


def _vllm_prompts(prompt_token_ids: list[list[int]]):
    try:
        from vllm import TokensPrompt

        return [TokensPrompt(prompt_token_ids=ids) for ids in prompt_token_ids]
    except Exception:
        return [{"prompt_token_ids": ids} for ids in prompt_token_ids]


def generate_phase(
    llm,
    prompt_token_ids: list[list[int]],
    max_tokens: int,
    temperature: float,
    eos_id: int | list[int],
    rng: IsolatedRNG,
    stream: str,
    lora_request=None,
) -> PhaseResult:
    """Generate from raw token IDs. Prefix caching reuses KV, not RNG state."""
    _LLM, _SP = _require_vllm()
    prompts = _vllm_prompts(prompt_token_ids)
    seeds = [int(rng.integers(stream, 0, 2**31 - 1)) for _ in prompt_token_ids]
    param_list = [
        build_sampling_params(
            max_tokens,
            temperature,
            seed,
            eos_id=eos_id,
        )
        for seed in seeds
    ]
    kwargs = {"use_tqdm": False}
    if lora_request is not None:
        kwargs["lora_request"] = lora_request
    try:
        outputs = llm.generate(prompts, sampling_params=param_list, **kwargs)
    except TypeError:
        outputs = []
        for prompt, params in zip(prompts, param_list):
            outputs.extend(llm.generate([prompt], sampling_params=params, **kwargs))
    if len(outputs) != len(prompt_token_ids):
        raise ValueError(f"vLLM returned {len(outputs)} outputs for {len(prompt_token_ids)} prompts")
    token_ids = []
    finished = []
    prompt_lens = []
    finish_reasons = []
    stop_reasons = []
    for prompt, out in zip(prompt_token_ids, outputs):
        out_prompt = getattr(out, "prompt_token_ids", None)
        if out_prompt is None:
            raise ValueError("vLLM output is missing prompt_token_ids; cannot verify request order")
        if [int(x) for x in out_prompt] != [int(x) for x in prompt]:
            raise ValueError("vLLM output prompt_token_ids do not match the request order")
        completions = getattr(out, "outputs", None)
        if not completions:
            raise ValueError("vLLM returned a request with no completions")
        completion = completions[0]
        raw_finish = getattr(completion, "finish_reason", None)
        raw_stop = getattr(completion, "stop_reason", None)
        _require_known_finish(raw_finish)
        gen, ended = trim_generated_tokens(
            completion.token_ids,
            eos_id,
            raw_finish,
            raw_stop,
        )
        full = list(prompt) + gen
        token_ids.append(full)
        prompt_lens.append(len(prompt))
        finished.append(ended)
        finish_reasons.append(None if raw_finish is None else str(raw_finish))
        stop_reasons.append(None if raw_stop is None else raw_stop)
    # Independent draws can coincide, especially for short format-SFT answers.
    sampling = getattr(build_sampling_params, "last", None)
    if sampling is not None:
        sampling = {**sampling, "request_seeds": seeds}
    return PhaseResult(
        token_ids=token_ids,
        natural_finish=np.asarray(finished, dtype=bool),
        prompt_lens=np.asarray(prompt_lens, dtype=np.int64),
        finish_reasons=finish_reasons,
        stop_reasons=stop_reasons,
        sampling=None if sampling is None else dict(sampling),
    )


def continue_selected(
    llm,
    prefix_token_ids: list[list[int]],
    selected: np.ndarray,
    max_tokens: int,
    temperature: float,
    eos_id: int | list[int],
    rng: IsolatedRNG,
    lora_request=None,
) -> list[list[int] | None]:
    """Continue only selected prefixes on the same engine/snapshot."""
    from grace_gc.data.tokenize import is_stop_token

    selected = np.asarray(selected, dtype=bool)
    chosen = []
    chosen_idx = []
    already = {}
    for i, (ids, keep) in enumerate(zip(prefix_token_ids, selected)):
        if not keep:
            continue
        if ids and is_stop_token(ids[-1], eos_id):
            already[i] = list(ids)
            continue
        chosen.append(ids)
        chosen_idx.append(i)
    out: list[list[int] | None] = [None] * len(prefix_token_ids)
    for i, seq in already.items():
        out[i] = seq
    if not chosen:
        continue_selected.last_phase = None
        continue_selected.last_idx = []
        return out
    phase = generate_phase(llm, chosen, max_tokens, temperature, eos_id, rng, "continuation", lora_request=lora_request)
    continue_selected.last_phase = phase
    continue_selected.last_idx = list(chosen_idx)
    for j, i in enumerate(chosen_idx):
        out[i] = phase.token_ids[j]
    return out
