"""Compare HF actor token log-probs with vLLM on the same tokens. Record only."""

from __future__ import annotations

from typing import Any


def hf_response_logprobs(model, token_ids: list[int], prompt_len: int, pad_id: int, eos_id=None, max_n: int = 64):
    from grace_gc.backends.hf_actor import actor_forward
    from grace_gc.data.tokenize import as_stop_ids

    out, ids, _mask = actor_forward(model, [token_ids], pad_id, output_hidden_states=False)
    logp = out.logits[:, :-1].float().log_softmax(-1)
    target = ids[:, 1:]
    token_lp = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)[0]
    start = max(int(prompt_len) - 1, 0)
    end = min(len(token_ids) - 1, token_lp.shape[0])
    stops = set(as_stop_ids(eos_id))
    for j in range(int(prompt_len), len(token_ids)):
        if int(token_ids[j]) in stops:
            end = min(end, j)
            break
    sl = token_lp[start:end][: int(max_n)]
    return sl.detach().float().cpu().numpy()


def vllm_prompt_logprobs(llm, token_ids: list[int], lora_request=None, max_n: int = 64):
    from grace_gc.backends.vllm_two_phase import _vllm_prompts, build_sampling_params

    params = build_sampling_params(1, 0.0, seed=0)
    extra = {}
    for key in ("prompt_logprobs",):
        extra[key] = 0
    try:
        params = type(params)(**{**getattr(build_sampling_params, "last", {}).get("used", {}), **extra, "max_tokens": 1, "temperature": 0.0, "seed": 0})
    except TypeError as exc:
        raise ValueError(f"vLLM SamplingParams rejected prompt_logprobs: {exc}") from exc
    kwargs = {"sampling_params": params, "use_tqdm": False}
    if lora_request is not None:
        kwargs["lora_request"] = lora_request
    outputs = llm.generate(_vllm_prompts([list(token_ids)]), **kwargs)
    if not outputs:
        raise ValueError("vLLM prompt_logprobs returned no outputs")
    raw = getattr(outputs[0], "prompt_logprobs", None)
    if raw is None:
        raise ValueError("vLLM output has no prompt_logprobs")
    out = []
    for i, item in enumerate(raw):
        if i == 0 or item is None:
            continue
        tid = int(token_ids[i])
        rec = item.get(tid) if hasattr(item, "get") else None
        if rec is None and isinstance(item, dict):
            rec = item.get(str(tid))
        if rec is None:
            continue
        lp = getattr(rec, "logprob", rec)
        out.append(float(lp))
        if len(out) >= int(max_n):
            break
    return out


def compare_hf_vllm_logprob(
    model,
    llm,
    token_ids: list[int],
    prompt_len: int,
    pad_id: int,
    eos_id=None,
    lora_request=None,
    max_n: int = 64,
) -> dict[str, Any]:
    """Same tokens, two engines. Large mean_abs means rollout ≠ score."""
    payload: dict[str, Any] = {
        "status": "ok",
        "n_tokens": 0,
        "hf_sum": None,
        "vllm_sum": None,
        "mean_abs": None,
        "max_abs": None,
        "prompt_len": int(prompt_len),
        "seq_len": len(token_ids),
    }
    try:
        hf = hf_response_logprobs(model, token_ids, prompt_len, pad_id, eos_id, max_n=max_n)
        vllm = vllm_prompt_logprobs(llm, token_ids, lora_request=lora_request, max_n=len(hf) + int(prompt_len))
        # vLLM prompt_logprobs[i] is p(token i | prefix). HF slice starts at prompt_len-1
        # which is p(token[prompt_len] | prompt). Align on the response tokens.
        start = max(int(prompt_len) - 1, 0)
        vllm_resp = vllm[start : start + len(hf)] if len(vllm) > start else vllm[-len(hf) :]
        n = min(len(hf), len(vllm_resp))
        if n <= 0:
            payload["status"] = "empty"
            return payload
        import numpy as np

        a = np.asarray(hf[:n], dtype=np.float64)
        b = np.asarray(vllm_resp[:n], dtype=np.float64)
        diff = np.abs(a - b)
        payload.update(
            {
                "n_tokens": int(n),
                "hf_sum": float(a.sum()),
                "vllm_sum": float(b.sum()),
                "mean_abs": float(diff.mean()),
                "max_abs": float(diff.max()),
            }
        )
        return payload
    except Exception as exc:
        payload["status"] = "unavailable"
        payload["error"] = f"{type(exc).__name__}: {exc}"
        return payload


def probe_first_completed(model, llm, records, pad_id: int, eos_id=None, lora_request=None) -> dict[str, Any]:
    for rec in records or []:
        if rec.z < 1.0 or not rec.full_token_ids or rec.prompt_len <= 0:
            continue
        if len(rec.full_token_ids) <= int(rec.prompt_len) + 1:
            continue
        out = compare_hf_vllm_logprob(
            model,
            llm,
            rec.full_token_ids,
            int(rec.prompt_len),
            pad_id,
            eos_id=eos_id,
            lora_request=lora_request,
        )
        out["problem_id"] = rec.problem_id
        return out
    return {"status": "no_completed_sequence"}
