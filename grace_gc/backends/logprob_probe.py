"""Compare HF actor token log-probs with vLLM on the same tokens. Record only."""

from __future__ import annotations

from typing import Any


def logprob_error_summary(hf, behavior, token_ids, prompt_len, prefix_tokens=None):
    """Locate deviations in already computed scores; never rescore or resample."""
    import numpy as np

    a, b = np.asarray(hf, dtype=float), np.asarray(behavior, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or len(token_ids) != len(a):
        raise ValueError("logprob diagnostic positions must align")
    diff = a-b
    finite = np.isfinite(a) & np.isfinite(b)
    def stats(mask):
        values = diff[mask & finite]
        return {"n_tokens": int(len(values)), "signed_mean": float(values.mean()) if len(values) else None,
                "rms": float(np.sqrt(np.mean(values**2))) if len(values) else None,
                "mean_abs": float(np.abs(values).mean()) if len(values) else None,
                "absolute_quantiles": dict(zip(("p50", "p90", "p99"), np.quantile(np.abs(values), [.5, .9, .99]).tolist())) if len(values) else {}}
    result = {**stats(np.ones(len(a), dtype=bool)), "nonfinite_tokens": int((~finite).sum()),
              "max_error_token": None, "by_phase": {}, "position_base": "zero-based; logit position predicts sequence position"}
    if finite.any():
        index = int(np.flatnonzero(finite)[np.argmax(np.abs(diff[finite]))])
        phase = None if prefix_tokens is None else ("prefix" if index < prefix_tokens else "continuation")
        result["max_error_token"] = {"response_index": index, "sequence_index": int(prompt_len)+index,
            "logit_index": int(prompt_len)+index-1, "token_id": int(token_ids[index]), "phase": phase,
            "hf_logprob": float(a[index]), "behavior_logprob": float(b[index]),
            "signed_error": float(diff[index]), "absolute_error": float(abs(diff[index]))}
    if prefix_tokens is not None:
        prefix = np.arange(len(a)) < int(prefix_tokens)
        result["by_phase"] = {"prefix": stats(prefix), "continuation": stats(~prefix)}
    return result


def probe_numerics(model, llm, lora_request):
    """Small metadata only: no tensor copies/hashes or extra engine requests."""
    result = {"hf_compute_dtype": getattr(model, "_grace_compute_dtype", None),
              "hf_log_softmax_dtype": "float32", "hf_probe_microbatch_size": 1, "hf_use_cache": False,
              "hf_training": getattr(model, "training", None),
              "hf_attention_implementation": getattr(getattr(model, "config", None), "_attn_implementation", None),
              "hf_parameter_dtypes": [], "lora_parameter_dtypes": [], "active_adapters": None,
              "lora_configuration": {}, "lora_request": {}, "vllm_dtype": None, "vllm_lora_dtype": None,
              "scope": "Host metadata; not a vLLM worker tensor hash or kernel equivalence check. Join step with trajectory snapshot_sha for frozen actor provenance."}
    try:
        named = list(model.named_parameters()) if model is not None else []
        result["hf_parameter_dtypes"] = sorted({str(param.dtype) for _, param in named})
        result["lora_parameter_dtypes"] = sorted({str(param.dtype) for name, param in named if "lora_" in name.lower()})
        active = getattr(model, "active_adapters", None)
        result["active_adapters"] = active if isinstance(active, (list, str)) else None
        for name, config in (getattr(model, "peft_config", None) or {}).items():
            result["lora_configuration"][str(name)] = {key: getattr(config, key, None) for key in ("r", "lora_alpha", "lora_dropout", "use_rslora", "use_dora")}
        result["lora_request"] = {key: getattr(lora_request, key, None) for key in ("lora_name", "lora_int_id", "lora_path")}
        engine = getattr(llm, "llm_engine", None)
        engine_config = getattr(engine, "vllm_config", None)
        model_config = getattr(engine_config, "model_config", None) or getattr(engine, "model_config", None)
        lora_config = getattr(engine_config, "lora_config", None) or getattr(engine, "lora_config", None)
        result["vllm_dtype"] = None if model_config is None else str(getattr(model_config, "dtype", None))
        result["vllm_lora_dtype"] = None if lora_config is None else str(getattr(lora_config, "lora_dtype", None))
    except Exception as exc:
        result["metadata_error"] = f"{type(exc).__name__}: {exc}"
    return result


def hf_response_logprobs(model, token_ids: list[int], prompt_len: int, pad_id: int, eos_id=None, max_n: int = 64):
    from grace_gc.backends.hf_actor import actor_forward
    from grace_gc.data.tokenize import as_stop_ids

    import torch

    if max_n is not None and int(max_n) > 0:
        token_ids = token_ids[:int(prompt_len) + int(max_n)]
    with torch.no_grad():
        out, ids, _mask = actor_forward(model, [token_ids], pad_id, output_hidden_states=False)
        # Softmax only response positions, in bounded chunks.
        chunks = []
        for pos in range(max(int(prompt_len) - 1, 0), len(token_ids) - 1, 64):
            end = min(pos + 64, len(token_ids) - 1)
            logp = out.logits[:, pos:end].float().log_softmax(-1)
            targets = ids[:, pos + 1:end + 1]
            chunks.append(logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)[0].cpu())
        token_lp = torch.cat(chunks) if chunks else torch.empty(0)
    start = max(int(prompt_len) - 1, 0)
    end = len(token_ids) - 1
    stops = set(as_stop_ids(eos_id))
    for j in range(int(prompt_len), len(token_ids)):
        if int(token_ids[j]) in stops:
            end = min(end, j)
            break
    sl = token_lp[:max(end - start, 0)]
    return sl.detach().float().cpu().numpy()


def vllm_prompt_logprobs(llm, token_ids: list[int], lora_request=None, max_n: int = 64):
    from grace_gc.backends.vllm_two_phase import _vllm_prompts, build_sampling_params

    params = build_sampling_params(1, 0.0, seed=0)
    extra = {}
    for key in ("prompt_logprobs",):
        extra[key] = 1
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
        if i == 0:
            continue
        tid = int(token_ids[i])
        rec = item.get(tid) if hasattr(item, "get") else None
        if rec is None and isinstance(item, dict):
            rec = item.get(str(tid))
        lp = getattr(rec, "logprob", rec)
        # Keep token positions; dropping a missing entry silently shifts scores.
        out.append(float("nan") if lp is None else float(lp))
        if max_n is not None and int(max_n) > 0 and len(out) >= int(max_n):
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
        "status": "compared",
        "meaning": "numerical comparison only; no on-policy acceptance threshold",
        "n_tokens": 0,
        "hf_sum": None,
        "vllm_sum": None,
        "mean_abs": None,
        "max_abs": None,
        "prompt_len": int(prompt_len),
        "seq_len": len(token_ids),
    }
    try:
        checked_ids = token_ids if max_n is None or int(max_n) <= 0 else token_ids[:int(prompt_len) + int(max_n)]
        hf = hf_response_logprobs(model, checked_ids, prompt_len, pad_id, eos_id, max_n=max_n)
        vllm = vllm_prompt_logprobs(llm, checked_ids, lora_request=lora_request, max_n=len(hf) + int(prompt_len))
        # vLLM prompt_logprobs[i] is p(token i | prefix). HF slice starts at prompt_len-1
        # which is p(token[prompt_len] | prompt). Align on the response tokens.
        start = max(int(prompt_len) - 1, 0)
        if len(vllm) <= start:
            payload["status"] = "misaligned"
            payload["n_vllm"] = len(vllm)
            return payload
        vllm_resp = vllm[start : start + len(hf)]
        n = min(len(hf), len(vllm_resp))
        if n <= 0:
            payload["status"] = "empty"
            return payload
        import numpy as np

        a = np.asarray(hf[:n], dtype=np.float64)
        b = np.asarray(vllm_resp[:n], dtype=np.float64)
        if n != len(hf):
            payload.update(status="misaligned", n_hf=len(hf), n_vllm=len(vllm_resp))
            return payload
        if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
            payload.update(status="nonfinite", hf_finite=int(np.isfinite(a).sum()),
                           vllm_finite=int(np.isfinite(b).sum()))
            return payload
        diff = np.abs(a - b)
        payload.update(
            {
                "n_tokens": int(n),
                "hf_sum": float(a.sum()),
                "vllm_sum": float(b.sum()),
                "mean_abs": float(diff.mean()),
                "max_abs": float(diff.max()),
                "hf_token_logprobs": a.tolist(),
                "vllm_prompt_token_logprobs": b.tolist(),
                "response_token_ids": list(checked_ids[int(prompt_len):int(prompt_len) + n]),
                "full_response_checked": n == len(token_ids) - int(prompt_len),
                "deviation": logprob_error_summary(a, b, checked_ids[int(prompt_len):int(prompt_len)+n], prompt_len),
            }
        )
        return payload
    except Exception as exc:
        payload["status"] = "unavailable"
        payload["error"] = f"{type(exc).__name__}: {exc}"
        return payload


def _probe_first_completed(model, llm, records, pad_id: int, eos_id=None, lora_request=None, max_n=64) -> dict[str, Any]:
    for rec in records or []:
        if rec.z < 1.0 or not rec.full_token_ids or rec.prompt_len <= 0:
            continue
        if len(rec.full_token_ids) <= int(rec.prompt_len):
            continue
        behavior = getattr(rec, "rollout_token_logprobs", None)
        if behavior is not None:
            import numpy as np

            hf = hf_response_logprobs(model, rec.full_token_ids, int(rec.prompt_len), pad_id, eos_id, max_n=max_n)
            raw = behavior[:len(hf)]
            out = {"status": "misaligned", "source": "sampled_behavior",
                   "meaning": "measured difference, not an acceptance threshold",
                   "expected_tokens": len(hf), "behavior_tokens": len(raw),
                   "missing_behavior_positions": [i for i, value in enumerate(raw) if value is None]}
            if len(raw) == len(hf) and len(hf) and all(v is not None for v in raw):
                if not np.all(np.isfinite(hf)) or not np.all(np.isfinite(raw)):
                    return {"status": "nonfinite", "source": "sampled_behavior",
                            "hf_finite": int(np.isfinite(hf).sum()),
                            "vllm_finite": int(np.isfinite(raw).sum()),
                            "problem_id": rec.problem_id, "timing": "before_actor_update"}
                diff = np.asarray(hf, dtype=np.float64) - np.asarray(raw, dtype=np.float64)
                out.update(status="compared", n_tokens=len(hf), hf_sum=float(np.sum(hf, dtype=np.float64)),
                           vllm_sum=float(sum(raw)), mean_abs=float(np.mean(np.abs(diff))),
                           max_abs=float(np.max(np.abs(diff))), hf_token_logprobs=hf.tolist(),
                           behavior_token_logprobs=raw,
                           response_token_ids=list(rec.full_token_ids[rec.prompt_len:rec.prompt_len+len(hf)]),
                           full_response_checked=len(hf) == len(rec.full_token_ids) - rec.prompt_len)
        else:
            out = compare_hf_vllm_logprob(
                model, llm, rec.full_token_ids, int(rec.prompt_len), pad_id,
                eos_id=eos_id, lora_request=lora_request, max_n=max_n,
            )
            out["source"] = "teacher_forced_prompt_logprobs"
        out["problem_id"] = rec.problem_id
        out["timing"] = "before_actor_update"
        out["prompt_len"] = int(rec.prompt_len)
        if out.get("status") == "compared":
            scores = out.get("behavior_token_logprobs", out.get("vllm_prompt_token_logprobs"))
            out["deviation"] = logprob_error_summary(out["hf_token_logprobs"], scores,
                out["response_token_ids"], rec.prompt_len, getattr(rec, "prefix_tokens", None))
        out["behavior_logprob_sum"] = rec.rollout_logprob_sum
        if out.get("full_response_checked") and rec.rollout_logprob_sum is not None:
            out["hf_minus_behavior_sum"] = out["hf_sum"] - rec.rollout_logprob_sum
        return out
    return {"status": "no_completed_sequence"}


def probe_first_completed(model, llm, records, pad_id: int, eos_id=None, lora_request=None, max_n=64) -> dict[str, Any]:
    try:
        return _probe_first_completed(model, llm, records, pad_id, eos_id, lora_request, max_n)
    except Exception as exc:
        return {"status": "unavailable", "error": f"{type(exc).__name__}: {exc}",
                "timing": "before_actor_update"}


def probe_behavior_batch(model, llm, records, pad_id, eos_id=None, lora_request=None,
                         max_n=0, max_sequences=0, include_stopped=True):
    """Cover observed behavior, including stopped prefixes; never generate suffixes.

    Round-robin across problem/outcome groups if coverage is explicitly limited.
    The summary is token-weighted and keeps every failed comparison in details.
    """
    import copy

    groups = {}
    for index, rec in enumerate(records or []):
        stopped = rec.z < 1.
        ids = getattr(rec, "prefix_token_ids", None) if stopped else rec.full_token_ids
        if (stopped and not include_stopped) or not ids or len(ids) <= rec.prompt_len:
            continue
        key = (rec.problem_id, "stopped_prefix" if stopped else "completed")
        groups.setdefault(key, []).append((index, rec, ids))
    candidates = []
    while any(groups.values()):
        for group in groups.values():
            if group:
                candidates.append(group.pop(0))
    chosen = candidates if max_sequences is None or int(max_sequences) <= 0 else candidates[:int(max_sequences)]
    details = []
    for index, rec, ids in chosen:
        proxy = copy.copy(rec)
        stopped = rec.z < 1.
        proxy.z, proxy.full_token_ids = 1., list(ids)
        if getattr(proxy, "rollout_token_logprobs", None) is None:
            out = {"status": "unavailable", "source": "sampled_behavior",
                   "reason": "original_behavior_scores_missing", "problem_id": rec.problem_id,
                   "timing": "before_actor_update"}
        else:
            out = probe_first_completed(model, llm, [proxy], pad_id, eos_id, lora_request, max_n)
        out.update(record_index=index, observed_scope="stopped_prefix" if stopped else "completed_response",
                   request_seeds=getattr(rec, "request_seeds", None),
                   prefix_response_tokens=getattr(rec, "prefix_tokens", None))
        details.append(out)
    compared = [row for row in details if row.get("status") == "compared"]
    tokens = sum(row["n_tokens"] for row in compared)
    counts = {}
    for row in details:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    worst = max(compared, key=lambda row: row["max_abs"], default=None)
    worst_token = None if worst is None else {**worst["deviation"]["max_error_token"],
        "record_index": worst["record_index"], "problem_id": worst["problem_id"],
        "observed_scope": worst["observed_scope"], "request_seeds": worst["request_seeds"]}
    return {
        "status": ("no_observed_sequence" if not details else
                   "compared" if len(compared) == len(details) else "partial"),
        "timing": "before_actor_update", "source": "observed_behavior_batch",
        "meaning": "numerical evidence; coverage and missing scores do not certify policy equality",
        "eligible_sequences": len(candidates), "checked_sequences": len(details),
        "coverage_complete": len(details) == len(candidates), "status_counts": counts,
        "n_tokens": tokens,
        "mean_abs": sum(row["mean_abs"] * row["n_tokens"] for row in compared) / tokens if tokens else None,
        "max_abs": max((row["max_abs"] for row in compared), default=None),
        "full_observed_tokens_checked": bool(details) and all(row.get("full_response_checked", False) for row in details),
        "numerics": probe_numerics(model, llm, lora_request), "worst_token": worst_token,
        "details": details,
    }
