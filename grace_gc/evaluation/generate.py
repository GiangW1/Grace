"""Independent full-answer generation. Training stoppers are not used."""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
from typing import Any

from grace_gc.data.format_prompt import apply_solve_instruction
from grace_gc.data.math_data import MathRecord, looks_like_math500, select_records, selection_manifest
from grace_gc.data.reward import REWARD_PROTOCOL_VERSION, extract_answer
from grace_gc.evaluation.eval_full import EvalItem, evaluate_items
from grace_gc.logging_util.forensics import persist_eval_items, persist_load_report, write_failed
from grace_gc.logging_util.ledger import ComputeLedger, Timer
from grace_gc.logging_util.run_dir import RunDirectory, resolve_run_dir, utc_now
from grace_gc.logging_util.run_log import RunLog
from grace_gc.trainer.grace_step import finish_reason
from grace_gc.versions import collect_environment


def _load_tiny_actor(cfg: dict[str, Any] | None = None):
    from grace_gc.core.rng import seed_all
    from grace_gc.trainer.cpu_tiny import TinyLoRAActor
    from grace_gc.trainer.state_io import load_numpy_module_state
    from grace_gc.trainer.checkpoint import load_checkpoint

    seed_all(int((cfg or {}).get("seed", 17)))
    actor = TinyLoRAActor()
    ckpt = None if cfg is None else (cfg.get("checkpoint") or cfg.get("resume"))
    if ckpt:
        payload = load_checkpoint(ckpt)
        if not payload.get("actor_full"):
            raise ValueError("tiny eval needs actor_full; this checkpoint is LoRA-only")
        load_numpy_module_state(actor.named_all_params(), payload["actor_full"])
    return actor


def generate_answers_tiny(
    records: list[MathRecord],
    n: int,
    max_new: int,
    seed: int,
    cfg: dict[str, Any] | None = None,
) -> list[EvalItem]:
    from grace_gc.core.rng import IsolatedRNG
    from grace_gc.data.tokenize import encode_records_tiny
    from grace_gc.trainer.tiny_engine import make_tiny_engines

    actor = _load_tiny_actor(cfg)
    engines = make_tiny_engines(actor, actor.vocab)
    prompt_max = int((cfg or {}).get("prompt_max_tokens", 1024))
    from grace_gc.trainer.algorithm import _length_truncated, _traj_natural_finish

    items = []
    sample_offset = 0
    eos_id = getattr(engines, "eos_id", None)
    for rec in records:
        prompt_meta: list = []
        prompts, _pids, _golds = encode_records_tiny([rec], actor.vocab, prompt_max, prompt_meta=prompt_meta)
        answers = []
        truncated = []
        extracted = []
        response_tokens = []
        finish_reasons = []
        token_ids = []
        sample_seeds = []
        for sample_i in range(n):
            request_seed = int(seed) + sample_offset
            request_rng = IsolatedRNG.create(request_seed)
            full = engines.continue_selected(prompts, [True], max_new, request_rng)
            sample_offset += 1
            if not full or full[0] is None:
                raise ValueError("eval generate returned no sequence")
            seq = full[0]
            resp = seq[len(prompts[0]) :]
            text = engines.decode(resp) if engines.decode else ""
            answers.append(text)
            ended = _traj_natural_finish(False, seq, eos_id, generated=len(resp), requested=max_new)
            trunc = _length_truncated(seq, len(prompts[0]), max_new, ended, eos_id)
            truncated.append(trunc)
            extracted.append(extract_answer(text))
            response_tokens.append(len(resp))
            finish_reasons.append(finish_reason(1.0, ended, trunc))
            token_ids.append([int(x) for x in resp])
            sample_seeds.append(request_seed)
        items.append(
            EvalItem(
                rec.problem_id,
                rec.answer,
                answers,
                truncated,
                extracted=extracted,
                response_tokens=response_tokens,
                finish_reasons=finish_reasons,
                token_ids=token_ids,
                sample_seeds=sample_seeds,
                prompt_truncated=[bool((prompt_meta or [{}])[0].get("prompt_truncated"))] * n,
            )
        )
    return items


def generate_answers_vllm(
    records: list[MathRecord],
    tokenizer,
    llm,
    n: int,
    max_new: int,
    temperature: float,
    top_p: float,
    seed: int,
    lora_request=None,
    max_prompt: int = 1024,
    sample_batch_size: int = 1,
    request_recorder=None,
) -> list[EvalItem]:
    from grace_gc.backends.vllm_two_phase import (
        _require_known_finish,
        _require_vllm,
        _vllm_prompts,
        build_sampling_params,
        generated_was_truncated,
        trim_generated_tokens,
    )
    from grace_gc.data.tokenize import collect_stop_token_ids, decode_hf, encode_records_hf

    _require_vllm()
    items = []
    sample_i = 0
    chunk_i = 0
    eos_id = collect_stop_token_ids(tokenizer)
    for rec in records:
        prompt_meta: list = []
        prompts, _pids, _golds = encode_records_hf([rec] * n, tokenizer, int(max_prompt), prompt_meta=prompt_meta)
        answers = []
        truncated = []
        extracted = []
        response_tokens = []
        finish_reasons = []
        token_ids = []
        sample_seeds = []
        vllm_finish_reasons = []
        fulls = []
        generated = []
        execution = []
        request_seeds = [int(seed) + sample_i + i for i in range(len(prompts))]
        sample_i += len(prompts)
        batch_size = max(int(sample_batch_size), 1)
        for start in range(0, len(prompts), batch_size):
            chunk = prompts[start:start + batch_size]
            params = [build_sampling_params(int(max_new), float(temperature), request_seeds[i],
                       eos_id=eos_id, top_p=float(top_p)) for i in range(start, start + len(chunk))]
            gen_kwargs = {"use_tqdm": False}
            if lora_request is not None:
                gen_kwargs["lora_request"] = lora_request
            fallback = None
            try:
                outputs = llm.generate(_vllm_prompts(chunk),
                                       sampling_params=params[0] if len(chunk) == 1 else params, **gen_kwargs)
            except TypeError as exc:
                fallback = str(exc)
                outputs = []
                for prompt, param in zip(chunk, params):
                    outputs.extend(llm.generate(_vllm_prompts([prompt]), sampling_params=param, **gen_kwargs))
            if len(outputs) != len(chunk):
                raise ValueError(f"vLLM returned {len(outputs)} outputs for {len(chunk)} prompts")
            generated.extend(outputs)
            execution.extend({"chunk_index": chunk_i, "requested_batch_size": len(chunk),
                              "execution_batch_size": 1 if fallback is not None else len(chunk),
                              "serial_fallback": fallback is not None, "serial_fallback_reason": fallback}
                             for _ in outputs)
            chunk_i += 1
        for local_i, (prompt, output, request_seed) in enumerate(zip(prompts, generated, request_seeds)):
            out_prompt = getattr(output, "prompt_token_ids", None)
            if out_prompt is None:
                raise ValueError("vLLM output is missing prompt_token_ids; cannot verify request order")
            if [int(x) for x in out_prompt] != [int(x) for x in prompt]:
                raise ValueError("vLLM output prompt_token_ids do not match the request order")
            completions = getattr(output, "outputs", None)
            if not completions:
                raise ValueError("vLLM returned a request with no completions")
            completion = completions[0]
            finish_reason = getattr(completion, "finish_reason", None)
            _require_known_finish(finish_reason)
            gen, ended = trim_generated_tokens(
                completion.token_ids,
                eos_id,
                finish_reason,
                getattr(completion, "stop_reason", None),
            )
            fulls.append(list(prompt) + list(gen))
            text = decode_hf(tokenizer, gen)
            answers.append(text)
            trunc = generated_was_truncated(len(gen), max_new, ended, finish_reason)
            truncated.append(trunc)
            extracted.append(extract_answer(text))
            response_tokens.append(len(gen))
            finish_reasons.append("length" if trunc else ("eos" if ended else "completed"))
            vllm_finish_reasons.append(None if finish_reason is None else str(finish_reason))
            token_ids.append([int(x) for x in gen])
            sample_seeds.append(request_seed)
            if request_recorder is not None:
                token_hash = lambda ids: hashlib.sha256(json.dumps([int(x) for x in ids], separators=(",", ":")).encode()).hexdigest()
                request_recorder({"problem_id": rec.problem_id, "sample_index": local_i,
                    "request_index": sample_i-len(prompts)+local_i, "request_seed": request_seed,
                    "request_id": None if getattr(output, "request_id", None) is None else str(output.request_id),
                    "prompt_token_count": len(prompt), "prompt_token_sha256": token_hash(prompt),
                    "response_token_count": len(gen), "response_token_sha256": token_hash(gen),
                    "finish_reason": finish_reason, "stop_reason": getattr(completion, "stop_reason", None),
                    **execution[local_i],
                    "scope": "Observed execution metadata; identical prompts alone cannot verify the order of duplicate requests. No bitwise determinism claim."})
        # Equal answers are valid independent samples; keep their request seeds.
        items.append(
            EvalItem(
                rec.problem_id,
                rec.answer,
                answers,
                truncated,
                extracted=extracted,
                response_tokens=response_tokens,
                finish_reasons=finish_reasons,
                token_ids=token_ids,
                sample_seeds=sample_seeds,
                prompt_truncated=[bool(m.get("prompt_truncated")) for m in prompt_meta] or None,
                vllm_finish_reasons=vllm_finish_reasons,
            )
        )
        print(
            f"phase=eval_problem_done problems={len(items)}/{len(records)} "
            f"samples={sample_i} problem_id={rec.problem_id} "
            f"response_tokens={sum(response_tokens)} truncated={sum(truncated)}",
            flush=True,
        )
    return items


def _eval_hparams(cfg: dict[str, Any]) -> tuple[int, int, int, float, float]:
    ev = cfg.get("eval") or {}
    k = int(cfg.get("eval_k", ev.get("k", 1)))
    n = int(cfg.get("eval_n", ev.get("n", max(k, 2))))
    if cfg.get("eval_max_new_tokens") is not None:
        max_new = int(cfg["eval_max_new_tokens"])
    elif ev.get("max_new_tokens") is not None:
        max_new = int(ev["max_new_tokens"])
    else:
        from grace_gc.trainer.methods import method_spec

        spec = method_spec(str(cfg.get("method", "grace")))
        # Paper eval is 4096 for every method. GRPO-short's 1024 is a train cap.
        # GPU without an eval block must not inherit train 2048 while
        # GRPO-short still gets 4096 — that would confound the comparison.
        if spec.max_new_tokens is not None or str(cfg.get("backend", "cpu_tiny")) == "gpu_verl":
            max_new = 4096
        else:
            max_new = int(cfg.get("max_new_tokens", 16))
    if "temperature" in ev:
        temperature = float(ev["temperature"])
    else:
        temperature = float(cfg.get("eval_temperature", 0.6))
    if "top_p" in ev:
        top_p = float(ev["top_p"])
    else:
        top_p = float(cfg.get("eval_top_p", 0.95))
    return k, n, max_new, temperature, top_p


def limit_eval_records(records: list[MathRecord], cfg: dict[str, Any]) -> list[MathRecord]:
    """0 or missing means the full split. Smoke sets eval.n_problems."""
    ev = cfg.get("eval") or {}
    n_problems = int(cfg.get("eval_n_problems", ev.get("n_problems", 0)) or 0)
    if n_problems < 0:
        raise ValueError(f"eval.n_problems must be >= 0, got {n_problems}")
    return select_records(records, n_problems, str(ev.get("selection", "first")),
                          int(ev.get("selection_seed", cfg.get("split_seed", cfg.get("seed", 17)))))


def eval_engine_config(cfg, max_new):
    from grace_gc.backends.verl_trainer import vllm_needed_max_model_len

    options = dict(cfg.get("vllm") or {})
    options.setdefault("seed", int(cfg.get("seed", 17)))
    options["max_model_len"] = vllm_needed_max_model_len(cfg, max_new)
    return options


def _generate_eval_items(records, cfg, backend, n, max_new, temperature, top_p, run_dir):
    if backend == "gpu_verl":
        if not cfg.get("model_path"):
            raise ValueError("model_path is required for GPU evaluation")
        ckpt = cfg.get("checkpoint") or cfg.get("resume")
        if not ckpt:
            raise ValueError("GPU evaluation needs a training checkpoint")
        from grace_gc.backends.hf_actor import named_lora_params
        from grace_gc.backends.verl_trainer import build_vllm_engine, load_lora_actor
        from grace_gc.backends.weight_sync import apply_lora_request, make_lora_request, reset_vllm_prefix_cache, save_lora_adapter
        from grace_gc.data.tokenize import load_hf_tokenizer, tokenizer_inventory
        from grace_gc.data.reward import require_math_verify
        from grace_gc.trainer.checkpoint import load_checkpoint
        from grace_gc.trainer.state_io import check_snapshot_identity, load_numpy_module_state

        require_math_verify()
        from grace_gc.core.rng import seed_all

        seed_all(int(cfg.get("seed", 17)))
        tok = load_hf_tokenizer(str(cfg["model_path"]))
        RunDirectory(run_dir).write_json("tokenizer.json", tokenizer_inventory(tok))
        actor = load_lora_actor(str(cfg["model_path"]), cfg.get("lora", {}))
        payload = load_checkpoint(ckpt)
        check_snapshot_identity(payload, cfg)
        load_numpy_module_state(named_lora_params(actor), payload.get("actor") or {})
        from grace_gc.versions import sha256_named, sha256_file

        actor_source = {
            "checkpoint": str(ckpt), "checkpoint_step": payload.get("step"),
            "method": payload.get("spec"), "actor_sha256": sha256_named(named_lora_params(actor)),
            "hash_stage": "loaded_checkpoint_before_generation", "layout": "all_qv_lora_A_B",
        }
        adapter = Path(run_dir) / "eval_lora"
        save_lora_adapter(actor, adapter)
        actor_source["exported_adapter_files"] = {path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
                                                   for path in sorted(adapter.iterdir()) if path.is_file()}
        actor_source["identity_scope"] = "Loaded HF LoRA tensors and exported adapter files; no independent vLLM worker tensor hash or full base-weight hash."
        RunDirectory(run_dir).write_json("actor_source.json", actor_source)
        del actor
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        vllm_cfg = eval_engine_config(cfg, max_new)
        llm = build_vllm_engine(str(cfg["model_path"]), vllm_cfg, int(cfg.get("lora", {}).get("rank", 16)))
        if getattr(build_vllm_engine, "last", None):
            RunDirectory(run_dir).write_json("vllm_engine.json", build_vllm_engine.last)
        lora_request = make_lora_request(adapter, 1)
        apply_lora_request(llm, lora_request)
        reset_vllm_prefix_cache(llm)
        items = generate_answers_vllm(
            records,
            tok,
            llm,
            n,
            max_new,
            temperature,
            top_p,
            int(cfg.get("seed", 17)),
            lora_request=lora_request,
            max_prompt=int(cfg.get("prompt_max_tokens", 1024)),
            sample_batch_size=int((cfg.get("eval") or {}).get("sample_batch_size", 1)),
            request_recorder=lambda row: RunDirectory(run_dir).append_jsonl("eval_requests.jsonl", row),
        )
    else:
        items = generate_answers_tiny(records, n, max_new, int(cfg.get("seed", 17)), cfg=cfg)
    return items


def run_eval(records: list[MathRecord], cfg: dict[str, Any], run_dir: str | Path) -> dict:
    timer = Timer()
    started = utc_now()
    cfg = dict(cfg)
    backend = cfg.get("backend", "cpu_tiny")
    requested = Path(run_dir)
    run_dir = resolve_run_dir(run_dir)
    run = RunDirectory(run_dir)
    n_gpu = int((cfg.get("hardware") or {}).get("n_gpu", 1 if backend == "gpu_verl" else 0))
    hardware = str((cfg.get("hardware") or {}).get("name", "gpu" if backend == "gpu_verl" else "cpu"))
    ledger = ComputeLedger(n_gpu=n_gpu, hardware=hardware)
    versions, status = {}, "failed"
    try:
        ckpt = cfg.get("checkpoint") or cfg.get("resume")
        if ckpt:
            from grace_gc.trainer.checkpoint import load_checkpoint

            payload = load_checkpoint(ckpt)
            cfg["method"] = payload.get("spec") or cfg.get("method", "grace")
        n_source = len(records)
        math500 = looks_like_math500(records, path=cfg.get("data_path"), n_source=n_source)
        records = apply_solve_instruction(records)
        records = limit_eval_records(records, cfg)
        k, n, max_new, temperature, top_p = _eval_hparams(cfg)
        versions = collect_environment(cfg)
        versions["started"] = started
        run.write_run_meta(kind="eval", started=started, requested=requested)
        run.write_yaml("config.yaml", cfg)
        run.write_json("environment.json", versions)
        ev = cfg.get("eval") or {}
        manifest = selection_manifest(records, str(ev.get("selection", "first")),
                                      int(ev.get("selection_seed", cfg.get("split_seed", cfg.get("seed", 17)))))
        manifest.update(sample_seed_start=int(cfg.get("seed", 17)), samples_per_problem=n,
                        temperature=temperature, top_p=top_p, max_new_tokens=max_new,
                        sample_batch_size=int(ev.get("sample_batch_size", 1)),
                        reward_protocol_version=REWARD_PROTOCOL_VERSION)
        run.write_json("evaluation_manifest.json", manifest)
        with RunLog(run.root / "run.log"):
            print(f"eval start {started} backend={backend} n_problems={len(records)} n={n} k={k} dir={run.root}")
            if Path(run.root).resolve() != requested.resolve():
                print(f"run-dir {requested} already had artifacts; writing to {run.root}")
            items = _generate_eval_items(records, cfg, backend, n, max_new, temperature, top_p, run.root)
            result = evaluate_items(items, k=k)
            result["seed"] = int(cfg.get("seed", 17))
            result["checkpoint"] = cfg.get("checkpoint") or cfg.get("resume")
            result["checkpoint_step"] = payload.get("step") if ckpt else None
            result["actor_source"] = "checkpoint" if result["checkpoint"] else "base_or_random"
            result["method"] = cfg.get("method")
            result["evaluation_manifest"] = manifest
            result["measurement_version"] = 2
            result["n_problems_requested"] = len(records)
            result["started"] = started
            result["run_dir"] = str(run.root)
            persist_eval_items(run, items, cfg, result)
            from grace_gc.backends.vllm_two_phase import build_sampling_params

            if getattr(build_sampling_params, "last", None):
                run.write_json("sampling.json", build_sampling_params.last)
            report = persist_load_report(run, data_path=cfg.get("data_path"))
            run.write_json(
                "data_splits.json",
                {
                    "data_path": cfg.get("data_path"),
                    "eval_split": cfg.get("eval_split", "eval"),
                    "n_source": n_source,
                    "n_loaded": len(records),
                    "n_problems": result.get("n_problems"),
                    "looks_like_math500": math500,
                    "load": None
                    if report is None
                    else {
                        "n_raw": report.get("n_raw"),
                        "n_kept": report.get("n_kept"),
                        "n_conflict_groups": report.get("n_conflict_groups"),
                    },
                },
            )
            print(f"eval done avg={result.get('avg')} pass_at_k={result.get('pass_at_k')}")
        result["finished"] = utc_now()
        run.write_json("eval_summary.json", result)
        status = "completed"
        return result
    except Exception as exc:
        status = "failed"
        write_failed(run, exc, versions, started)
        raise
    finally:
        ledger.add("eval", timer.elapsed(), cpu_s=timer.cpu_elapsed(), status=status,
                   timing_scope="Function entry through setup, computation and result/failure persistence; excludes final ledger publication and caller-side CLI/data loading.")
        run.write_json("compute_ledger.json", ledger.summary())
        run.append_jsonl("compute_ledger.jsonl", ledger.rows[-1])
