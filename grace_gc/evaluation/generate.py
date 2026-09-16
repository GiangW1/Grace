"""Independent full-answer generation. Training stoppers are not used."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from grace_gc.data.format_prompt import apply_solve_instruction
from grace_gc.data.math_data import MathRecord, looks_like_math500
from grace_gc.data.reward import extract_answer
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
    rng = IsolatedRNG.create(seed)
    prompt_max = int((cfg or {}).get("prompt_max_tokens", 1024))
    from grace_gc.trainer.algorithm import _length_truncated, _traj_natural_finish

    items = []
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
            full = engines.continue_selected(prompts, [True], max_new, rng)
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
            sample_seeds.append(int(seed) + sample_i)
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
        for prompt in prompts:
            params = build_sampling_params(
                int(max_new),
                float(temperature),
                int(seed) + sample_i,
                eos_id=eos_id,
                top_p=float(top_p),
            )
            sample_i += 1
            gen_kwargs = {"sampling_params": params, "use_tqdm": False}
            if lora_request is not None:
                gen_kwargs["lora_request"] = lora_request
            outputs = llm.generate(_vllm_prompts([prompt]), **gen_kwargs)
            if len(outputs) != 1:
                raise ValueError(f"vLLM returned {len(outputs)} outputs for 1 prompt")
            out_prompt = getattr(outputs[0], "prompt_token_ids", None)
            if out_prompt is None:
                raise ValueError("vLLM output is missing prompt_token_ids; cannot verify request order")
            if [int(x) for x in out_prompt] != [int(x) for x in prompt]:
                raise ValueError("vLLM output prompt_token_ids do not match the request order")
            completions = getattr(outputs[0], "outputs", None)
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
            sample_seeds.append(int(seed) + sample_i - 1)
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
    if n_problems == 0:
        return list(records)
    return list(records[:n_problems])


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
        tok = load_hf_tokenizer(str(cfg["model_path"]))
        RunDirectory(run_dir).write_json("tokenizer.json", tokenizer_inventory(tok))
        actor = load_lora_actor(str(cfg["model_path"]), cfg.get("lora", {}))
        payload = load_checkpoint(ckpt)
        check_snapshot_identity(payload, cfg)
        load_numpy_module_state(named_lora_params(actor), payload.get("actor") or {})
        adapter = Path(run_dir) / "eval_lora"
        save_lora_adapter(actor, adapter)
        del actor
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        vllm_cfg = dict(cfg.get("vllm") or {})
        need = int(cfg.get("prompt_max_tokens", 1024)) + int(max_new)
        have = int(vllm_cfg.get("max_model_len", 0) or 0)
        vllm_cfg["max_model_len"] = max(have, 5120, need)
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
        )
    else:
        items = generate_answers_tiny(records, n, max_new, int(cfg.get("seed", 17)), cfg=cfg)
    return items


def run_eval(records: list[MathRecord], cfg: dict[str, Any], run_dir: str | Path) -> dict:
    backend = cfg.get("backend", "cpu_tiny")
    n_source = len(records)
    math500 = looks_like_math500(records, path=cfg.get("data_path"), n_source=n_source)
    records = apply_solve_instruction(records)
    records = limit_eval_records(records, cfg)
    k, n, max_new, temperature, top_p = _eval_hparams(cfg)
    requested = Path(run_dir)
    run_dir = resolve_run_dir(run_dir)
    run = RunDirectory(run_dir)
    started = utc_now()
    versions = collect_environment(cfg)
    versions["started"] = started
    run.write_run_meta(kind="eval", started=started, requested=requested)
    run.write_yaml("config.yaml", cfg)
    run.write_json("environment.json", versions)
    n_gpu = int((cfg.get("hardware") or {}).get("n_gpu", 1 if backend == "gpu_verl" else 0))
    hardware = str((cfg.get("hardware") or {}).get("name", "gpu" if backend == "gpu_verl" else "cpu"))
    ledger = ComputeLedger(n_gpu=n_gpu, hardware=hardware)
    timer = Timer()
    try:
        with RunLog(run.root / "run.log"):
            print(f"eval start {started} backend={backend} n_problems={len(records)} n={n} k={k} dir={run.root}")
            if Path(run.root).resolve() != requested.resolve():
                print(f"run-dir {requested} already had artifacts; writing to {run.root}")
            items = _generate_eval_items(records, cfg, backend, n, max_new, temperature, top_p, run.root)
            result = evaluate_items(items, k=k)
            result["seed"] = int(cfg.get("seed", 17))
            result["checkpoint"] = cfg.get("checkpoint") or cfg.get("resume")
            result["actor_source"] = "checkpoint" if result["checkpoint"] else "base_or_random"
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
        ledger.add("eval", timer.elapsed())
        result["finished"] = utc_now()
        run.write_json("eval_summary.json", result)
        run.write_json("compute_ledger.json", ledger.summary())
        run.append_jsonl("compute_ledger.jsonl", ledger.rows[-1])
        return result
    except Exception as exc:
        write_failed(run, exc, versions, started)
        run.write_json("compute_ledger.json", ledger.summary())
        raise
