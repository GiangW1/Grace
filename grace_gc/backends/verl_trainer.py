"""GPU training loop: data → two-phase rollout → dual-stream update."""

from __future__ import annotations

import atexit
from pathlib import Path
import sys
from typing import Any

import numpy as np

from grace_gc.backends import require_gpu_stack
from grace_gc.backends.fsdp_actor import maybe_wrap_fsdp
from grace_gc.backends.gpu_engine import make_gpu_engines
from grace_gc.backends.hf_actor import actor_numerics, named_lora_params
from grace_gc.core.layout import collect_lora_layout
from grace_gc.core.rng import IsolatedRNG, seed_all
from grace_gc.data.math_data import load_training_data
from grace_gc.data.tokenize import encode_records_hf, load_hf_tokenizer, tokenizer_inventory
from grace_gc.trainer.format_warmup import format_warmup_steps, run_format_warmup_hf
from grace_gc.logging_util.forensics import persist_final_checkpoint, persist_initial_checkpoint, persist_training_step, write_data_inventory
from grace_gc.logging_util.ledger import ComputeLedger, Timer
from grace_gc.logging_util.run_dir import RunDirectory
from grace_gc.predictor.heads import predictor_from_spec
from grace_gc.predictor.reservoir import GradientReservoir
from grace_gc.trainer.algorithm import TrainState, run_algorithm1_step
from grace_gc.trainer.baseline import baseline_from_config
from grace_gc.trainer.loop import resolve_start_counts, resume_start_counts, sample_prompt_indices, sample_starts
from grace_gc.trainer.methods import apply_method_defaults, method_spec
from grace_gc.trainer.state_io import restore_train_state
from grace_gc.trainer.cost_control import cost_start_counts, ensure_cost_control, observe_batch_cost, start_cost_control, start_count_mode


def _ensure_bf16(model):
    import torch

    try:
        current = next(model.parameters()).dtype
    except StopIteration:
        return model
    if current != torch.bfloat16:
        model = model.to(dtype=torch.bfloat16)
    return model


def load_lora_actor(model_path: str, lora_cfg: dict[str, Any]):
    require_gpu_stack()
    import torch
    from transformers import AutoModelForCausalLM

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, trust_remote_code=True
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
    model = _ensure_bf16(model)
    try:
        from peft import LoraConfig, get_peft_model
    except Exception as exc:
        raise ImportError("peft is required to attach q/v LoRA on the actor") from exc
    cfg = LoraConfig(
        r=int(lora_cfg.get("rank", 16)),
        lora_alpha=float(lora_cfg.get("alpha", 32)),
        lora_dropout=float(lora_cfg.get("dropout", 0.0)),
        target_modules=list(lora_cfg.get("targets", ["q_proj", "v_proj"])),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, cfg)
    model._grace_compute_dtype = str(lora_cfg.get("compute_dtype", "native"))
    # Dropout-free behavior for scoring and generation; gradients remain enabled.
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model


class VLLMWorkerCleanup:
    """Expose process-group cleanup through vLLM's named worker RPC API."""

    def grace_destroy_process_groups(self) -> None:
        from vllm.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel

        destroy_model_parallel()
        destroy_distributed_environment()


def _shutdown_vllm_engine(llm) -> None:
    core = llm.llm_engine.engine_core
    try:
        rpc = getattr(llm, "collective_rpc", None)
        if callable(rpc):
            rpc("grace_destroy_process_groups")
    finally:
        core.shutdown()


def vllm_needed_max_model_len(cfg: dict[str, Any], max_new: int | None = None) -> int:
    """Leave slack so prompt + eval 4096 is not exactly max_model_len."""
    prompt = int(cfg.get("prompt_max_tokens", 1024) or 1024)
    ev = cfg.get("eval") or {}
    gen = int(max_new if max_new is not None else ev.get("max_new_tokens") or cfg.get("max_new_tokens") or 4096)
    have = int((cfg.get("vllm") or {}).get("max_model_len") or cfg.get("max_model_len") or 0)
    return max(have, 5120, prompt + int(gen) + 64)


def build_vllm_engine(model_path: str, cfg: dict[str, Any], lora_rank: int):
    from grace_gc.backends.vllm_two_phase import _require_vllm

    LLM, _SP = _require_vllm()
    kwargs = {
        "model": model_path,
        "worker_extension_cls": "grace_gc.backends.verl_trainer.VLLMWorkerCleanup",
        "enable_prefix_caching": True,
        "enable_lora": True,
        "max_lora_rank": int(lora_rank),
        "max_loras": 4,
        "tensor_parallel_size": int(cfg.get("tensor_parallel", 1)),
        "dtype": cfg.get("dtype", "bfloat16"),
        "lora_dtype": cfg.get("lora_dtype", "bfloat16"),
        "seed": int(cfg.get("seed", 17)),
        "trust_remote_code": True,
        # Qwen3-*-Base ships max_new_tokens=2048. vLLM's default
        # generation_config="auto" can cap every request at 2048, so
        # paper eval (4096) would silently truncate.
        "generation_config": "vllm",
        # Default 0.9 would pre-allocate almost the whole GPU. Train and
        # audit keep a second 4B HF actor on the same card.
        "gpu_memory_utilization": float(cfg.get("gpu_memory_utilization", 0.5)),
        # Prompt 1024 + paper eval 4096. A profile-time shrink below this
        # would silently cap eval.
        "max_model_len": int(cfg.get("max_model_len", 5120)),
    }
    dropped: list[str] = []
    try:
        llm = LLM(**kwargs)
    except TypeError:
        kwargs.pop("worker_extension_cls", None)
        dropped.append("worker_extension_cls")
        try:
            llm = LLM(**kwargs)
        except TypeError:
            kwargs.pop("max_loras", None)
            dropped.append("max_loras")
            try:
                llm = LLM(**kwargs)
            except TypeError:
                kwargs.pop("trust_remote_code", None)
                dropped.append("trust_remote_code")
                try:
                    llm = LLM(**kwargs)
                except TypeError as exc:
                    raise ValueError(
                        'vLLM LLM rejected generation_config="vllm"; '
                        "Qwen3-Base max_new_tokens=2048 would silently cap eval 4096"
                    ) from exc
    build_vllm_engine.last = {"accepted": dict(kwargs), "dropped": dropped}
    # Stop the worker before Python's multiprocessing finalizers terminate it.
    core = getattr(getattr(llm, "llm_engine", None), "engine_core", None)
    shutdown = getattr(core, "shutdown", None)
    if callable(shutdown):
        atexit.register(_shutdown_vllm_engine, llm)
    return llm


def train(cfg: dict[str, Any], run: RunDirectory, ledger: ComputeLedger | None = None) -> dict[str, Any]:
    resources = {}
    try:
        return _train(cfg, run, ledger, resources)
    finally:
        pool = resources.get("pool")
        if pool is not None:
            pool.close()


def _train(cfg: dict[str, Any], run: RunDirectory, ledger: ComputeLedger | None, resources) -> dict[str, Any]:
    start_cost_control(cfg, Timer())
    n_gpu = int(cfg.get("hardware", {}).get("n_gpu", 1))
    from grace_gc.backends.rollout_pool import RolloutPool, validate_layout
    try:
        workers = validate_layout(cfg)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    versions = require_gpu_stack()
    if workers:
        import torch
        torch.cuda.set_device(0)
    cfg = apply_method_defaults(cfg)
    spec = method_spec(cfg.get("method", "grace"))
    if spec.max_new_tokens:
        cfg = {**cfg, "max_new_tokens": spec.max_new_tokens}
    model_path = cfg.get("model_path")
    if not model_path:
        raise ValueError("model_path is required for GPU training")
    if Path(str(model_path)).is_absolute() and not Path(str(model_path)).exists():
        raise FileNotFoundError(f"model path is not readable: {model_path}")

    data_path = cfg.get("data_path")
    tokenizer = load_hf_tokenizer(str(model_path))
    run.write_json("tokenizer.json", tokenizer_inventory(tokenizer))
    if cfg.get("prompt_token_ids"):
        prompt_pool = list(cfg["prompt_token_ids"])
        problem_pool = [str(i) for i in range(len(prompt_pool))]
        gold_pool = list(cfg.get("golds", [""] * len(prompt_pool)))
        class _Rec:
            def __init__(self, pid, prompt, answer):
                self.problem_id = pid
                self.prompt = prompt
                self.answer = answer

        # Keep a record-shaped pool so sample_starts still works.
        train_recs = None
        packed = list(zip(prompt_pool, problem_pool, gold_pool))
    else:
        packed = None
        if not data_path:
            raise ValueError("data_path or prompt_token_ids is required for GPU training")
        buckets, report = load_training_data(data_path, cfg.get("eval_data_path"), seed=int(cfg.get("split_seed", 17)))
        train_recs = buckets["train"]
        write_data_inventory(run, buckets, data_path, source="file", load_report=report,
                             split_seed=int(cfg.get("split_seed", 17)))
        if not train_recs:
            raise ValueError("training split is empty")
        if report and int(report.get("n_conflict_groups") or 0) > 0:
            print(
                f"dropped {report['n_conflict_groups']} conflicting prompt groups; "
                "see data_conflicts.json"
            )

    seed_all(int(cfg.get("seed", 17)))
    from grace_gc.data.reward import require_math_verify

    require_math_verify()
    if ledger is None:
        ledger = ComputeLedger(n_gpu=max(n_gpu, 1), hardware=str(cfg.get("hardware", {}).get("name", "gpu")))
    actor = load_lora_actor(str(model_path), cfg.get("lora", {}))
    actor = maybe_wrap_fsdp(actor, 1)
    from grace_gc.trainer.initialization import initialize_actor

    initial_actor = initialize_actor(actor, cfg, run)
    if format_warmup_steps(cfg) and not cfg.get("resume") and not initial_actor and train_recs:
        info = run_format_warmup_hf(actor, tokenizer, train_recs, cfg, ledger)
        run.write_json("format_warmup.json", info)
        print(f"format warmup steps={info.get('steps')} last_loss={info.get('last_loss')}")
    vllm_cfg = dict(cfg.get("vllm") or {})
    vllm_cfg.setdefault("seed", int(cfg.get("seed", 17)))
    vllm_cfg["max_model_len"] = vllm_needed_max_model_len(cfg)
    if workers:
        llm = RolloutPool.launch(str(model_path), vllm_cfg, int(cfg.get("lora", {}).get("rank", 16)), n_gpu)
        resources["pool"] = llm
        run.write_json("vllm_engine.json", llm.metadata)
    else:
        llm = build_vllm_engine(str(model_path), vllm_cfg, int(cfg.get("lora", {}).get("rank", 16)))
    if not workers and getattr(build_vllm_engine, "last", None):
        run.write_json("vllm_engine.json", build_vllm_engine.last)
    engines, extra = make_gpu_engines(actor, llm, tokenizer, cfg, Path(run.root) / "lora")
    def observed_call(phase, step, action):
        timer = Timer()
        try:
            return action()
        except Exception as exc:
            roll = engines.last_rollout or {}
            failed_attempt = None
            if phase == "algorithm_step":
                phases = [("prefix", 1, roll.get("prefix_execution") or {})]
                phases.extend(("continue", i + 1, item)
                              for i, item in enumerate(roll.get("continue_execution") or []))
                for name, index, item in phases:
                    calls = item.get("generate_calls") or []
                    if calls and calls[-1].get("status") == "failed":
                        failed_attempt = {**calls[-1], "phase": name, "phase_call_index": index,
                                          "generate_call_index": len(calls),
                                          "request_indices": item.get("request_indices")}
            payload = {
                "step": int(step), "phase": phase,
                "error_type": type(exc).__name__, "error": str(exc),
                "phase_attempt_wall_seconds": timer.elapsed(),
                "failed_attempt": failed_attempt,
                "rollout_execution": {key: roll[key] for key in (
                    "prefix_execution", "continue_execution", "prefix_num_cached_tokens",
                    "continue_num_cached_tokens") if key in roll},
                "last_sync": extra.get("last_sync"),
                "scope": "Failure observations only; timings remain included in failed-run costs; call indices are one-based",
            }
            try:
                run.write_json("failed_execution.json", payload)
            except Exception as write_error:
                note = f"failed_execution.json could not be written: {write_error}"
                if hasattr(exc, "add_note"):
                    exc.add_note(note)
                else:
                    print(note, file=sys.stderr)
            raise

    from grace_gc.backends.logprob_probe import probe_behavior_batch

    engines.before_update = lambda records: probe_behavior_batch(
        actor, llm, records, int(extra.get("pad_id") or 0), eos_id=extra.get("eos_id"),
        lora_request=extra.get("lora_request"), max_n=(cfg.get("diagnostics") or {}).get("probe_tokens", 64),
        max_sequences=(cfg.get("diagnostics") or {}).get("probe_sequences", 1),
        include_stopped=bool((cfg.get("diagnostics") or {}).get("probe_stopped_prefixes", True)),
    )
    observed_call("initial_sync", 0, extra["sync"])
    print(
        "phase=post_sync implementation=gpu_vllm_hf "
        f"adapter={extra.get('adapter_path')} lora_id={extra.get('lora_id')} "
        "(lora/step-N is the vLLM adapter id, not state.step; first sync is step-2)",
        flush=True,
    )

    import torch

    named = named_lora_params(actor)
    layout = collect_lora_layout(named)
    rng = IsolatedRNG.create(int(cfg.get("seed", 17)))
    n_prompts, starts_per, n_start = resolve_start_counts(cfg, spec)

    def _batch_ids(draw_rng):
        meta: list = []
        if packed is not None:
            pr, pi, go = [], [], []
            for i in sample_prompt_indices(len(packed), n_prompts, draw_rng):
                i = int(i)
                for _j in range(starts_per):
                    pr.append(packed[i][0])
                    pi.append(packed[i][1])
                    go.append(packed[i][2])
            return pr, pi, go, meta
        batch = sample_starts(train_recs, n_prompts, starts_per, draw_rng)
        pr, pi, go = encode_records_hf(
            batch, tokenizer, int(cfg.get("prompt_max_tokens", 1024)), prompt_meta=meta
        )
        return pr, pi, go, meta

    pcfg = cfg.get("predictor") or {}
    k = min(int(pcfg.get("k", 8)), layout.dim)
    in_dim = 1
    predictor = None
    if spec.use_predictor:
        # The feature schema is 3 hidden vectors + 3 entropy + length + b.
        # Avoid a method-specific generation request before the common start.
        in_dim = 3 * int(actor.config.hidden_size) + 5
        predictor = predictor_from_spec(in_dim, k, {**pcfg, "train_auxiliary": spec.risk_mode == "reward"})
    opt = torch.optim.AdamW(
        [p for _, p in named],
        lr=float(cfg.get("optim", {}).get("lr", 1e-4)),
        betas=tuple(cfg.get("optim", {}).get("betas", [0.9, 0.99])),
        weight_decay=float(cfg.get("optim", {}).get("weight_decay", 0.0)),
    )
    state = TrainState(
        spec=spec,
        baseline=baseline_from_config(cfg.get("baseline")),
        rng=rng,
        layout=layout,
        u=np.eye(layout.dim, k, dtype=np.float64),
        predictor=predictor,
        reservoir=GradientReservoir(capacity=int(cfg.get("predictor", {}).get("reservoir_size", 512))),
        n_ref=n_start,
    )
    if cfg.get("resume"):
        from grace_gc.trainer.checkpoint import load_checkpoint
        from grace_gc.trainer.state_io import check_snapshot_identity

        resume_payload = cfg.pop("_resume_payload", None)
        if resume_payload is None:
            resume_payload = load_checkpoint(cfg["resume"])
        check_snapshot_identity(resume_payload, cfg)
        state = restore_train_state(cfg["resume"], actor, opt, in_dim, k, spec.name, payload=resume_payload)
        from grace_gc.logging_util.forensics import record_effective_config

        record_effective_config(run, cfg, state, opt)
        observed_call("resume_sync", state.step, extra["sync"])
        print(
            f"phase=resume_sync adapter={extra.get('adapter_path')} lora_id={extra.get('lora_id')}",
            flush=True,
        )
        n_prompts, starts_per, n_start = resume_start_counts(cfg, spec, state)

    from grace_gc.predictor.offline import attach_predictor
    attach_predictor(state, cfg, run)
    run.write_json(
        "startup.json",
        {
            "implementation": "gpu_vllm_hf",
            "backend_alias": "gpu_verl",
            "note": "HF actor + vLLM two-phase generation; config backend gpu_verl is not verl PPO",
            "method": spec.name,
            "n_start": n_start,
            "n_prompts": n_prompts,
            "starts_per": starts_per,
            "adapter_path": None if extra.get("adapter_path") is None else str(extra.get("adapter_path")),
            "lora_id": extra.get("lora_id"),
            "format_warmup_steps": 0 if cfg.get("resume") or initial_actor else format_warmup_steps(cfg),
            "initial_actor": initial_actor,
            "actor_numerics": actor_numerics(actor),
            "has_predictor": predictor is not None,
            "grpo_advantage": "group_mean_no_std",
        },
    )
    print(
        f"phase=train_loop_begin method={spec.name} n_steps={cfg.get('num_steps')} "
        f"n_start={n_start} n_prompts={n_prompts}",
        flush=True,
    )

    controller = ensure_cost_control(cfg, state)
    persist_initial_checkpoint(
        run,
        cfg,
        state,
        named,
        opt,
        extra={"model_path": str(model_path), "lora": dict(cfg.get("lora") or {})},
    )
    controller.record(run, "setup_complete", state=state, cfg=cfg)
    last = {}
    steps, completed, allocation_steps = int(cfg.get("num_steps", 1)), 0, 0
    checkpoint_extra = {"model_path": str(model_path), "lora": dict(cfg.get("lora") or {})}
    for _step in range(steps):
        if controller.stop_reason():
            break
        if controller.enabled:
            n_prompts, starts_per, n_start = cost_start_counts(cfg, spec, state, (n_prompts, starts_per, n_start))
        elif start_count_mode(cfg) != "fixed" and last.get("next_n"):
            n_prompts, starts_per, n_start = resolve_start_counts(cfg, spec, n_start=max(1, int(last["next_n"])))
        step_timer = Timer()
        prompts, pids, golds, prompt_meta = _batch_ids(state.rng)
        print(
            f"phase=train_step_begin next_step={int(state.step) + 1} "
            f"n_prompts={n_prompts} starts_per={starts_per}",
            flush=True,
        )
        last = observed_call("algorithm_step", state.step + 1, lambda: run_algorithm1_step(
            engines, state, prompts, pids, golds, cfg, opt, prompt_meta=prompt_meta))
        sync_timer = Timer()
        observed_call("step_sync", state.step, extra["sync"])
        last["timings"]["sync"] = sync_timer.elapsed()
        last["force_final_checkpoint"] = controller.stop_reason() is not None
        persist_training_step(
            run,
            cfg,
            state,
            last,
            ledger,
            actor_named=named,
            optimizer=opt,
            extra={
                "model_path": str(model_path),
                "lora": dict(cfg.get("lora") or {}),
                "adapter_path": None if extra.get("adapter_path") is None else str(extra.get("adapter_path")),
                "lora_id": extra.get("lora_id"),
                "last_sync": extra.get("last_sync"),
            },
            wall_s=step_timer.elapsed(),
            cpu_s=step_timer.cpu_elapsed(),
        )
        if controller.stop_reason():
            persist_final_checkpoint(run, cfg, state, named, opt, ledger, extra=checkpoint_extra)
        observe_batch_cost(controller, cfg, state, last, step_timer.elapsed())
        controller.record(run, "batch_complete", state=state, cfg=cfg)
        completed += 1
        allocation_steps += int(bool(last.get("allocation_ready")))
        print(
            f"step {state.step} n={last['n']} completed={last['n_completed']} "
            f"audited={last['n_audited']} loss={last['loss']} "
            f"n_prescan={last.get('n_prescan')} timings={last.get('timings')}",
            flush=True,
        )
    stop_reason = controller.stop_reason() or "num_steps"
    persist_final_checkpoint(run, cfg, state, named, opt, ledger, extra=checkpoint_extra)
    cost = controller.record(run, "terminal", state=state, cfg=cfg, stop_reason=stop_reason,
                             num_steps_cap_reached=completed >= steps,
                             cap_before_wall_budget=stop_reason == "num_steps" and (controller.run_budget is not None or controller.session_budget is not None))
    return {
        "versions": versions,
        "method": spec.name,
        "backend": "gpu_vllm_hf",
        "n": last.get("n"),
        "n_completed": last.get("n_completed"),
        "steps": completed,
        "num_steps_cap": steps,
        "stop_reason": stop_reason,
        "cost_control": cost,
        "allocation_ready_last_batch": bool(last.get("allocation_ready", False)),
        "allocation_ready_steps_this_session": allocation_steps,
        "post_warmup_steps": state.step if state.offline_predictor else max(0, state.step - int((cfg.get("predictor") or {}).get("warmup_steps", 0))),
        "step": state.step,
        "status": "gpu_loop_ran",
        "checkpoint": str(Path(run.root) / "checkpoint.npz"),
    }
