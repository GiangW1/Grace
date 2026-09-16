"""GPU training loop: data → two-phase rollout → dual-stream update."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from grace_gc.backends import require_gpu_stack
from grace_gc.backends.fsdp_actor import maybe_wrap_fsdp
from grace_gc.backends.gpu_engine import make_gpu_engines
from grace_gc.backends.hf_actor import named_lora_params
from grace_gc.core.layout import collect_lora_layout
from grace_gc.core.rng import IsolatedRNG, seed_all
from grace_gc.data.math_data import load_math_records, split_records
from grace_gc.data.tokenize import encode_records_hf, load_hf_tokenizer, tokenizer_inventory
from grace_gc.logging_util.forensics import persist_training_step, write_data_inventory
from grace_gc.logging_util.ledger import ComputeLedger, Timer
from grace_gc.logging_util.run_dir import RunDirectory
from grace_gc.predictor.heads import PredictorHeads
from grace_gc.predictor.reservoir import GradientReservoir
from grace_gc.trainer.algorithm import TrainState, run_algorithm1_step
from grace_gc.trainer.baseline import HistoricalBaseline
from grace_gc.trainer.loop import resolve_start_counts, resume_start_counts, sample_prompt_indices, sample_starts
from grace_gc.trainer.methods import apply_method_defaults, method_spec
from grace_gc.trainer.state_io import restore_train_state


def load_lora_actor(model_path: str, lora_cfg: dict[str, Any]):
    require_gpu_stack()
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
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
    if torch.cuda.is_available():
        model = model.cuda()
    return model


def build_vllm_engine(model_path: str, cfg: dict[str, Any], lora_rank: int):
    from grace_gc.backends.vllm_two_phase import _require_vllm

    LLM, _SP = _require_vllm()
    kwargs = {
        "model": model_path,
        "enable_prefix_caching": True,
        "enable_lora": True,
        "max_lora_rank": int(lora_rank),
        "max_loras": 4,
        "tensor_parallel_size": int(cfg.get("tensor_parallel", 1)),
        "dtype": cfg.get("dtype", "bfloat16"),
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
    return llm


def train(cfg: dict[str, Any], run: RunDirectory, ledger: ComputeLedger | None = None) -> dict[str, Any]:
    n_gpu = int(cfg.get("hardware", {}).get("n_gpu", 1))
    if n_gpu > 1:
        raise RuntimeError(
            "n_gpu>1 needs a real distributed reduce that is not wired. "
            "Set hardware.n_gpu=1 until FSDP allreduce writes .grad once."
        )
    versions = require_gpu_stack()
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
        buckets = split_records(load_math_records(data_path), seed=int(cfg.get("split_seed", 17)))
        train_recs = buckets["train"]
        if not train_recs:
            raise ValueError("training split is empty")
        write_data_inventory(run, buckets, data_path, source="file")

    seed_all(int(cfg.get("seed", 17)))
    from grace_gc.data.reward import require_math_verify

    require_math_verify()
    actor = load_lora_actor(str(model_path), cfg.get("lora", {}))
    actor = maybe_wrap_fsdp(actor, n_gpu)
    llm = build_vllm_engine(str(model_path), cfg.get("vllm", {}), int(cfg.get("lora", {}).get("rank", 16)))
    if getattr(build_vllm_engine, "last", None):
        run.write_json("vllm_engine.json", build_vllm_engine.last)
    engines, extra = make_gpu_engines(actor, llm, tokenizer, cfg, Path(run.root) / "lora")
    extra["sync"]()

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
        probe_ids, _pp, _gg, _mm = _batch_ids(rng)
        prefixes, _fin = engines.generate_prefix(probe_ids[:1], 2, rng, "token")
        feat = engines.prefix_features(prefixes, np.array([len(probe_ids[0])]), [0.5])["features"]
        in_dim = int(feat.shape[1])
        predictor = PredictorHeads(
            in_dim=in_dim,
            k=k,
            hidden_coord=int(pcfg.get("hidden_coord", 256)),
            hidden_risk=int(pcfg.get("hidden_risk", 64)),
            constant_cost=bool(pcfg.get("constant_cost", True)),
            lr=float(pcfg.get("lr", 1e-3)),
        )
    opt = torch.optim.AdamW(
        [p for _, p in named],
        lr=float(cfg.get("optim", {}).get("lr", 1e-4)),
        betas=tuple(cfg.get("optim", {}).get("betas", [0.9, 0.99])),
        weight_decay=float(cfg.get("optim", {}).get("weight_decay", 0.0)),
    )
    state = TrainState(
        spec=spec,
        baseline=HistoricalBaseline(alpha=float(cfg.get("baseline", {}).get("ema_alpha", 0.7))),
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

        check_snapshot_identity(load_checkpoint(cfg["resume"]), cfg)
        state = restore_train_state(cfg["resume"], actor, opt, in_dim, k, spec.name)
        extra["sync"]()
        n_prompts, starts_per, n_start = resume_start_counts(cfg, spec, state)

    if ledger is None:
        ledger = ComputeLedger(n_gpu=max(n_gpu, 1), hardware=str(cfg.get("hardware", {}).get("name", "gpu")))
    last = {}
    for _step in range(int(cfg.get("num_steps", 1))):
        if last.get("next_n"):
            n_prompts, starts_per, n_start = resolve_start_counts(cfg, spec, n_start=max(1, int(last["next_n"])))
        prompts, pids, golds, prompt_meta = _batch_ids(state.rng)
        step_timer = Timer()
        last = run_algorithm1_step(engines, state, prompts, pids, golds, cfg, opt, prompt_meta=prompt_meta)
        extra["sync"]()
        from grace_gc.backends.logprob_probe import probe_first_completed

        last["logprob_probe"] = probe_first_completed(
            actor,
            llm,
            last.get("records"),
            int(extra.get("pad_id") or 0),
            eos_id=extra.get("eos_id"),
            lora_request=extra.get("lora_request"),
        )
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
            },
            wall_s=step_timer.elapsed(),
        )
        print(
            f"step {state.step} n={last['n']} completed={last['n_completed']} "
            f"audited={last['n_audited']} loss={last['loss']}"
        )
    return {
        "versions": versions,
        "method": spec.name,
        "backend": "gpu_vllm_hf",
        "n": last.get("n"),
        "n_completed": last.get("n_completed"),
        "steps": int(cfg.get("num_steps", 1)),
        "step": state.step,
        "status": "gpu_loop_ran",
        "checkpoint": str(Path(run.root) / "checkpoint.npz"),
    }
