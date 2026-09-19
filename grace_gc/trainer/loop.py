"""Multi-step training: data → starts → Algorithm 1 → checkpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import shutil

import numpy as np

from grace_gc.config import default_config, load_config, merge_configs, validate_config
from grace_gc.core.layout import collect_lora_layout
from grace_gc.core.rng import IsolatedRNG, seed_all
from grace_gc.data.math_data import MathRecord, load_training_data
from grace_gc.data.reward import rule_reward
from grace_gc.data.tokenize import encode_records_tiny
from grace_gc.logging_util.forensics import persist_final_checkpoint, persist_initial_checkpoint, persist_training_step, record_effective_config, reset_run_artifacts, trajectory_row, write_data_inventory, write_failed
from grace_gc.logging_util.ledger import ComputeLedger, Timer
from grace_gc.logging_util.run_dir import RunDirectory, resolve_run_dir, utc_now
from grace_gc.logging_util.run_log import RunLog
from grace_gc.predictor.heads import predictor_from_spec
from grace_gc.predictor.reservoir import GradientReservoir
from grace_gc.trainer.algorithm import TrainState, run_algorithm1_step
from grace_gc.trainer.grace_step import next_start_count
from grace_gc.trainer.baseline import baseline_from_config
from grace_gc.trainer.cpu_tiny import TinyLoRAActor, TinyTrainConfig, run_tiny_batch
from grace_gc.trainer.methods import apply_method_defaults, method_spec, start_group_size
from grace_gc.trainer.state_io import restore_train_state
from grace_gc.trainer.cost_control import cost_start_counts, ensure_cost_control, observe_batch_cost, restore_cost_control, start_cost_control, start_count_mode
from grace_gc.trainer.tiny_engine import make_tiny_engines
from grace_gc.versions import collect_environment


def build_run_config(paths: list[str], overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = default_config()
    for path in paths:
        cfg = merge_configs(cfg, load_config(path))
    if overrides:
        cfg = merge_configs(cfg, overrides)
    return validate_config(apply_method_defaults(cfg))


def _synthetic_records(n: int = 8) -> list[MathRecord]:
    return [
        MathRecord(problem_id=str(i % 4), prompt=f"q{i} add two", answer=str(i % 3))
        for i in range(n)
    ]


def resolve_start_counts(cfg: dict[str, Any], spec, n_start: int | None = None) -> tuple[int, int, int]:
    """Return (n_prompts, starts_per_prompt, global_n).

    GRPO-short keeps 16 starts per prompt. next_n / resume change how many
    prompts are drawn, not the group size.
    """
    n_prompts = max(1, int(cfg.get("n_prompts", 4)))
    if n_start is None:
        n_start = int(cfg.get("n_start", spec.n_start or 8))
    else:
        n_start = int(n_start)
    if n_start <= 0:
        raise ValueError(f"n_start must be positive, got {n_start}")
    if getattr(spec, "starts_per_prompt", None):
        starts_per = int(spec.starts_per_prompt)
        if n_start % starts_per != 0:
            raise ValueError(
                f"n_start={n_start} is not a multiple of starts_per_prompt={starts_per}"
            )
        n_prompts = n_start // starts_per
        return n_prompts, starts_per, n_start
    n_prompts = min(n_prompts, n_start)
    if n_start % n_prompts != 0:
        raise ValueError(f"n_start={n_start} is not a multiple of n_prompts={n_prompts}")
    starts_per = n_start // n_prompts
    if getattr(spec, "objective", None) == "grpo" and starts_per < 2:
        raise ValueError(
            f"GRPO/GRPO-short needs at least 2 starts of the same prompt; N={n_start} cannot form a group"
        )
    return n_prompts, starts_per, n_start


def resume_start_counts(cfg: dict[str, Any], spec, state: TrainState) -> tuple[int, int, int]:
    """First resumed batch uses the N that next_n already scheduled."""
    if start_count_mode(cfg) == "fixed":
        return resolve_start_counts(cfg, spec, state.n_ref or int(cfg.get("n_start", 8)))
    controller = cfg.get("_cost_controller")
    if controller is not None and controller.enabled and controller.state.get("next_n"):
        return cost_start_counts(cfg, spec, state, resolve_start_counts(cfg, spec, state.n_ref))
    if state.history_costs:
        n_ref = state.n_ref or int(cfg.get("n_start", 8))
        group = start_group_size(spec, int(cfg.get("n_prompts", 4)), n_ref)
        nxt = next_start_count(state.history_costs, n_ref, c_full=1.0, group=group)
        return resolve_start_counts(cfg, spec, n_start=nxt)
    if state.n_ref > 0:
        return resolve_start_counts(cfg, spec, n_start=state.n_ref)
    return resolve_start_counts(cfg, spec)


def sample_prompt_indices(n_avail: int, n_prompts: int, rng: IsolatedRNG) -> np.ndarray:
    """Paper batch is K distinct problems. Replacement only if K exceeds the pool."""
    if n_avail <= 0:
        raise ValueError("no training records")
    if n_prompts <= 0:
        raise ValueError("n_prompts must be positive")
    replace = int(n_prompts) > int(n_avail)
    return rng.generator("eval").choice(int(n_avail), size=int(n_prompts), replace=replace)


def sample_starts(
    records: list[MathRecord],
    n_prompts: int,
    starts_per_prompt: int,
    rng: IsolatedRNG,
) -> list[MathRecord]:
    if not records:
        raise ValueError("no training records")
    out = []
    for idx in sample_prompt_indices(len(records), n_prompts, rng):
        rec = records[int(idx)]
        for _j in range(starts_per_prompt):
            out.append(rec)
    return out


def run_tiny_training(cfg: dict[str, Any], run: RunDirectory, ledger: ComputeLedger | None = None) -> dict[str, Any]:
    start_cost_control(cfg, Timer())
    spec = method_spec(cfg.get("method", "grace"))
    if spec.max_new_tokens:
        cfg = {**cfg, "max_new_tokens": spec.max_new_tokens}
    seed_all(int(cfg.get("seed", 17)))
    actor = TinyLoRAActor()
    from grace_gc.trainer.initialization import initialize_actor

    initial_actor = initialize_actor(actor, cfg, run)
    layout = collect_lora_layout(actor.named_lora_params())
    rng = IsolatedRNG.create(int(cfg.get("seed", 17)))
    data_source = "synthetic"
    if cfg.get("data_path"):
        buckets, load_report = load_training_data(
            cfg["data_path"], cfg.get("eval_data_path"),
            seed=int(cfg.get("split_seed", 17)),
        )
        train_recs = buckets["train"]
        write_data_inventory(run, buckets, cfg.get("data_path"), source="file", load_report=load_report, split_seed=int(cfg.get("split_seed", 17)))
        if not train_recs:
            raise ValueError("training split is empty")
        data_source = "file"
    else:
        train_recs = _synthetic_records()
        write_data_inventory(run, {"train": train_recs}, None, source="synthetic")
    n_prompts, starts_per, n_start = resolve_start_counts(cfg, spec)
    engines = make_tiny_engines(actor, actor.vocab)
    in_dim = 1
    if spec.use_predictor:
        probe_rng = IsolatedRNG.create(rng.seed)
        probe = sample_starts(train_recs, 1, 1, probe_rng)
        prompt_ids, _pids, _golds = encode_records_tiny(probe, actor.vocab, 8)
        prefixes, _ = engines.generate_prefix(prompt_ids, 2, probe_rng, "token")
        feat = engines.prefix_features(prefixes, np.array([len(prompt_ids[0])]), [0.5])["features"]
        in_dim = int(feat.shape[1])
    k = min(int(cfg.get("predictor", {}).get("k", 2)), layout.dim)
    predictor = None
    if spec.use_predictor:
        predictor = predictor_from_spec(
            in_dim,
            k,
            cfg.get("predictor") or {},
            hidden_coord=min(32, int(cfg.get("predictor", {}).get("hidden_coord", 256))),
            hidden_risk=min(16, int(cfg.get("predictor", {}).get("hidden_risk", 64))),
            train_auxiliary=(spec.risk_mode == "reward"),
        )
    torch = __import__("torch")
    opt = torch.optim.SGD(actor.trainable_params(), lr=float(cfg.get("optim", {}).get("lr", 0.05)))
    state = TrainState(
        spec=spec,
        baseline=baseline_from_config(cfg.get("baseline")),
        rng=rng,
        layout=layout,
        u=np.eye(layout.dim, k, dtype=np.float64),
        predictor=predictor,
        reservoir=GradientReservoir(capacity=int(cfg.get("predictor", {}).get("reservoir_size", 64))),
        n_ref=n_start,
    )
    resume = cfg.get("resume")
    if resume:
        state = restore_train_state(resume, actor, opt, in_dim, k, spec.name, payload=cfg.pop("_resume_payload", None))
        engines = make_tiny_engines(actor, actor.vocab)
        n_prompts, starts_per, n_start = resume_start_counts(cfg, spec, state)
        record_effective_config(run, cfg, state, opt)
    steps = int(cfg.get("num_steps", 1))
    if ledger is None:
        ledger = ComputeLedger(n_gpu=0, hardware="cpu")
    controller = ensure_cost_control(cfg, state)
    if not resume:
        from grace_gc.trainer.format_warmup import format_warmup_steps, run_format_warmup_tiny

        if format_warmup_steps(cfg) and not initial_actor:
            info = run_format_warmup_tiny(actor, train_recs, actor.vocab, cfg, ledger)
            run.write_json("format_warmup.json", info)
        persist_initial_checkpoint(
            run,
            cfg,
            state,
            actor.named_lora_params(),
            opt,
            actor_full=actor.named_all_params(),
        )
    run.write_json("startup.json", {"implementation": "cpu_tiny", "initial_actor": initial_actor,
                                     "method": spec.name, "n_start": n_start, "n_prompts": n_prompts})
    controller.record(run, "setup_complete", state=state, cfg=cfg)
    last = {}
    completed, allocation_steps = 0, 0
    for _ in range(steps):
        if controller.stop_reason():
            break
        if controller.enabled:
            n_prompts, starts_per, n_start = cost_start_counts(cfg, spec, state, (n_prompts, starts_per, n_start))
        elif start_count_mode(cfg) != "fixed" and last.get("next_n"):
            n_prompts, starts_per, n_start = resolve_start_counts(cfg, spec, n_start=max(1, int(last["next_n"])))
        step_timer = Timer()
        batch = sample_starts(train_recs, n_prompts, starts_per, state.rng)
        prompt_meta: list = []
        prompts, pids, golds = encode_records_tiny(
            batch, actor.vocab, int(cfg.get("prompt_max_tokens", 1024)), prompt_meta=prompt_meta
        )
        last = run_algorithm1_step(engines, state, prompts, pids, golds, cfg, opt, prompt_meta=prompt_meta)
        last["force_final_checkpoint"] = controller.stop_reason() is not None
        persist_training_step(
            run,
            cfg,
            state,
            last,
            ledger,
            actor_named=actor.named_lora_params(),
            optimizer=opt,
            actor_full=actor.named_all_params(),
            wall_s=step_timer.elapsed(),
            cpu_s=step_timer.cpu_elapsed(),
        )
        if controller.stop_reason():
            persist_final_checkpoint(run, cfg, state, actor.named_lora_params(), opt, ledger,
                                     actor_full=actor.named_all_params())
        observe_batch_cost(controller, cfg, state, last, step_timer.elapsed())
        controller.record(run, "batch_complete", state=state, cfg=cfg)
        completed += 1
        allocation_steps += int(bool(last.get("allocation_ready")))
        print(
            f"step {state.step} n={last['n']} completed={last['n_completed']} "
            f"audited={last['n_audited']} loss={last['loss']}"
        )
    stop_reason = controller.stop_reason() or "num_steps"
    persist_final_checkpoint(run, cfg, state, actor.named_lora_params(), opt, ledger,
                             actor_full=actor.named_all_params())
    cost = controller.record(run, "terminal", state=state, cfg=cfg, stop_reason=stop_reason,
                             num_steps_cap_reached=completed >= steps,
                             cap_before_wall_budget=stop_reason == "num_steps" and (controller.run_budget is not None or controller.session_budget is not None))
    return {
        "method": spec.name,
        "backend": "cpu_tiny",
        "n": last.get("n", n_start),
        "n_completed": last.get("n_completed", 0),
        "steps": completed,
        "num_steps_cap": steps,
        "stop_reason": stop_reason,
        "cost_control": cost,
        "allocation_ready_last_batch": bool(last.get("allocation_ready", False)),
        "allocation_ready_steps_this_session": allocation_steps,
        "post_warmup_steps": max(0, state.step - int((cfg.get("predictor") or {}).get("warmup_steps", 0))),
        "step": state.step,
        "actor_moved": True,
        "n_audited": last.get("n_audited", 0),
        "checkpoint": str(Path(run.root) / "checkpoint.npz"),
        "data_source": data_source,
    }


def run_training(cfg: dict[str, Any], run_dir: str | Path) -> dict[str, Any]:
    started, timer = utc_now(), Timer()
    cfg = apply_method_defaults(cfg)
    spec = method_spec(cfg.get("method", "grace"))
    requested = Path(run_dir)
    resume_payload = None
    branch_resume = False
    if cfg.get("resume"):
        from grace_gc.trainer.checkpoint import load_checkpoint

        resume_payload = load_checkpoint(cfg["resume"])
        saved_step = int(resume_payload.get("step", 0))
        # Fork a rewind instead of retaining logs/checkpoints from its abandoned future.
        observed_steps = []
        source_path = Path(cfg["resume"]).resolve()
        source_root = source_path.parent.parent if source_path.parent.name == "checkpoints" else source_path.parent
        if (requested / "config.yaml").is_file() and source_root != requested.resolve():
            branch_resume = True
        for name in ("steps.jsonl", "persistence.jsonl", "prescan.jsonl", "fresh_supervision.jsonl", "trajectories.jsonl"):
            path = requested / name
            if path.is_file():
                for line in path.read_text(encoding="utf-8").splitlines():
                    try:
                        row_step = int(json.loads(line).get("step", 0))
                        branch_resume |= row_step > saved_step
                        if name == "steps.jsonl":
                            observed_steps.append(row_step)
                    except (ValueError, json.JSONDecodeError):
                        branch_resume = True
        if observed_steps and max(observed_steps) != saved_step:
            branch_resume = True
        index_path = requested / "checkpoints.json"
        if index_path.is_file():
            index = json.loads(index_path.read_text(encoding="utf-8"))
            branch_resume |= any(int(row["step"]) > saved_step for row in index.get("steps", []))
    controller = restore_cost_control(cfg, timer, resume_payload)
    run_dir = resolve_run_dir(run_dir, resume=bool(cfg.get("resume")) and not branch_resume)
    run = RunDirectory(run_dir)
    if cfg.get("resume") and (run.root / "summary.json").is_file():
        old = json.loads((run.root / "summary.json").read_text(encoding="utf-8"))
        archive_name = str(old.get("session_id") or old.get("started", "previous"))
        archive_name = "session-" + "".join(char if char.isalnum() or char in "._-" else "-" for char in archive_name)
        archive = run.root / "attempts" / archive_name
        archive.mkdir(parents=True, exist_ok=True)
        for name in ("summary.json", "config.yaml", "effective_config.yaml", "environment.json", "run_meta.json", "startup.json", "resume.json", "compute_ledger.json", "cost_control.json"):
            path = run.root / name
            if path.is_file():
                shutil.copy2(path, archive / name)
    versions = collect_environment(cfg)
    versions["started"] = started
    run.write_run_meta(kind="train", started=started, requested=requested)
    run.write_yaml("config.yaml", cfg)
    run.write_json("environment.json", versions)
    cfg["_cost_controller"] = controller
    cfg["_run_identity"] = {"model": versions.get("model"), "data": versions.get("data"),
                              "source_file_hashes": versions.get("file_hashes")}
    if resume_payload is not None:
        cfg["_resume_payload"] = resume_payload
        cfg["_resume_metadata"] = {key: resume_payload.get(key) for key in ("step", "run_config", "identity")}
        cfg["_resume_metadata"]["global_rng"] = bool(resume_payload.get("global_rng"))
        resume_payload = None
    reset_run_artifacts(run, resume=bool(cfg.get("resume")))
    backend = cfg.get("backend", "cpu_tiny")
    n_gpu = int(cfg.get("hardware", {}).get("n_gpu", 0))
    hardware = str(cfg.get("hardware", {}).get("name", "cpu"))
    if backend == "gpu_verl":
        n_gpu = max(n_gpu, 1)
        if hardware == "cpu":
            hardware = "gpu"
    ledger = ComputeLedger(n_gpu=n_gpu, hardware=hardware)
    if cfg.get("resume"):
        prev_ledger = Path(run.root) / "compute_ledger.json"
        if prev_ledger.is_file():
            prev = json.loads(prev_ledger.read_text(encoding="utf-8"))
            ledger.rows.extend(prev.get("rows") or [])
            if ledger.rows and not any(row.get("session_id") for row in ledger.rows):
                # Legacy failed runs had no enclosing timer. Recover only the
                # known missing interval, retaining the original ledger archive.
                from datetime import datetime

                old_summaries = [json.loads(path.read_text(encoding="utf-8"))
                                 for path in (run.root / "attempts").glob("*/summary.json")]
                known_wall = sum((datetime.fromisoformat(item["finished"]) - datetime.fromisoformat(item["started"])).total_seconds()
                                 for item in old_summaries if item.get("started") and item.get("finished"))
                missing_wall = max(0.0, known_wall - ledger.summary()["wall_seconds"])
                if missing_wall:
                    ledger.add("train", missing_wall, session_id="legacy-recovered-" + ledger.session_id,
                               status="recovered_timing", source="archived_summary_intervals")
    run.write_json("summary.json", {"run_status": "running", "started": started,
                                   "session_id": ledger.session_id, "run_dir": str(run.root),
                                   "resume_branch": branch_resume})
    try:
        with RunLog(run.root / "run.log"):
            print(f"run start {started} method={spec.name} backend={backend} dir={run.root}")
            if backend == "gpu_verl":
                print(
                    "implementation=gpu_vllm_hf "
                    "(HF actor + vLLM two-phase; config alias gpu_verl is not verl PPO)"
                )
            if Path(run.root).resolve() != requested.resolve():
                print(f"run-dir {requested} already had artifacts; writing to {run.root}")
            if backend == "gpu_verl":
                from grace_gc.backends.verl_trainer import train as gpu_train

                summary = gpu_train(cfg, run, ledger)
            elif cfg.get("single_batch_smoke"):
                tiny_cfg = TinyTrainConfig(
                    n=int(cfg.get("n_start", 8)),
                    decision_tokens=int(cfg.get("decision_tokens", 4)),
                    max_new=int(cfg.get("max_new_tokens", 8)),
                    p_min=float(cfg.get("allocation", {}).get("p_min", 0.2)),
                    beta=float(cfg.get("allocation", {}).get("beta", 0.5)),
                    warmup=bool(cfg.get("warmup", False)),
                    method=spec.name,
                    seed=int(cfg.get("seed", 17)),
                )
                result = run_tiny_batch(tiny_cfg)
                ctx = {
                    "run_id": run.root.name,
                    "domain": str(cfg.get("domain", "math")),
                    "method": spec.name,
                    "seed": int(cfg.get("seed", 17)),
                    "decision_t": int(cfg.get("decision_tokens", 4)),
                    "data_split": "train",
                    "warmup": bool(cfg.get("warmup", False)),
                }
                for i, rec in enumerate(result["records"]):
                    run.append_jsonl(
                        "trajectories.jsonl",
                        trajectory_row(rec, index=i, step=1, n=result["n"], ctx=ctx),
                    )
                run.append_jsonl(
                    "steps.jsonl",
                    {"step": 1, "n": result["n"], "n_completed": result["n_completed"], "method": spec.name},
                )
                summary = {
                    "method": spec.name,
                    "backend": "cpu_tiny_smoke",
                    "n": result["n"],
                    "n_completed": result["n_completed"],
                    "actor_moved": result["actor_moved"],
                }
            else:
                summary = run_tiny_training(cfg, run, ledger)
        ledger.add("train", timer.elapsed(), cpu_s=timer.cpu_elapsed(), status="complete")
        run.append_jsonl("compute_ledger.jsonl", ledger.rows[-1])
        run.write_json("compute_ledger.json", ledger.summary())
        cost = controller.record(run, "session_complete", state=cfg.get("_cost_state"), cfg=cfg)
        payload = {
            "run_status": "complete",
            "started": started,
            "finished": utc_now(),
            "run_dir": str(run.root),
            "summary": summary,
            "step": summary.get("step"),
            "stop_reason": summary.get("stop_reason"),
            "num_steps_cap": int(cfg.get("num_steps", 1)),
            "versions": versions,
            "missing": versions.get("missing", []),
            "session_id": ledger.session_id,
            "session_wall_seconds": ledger.rows[-1]["wall_seconds"],
            "cumulative_wall_seconds": ledger.summary()["wall_seconds"],
            "resume_branch": branch_resume,
            "cost_control": cost,
        }
        run.write_json("summary.json", payload)
        return payload
    except BaseException as exc:
        ledger.add("train", timer.elapsed(), cpu_s=timer.cpu_elapsed(), status="failed")
        run.append_jsonl("compute_ledger.jsonl", ledger.rows[-1])
        write_failed(run, exc, versions, started)
        failed = json.loads((run.root / "summary.json").read_text(encoding="utf-8"))
        cost = controller.record(run, "session_failed", state=cfg.get("_cost_state"), cfg=cfg, error=str(exc))
        failed.update(session_id=ledger.session_id, session_wall_seconds=ledger.rows[-1]["wall_seconds"],
                      cumulative_wall_seconds=ledger.summary()["wall_seconds"], resume_branch=branch_resume)
        failed["cost_control"] = cost
        run.write_json("summary.json", failed)
        run.write_json("compute_ledger.json", ledger.summary())
        raise


def maybe_load_data(cfg: dict[str, Any]) -> dict[str, list] | None:
    path = cfg.get("data_path")
    if not path:
        return None
    return load_training_data(path, cfg.get("eval_data_path"), seed=int(cfg.get("split_seed", 17)))[0]


def score_text(text: str | None, gold: str, truncated: bool = False) -> float | None:
    return rule_reward(text, gold, truncated=truncated)
