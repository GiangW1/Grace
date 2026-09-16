"""Write already-computed train/eval fields. No extra analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from grace_gc.core.layout import layout_hash
from grace_gc.data.reward import extract_answer
from grace_gc.logging_util.ledger import ComputeLedger
from grace_gc.logging_util.run_dir import RunDirectory, utc_now
from grace_gc.trainer.grace_step import StartRecord, finish_reason
from grace_gc.trainer.state_io import dump_train_state
from grace_gc.versions import sha256_array, sha256_file, sha256_mapping, sha256_named


def _as_list(value):
    if value is None:
        return None
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [int(x) if isinstance(x, (np.integer, int)) else x for x in value]
    return value


def _mean(values) -> float | None:
    nums = [float(v) for v in values if v is not None]
    if not nums:
        return None
    return float(sum(nums) / len(nums))


def reset_run_artifacts(run: RunDirectory, resume: bool) -> None:
    if resume:
        return
    for name in ("trajectories.jsonl", "steps.jsonl", "compute_ledger.jsonl", "logprob_probe.jsonl"):
        path = run.root / name
        if path.is_file():
            path.unlink()
    ckpt_dir = run.root / "checkpoints"
    if ckpt_dir.is_dir():
        for path in ckpt_dir.glob("step_*.npz"):
            path.unlink()


def write_failed(run: RunDirectory, exc: BaseException, versions: dict[str, Any], started: str) -> None:
    run.write_json(
        "summary.json",
        {
            "run_status": "failed",
            "error": str(exc),
            "started": started,
            "finished": utc_now(),
            "run_dir": str(run.root),
            "versions": versions,
            "missing": versions.get("missing", []),
        },
    )


def step_context(run: RunDirectory, cfg: dict[str, Any], state, last: dict[str, Any]) -> dict[str, Any]:
    named = last.get("actor_named")
    predictor = getattr(state, "predictor", None)
    u = last.get("u_frozen")
    if u is None:
        u = getattr(state, "u", None)
    return {
        "run_id": run.root.name,
        "domain": str(cfg.get("domain", "math")),
        "method": getattr(getattr(state, "spec", None), "name", cfg.get("method", "grace")),
        "seed": int(cfg.get("seed", 17)),
        "decision_t": int(cfg.get("decision_tokens", 0)),
        "data_split": "train",
        "warmup": bool(last.get("warmup", False)),
        "rng_counters": dict(getattr(state.rng, "counters", {})),
        "snapshot_sha": None if named is None else sha256_named(named),
        "basis_sha": None if u is None else sha256_array(u),
        "predictor_sha": None if predictor is None else sha256_mapping(predictor),
        "layout_sha": None if getattr(state, "layout", None) is None else layout_hash(state.layout),
    }


def trajectory_row(
    rec: StartRecord,
    *,
    index: int,
    step: int,
    n: int,
    ctx: dict[str, Any],
    u=None,
    wall_s: float = 0.0,
    n_gpu: int = 0,
) -> dict[str, Any]:
    reason = rec.finish_reason
    if reason == "unknown":
        reason = finish_reason(rec.z, rec.finished, rec.truncated)
    coords = None
    norm = rec.g_norm_sq
    if rec.g is not None:
        g = np.asarray(rec.g, dtype=np.float64)
        if norm is None:
            norm = float(np.dot(g, g))
        if u is not None:
            coords = (np.asarray(u, dtype=np.float64).T @ g).tolist()
    return {
        "run_id": ctx.get("run_id"),
        "domain": ctx.get("domain", "math"),
        "method": ctx.get("method"),
        "seed": ctx.get("seed"),
        "step": int(step),
        "problem_id": rec.problem_id,
        "trajectory_id": f"{ctx.get('run_id')}:{step}:{index}:{rec.problem_id}",
        "snapshot_sha": ctx.get("snapshot_sha"),
        "basis_sha": ctx.get("basis_sha"),
        "predictor_sha": ctx.get("predictor_sha"),
        "layout_sha": ctx.get("layout_sha"),
        "decision_t": ctx.get("decision_t"),
        "prompt_len": rec.prompt_len,
        "prefix_tokens": rec.prefix_tokens,
        "response_tokens": rec.response_tokens,
        "suffix_tokens": rec.suffix_tokens,
        "natural_finish": rec.finished,
        "finish_reason": reason,
        "truncated": rec.truncated,
        "p_continue": rec.p,
        "z_continue": rec.z,
        "audit_selected": rec.audited,
        "prediction_coords": _as_list(rec.f),
        "r_hat_full": rec.r_hat,
        "c_hat_remaining": rec.c_hat,
        "q_hat": rec.q_hat,
        "reward": rec.reward,
        "advantage": rec.advantage,
        "extracted": rec.extracted,
        "gold": rec.gold,
        "text": rec.text,
        "prefix_text": rec.prefix_text,
        "answer_first_token": rec.answer_first_token,
        "baseline_b": rec.baseline_b,
        "rng_counters": ctx.get("rng_counters"),
        "global_starts_denominator": int(n),
        "gpu_reserved_seconds": None,
        "cpu_verifier_seconds": None,
        "true_grad_norm_sq": norm,
        "true_grad_coords": coords,
        "data_split": ctx.get("data_split", "train"),
        "warmup": ctx.get("warmup", False),
        "leak_flag": rec.leak_flag,
        "prompt_token_ids": _as_list(rec.prompt_token_ids),
        "prefix_token_ids": _as_list(rec.prefix_token_ids),
        "full_token_ids": _as_list(rec.full_token_ids),
        "prompt_truncated": rec.prompt_truncated,
        "untruncated_prompt_len": rec.untruncated_prompt_len,
        "used_chat_template": rec.used_chat_template,
        "thinking_closed": rec.thinking_closed,
        "vllm_finish_reason": rec.vllm_finish_reason,
        "vllm_stop_reason": rec.vllm_stop_reason,
        "step_wall_seconds": float(wall_s),
        "step_gpu_reserved_seconds": float(wall_s) * max(int(n_gpu), 0),
    }


def persist_load_report(run: RunDirectory, report=None, data_path=None) -> dict | None:
    explicit = report is not None
    if report is None:
        from grace_gc.data.math_data import last_load_report

        report = last_load_report()
    if not report:
        return None
    if not explicit:
        if data_path is None:
            return None
        if str(Path(str(report.get("path") or ""))) != str(Path(str(data_path))):
            return None
    if int(report.get("n_conflict_groups") or 0) > 0:
        run.write_json("data_conflicts.json", report)
    return report


def write_data_inventory(run: RunDirectory, records_or_buckets, data_path=None, source: str = "file", load_report=None) -> None:
    if isinstance(records_or_buckets, dict):
        buckets = records_or_buckets
    else:
        buckets = {"train": list(records_or_buckets or [])}
    if load_report is not None:
        report = persist_load_report(run, load_report)
    elif source == "file":
        report = persist_load_report(run, data_path=data_path)
    else:
        report = None
    payload = {
        "data_path": None if data_path is None else str(data_path),
        "source": source,
        "counts": {name: len(rows) for name, rows in buckets.items()},
        "n_unique": {
            name: len({getattr(row, "problem_id", str(i)) for i, row in enumerate(rows)})
            for name, rows in buckets.items()
        },
    }
    if report:
        payload["load"] = {
            "n_raw": report.get("n_raw"),
            "n_kept": report.get("n_kept"),
            "n_same_answer_dedup": report.get("n_same_answer_dedup"),
            "n_conflict_groups": report.get("n_conflict_groups"),
        }
    run.write_json("data_splits.json", payload)


def lora_param_health(named) -> dict[str, Any]:
    a_sq = 0.0
    b_sq = 0.0
    a_g = 0.0
    b_g = 0.0
    n_a = 0
    n_b = 0
    n_a_missing = 0
    n_b_missing = 0
    n_a_zero = 0
    n_b_zero = 0
    for name, param in named or []:
        arr = param.detach().float().cpu().numpy() if hasattr(param, "detach") else np.asarray(param)
        nrm = float(np.square(arr).sum())
        g = getattr(param, "grad", None)
        missing = g is None
        g_n = 0.0 if missing else float(np.square(g.detach().float().cpu().numpy()).sum())
        key = str(name)
        if "lora_A" in key or "lora_a" in key:
            a_sq += nrm
            a_g += g_n
            n_a += 1
            n_a_missing += int(missing)
            n_a_zero += int(not missing and g_n < 1e-12)
        elif "lora_B" in key or "lora_b" in key:
            b_sq += nrm
            b_g += g_n
            n_b += 1
            n_b_missing += int(missing)
            n_b_zero += int(not missing and g_n < 1e-12)
    b_near_zero = n_b > 0 and b_sq < 1e-12
    return {
        "lora_A_norm_sq": a_sq,
        "lora_B_norm_sq": b_sq,
        "lora_A_grad_norm_sq": a_g,
        "lora_B_grad_norm_sq": b_g,
        "n_lora_A": n_a,
        "n_lora_B": n_b,
        "n_lora_A_grad_missing": n_a_missing,
        "n_lora_B_grad_missing": n_b_missing,
        "n_lora_A_grad_zero": n_a_zero,
        "n_lora_B_grad_zero": n_b_zero,
        "lora_A_near_zero": n_a > 0 and a_sq < 1e-12,
        "lora_B_near_zero": b_near_zero,
        # PEFT ΔW=B·A: B≈0 ⇒ ∂L/∂A=0. First-step A grad can be zero without a bug.
        "lora_A_grad_zero_expected": b_near_zero,
    }


def step_health(state, last: dict[str, Any], cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    records = last.get("records") or []
    completed = [r for r in records if r.reward is not None]
    rewards = [float(r.reward) for r in completed]
    advs = [float(r.advantage) for r in completed if r.advantage is not None]
    pred = last.get("predictor") or {}
    warmup_steps = int((cfg or {}).get("predictor", {}).get("warmup_steps", 0) or 0)
    step = int(getattr(state, "step", 0) or 0)
    named = last.get("actor_named")
    baseline_vals = list((getattr(getattr(state, "baseline", None), "values", None) or {}).values())
    n_baseline_zero = sum(1 for v in baseline_vals if abs(float(v)) < 1e-12)
    batch_b = [float(r.baseline_b) for r in completed if getattr(r, "baseline_b", None) is not None]
    n_batch_b_zero = sum(1 for v in batch_b if abs(v) < 1e-12)
    reservoir_n = len(getattr(getattr(state, "reservoir", None), "items", []) or [])
    out = {
        "basis_id": getattr(state, "basis_id", None),
        "reservoir_n": reservoir_n,
        "n_parsed": sum(1 for r in records if r.extracted),
        "n_truncated": sum(1 for r in records if r.truncated),
        "n_prompt_truncated": sum(1 for r in records if r.prompt_truncated),
        "n_reward_pos": sum(1 for r in rewards if r >= 1.0),
        "mean_response_tokens": _mean(getattr(r, "response_tokens", None) for r in records),
        "mean_suffix_tokens": last.get(
            "mean_suffix_tokens",
            _mean(getattr(r, "suffix_tokens", None) for r in records),
        ),
        "n_short_response": sum(1 for r in records if int(getattr(r, "response_tokens", 0) or 0) < 16),
        "n_prefix_finished": last.get(
            "n_prefix_finished",
            sum(1 for r in records if r.z >= 1.0 and r.finished and int(getattr(r, "suffix_tokens", 0) or 0) == 0),
        ),
        "n_eligible": last.get("n_eligible"),
        "n_continued": last.get(
            "n_continued",
            sum(1 for r in records if int(getattr(r, "suffix_tokens", 0) or 0) > 0),
        ),
        "n_zero_adv": sum(1 for a in advs if abs(a) < 1e-12),
        "all_reward_zero": bool(completed) and all(r == 0.0 for r in rewards),
        "all_completed_adv_zero": bool(advs) and all(abs(a) < 1e-12 for a in advs),
        "all_p_one": bool(records) and all(abs(float(r.p) - 1.0) < 1e-12 for r in records),
        "predictor_loss_null": pred.get("coord_loss") is None,
        "warmup_steps": warmup_steps,
        "steps_after_warmup": max(0, step - warmup_steps),
        "predictor_untrained": last.get("warmup") or reservoir_n < 2,
        "grad_norm": last.get("grad_norm"),
        "grad_norm_preclip": last.get("grad_norm_preclip"),
        "logprob_probe": last.get("logprob_probe"),
        "adapter_path": last.get("adapter_path"),
        "lora_id": last.get("lora_id"),
        "sampling": last.get("sampling"),
        "allocating_with_untrained_predictor": bool(
            getattr(getattr(state, "spec", None), "use_predictor", False)
            and not last.get("warmup")
            and reservoir_n < 2
        ),
        "mean_baseline_b": _mean(baseline_vals),
        "n_baseline": len(baseline_vals),
        "n_baseline_zero": n_baseline_zero,
        "baseline_collapsed_with_zero_reward": bool(completed)
        and all(r == 0.0 for r in rewards)
        and n_batch_b_zero > 0,
    }
    if named is not None:
        out.update(lora_param_health(named))
    from grace_gc.versions import resource_snapshot

    out["resources"] = resource_snapshot()
    return out


def step_metrics_row(state, last: dict[str, Any], ctx: dict[str, Any], wall_s: float) -> dict[str, Any]:
    records = last.get("records") or []
    pred = last.get("predictor") or {}
    health = step_health(state, last, last.get("cfg"))
    return {
        "run_id": ctx.get("run_id"),
        "step": int(state.step),
        "n": last.get("n"),
        "n_completed": last.get("n_completed"),
        "n_stopped": last.get("n_stopped", sum(1 for r in records if r.z < 1.0)),
        "n_audited": last.get("n_audited"),
        "n_prescan": last.get("n_prescan"),
        "mean_p": _mean(r.p for r in records),
        "mean_reward": _mean(r.reward for r in records),
        "loss": last.get("loss"),
        "warmup": last.get("warmup"),
        "clip_triggered": last.get("clip_triggered"),
        "next_n": last.get("next_n"),
        "deviation": last.get("deviation"),
        "coord_loss": pred.get("coord_loss"),
        "risk_loss": pred.get("risk_loss"),
        "residual_mean": last.get("residual_mean"),
        "token_cost_used": last.get("token_cost_used"),
        "token_cost_full": last.get("token_cost_full"),
        "wall_seconds": float(wall_s),
        "written_at": utc_now(),
        "phases": last.get("timings") or {},
        "snapshot_sha": ctx.get("snapshot_sha"),
        "basis_sha": ctx.get("basis_sha"),
        "predictor_sha": ctx.get("predictor_sha"),
        "rng_counters": ctx.get("rng_counters"),
        **health,
    }


def _append_ledger(run: RunDirectory, ledger: ComputeLedger, n_before: int) -> None:
    for row in ledger.rows[n_before:]:
        run.append_jsonl("compute_ledger.jsonl", row)
    run.write_json("compute_ledger.json", ledger.summary())


def _update_checkpoint_index(run: RunDirectory, step: int, sha: str | None) -> None:
    path = run.root / "checkpoints.json"
    data: dict[str, Any] = {"steps": []}
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {"steps": []}
    steps = [row for row in data.get("steps", []) if int(row.get("step", -1)) != int(step)]
    steps.append({"step": int(step), "path": f"checkpoints/step_{step}.npz", "sha256": sha})
    steps.sort(key=lambda row: int(row["step"]))
    run.write_json(
        "checkpoints.json",
        {"latest": "checkpoint.npz", "latest_sha256": sha, "steps": steps},
    )


def persist_training_step(
    run: RunDirectory,
    cfg: dict[str, Any],
    state,
    last: dict[str, Any],
    ledger: ComputeLedger,
    *,
    actor_named,
    optimizer,
    actor_full=None,
    extra=None,
    wall_s: float = 0.0,
) -> None:
    last = dict(last)
    last["actor_named"] = actor_named
    last["cfg"] = cfg
    if extra:
        last.setdefault("adapter_path", None if extra.get("adapter_path") is None else str(extra.get("adapter_path")))
        last.setdefault("lora_id", extra.get("lora_id"))
    ctx = step_context(run, cfg, state, last)
    n = int(last.get("n") or len(last.get("records") or []))
    n_gpu = int(ledger.n_gpu)
    u = last.get("u_frozen")
    for i, rec in enumerate(last.get("records") or []):
        run.append_jsonl(
            "trajectories.jsonl",
            trajectory_row(
                rec,
                index=i,
                step=int(state.step),
                n=n,
                ctx=ctx,
                u=u,
                wall_s=wall_s,
                n_gpu=n_gpu,
            ),
        )
    step_row = step_metrics_row(state, last, ctx, wall_s)
    run.append_jsonl("steps.jsonl", step_row)
    health = {"step": int(state.step), "written_at": utc_now(), **step_health(state, last, cfg)}
    run.write_json("health.json", health)
    if last.get("logprob_probe") is not None:
        run.append_jsonl("logprob_probe.jsonl", {"step": int(state.step), **dict(last["logprob_probe"])})
    if last.get("sampling") is not None:
        run.write_json("sampling.json", last["sampling"])
    n_before = len(ledger.rows)
    phases = last.get("timings") or {}
    ledger.add(
        "train_step",
        float(wall_s),
        step=int(state.step),
        n=n,
        n_completed=last.get("n_completed"),
        n_audited=last.get("n_audited"),
        phases=phases,
    )
    for name, seconds in phases.items():
        ledger.add(f"phase_{name}", float(seconds), step=int(state.step))
    _append_ledger(run, ledger, n_before)
    dump_train_state(
        Path(run.root) / "checkpoint.npz",
        state,
        actor_named,
        optimizer,
        actor_full=actor_full,
        extra=extra,
    )
    step_path = Path(run.root) / "checkpoints" / f"step_{int(state.step)}.npz"
    dump_train_state(
        step_path,
        state,
        actor_named,
        optimizer,
        actor_full=actor_full,
        extra=extra,
    )
    _update_checkpoint_index(run, int(state.step), sha256_file(step_path) if step_path.is_file() else None)


def persist_eval_items(run: RunDirectory, items, cfg: dict[str, Any], result: dict[str, Any]) -> None:
    from grace_gc.data.reward import rule_reward

    rows = []
    for item in items:
        extracted = item.extracted
        if extracted is None:
            extracted = [extract_answer(ans) for ans in item.answers]
        rows.append(
            {
                "problem_id": item.problem_id,
                "gold": item.gold,
                "answers": item.answers,
                "truncated": item.truncated,
                "rewards": [rule_reward(a, item.gold, t) for a, t in zip(item.answers, item.truncated)],
                "extracted": extracted,
                "response_tokens": item.response_tokens,
                "finish_reasons": item.finish_reasons,
                "token_ids": item.token_ids,
                "sample_seeds": item.sample_seeds,
                "prompt_truncated": item.prompt_truncated,
                "vllm_finish_reasons": item.vllm_finish_reasons,
            }
        )
    run.write_jsonl("eval_per_problem.jsonl", rows)
    run.write_json("eval_summary.json", result)
    _ = cfg
