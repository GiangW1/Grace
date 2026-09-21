"""Measured wall-clock feedback and batch-boundary budgets shared by both loops."""

from __future__ import annotations

import json
import math
from pathlib import Path

from grace_gc.logging_util.ledger import Timer


def start_count_mode(cfg):
    options = cfg.get("cost_control") or {}
    mode = options.get("mode")
    if mode is None:
        mode = "wall" if options.get("enabled", False) else "token"
    if mode not in {"fixed", "wall", "token"}:
        raise ValueError("cost_control.mode must be fixed, wall or token")
    return mode


class CostControl:
    def __init__(self, cfg, timer=None, restored=None, prior_seconds=0.0, recovery_source="new_run"):
        options = cfg.get("cost_control") or {}
        self.mode = start_count_mode(cfg)
        self.enabled = self.mode == "wall"
        self.alpha = float(options.get("ema_alpha", 0.3))
        self.max_n = options.get("max_n")
        self.run_budget = cfg.get("run_wall_seconds")
        self.session_budget = cfg.get("session_wall_seconds")
        for name, value in (("target_step_seconds", options.get("target_step_seconds")),
                            ("run_wall_seconds", self.run_budget), ("session_wall_seconds", self.session_budget)):
            if value is not None and (not math.isfinite(float(value)) or float(value) <= 0):
                raise ValueError(f"{name} must be finite and positive")
        if not 0 < self.alpha <= 1:
            raise ValueError("cost_control.ema_alpha must be in (0, 1]")
        if self.max_n is not None and int(self.max_n) <= 0:
            raise ValueError("cost_control.max_n must be positive")
        self.timer = timer if timer is not None else Timer()
        self.prior_seconds = float(prior_seconds)
        self.recovery_source = recovery_source
        self.state = dict(restored or {})
        explicit_target = options.get("target_step_seconds")
        if explicit_target is not None:
            self.state["target_step_seconds"] = float(explicit_target)
            self.state["target_source"] = "explicit"
        self.state.setdefault("observations", 0)
        self.state.setdefault("next_n", None)

    def snapshot(self):
        session = float(self.timer.elapsed())
        total = self.prior_seconds + session
        return {**self.state, "mode": self.mode, "enabled": self.enabled, "global_wall_seconds": total,
                "session_wall_seconds": session, "prior_wall_seconds": self.prior_seconds,
                "run_wall_seconds_budget": self.run_budget, "session_wall_seconds_budget": self.session_budget,
                "run_overshoot_seconds": max(0., total - float(self.run_budget)) if self.run_budget is not None else None,
                "session_overshoot_seconds": max(0., session - float(self.session_budget)) if self.session_budget is not None else None,
                "recovery_source": self.recovery_source,
                "recovery_cost_note": "Spend uses recorded source-run envelopes, including failed attempts; an unrecorded process-kill tail remains unmeasured.",
                "scope": "run entry through measured boundary, including setup, SFT, sync, predictor and persistence; excludes process launch/shutdown",
                "clock_boundary": "checkpoint embeds pre-save clock; cost_control.jsonl/json records post-save boundary; final metadata write itself is outside that boundary"}

    def stop_reason(self):
        measured = self.snapshot()
        if self.run_budget is not None and measured["global_wall_seconds"] >= float(self.run_budget):
            return "run_wall_seconds"
        if self.session_budget is not None and measured["session_wall_seconds"] >= float(self.session_budget):
            return "session_wall_seconds"
        return None

    def observe(self, n, wall_seconds, step, group=1, minimum_n=None, *, main_seconds=None):
        if not math.isfinite(wall_seconds) or wall_seconds <= 0 or n <= 0:
            raise ValueError("observed batch wall time and N must be positive")
        if main_seconds is not None and (not math.isfinite(main_seconds) or not 0 <= main_seconds <= wall_seconds):
            raise ValueError("main_seconds must be finite and within the complete batch wall time")
        self.state.update(last_step=int(step), last_n=int(n), last_batch_wall_seconds=float(wall_seconds))
        if not self.enabled:
            return None
        group = max(1, int(group))
        minimum = max(group, int(minimum_n or group))
        if self.max_n is not None and int(self.max_n) < minimum:
            raise ValueError("cost_control.max_n cannot form the required prompt group")
        model = ("main_work_plus_amortized_fixed_seconds" if main_seconds is not None
                 else "complete_batch_wall_seconds_per_start_ema")
        previous = self.state.get("ema_seconds_per_start")
        if self.state.get("model", "complete_batch_wall_seconds_per_start_ema") != model:
            # An old complete-batch slope cannot initialize a marginal slope.
            previous = None
            self.state.update(main_observations=0, fixed_seconds_sum=0.)
        fixed_mean = 0.
        if main_seconds is not None:
            fixed = float(wall_seconds) - float(main_seconds)
            count = int(self.state.get("main_observations", 0)) + 1
            fixed_sum = float(self.state.get("fixed_seconds_sum", 0.)) + fixed
            fixed_mean = fixed_sum / count
            self.state.update(main_observations=count, fixed_seconds_sum=fixed_sum,
                              mean_fixed_seconds=fixed_mean, last_main_seconds=float(main_seconds),
                              last_fixed_seconds=fixed,
                              feedback_note="Main phase seconds/N is an approximate marginal cost; remaining measured wall is amortized across observed batches. No linear throughput guarantee; every second remains in the run budget.")
        per_start = float(wall_seconds if main_seconds is None else main_seconds) / int(n)
        ema = per_start if previous is None else self.alpha * per_start + (1 - self.alpha) * float(previous)
        if self.state.get("target_step_seconds") is None:
            self.state.update(target_step_seconds=float(wall_seconds), target_source="first_observed_complete_batch")
        available = max(0., float(self.state["target_step_seconds"]) - fixed_mean)
        # The tolerance only avoids rounding a mathematically integral group
        # down by one after subtracting the measured fixed component.
        next_n = max(minimum, math.floor(available / ema / group + 1e-12) * group) if ema > 0 else max(minimum, int(n))
        if self.max_n is not None:
            next_n = min(next_n, (int(self.max_n) // group) * group)
        self.state.update(ema_seconds_per_start=ema, observations=int(self.state["observations"]) + 1,
                          next_n=next_n, group_size=group, model=model)
        return next_n

    def record(self, run, event, *, state=None, cfg=None, **extra):
        if event == "terminal":
            self.state.update({key: extra[key] for key in ("stop_reason", "num_steps_cap_reached", "cap_before_wall_budget") if key in extra})
        row = {"event": event, **self.snapshot(), **extra}
        if state is not None:
            row["step"] = int(state.step)
            state.cost_control = self.snapshot()
        if cfg is not None:
            row["last_saved_step"] = cfg.get("_last_saved_step")
            row["checkpoint_sha256"] = cfg.get("_last_checkpoint_sha")
        run.append_jsonl("cost_control.jsonl", row)
        run.write_json("cost_control.json", row)
        return row


def restore_cost_control(cfg, timer, checkpoint_payload=None):
    """Restore decisions at the checkpoint, and actual spend from the source run."""
    payload = checkpoint_payload or {}
    restored = payload.get("cost_control") or {}
    prior = float(restored.get("global_wall_seconds", 0.0))
    source = "checkpoint_pre_save" if payload else "new_run"
    if cfg.get("resume"):
        path = Path(cfg["resume"])
        root = path.parent.parent if path.parent.name == "checkpoints" else path.parent
        log = root / "cost_control.jsonl"
        if log.is_file():
            for line in log.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # An interrupted tail is not a complete observation.
                prior = max(prior, float(row.get("global_wall_seconds", 0.0)))
                if row.get("step") == payload.get("step") and row.get("event") in {"batch_complete", "terminal", "setup_complete"}:
                    restored = row
                    source = "checkpoint_step_post_save_cost_log"
        summary = root / "summary.json"
        if summary.is_file():
            old = json.loads(summary.read_text(encoding="utf-8"))
            prior = max(prior, float((old.get("cost_control") or {}).get("global_wall_seconds", 0.0)),
                        float(old.get("cumulative_wall_seconds", 0.0)))
    # External records contain metadata, not additional controller parameters.
    keys = ("target_step_seconds", "target_source", "observations", "next_n", "ema_seconds_per_start",
            "last_step", "last_n", "last_batch_wall_seconds", "group_size", "model",
            "main_observations", "fixed_seconds_sum", "mean_fixed_seconds", "last_main_seconds",
            "last_fixed_seconds", "feedback_note")
    return CostControl(cfg, timer, {key: restored[key] for key in keys if key in restored}, prior, source)


def ensure_cost_control(cfg, state):
    controller = cfg.get("_cost_controller")
    if controller is None:
        controller = CostControl(cfg, restored=state.cost_control)
        cfg["_cost_controller"] = controller
    state.cost_control = controller.snapshot()
    cfg["_cost_state"] = state
    return controller


def start_cost_control(cfg, timer=None):
    """Direct backend calls also start measuring before loading actors and data."""
    if cfg.get("_cost_controller") is None:
        payload = cfg.get("_resume_payload")
        if cfg.get("resume") and payload is None:
            from grace_gc.trainer.checkpoint import load_checkpoint

            payload = load_checkpoint(cfg["resume"])
            cfg["_resume_payload"] = payload
        cfg["_cost_controller"] = restore_cost_control(cfg, timer or Timer(), payload)
    return cfg["_cost_controller"]


def cost_start_counts(cfg, spec, state, fallback):
    """Cost feedback changes prompt groups only for the next batch."""
    controller = cfg.get("_cost_controller")
    if controller is None or not controller.enabled:
        return fallback
    n = int(controller.state.get("next_n") or fallback[2])
    group = int(spec.starts_per_prompt or min(int(cfg.get("n_prompts", 4)), state.n_ref))
    minimum = group * (2 if spec.objective == "grpo" and not spec.starts_per_prompt else 1)
    if controller.max_n is not None:
        if int(controller.max_n) < minimum:
            raise ValueError("cost_control.max_n cannot form the required prompt group")
        n = min(n, int(controller.max_n) // group * group)
    if spec.starts_per_prompt:
        group = int(spec.starts_per_prompt)
        return n // group, group, n
    prompts = min(int(cfg.get("n_prompts", 4)), n)
    return prompts, n // prompts, n


def observe_batch_cost(controller, cfg, state, last, seconds):
    if state.spec.starts_per_prompt:
        group = int(state.spec.starts_per_prompt)
        minimum = group
    else:
        group = min(int(cfg.get("n_prompts", 4)), state.n_ref)
        minimum = group * (2 if state.spec.objective == "grpo" else 1)
    timings = last.get("timings")
    # These phases operate on main trajectories. Prescan, fresh supervision,
    # predictor fitting, update/sync and persistence stay in the fixed account.
    # This grouping is a scheduling approximation, not a GPU kernel model.
    main = None if not timings else sum(float(timings.get(name, 0.)) for name in
                                       ("prefix", "allocate", "continue", "backward", "audit", "behavior_probe"))
    nxt = controller.observe(int(last["n"]), seconds, state.step, group, minimum, main_seconds=main)
    if nxt is not None:
        last["next_n"] = nxt
    state.cost_control = controller.snapshot()
