"""Standalone ST experiment. Reuses HF/vLLM engines, reward and initialization.

The historical HT trainer is unchanged. Audit spends extra compute on controls;
only separately timed train runs compare efficient Full-PG/donor-only/ST costs.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np

from grace_gc.audit.suffix_transport import PairedMoments, summarize_groups, summarize_population
from grace_gc.core.rng import IsolatedRNG
from grace_gc.core.suffix_transport import suffix_posterior, transport_gradient
from grace_gc.data.tokenize import as_stop_ids


def validate_experiment(cfg):
    st = cfg["suffix_transport"]
    if cfg.get("resume"):
        raise ValueError("ST currently supports actor initialization, not optimizer/RNG resume")
    if cfg.get("checkpoint"):
        raise ValueError("ST uses init_checkpoint for actor initialization; checkpoint is not read by this entry")
    if float(cfg.get("temperature", 1)) != 1 or float(cfg.get("lora", {}).get("dropout", 0)) != 0:
        raise ValueError("exact ST requires temperature=1 and dropout=0")
    if set(cfg.get("lora", {}).get("targets", [])) != {"q_proj", "v_proj"}:
        raise ValueError("ST gradient target is all q/v LoRA A/B parameters")
    if not 1 <= int(cfg["decision_tokens"]) < int(cfg["max_new_tokens"]):
        raise ValueError("require 1 <= decision_tokens < max_new_tokens")
    for name in ("group_size", "groups_per_problem", "audit_draws", "train_steps", "prompts_per_step", "checkpoint_every"):
        if int(st[name]) < 1:
            raise ValueError(f"suffix_transport.{name} must be positive")
    if int(st["n_problems"]) < 0 or int(st["bootstrap"]) < 0:
        raise ValueError("n_problems and bootstrap must be nonnegative")
    if int(st.get("audit_draw_batch_size", 1)) < 1:
        raise ValueError("audit_draw_batch_size must be positive")
    if not np.isfinite(float(st["strength"])) or not np.isfinite(float(st["baseline"])):
        raise ValueError("strength and fixed baseline must be finite")
    if st["method"] not in {"full_pg", "donor_only", "suffix_transport"}:
        raise ValueError("unknown ST training control")
    budget = st.get("budget_seconds")
    if budget is not None and (not np.isfinite(float(budget)) or float(budget) <= 0):
        raise ValueError("budget_seconds must be finite and positive")
    optim = cfg.get("optim", {})
    if (not np.isfinite(float(optim.get("lr", 1e-4))) or float(optim.get("lr", 1e-4)) <= 0
            or not np.isfinite(float(optim.get("grad_clip", 1))) or float(optim.get("grad_clip", 1)) < 0):
        raise ValueError("optimizer lr must be positive and grad_clip nonnegative")


def reward_for_tokens(tokens, prompt_len, gold, decode, eos_ids, thinking_open=False):
    """Only actual EOS completion earns reward; a correct answer at a cap is zero."""
    from grace_gc.data.reward import rule_reward

    response = tokens[prompt_len:]
    stops = set(as_stop_ids(eos_ids))
    ended = bool(response and response[-1] in stops)
    if any(token in stops for token in response[:-1]):
        raise ValueError("trajectory contains tokens after EOS")
    text = decode(response)
    if thinking_open:
        text = "<think>\n" + text
    reward = rule_reward(text, gold, truncated=not ended, require_complete=True)
    return float(reward), ended, text


def live_groups(prefixes, finished, starts_per_prompt):
    """Never transport across prompts; keep ragged groups after natural finishes."""
    if len(prefixes) != len(finished) or len(prefixes) % starts_per_prompt:
        raise ValueError("prefix batch shape disagrees with prompt groups")
    return [[i for i in range(begin, begin + starts_per_prompt) if not finished[i]]
            for begin in range(0, len(prefixes), starts_per_prompt)]


def draw_donors(groups, rng):
    """Selection stream is independent of tokens; called before any suffix draw."""
    return [None if not group else int(rng.integers("selection", 0, len(group))) for group in groups]


class Experiment:
    def __init__(self, cfg, run):
        self.cfg, self.run = cfg, run
        self.timings = Counter()
        self.pool = None
        self.llm = None
        self.actual_completions = self.actual_successes = self.generated_tokens = 0
        self.first_successes = set()
        self.budget_checkpoint = None
        self.rng = IsolatedRNG.create(int(cfg.get("seed", 17)))

    def timed(self, name, action):
        import torch

        torch.cuda.synchronize()
        tick = perf_counter()
        try:
            return action()
        finally:
            torch.cuda.synchronize()
            self.timings[name] += perf_counter() - tick

    def setup(self, mode):
        import torch
        from grace_gc.backends import require_gpu_stack
        from grace_gc.backends.gpu_engine import make_gpu_engines
        from grace_gc.backends.hf_actor import actor_numerics, named_lora_params
        from grace_gc.backends.rollout_pool import RolloutPool, validate_layout
        from grace_gc.backends.verl_trainer import (build_vllm_engine, cap_colocated_vllm_config,
                                                   load_lora_actor, vllm_needed_max_model_len)
        from grace_gc.core.layout import collect_lora_layout
        from grace_gc.core.rng import seed_all
        from grace_gc.data.format_prompt import apply_solve_instruction
        from grace_gc.data.math_data import (last_load_report, load_math_records, load_training_data,
                                             records_for_split, select_records, selection_manifest)
        from grace_gc.data.reward import require_math_verify
        from grace_gc.data.tokenize import encode_records_hf, load_hf_tokenizer
        from grace_gc.trainer.initialization import initialize_actor
        from grace_gc.versions import sha256_named

        cfg, st = self.cfg, self.cfg["suffix_transport"]
        require_gpu_stack()
        require_math_verify()
        workers = validate_layout(cfg)
        torch.cuda.set_device(0)
        seed_all(int(cfg.get("seed", 17)))
        if mode == "train":
            buckets, report = load_training_data(cfg["data_path"], cfg.get("eval_data_path"), int(cfg.get("split_seed", 17)))
            records = buckets["train"]
        else:
            records = load_math_records(cfg["data_path"])
            report = last_load_report()
            if mode == "audit":
                records = records_for_split(records, st.get("split", "audit"), seed=int(cfg.get("split_seed", 17)))
        self.run.write_json("data_conflicts.json", report or {})
        records = select_records(records, int(st["n_problems"]) if mode != "smoke" else 1,
                                 "seeded", int(st.get("selection_seed", 17)))
        if not records:
            raise ValueError("selected data split contains no problems")
        self.run.write_json("data_manifest.json", selection_manifest(records, "seeded", int(st.get("selection_seed", 17))))
        self.tokenizer = load_hf_tokenizer(cfg["model_path"])
        self.prompt_meta = []
        self.records = apply_solve_instruction(records)
        self.prompts, _, _ = encode_records_hf(self.records, self.tokenizer, int(cfg.get("prompt_max_tokens", 1024)),
                                              self.prompt_meta, bool(cfg.get("enable_thinking", False)))
        self.actor = load_lora_actor(cfg["model_path"], cfg["lora"])
        # Checkpointing needs train mode. Sampling engines use eval mode, so
        # explicitly disable base-model dropout on the scoring actor as well.
        for module in self.actor.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
            if isinstance(getattr(module, "attention_dropout", None), (float, int)):
                module.attention_dropout = 0.0
        self.named = named_lora_params(self.actor)
        # Shared FP32 masters, BF16 forward kernels, no quantization of optimizer updates.
        for _, parameter in self.named:
            parameter.data = parameter.data.float()
        self.actor._grace_compute_dtype = "bfloat16"
        initialize_actor(self.actor, cfg, self.run)
        self.layout = collect_lora_layout(self.named)
        self.initial_sha = sha256_named(self.named)
        self.run.write_json("actor.json", {"initial_sha256": self.initial_sha,
                                           "layout": [vars(entry) for entry in self.layout.entries],
                                           "dim": self.layout.dim, "numerics": actor_numerics(self.actor)})
        vcfg = dict(cfg.get("vllm") or {})
        vcfg.update(seed=int(cfg.get("seed", 17)), max_model_len=vllm_needed_max_model_len(cfg))
        if workers:
            self.pool = RolloutPool.launch(cfg["model_path"], vcfg, int(cfg["lora"]["rank"]), int(cfg["hardware"]["n_gpu"]))
            llm = self.pool
            self.run.write_json("vllm_engine.json", llm.metadata)
        else:
            llm = build_vllm_engine(cfg["model_path"], cap_colocated_vllm_config(cfg, vcfg), int(cfg["lora"]["rank"]))
            self.run.write_json("vllm_engine.json", build_vllm_engine.last)
        self.llm = llm
        self.engines, self.extra = make_gpu_engines(self.actor, llm, self.tokenizer, cfg, self.run.root / "lora")
        self.timed("sync", self.extra["sync"])

    def score(self, tokens, start, *, split=None, end=None):
        from grace_gc.backends.suffix_transport import range_logprob, split_logprob

        if split is not None:
            return split_logprob(self.actor, tokens, start, split, self.extra["pad_id"])
        return range_logprob(self.actor, tokens, start, self.extra["pad_id"], end=end)

    def gradient(self, tokens, start):
        from grace_gc.backends.suffix_transport import gradient_vector

        return self.timed("gradient", lambda: gradient_vector(self.score(tokens, start), self.named))

    def reward(self, tokens, record_index):
        return reward_for_tokens(tokens, len(self.prompts[record_index]), self.records[record_index].answer,
                                 self.engines.decode, self.engines.eos_id,
                                 self.prompt_meta[record_index].get("thinking_closed") is False)

    def prefixes(self, record_indices, group_size):
        prompt_ids = [self.prompts[j] for j in record_indices for _ in range(group_size)]
        prefixes, finished = self.timed("prefix_generation", lambda: self.engines.generate_prefix(
            prompt_ids, int(self.cfg["decision_tokens"]), self.rng, "token"))
        if len(prefixes) != len(prompt_ids) or len(finished) != len(prefixes):
            raise ValueError("rollout prefix count mismatch")
        stops = set(as_stop_ids(self.engines.eos_id))
        for prefix, prompt, ended in zip(prefixes, prompt_ids, finished):
            generated = prefix[len(prompt):]
            if prefix[:len(prompt)] != prompt or not generated:
                raise ValueError("prefix rollout changed prompt or generated no tokens")
            if (bool(generated[-1] in stops) != bool(ended)
                    or any(token in stops for token in generated[:-1])):
                raise ValueError("natural completion must preserve the actual EOS token")
            if not ended and len(generated) != int(self.cfg["decision_tokens"]):
                raise ValueError("live prefix stopped before fixed decision position")
            self.generated_tokens += len(generated)
        return prefixes, finished

    def continue_batch(self, prefixes, chosen):
        remaining = int(self.cfg["max_new_tokens"]) - int(self.cfg["decision_tokens"])
        full = self.timed("suffix_generation", lambda: self.engines.continue_selected(prefixes, chosen, remaining, self.rng))
        if len(full) != len(prefixes):
            raise ValueError("continuation output count mismatch")
        for i, selected in enumerate(chosen):
            if not selected:
                if full[i] is not None:
                    raise ValueError("unselected prefix received a generated suffix")
                continue
            if full[i] is None or full[i][:len(prefixes[i])] != prefixes[i]:
                raise ValueError("continuation does not preserve donor prefix")
            suffix = full[i][len(prefixes[i]):]
            stops = set(as_stop_ids(self.engines.eos_id))
            if (not suffix or len(suffix) > remaining or any(token in stops for token in suffix[:-1])
                    or (suffix[-1] not in stops and len(suffix) != remaining)):
                raise ValueError("suffix has invalid EOS or fixed-cap semantics")
            self.generated_tokens += len(suffix)
        return full

    def posterior(self, prefixes, suffix):
        import torch

        def evaluate():
            with torch.no_grad():
                return [float(self.score(prefix + suffix, len(prefix)).detach().cpu()) for prefix in prefixes]
        likelihoods = self.timed("cross_score", evaluate)
        return suffix_posterior(likelihoods), likelihoods

    def observe(self, tokens, record_index, **metadata):
        reward, ended, text = self.reward(tokens, record_index)
        self.actual_completions += 1
        self.actual_successes += int(reward == 1)
        pid = self.records[record_index].problem_id
        if reward == 1:
            self.first_successes.add(pid)
        self.run.append_jsonl("trajectories.jsonl", {**metadata, "problem_id": pid,
                              "token_ids": tokens, "response": text, "reward": reward,
                              "natural_finish": ended, "truncated": not ended,
                              "kind": "actual_rollout", "prompt_len": len(self.prompts[record_index])})
        return reward

    def checkpoint(self, step, optimizer):
        from grace_gc.trainer.checkpoint import save_checkpoint
        from grace_gc.trainer.state_io import numpy_module_state, optimizer_state

        path = self.run.root / "checkpoints" / f"step-{step}.npz"
        payload = {"actor": numpy_module_state(self.named), "step": step,
                   "model_path": self.cfg["model_path"], "lora": self.cfg["lora"],
                   "config": self.cfg, "rng": self.rng.state_dict(), "algorithm": self.cfg["suffix_transport"]["method"],
                   "initial_actor_sha256": self.initial_sha}
        if optimizer is not None:
            payload["optimizer"] = optimizer_state(optimizer)
        save_checkpoint(path, payload)
        elapsed = perf_counter() - getattr(self, "run_started", perf_counter())
        metadata = {"path": str(path), "step": step, "wall_seconds": elapsed,
                    "reserved_gpu_seconds": elapsed * int(self.cfg.get("hardware", {}).get("n_gpu", 1)),
                    "scope": "actor initialization/evaluation; no ST resume entry point"}
        budget = self.cfg["suffix_transport"].get("budget_seconds")
        if budget is not None:
            metadata["within_budget"] = elapsed <= float(budget)
            if metadata["within_budget"]:
                self.budget_checkpoint = dict(metadata)
            self.run.write_json("budget_checkpoint.json", self.budget_checkpoint or {
                "path": None, "reason": "No checkpoint persisted within the configured wall budget"})
        self.run.write_json("latest_checkpoint.json", metadata)
        self.run.append_jsonl("checkpoints.jsonl", metadata)
        return path

    def train(self):
        import torch
        from grace_gc.backends.suffix_transport import backward_group

        st, cfg = self.cfg["suffix_transport"], self.cfg
        opt_cfg = cfg.get("optim", {})
        optimizer = torch.optim.Adam([p for _, p in self.named], lr=float(opt_cfg.get("lr", 1e-4)),
                                     betas=tuple(opt_cfg.get("betas", [.9, .99])),
                                     eps=float(opt_cfg.get("eps", 1e-8)))
        baseline, strength = float(st["baseline"]), float(st["strength"])
        method = st["method"]
        # Efficient donor-only generates no discarded prefixes or cross scores.
        m = 1 if method == "donor_only" else int(st["group_size"])
        self.timed("checkpoint", lambda: self.checkpoint(0, optimizer))
        completed_steps = 0
        for step in range(1, int(st["train_steps"]) + 1):
            budget = st.get("budget_seconds")
            if budget is not None and perf_counter() - self.run_started >= float(budget):
                break
            started = perf_counter()
            indices = self.rng.integers("eval", 0, len(self.prompts), size=int(st["prompts_per_step"])).tolist()
            prefixes, finished = self.prefixes(indices, m)
            groups = live_groups(prefixes, finished, m)
            donors = draw_donors(groups, self.rng)
            chosen = [False] * len(prefixes)
            for group, donor in zip(groups, donors):
                for i in (group if method == "full_pg" else ([] if donor is None else [group[donor]])):
                    chosen[i] = True
            full = self.continue_batch(prefixes, chosen) if any(chosen) else [None] * len(prefixes)
            n_start = len(prefixes)  # Never divide by donor count or surviving starts.
            for i, (ended, selected) in enumerate(zip(finished, chosen)):
                if not ended and not selected:
                    rec = indices[i // m]
                    self.run.append_jsonl("trajectories.jsonl", {"step": step, "start": i,
                         "problem_id": self.records[rec].problem_id, "token_ids": prefixes[i],
                         "prompt_len": len(self.prompts[rec]), "kind": "stopped_prefix", "reward": None,
                         "natural_finish": False, "scope": "No independently generated future; counterfactual rewards live in steps.jsonl."})
            optimizer.zero_grad(set_to_none=True)
            posterior_rows = []
            for i, ended in enumerate(finished):
                if ended:
                    rec = indices[i // m]
                    advantage = self.observe(prefixes[i], rec, step=step, start=i, phase="prefix") - baseline
                    self.timed("backward", lambda i=i, rec=rec, a=advantage:
                               (-a / n_start * self.score(prefixes[i], len(self.prompts[rec]))).backward())
            for group_number, (group, donor) in enumerate(zip(groups, donors)):
                if not group:
                    continue
                rec = indices[group_number]
                plen = len(self.prompts[rec])
                if method in {"full_pg", "donor_only"}:
                    for i in group:
                        advantage = self.observe(full[i], rec, step=step, start=i, phase="suffix") - baseline
                        self.timed("backward", lambda i=i, a=advantage:
                                   (-a / n_start * self.score(full[i], plen)).backward())
                else:
                    prefix_group = [prefixes[i] for i in group]
                    d = group[donor]
                    donor_reward = self.observe(full[d], rec, step=step, start=d, phase="suffix")
                    suffix = full[d][len(prefixes[d]):]
                    # lambda=0 is a cheap paired-mechanism ablation, not efficient donor-only.
                    if strength == 0:
                        alpha = np.eye(len(group))[donor]
                        advantages = np.zeros(len(group))
                        advantages[donor] = donor_reward - baseline
                        likelihoods = None
                    else:
                        alpha, likelihoods = self.posterior(prefix_group, suffix)
                        advantages = np.array([self.reward(prefix + suffix, rec)[0] - baseline for prefix in prefix_group])
                    self.timed("backward", lambda: backward_group(self.score, prefix_group, plen, suffix,
                               alpha, advantages, donor, len(group) / n_start, strength))
                    posterior_rows.append({"problem_id": self.records[rec].problem_id, "donor": donor,
                                           "n_live": len(group), "donor_reward": donor_reward,
                                           "posterior_computed": strength != 0,
                                           "alpha": alpha.tolist() if strength != 0 else None,
                                           "log_likelihoods": likelihoods,
                                           "counterfactual_rewards": (advantages + baseline).tolist() if strength != 0 else None,
                                           "ess": float(1 / (alpha @ alpha)) if strength != 0 else None})
            def update():
                if any(p.grad is None or not torch.isfinite(p.grad).all() for _, p in self.named):
                    raise ValueError("missing or nonfinite q/v LoRA optimizer gradient")
                norm = float(torch.nn.utils.clip_grad_norm_([p for _, p in self.named],
                             float(opt_cfg.get("grad_clip", 1)) or float("inf")))
                if not np.isfinite(norm):
                    raise ValueError("nonfinite full-space gradient norm")
                optimizer.step()
                return norm
            norm = self.timed("optimizer", update)
            self.timed("sync", self.extra["sync"])
            # A later batch can cross the budget. Keep every preceding budgeted
            # step so evaluation never falls back ten updates just due to cadence.
            if budget is not None or step % int(st["checkpoint_every"]) == 0 or step == int(st["train_steps"]):
                self.timed("checkpoint", lambda: self.checkpoint(step, optimizer))
            self.run.append_jsonl("steps.jsonl", {"step": step, "method": method, "n_start": n_start,
                                  "natural_prefix_finishes": sum(finished), "n_suffixes": sum(chosen),
                                  "gradient_norm_before_clip": norm, "wall_seconds": perf_counter() - started,
                                  "posterior_groups": posterior_rows, "sampling": self.engines.last_rollout,
                                  "actual_completions_cumulative": self.actual_completions,
                                  "actual_successes_cumulative": self.actual_successes,
                                  "first_success_problem_ids": sorted(self.first_successes)})
            print(f"step={step} method={method} starts={n_start} suffixes={sum(chosen)} grad_norm={norm:.6g}", flush=True)
            completed_steps = step
        if (st.get("budget_seconds") is None and completed_steps and completed_steps < int(st["train_steps"])
                and completed_steps % int(st["checkpoint_every"]) != 0):
            self.timed("checkpoint", lambda: self.checkpoint(completed_steps, optimizer))
        return {"mode": "train", "method": method, "steps": completed_steps,
                "requested_steps": int(st["train_steps"]), "budget_seconds": st.get("budget_seconds"),
                "budget_checkpoint": self.budget_checkpoint,
                "note": "A batch already begun finishes; the budget can overshoot by one batch plus persistence/cleanup. Use timed checkpoints within budget for quality comparisons."}

    def audit_completions(self, prefixes, group):
        st = self.cfg["suffix_transport"]
        draws = int(st["audit_draws"])
        batch_size = int(st.get("audit_draw_batch_size", 1))
        width = len(prefixes)
        selected = [i in group for i in range(width)]
        for begin in range(0, draws, batch_size):
            count = min(batch_size, draws - begin)
            # Selection and continuation have independent RNG streams. Preserve
            # draw-major request order while selecting before suffix sampling.
            donors = [draw_donors([group], self.rng)[0] for _ in range(count)]
            full = self.continue_batch(prefixes * count, selected * count)
            seeds = (self.engines.last_rollout or {}).get("continue_request_seeds", {})
            for offset, donor in enumerate(donors):
                first = offset * width
                yield (begin + offset, donor, full[first:first + width],
                       [seeds.get(first + i) for i in group])

    def audit(self):
        from grace_gc.versions import sha256_named

        cfg, st = self.cfg, self.cfg["suffix_transport"]
        m, baseline, strength = int(st["group_size"]), float(st["baseline"]), float(st["strength"])
        rows = []
        for rec in range(len(self.prompts)):
            for group_id in range(int(st["groups_per_problem"])):
                prefixes, finished = self.prefixes([rec], m)
                group = live_groups(prefixes, finished, m)[0]
                row = {"problem_id": self.records[rec].problem_id, "group_id": group_id,
                       "n_start": m, "n_live": len(group), "prefix_token_ids": prefixes,
                       "prefix_finished": finished, "baseline": baseline}
                early = np.zeros(self.layout.dim)
                for i, ended in enumerate(finished):
                    if ended:
                        reward = self.observe(prefixes[i], rec, group_id=group_id, start=i, phase="prefix")
                        early += (reward - baseline) * self.gradient(prefixes[i], len(self.prompts[rec])) / m
                moments = PairedMoments(self.layout.dim)
                if group:
                    prefix_group = [prefixes[i] for i in group]
                    plen = len(self.prompts[rec])
                    prefix_grads = np.stack([self.gradient(prefix, plen) for prefix in prefix_group])
                    ess_values, donor_masses = [], []
                    for draw, donor, full, request_seeds in self.audit_completions(prefixes, group):
                        # Independent on-policy completions supply a real Full-PG control.
                        own_grads, rewards = [], []
                        for i in group:
                            reward = self.observe(full[i], rec, group_id=group_id, draw=draw, start=i, phase="suffix")
                            rewards.append(reward)
                            own_grads.append((reward - baseline) * self.gradient(full[i], plen))
                        d = group[donor]
                        suffix = full[d][len(prefixes[d]):]
                        alpha, likelihoods = self.posterior(prefix_group, suffix)
                        advantages = np.array([self.reward(prefix + suffix, rec)[0] - baseline for prefix in prefix_group])
                        gd = own_grads[donor]
                        shared = transport_gradient(prefix_grads, gd, alpha, advantages, donor)
                        controls = {"full_pg": np.mean(own_grads, axis=0)}
                        if st.get("full_rb", True):
                            rb = np.zeros_like(gd)
                            for j, prefix in enumerate(prefix_group):
                                gradient = own_grads[j] if j == donor else advantages[j] * self.gradient(prefix + suffix, plen)
                                rb += alpha[j] * gradient
                            controls["full_rb"] = rb
                        weight = len(group) / m
                        moments.add(early + weight * gd, weight * (shared - gd),
                                    **{name: early + weight * value for name, value in controls.items()})
                        ess_values.append(float(1 / (alpha @ alpha)))
                        donor_masses.append(float(alpha[donor]))
                        self.run.append_jsonl("transport_draws.jsonl", {"problem_id": row["problem_id"],
                              "group_id": group_id, "draw": draw, "donor": donor, "live_start_indices": group,
                              "request_seeds": request_seeds,
                              "suffix_token_ids": suffix, "alpha": alpha.tolist(), "log_likelihoods": likelihoods,
                              "counterfactual_rewards": (advantages + baseline).tolist(), "actual_rewards": rewards,
                              "counterfactuals_are_exploration_successes": False})
                    row.update(moments=moments.report(sorted({0.0, 1.0, strength})),
                               mean_ess=float(np.mean(ess_values)), mean_donor_posterior=float(np.mean(donor_masses)))
                else:
                    controls = {"full_pg": early}
                    if st.get("full_rb", True):
                        controls["full_rb"] = early
                    for _ in range(int(st["audit_draws"])):
                        moments.add(early, np.zeros_like(early), **controls)
                    row.update(moments=moments.report(sorted({0.0, 1.0, strength})),
                               mean_ess=None, mean_donor_posterior=None)
                # Full means allow population analysis without storing each dense draw.
                path = self.run.root / "moments" / f"problem-{rec}-group-{group_id}.npz"
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(path, donor_correction_sum=moments.sum, donor_correction_gram=moments.gram,
                         n=moments.n, **{f"{name}_sum": value for name, value in moments.other_sum.items()})
                row["moments_file"] = str(path.relative_to(self.run.root))
                rows.append(row)
                self.run.append_jsonl("groups.jsonl", row)
                print(f"audit problem={rec + 1}/{len(self.prompts)} group={group_id} live={len(group)}", flush=True)
        if sha256_named(self.named) != self.initial_sha:
            raise ValueError("frozen audit mutated actor parameters")
        return {"mode": "audit", **summarize_groups(rows, int(st["bootstrap"]), int(cfg.get("seed", 17))),
                "population": summarize_population(rows, self.run.root),
                "full_rb": bool(st.get("full_rb", True)),
                "note": "Lambda optimum is descriptive; calibrate on separate problems before held-out testing. Short cap runs change the reward horizon."}

    def close(self):
        pool, llm = self.pool, self.llm
        self.pool = self.llm = None
        if pool is not None:
            pool.close()
        elif llm is not None and getattr(getattr(llm, "llm_engine", None), "engine_core", None) is not None:
            from grace_gc.backends.verl_trainer import _shutdown_vllm_engine

            _shutdown_vllm_engine(llm)
