"""ST correctness: exact expectations, full-space loss and run wiring."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from grace_gc.audit.suffix_transport import PairedMoments, summarize_groups
from grace_gc.audit.suffix_transport_smoke import math_smoke
from grace_gc.core.rng import IsolatedRNG
from grace_gc.core.suffix_transport import aggregate_groups, prefix_coefficients, suffix_posterior, transport_gradient
from grace_gc.trainer.suffix_transport import draw_donors, live_groups, reward_for_tokens


def config():
    from grace_gc.config import default_config, merge_configs

    return merge_configs(default_config(), {"decision_tokens": 2, "max_new_tokens": 6,
             "model_path": "synthetic-model", "optim": {"lr": 1e-4, "grad_clip": 0},
             "suffix_transport": {"method": "suffix_transport", "group_size": 3, "strength": 1.,
             "baseline": .5, "n_problems": 1, "groups_per_problem": 1, "audit_draws": 3,
             "train_steps": 1, "prompts_per_step": 1, "checkpoint_every": 1, "bootstrap": 20}})


def test_exact_unbiased_expectation_and_numerics():
    assert max(math_smoke()["exact_enumeration_errors"].values()) < 1e-14


@pytest.mark.parametrize("values", [[], [float("nan")], [float("inf")], [-np.inf, -np.inf], [[1, 2]]])
def test_invalid_likelihoods_fail(values):
    with pytest.raises(ValueError):
        suffix_posterior(values)


def test_posterior_is_sequence_sum_not_mean_or_donor_ratio():
    alpha = suffix_posterior([-60, -62, -70])
    np.testing.assert_allclose(alpha, np.exp([0, -2, -10]) / np.exp([0, -2, -10]).sum())
    assert alpha[0] > .8
    np.testing.assert_allclose(prefix_coefficients(alpha, [.5, -.5, .7], 1, 0), [0, -.5, 0])
    with pytest.raises(ValueError):
        aggregate_groups([(1, np.ones(2))], [], 4)


def test_group_boundaries_and_ragged_donors():
    groups = live_groups([[1]] * 6, [True, False, False, False, True, True], 3)
    assert groups == [[1, 2], [3]]
    rng = IsolatedRNG.create(7)
    donors = draw_donors([groups[0], [], groups[1]], rng)
    assert donors[0] in (0, 1) and donors[1:] == [None, 0]
    assert rng.state_dict()["counters"]["continuation"] == 0


def test_reward_eos_cap_open_thinking_and_receiver_specific():
    decode = lambda _: "Answer: 4"
    assert reward_for_tokens([1, 2, 9], 1, "4", decode, [9])[0] == 1
    assert reward_for_tokens([1, 2], 1, "4", decode, [9])[0] == 0
    assert reward_for_tokens([1, 2, 9], 1, "4", decode, [9], True)[0] == 0
    assert reward_for_tokens([1, 2, 9], 1, "4", lambda _: "</think>\nAnswer: 4", [9], True)[0] == 1
    with pytest.raises(ValueError):
        reward_for_tokens([1, 9, 2], 1, "4", decode, [9])


def test_full_space_moments_and_cluster_bootstrap():
    rng = np.random.default_rng(17)
    gd, correction, full = rng.normal(size=(3, 5, 4))
    stats = PairedMoments(4)
    for a, b, c in zip(gd, correction, full):
        stats.add(a, b, full_pg=c)
    report = stats.report([0, .4, 1])
    for strength in (0, .4, 1):
        expected = np.var(gd + strength * correction, axis=0, ddof=1).sum()
        assert report["methods"][f"st_lambda_{strength:g}"]["conditional_trace_variance"] == pytest.approx(expected)
    row = {"problem_id": "same", "n_live": 2, "n_start": 4, "moments": report}
    output = summarize_groups([row, row], 20)
    value = report["methods"]["full_pg"]["conditional_trace_variance"]
    assert output["methods"]["full_pg"]["problem_bootstrap_95ci"] == pytest.approx([value, value])
    single = PairedMoments(4)
    single.add(gd[0], correction[0])
    assert single.report()["methods"]["st_lambda_1"]["conditional_trace_variance"] is None
    assert summarize_groups([], 0)["methods"] == {}


@pytest.fixture
def tiny():
    import torch
    from grace_gc.trainer.cpu_tiny import TinyLoRAActor

    torch.manual_seed(4)
    actor = TinyLoRAActor(vocab=12, dim=8, n_layers=2, n_heads=2, n_kv=1, rank=2)
    # Exercise nonzero A gradients as well as zero-B initialization.
    for name, parameter in actor.named_lora_params():
        if "lora_B" in name:
            parameter.data.normal_(0, .1)

    class Wrapped(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.masters = torch.nn.ParameterList(actor.trainable_params())

        def forward(self, input_ids, **kwargs):
            logits, _ = actor.forward(input_ids)
            return SimpleNamespace(logits=logits)

    return Wrapped(), actor.named_lora_params()


def test_autograd_loss_sign_full_layout_and_causal_prefix(tiny):
    from grace_gc.audit.suffix_transport_smoke import autograd_smoke
    from grace_gc.backends.suffix_transport import range_logprob, split_logprob

    model, named = tiny
    def score(ids, start, split=None, **kwargs):
        return (range_logprob(model, ids, start, 0, chunk_size=2) if split is None else
                split_logprob(model, ids, start, split, 0, chunk_size=2))
    result = autograd_smoke(score, named, [[1, 2, 3], [1, 4, 5]], 1, [6, 7, 11])
    assert result["n_qv_lora_tensors"] == 8
    assert all(check["relative_l2"] < 1e-4 for check in result["loss_checks"].values())


def test_token_ranges_include_actual_eos_and_empty_suffix(tiny):
    import torch
    from grace_gc.backends.suffix_transport import gradient_vector, range_logprob, split_logprob

    model, named = tiny
    ids = [1, 2, 3, 11]
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([ids])).logits
        expected = logits[0, :-1].float().log_softmax(-1)[range(3), ids[1:]].sum()
        assert float(range_logprob(model, ids, 1, 0)) == pytest.approx(float(expected), abs=1e-6)
        _, empty = split_logprob(model, ids, 1, len(ids), 0)
        assert float(empty) == 0
    assert gradient_vector(range_logprob(model, ids, len(ids), 0), named).shape[0] > 0


def experiment_fixture(tiny, tmp_path, monkeypatch, method="suffix_transport", early=True):
    from grace_gc.logging_util.run_dir import RunDirectory
    from grace_gc.trainer.suffix_transport import Experiment
    from grace_gc.core.layout import collect_lora_layout
    from grace_gc.versions import sha256_named
    import torch

    cfg = config()
    cfg["suffix_transport"]["method"] = method
    exp = Experiment(cfg, RunDirectory(tmp_path))
    exp.actor, exp.named = tiny
    exp.layout = collect_lora_layout(exp.named)
    exp.initial_sha = sha256_named(exp.named)
    exp.prompts = [[1]]
    exp.records = [SimpleNamespace(problem_id="one", answer="4")]
    exp.prompt_meta = [{"thinking_closed": True}]
    exp.extra = {"pad_id": 0, "sync": lambda: None}
    exp.timed = lambda name, action: action()
    exp.run_started = 0
    counts = {"prefix": 0, "suffix": 0, "sync": 0}
    history = []
    def generate(prompts, max_new, rng, stream):
        counts["prefix"] += len(prompts)
        prefixes = [p + [2 + i, 11 if early and i == 0 else 5] for i, p in enumerate(prompts)]
        return prefixes, [early and i == 0 for i in range(len(prompts))]
    def continuation(prefixes, chosen, max_new, rng):
        counts["suffix"] += sum(chosen)
        full = [p + [6, 11] if chosen[i] else None for i, p in enumerate(prefixes)]
        history.append((copy.deepcopy(prefixes), list(chosen), copy.deepcopy(full)))
        return full
    exp.engines = SimpleNamespace(generate_prefix=generate, continue_selected=continuation,
                                   decode=lambda _: "Answer: 4", eos_id=[11], last_rollout={})
    def sync():
        counts["sync"] += 1
    exp.extra["sync"] = sync
    gradients = []
    class RecordingSGD(torch.optim.SGD):
        def step(self, closure=None):
            gradients.append(-np.concatenate([p.grad.detach().numpy().ravel() for _, p in exp.named]).copy())
            return super().step(closure)
    monkeypatch.setattr(torch.optim, "Adam", lambda params, lr, **kwargs: RecordingSGD(params, lr=lr))
    return exp, counts, history, gradients


@pytest.mark.parametrize("method", ["suffix_transport", "full_pg", "donor_only"])
def test_training_entry_actual_fixed_n_and_efficient_control(tiny, tmp_path, monkeypatch, method):
    from grace_gc.backends.suffix_transport import gradient_vector
    from grace_gc.core.suffix_transport import aggregate_groups
    from grace_gc.trainer.checkpoint import load_checkpoint

    exp, counts, history, gradients = experiment_fixture(tiny, tmp_path, monkeypatch, method, early=True)
    result = exp.train()
    assert result["steps"] == 1
    assert counts["prefix"] == (1 if method == "donor_only" else 3)
    assert counts["suffix"] == {"donor_only": 0, "suffix_transport": 1, "full_pg": 2}[method]
    assert counts["sync"] == 1
    assert exp.actual_completions == {"donor_only": 1, "suffix_transport": 2, "full_pg": 3}[method]
    assert exp.actual_successes == exp.actual_completions  # No receiver duplication.
    step = json.loads((tmp_path / "steps.jsonl").read_text())
    assert step["n_start"] == counts["prefix"]
    loaded = load_checkpoint(tmp_path / "checkpoints" / "step-1.npz")
    assert set(loaded["actor"]) == {name for name, _ in exp.named}
    initial = load_checkpoint(tmp_path / "checkpoints" / "step-0.npz")
    for name, parameter in exp.named:
        parameter.data.copy_(__import__("torch").as_tensor(initial["actor"][name]))
    early_g = .5 * gradient_vector(exp.score([1, 2, 11], 1), exp.named)
    if method == "donor_only":
        expected = early_g
    else:
        prefixes, chosen, full = history[0]
        live = [1, 2]
        if method == "full_pg":
            expected = (early_g + sum(.5 * gradient_vector(exp.score(full[i], 1), exp.named) for i in live)) / 3
        else:
            donor = next(i for i in live if chosen[i])
            suffix = full[donor][len(prefixes[donor]):]
            alpha, _ = exp.posterior([prefixes[i] for i in live], suffix)
            pg = np.stack([gradient_vector(exp.score(prefixes[i], 1), exp.named) for i in live])
            gd = .5 * gradient_vector(exp.score(full[donor], 1), exp.named)
            expected = aggregate_groups([(2, transport_gradient(pg, gd, alpha, [.5, .5], live.index(donor)))], [early_g], 3)
    np.testing.assert_allclose(gradients[0], expected, rtol=2e-4, atol=1e-6)


def test_audit_frozen_pairing_and_full_pg_control(tiny, tmp_path, monkeypatch):
    exp, counts, _, _ = experiment_fixture(tiny, tmp_path, monkeypatch)
    report = exp.audit()
    assert counts["suffix"] == 6  # Independent 2 live completions per draw, not repeated donor.
    assert report["n_groups"] == 1 and report["n_live_groups"] == 1
    assert report["methods"]["full_rb"]["mean_conditional_trace_variance"] is not None
    assert exp.actual_completions == 7
    assert len((tmp_path / "transport_draws.jsonl").read_text().splitlines()) == 3
    assert (tmp_path / "moments" / "problem-0-group-0.npz").exists()
    assert report["population"]["methods"]["full_pg"]["population_trace_variance"] is None


def test_population_statistics_include_between_prefix_noise(tmp_path):
    from grace_gc.audit.suffix_transport import summarize_population

    rows = []
    means = []
    for i, group_mean in enumerate(([1., 0.], [3., 0.], [-2., 1.], [0., 1.])):
        stats = PairedMoments(2)
        for offset in (-1., 1.):
            stats.add(np.array(group_mean) + [0, offset], [0., 0.], full_pg=np.array(group_mean) + [0, offset])
        report = stats.report()
        name = f"group-{i}.npz"
        np.savez(tmp_path / name, n=stats.n, donor_correction_sum=stats.sum,
                 full_pg_sum=stats.other_sum["full_pg"])
        rows.append({"problem_id": str(i // 2), "moments": report, "moments_file": name})
        means.append(group_mean)
    output = summarize_population(rows, tmp_path)["methods"]["full_pg"]
    # Two problems, two independent groups each: noise_x=1, sum/P^2=.5.
    mean_norm = np.linalg.norm(np.mean(means, axis=0)) ** 2
    assert output["mean_norm_squared_noise_corrected"] == pytest.approx(mean_norm - .5)
    second = np.mean([r["moments"]["methods"]["full_pg"]["second_moment"] for r in rows])
    assert output["population_trace_variance"] == pytest.approx(second - mean_norm + .5)


def test_all_natural_finishes_are_retained_in_population(tiny, tmp_path, monkeypatch):
    exp, counts, _, _ = experiment_fixture(tiny, tmp_path, monkeypatch)
    exp.cfg["suffix_transport"]["groups_per_problem"] = 2
    exp.engines.generate_prefix = lambda prompts, *args: ([p + [2, 11] for p in prompts], [True] * len(prompts))
    report = exp.audit()
    assert counts["suffix"] == 0 and report["n_live_groups"] == 0
    assert exp.actual_completions == 6
    assert report["methods"]["st_lambda_1"]["mean_conditional_trace_variance"] == pytest.approx(0, abs=1e-12)
    assert report["population"]["methods"]["st_lambda_1"]["population_trace_variance"] == pytest.approx(0, abs=1e-12)


def test_gpu_smoke_workflow_with_cpu_engine_fixture(tiny, tmp_path, monkeypatch):
    from grace_gc.audit.suffix_transport_smoke import gpu_smoke
    from grace_gc.backends.logprob_probe import hf_response_logprobs
    from grace_gc.versions import sha256_named

    exp, counts, _, _ = experiment_fixture(tiny, tmp_path, monkeypatch)
    exp.tokenizer = SimpleNamespace(vocab_size=12)
    exp.extra["lora_id"] = 2
    original_generate = exp.engines.generate_prefix
    def generate(*args):
        prefixes, finished = original_generate(*args)
        behavior = [hf_response_logprobs(exp.actor, tokens, 1, 0, [11], max_n=None).tolist() for tokens in prefixes]
        # Exercise the vLLM omitted/restored stop logprob case.
        behavior[0][-1] = None
        exp.engines.last_rollout = {"prefix_token_logprobs": behavior}
        return prefixes, finished
    exp.engines.generate_prefix = generate
    original_continue = exp.engines.continue_selected
    def continuation(prefixes, *args):
        full = original_continue(prefixes, *args)
        exp.engines.last_rollout["continue_token_logprobs"] = {
            i: hf_response_logprobs(exp.actor, tokens, len(prefixes[i]), 0, [11], max_n=None).tolist()
            for i, tokens in enumerate(full) if tokens is not None}
        return full
    exp.engines.continue_selected = continuation
    def sync():
        exp.extra["lora_id"] += 1
        counts["sync"] += 1
    exp.extra["sync"] = sync
    result = gpu_smoke(exp)
    assert result["checkpoint_roundtrip"] and result["optimizer_and_updated_adapter_sync"]
    assert result["hf_behavior_logprob_errors"][0]["restored_eos_without_behavior_logprob"]
    assert len(result["suffix_hf_behavior_logprob_errors"]) == 1
    assert result["suffix_hf_behavior_logprob_errors"][0]["signed_sequence_logprob_error"] == pytest.approx(0)
    assert counts["sync"] == 2
    assert sha256_named(exp.named) == exp.initial_sha


def test_budget_finishes_without_invented_steps(tiny, tmp_path, monkeypatch):
    import time
    exp, counts, _, _ = experiment_fixture(tiny, tmp_path, monkeypatch)
    exp.cfg["suffix_transport"]["budget_seconds"] = 1
    exp.run_started = time.perf_counter() - 2
    result = exp.train()
    assert result["steps"] == 0 and counts["prefix"] == 0
    assert (tmp_path / "checkpoints" / "step-0.npz").exists()


def test_budget_keeps_last_valid_step_before_overshooting_batch(tiny, tmp_path, monkeypatch):
    import grace_gc.trainer.suffix_transport as module

    exp, _, _, _ = experiment_fixture(tiny, tmp_path, monkeypatch)
    exp.cfg["suffix_transport"].update(budget_seconds=6., train_steps=100, checkpoint_every=10)
    clock = [0.]
    monkeypatch.setattr(module, "perf_counter", lambda: clock[0])
    old_sync = exp.extra["sync"]
    def sync():
        old_sync()
        clock[0] += 4
    exp.extra["sync"] = sync
    result = exp.train()
    assert result["steps"] == 2
    assert (tmp_path / "checkpoints" / "step-1.npz").exists()
    assert result["budget_checkpoint"]["step"] == 1
    assert json.loads((tmp_path / "budget_checkpoint.json").read_text())["step"] == 1
    latest = json.loads((tmp_path / "latest_checkpoint.json").read_text())
    assert latest["step"] == 2 and not latest["within_budget"]


def test_lambda_zero_does_not_invent_posterior_or_receiver_rewards(tiny, tmp_path, monkeypatch):
    exp, _, _, _ = experiment_fixture(tiny, tmp_path, monkeypatch)
    exp.cfg["suffix_transport"]["strength"] = 0
    def no_scoring(*args):
        raise AssertionError("lambda zero should not cross-score")
    exp.posterior = no_scoring
    exp.train()
    group = json.loads((tmp_path / "steps.jsonl").read_text())["posterior_groups"][0]
    assert group["donor_reward"] == 1
    assert not group["posterior_computed"]
    assert group["alpha"] is None and group["ess"] is None and group["counterfactual_rewards"] is None


@pytest.mark.parametrize("method", ["suffix_transport", "donor_only", "full_pg"])
def test_eval_reports_checkpoint_method_not_decoding_alias(tmp_path, monkeypatch, method):
    from grace_gc.data.math_data import MathRecord
    from grace_gc.evaluation import generate
    from grace_gc.evaluation.eval_full import EvalItem
    from grace_gc.trainer.checkpoint import save_checkpoint

    checkpoint = tmp_path / "weights.npz"
    estimator = {"method": method, "strength": .4, "group_size": 3}
    save_checkpoint(checkpoint, {"actor": {"test": np.ones(2)}, "algorithm": method,
                                "step": 1, "config": {"suffix_transport": estimator}})
    cfg = config()
    cfg.update(method="full_pg", backend="gpu_verl", checkpoint=str(checkpoint))
    cfg["eval"] = {"max_new_tokens": 16, "n": 1, "k": 1}
    monkeypatch.setattr(generate, "collect_environment", lambda *args: {})
    monkeypatch.setattr(generate, "_generate_eval_items", lambda *args:
                        [EvalItem("p", "4", ["Answer: 4"], [False])])
    output = generate.run_eval([MathRecord("p", "2+2?", "4")], cfg, tmp_path / "evaluation")
    assert output["method"] == method
    assert output["training_estimator"] == estimator
    assert json.loads((tmp_path / "evaluation" / "eval_summary.json").read_text())["method"] == method


def test_nonfinite_st_backward_is_not_a_successful_smoke(tiny, monkeypatch):
    from grace_gc.audit.suffix_transport_smoke import autograd_smoke
    from grace_gc.backends import suffix_transport as backend

    model, named = tiny
    def score(ids, start, split=None, **kwargs):
        return (backend.range_logprob(model, ids, start, 0) if split is None else
                backend.split_logprob(model, ids, start, split, 0))
    original = backend.backward_group
    def corrupt(*args, **kwargs):
        original(*args, **kwargs)
        named[0][1].grad.flatten()[0] = float("nan")
    monkeypatch.setattr(backend, "backward_group", corrupt)
    with pytest.raises(AssertionError, match="nonfinite ST smoke gradient"):
        autograd_smoke(score, named, [[1, 2, 3], [1, 4, 5]], 1, [6, 11])


def test_behavior_probe_checks_suffix_sequence_drift_and_nan(monkeypatch):
    from grace_gc.audit.suffix_transport_smoke import behavior_probe
    from grace_gc.backends import logprob_probe

    exp = SimpleNamespace(actor=None, extra={"pad_id": 0}, engines=SimpleNamespace(eos_id=[9]),
                          cfg={"suffix_transport": {}})
    monkeypatch.setattr(logprob_probe, "hf_response_logprobs", lambda *a, **kw: np.array([-1.01, -2.02]))
    rollout = {"continue_token_logprobs": {3: [-1., -2.]}}
    args = (exp, [[1, 2, 3, 4]], rollout, 1, [9])
    report = behavior_probe(*args, phase="continue", starts=[2], indices=[3])[0]
    assert report["signed_sequence_logprob_error"] == pytest.approx(-.03)
    assert report["phase"] == "continue" and report["n_tokens"] == 2
    monkeypatch.setattr(logprob_probe, "hf_response_logprobs", lambda *a, **kw: np.array([np.nan, -2.02]))
    with pytest.raises(AssertionError, match="nonfinite HF continue"):
        behavior_probe(*args, phase="continue", starts=[2], indices=[3])


def test_variance_contrast_bootstraps_the_paired_difference():
    rows = []
    for i, donor in enumerate((1., 100., 10000.)):
        methods = {"st_lambda_0": {"conditional_trace_variance": donor},
                   "st_lambda_1": {"conditional_trace_variance": donor - .5}}
        rows.append({"problem_id": str(i), "n_live": 2, "moments": {"methods": methods}})
    contrast = summarize_groups(rows, 100)["contrasts"]["st_lambda_1_minus_donor"]
    assert contrast["mean_conditional_variance_difference"] == pytest.approx(-.5)
    assert contrast["paired_problem_bootstrap_95ci"] == pytest.approx([-.5, -.5])


def test_cli_cpu_smoke_and_preserve_previous_run(tmp_path):
    from scripts.suffix_transport import main

    root = tmp_path / "smoke"
    assert main(["--mode", "smoke", "--cpu", "--run-dir", str(root)]) == 0
    assert json.loads((root / "summary.json").read_text())["gpu_checks"] == "not executed"
    assert main(["--mode", "smoke", "--cpu", "--run-dir", str(root)]) == 0
    assert len(list(tmp_path.glob("smoke*"))) == 2


def test_invalid_input_and_failed_run_are_explicit(tmp_path):
    from scripts.suffix_transport import main
    from grace_gc.trainer.suffix_transport import validate_experiment

    cfg = config()
    cfg["lora"]["dropout"] = .1
    with pytest.raises(ValueError):
        validate_experiment(cfg)
    with pytest.raises(ValueError, match="model-path"):
        main(["--mode", "audit", "--run-dir", str(tmp_path / "bad")])
    meta = json.loads((tmp_path / "bad" / "run_meta.json").read_text())
    assert meta["status"] == "failed" and meta["exit_code"] == 1
