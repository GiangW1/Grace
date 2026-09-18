import json
from pathlib import Path

import numpy as np
import pytest

from grace_gc.audit.variance_cost import variance_times_cost
from grace_gc.data.math_data import load_math_records, split_records
from grace_gc.evaluation.eval_full import EvalItem, evaluate_items
from grace_gc.trainer.grace_step import StartRecord, assemble_ghat, next_start_count
from grace_gc.trainer.methods import method_spec


def test_variance_times_cost_includes_var_g():
    g = np.array([[-1.0], [1.0]])
    m = np.zeros((2, 1))
    suffix = np.ones(2)
    one = variance_times_cost(g, m, np.ones(2), prefix_cost=0.1, suffix_cost=suffix)
    assert one["ratio"] == pytest.approx(1.0)
    half = variance_times_cost(g, m, np.full(2, 0.5), prefix_cost=0.1, suffix_cost=suffix)
    assert half["ratio"] == pytest.approx(1.0909091)


def test_answer_zero_is_kept(tmp_path: Path):
    path = tmp_path / "z.jsonl"
    path.write_text(json.dumps({"problem_id": "z", "prompt": "q", "answer": 0}) + "\n", encoding="utf-8")
    recs = load_math_records(path)
    assert recs[0].answer == "0"


def test_conflict_split_raises(tmp_path: Path):
    path = tmp_path / "c.jsonl"
    rows = [
        {"problem_id": "p", "prompt": "a", "answer": "1", "split": "train"},
        {"problem_id": "p", "prompt": "b", "answer": "2", "split": "eval"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="conflict split"):
        split_records(load_math_records(path), seed=0)


def test_eval_avg_is_mean_reward_not_any_success():
    """MATH-500 headline is avg@4, not 'at least one of 4 correct'."""
    items = [EvalItem("a", "1", ["\\boxed{1}", "no", "no", "no"], [False] * 4)]
    out = evaluate_items(items, k=4)
    assert out["avg"] == pytest.approx(0.25)
    assert out["problem_success_rate"] == pytest.approx(1.0)
    assert out["pass_at_k"] == pytest.approx(1.0)


def test_evaluate_cli_prints_avg():
    import inspect

    from scripts import evaluate

    src = inspect.getsource(evaluate.main)
    assert 'result["avg"]' in src
    assert 'result["problem_success_rate"]' not in src


def test_evaluate_answers_uses_stored_n_as_k(tmp_path: Path):
    path = tmp_path / "a.jsonl"
    path.write_text(
        json.dumps(
            {
                "problem_id": "a",
                "gold": "1",
                "answers": ["\\boxed{1}", "no", "no", "no"],
                "truncated": [False] * 4,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    from scripts.evaluate import main

    assert main(["--answers", str(path), "--run-dir", str(tmp_path / "e")]) == 0
    summary = json.loads((tmp_path / "e" / "eval_summary.json").read_text(encoding="utf-8"))
    assert summary["avg"] == pytest.approx(0.25)
    assert summary["requested_k"] == 4
    assert summary["pass_at_k"] == pytest.approx(1.0)


def test_pass_at_k_null_when_n_lt_k():
    items = [EvalItem("a", "1", ["\\boxed{1}", "no"], [False, False])]
    out = evaluate_items(items, k=8)
    assert out["pass_at_k"] is None
    assert out["per_problem"][0]["pass_at_k"] is None


def test_answers_truncated_mismatch_raises():
    with pytest.raises(ValueError, match="truncated"):
        evaluate_items([EvalItem("a", "1", ["x", "y"], [False])], k=1)


def test_assemble_ghat_requires_completed_g():
    spec = method_spec("full_pg")
    recs = [
        StartRecord("a", False, 1.0, 1.0, np.zeros(2), 1, 1, 1.0, 1.0, np.array([2.0, 0.0]), True),
        StartRecord("b", False, 1.0, 1.0, np.zeros(2), 1, 1, 1.0, 1.0, None, False),
    ]
    with pytest.raises(ValueError, match="needs G"):
        assemble_ghat(spec, recs, np.eye(2), 2)


def test_next_n_uses_fixed_ref():
    hist = [0.5, 0.5, 0.5, 0.5]
    assert next_start_count(hist, 8, 1.0) == 16
    assert next_start_count(hist + [0.5], 8, 1.0) == 16
    assert next_start_count([0.5], 8, 1.0, group=16) == 16


def test_resolve_rejects_silent_n_change():
    from grace_gc.trainer.loop import resolve_start_counts

    with pytest.raises(ValueError, match="multiple"):
        resolve_start_counts({"n_prompts": 4, "n_start": 11}, method_spec("grace"))


def test_jl_project_large_d_is_deterministic():
    from grace_gc.audit.prefix_audit import jl_project

    rng = np.random.default_rng(0)
    g = rng.normal(size=(3, 20_000))
    a = jl_project(g, 8, 1)
    b = jl_project(g, 8, 1)
    c = jl_project(g, 8, 2)
    assert a.shape == (3, 8)
    np.testing.assert_allclose(a, b)
    assert not np.allclose(a, c)


def test_audit_jl_dim_shrinks_stored_grads():
    pytest.importorskip("torch")
    from grace_gc.audit.run import generate_bundles_tiny
    from grace_gc.data.math_data import MathRecord

    recs = [MathRecord("p", "1+1", "2")]
    raw = generate_bundles_tiny(recs, n_prefixes=1, n_cont=2, decision=1, max_new=2, seed=0)
    assert raw[0].grads.shape[1] > 8
    proj = generate_bundles_tiny(
        recs,
        n_prefixes=1,
        n_cont=2,
        decision=1,
        max_new=2,
        seed=0,
        cfg={"audit": {"jl_dim": 8, "jl_seed": 1}},
    )
    assert proj[0].grads.shape == (raw[0].grads.shape[0], 8)


def test_continue_uses_actual_prefix_remaining():
    pytest.importorskip("torch")
    import torch

    from grace_gc.core.layout import collect_lora_layout
    from grace_gc.core.rng import IsolatedRNG
    from grace_gc.predictor.reservoir import GradientReservoir
    from grace_gc.trainer.algorithm import TrainState, continuation_remainings, run_algorithm1_step
    from grace_gc.trainer.baseline import HistoricalBaseline
    from grace_gc.trainer.cpu_tiny import TinyLoRAActor
    from grace_gc.trainer.tiny_engine import make_tiny_engines

    prefixes = [[1, 2, 9], [1, 2, 9, 8]]
    prompt_lens = [2, 2]
    rem = continuation_remainings(prefixes, prompt_lens, np.array([False, False]), 8)
    np.testing.assert_array_equal(rem, [7, 6])
    rem_fin = continuation_remainings(prefixes, prompt_lens, np.array([True, False]), 8)
    np.testing.assert_array_equal(rem_fin, [0, 6])

    actor = TinyLoRAActor()
    engines = make_tiny_engines(actor, actor.vocab)
    seen = []
    orig = engines.continue_selected

    def wrap(prefs, selected, max_new, rng):
        if np.any(selected):
            seen.append(int(max_new))
        return orig(prefs, selected, max_new, rng)

    engines.continue_selected = wrap

    def short_prefix(prompt_ids, max_new, rng, stream):
        _ = max_new, rng, stream
        return [list(p) + [1] for p in prompt_ids], np.zeros(len(prompt_ids), dtype=bool)

    engines.generate_prefix = short_prefix
    opt = torch.optim.SGD(actor.trainable_params(), lr=0.01)
    layout = collect_lora_layout(actor.named_lora_params())
    state = TrainState(
        spec=method_spec("full_pg"),
        baseline=HistoricalBaseline(),
        rng=IsolatedRNG.create(0),
        layout=layout,
        u=np.eye(layout.dim, 2, dtype=np.float64),
        predictor=None,
        reservoir=GradientReservoir(capacity=4),
    )
    run_algorithm1_step(
        engines,
        state,
        [[1, 2], [3, 4]],
        ["a", "b"],
        ["1", "2"],
        {"decision_tokens": 4, "max_new_tokens": 8, "predictor": {"warmup_steps": 0, "audit_s": 0.0}},
        opt,
    )
    assert seen == [7]


def test_next_n_counts_prefix_in_token_budget():
    from grace_gc.trainer.grace_step import batch_token_costs

    prefixes = [[1, 2, 10, 11], [1, 2, 10, 11]]
    prompt_lens = [2, 2]
    used, full = batch_token_costs(prompt_lens, prefixes, np.array([False, False]), np.array([0.5, 0.5]), 8)
    assert used == pytest.approx(5.0)
    assert full == pytest.approx(8.0)
    assert next_start_count([used / full], 8, 1.0) == 13
    used_pg, full_pg = batch_token_costs(prompt_lens, prefixes, np.array([False, False]), np.array([1.0, 1.0]), 8)
    assert used_pg == pytest.approx(full_pg)
    assert next_start_count([used_pg / full_pg], 8, 1.0) == 8
    used_fin, full_fin = batch_token_costs(
        prompt_lens, prefixes, np.array([True, False]), np.array([1.0, 0.5]), 8
    )
    assert used_fin == pytest.approx(3.5)
    assert full_fin == pytest.approx(5.0)


def test_eos_prefix_is_not_extended():
    pytest.importorskip("torch")
    import torch

    from grace_gc.core.rng import IsolatedRNG
    from grace_gc.trainer.cpu_tiny import TinyLoRAActor, _sample_tokens

    actor = TinyLoRAActor()
    tok = torch.tensor([[1, 3, actor.eos_id]])
    out, fin = _sample_tokens(actor, tok, 4, IsolatedRNG.create(0), "continuation")
    assert out[0].tolist() == [1, 3, actor.eos_id]
    assert bool(fin[0])


def test_audit_g_uses_original_prompt_len():
    pytest.importorskip("torch")
    from grace_gc.audit.run import _policy_grad_vec
    from grace_gc.core.layout import collect_lora_layout
    from grace_gc.trainer.cpu_tiny import TinyLoRAActor
    from grace_gc.trainer.tiny_engine import make_tiny_engines

    actor = TinyLoRAActor()
    layout = collect_lora_layout(actor.named_lora_params())
    engines = make_tiny_engines(actor, actor.vocab)
    full = [1, 2, 3, 4, 5, 6]
    g_full = _policy_grad_vec(engines, layout, full, 2, 1.0, 0.0)
    g_suf = _policy_grad_vec(engines, layout, full, 4, 1.0, 0.0)
    assert not np.allclose(g_full, g_suf)
    assert np.linalg.norm(g_full) > 0


def test_real_stream_backward_each_matches_batched():
    from grace_gc.trainer.actor_update import real_stream_backward, real_stream_backward_each
    from grace_gc.trainer.cpu_tiny import TinyLoRAActor

    actor = TinyLoRAActor()
    seqs = [[1, 2, 3, 4], [2, 3, 4, 5, 6]]
    adv = np.array([0.5, -0.25], dtype=np.float64)
    p = np.array([0.4, 1.0], dtype=np.float64)
    z = np.array([1.0, 1.0], dtype=np.float64)
    for param in actor.trainable_params():
        param.grad = None
    lps = []
    for seq in seqs:
        t = __import__("torch").as_tensor(seq, dtype=__import__("torch").long).unsqueeze(0)
        lp, _tlp, _h = actor.token_logprob_sum(t, 2)
        lps.append(lp[0])
    stacked = __import__("torch").stack(lps)
    real_stream_backward(stacked, adv, p, z, 2)
    batched = [param.grad.detach().clone() for param in actor.trainable_params() if param.grad is not None]
    for param in actor.trainable_params():
        param.grad = None

    def _lp(i):
        t = __import__("torch").as_tensor(seqs[i], dtype=__import__("torch").long).unsqueeze(0)
        lp, _tlp, _h = actor.token_logprob_sum(t, 2)
        return lp[0]

    real_stream_backward_each(_lp, adv, p, z, 2, z >= 1.0)
    each = [param.grad.detach().clone() for param in actor.trainable_params() if param.grad is not None]
    assert len(batched) == len(each)
    for a, b in zip(batched, each):
        assert __import__("torch").allclose(a, b, atol=1e-6)


def test_natural_eos_at_budget_is_not_truncated():
    from grace_gc.trainer.algorithm import _length_truncated, _traj_natural_finish

    assert _length_truncated([1, 2, 3, 9], 2, 2, False, 9) is False
    assert _length_truncated([1, 2, 3, 4], 2, 2, False, 9) is True
    assert _length_truncated([1, 2, 9], 2, 8, True, 9) is False
    assert _traj_natural_finish(True, [1, 2], 9) is True
    assert _traj_natural_finish(False, [1, 2, 3, 9], 9) is True
    assert _traj_natural_finish(False, [1, 2, 3, 4], 9) is False
    assert _traj_natural_finish(False, None, 9) is False
    assert _traj_natural_finish(False, [1, 2, 3, 4], 9, generated=2, requested=8) is True
    assert _traj_natural_finish(False, [1, 2, 3, 4], 9, generated=8, requested=8) is False
    assert _traj_natural_finish(False, [1, 2], 9, generated=0, requested=8) is True
    assert _traj_natural_finish(False, [1, 2, 3, 4], 9, generated=2, requested=8, finish_reason="length") is False
    assert _traj_natural_finish(False, [1, 2, 3, 4], 9, generated=2, requested=8, finish_reason="stop") is True
    assert _traj_natural_finish(False, [1, 2, 3, 4], 9, generated=2, requested=8, finish_reason="FinishReason.LENGTH") is False
    assert _length_truncated([1, 2, 3, 4], 2, 8, False, 9, finish_reason="length") is True
    assert _length_truncated([1, 2, 3, 9], 2, 8, True, 9, finish_reason="stop") is False
    assert _length_truncated([1, 2, 3, 4], 2, 8, False, 9, finish_reason="FinishReason.LENGTH") is True


def test_unknown_split_raises(tmp_path: Path):
    path = tmp_path / "u.jsonl"
    path.write_text(json.dumps({"problem_id": "x", "prompt": "q", "answer": "1", "split": "test"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown split"):
        load_math_records(path)


def test_duplicate_prompt_conflict(tmp_path: Path):
    from grace_gc.data.math_data import last_load_report

    path = tmp_path / "d.jsonl"
    rows = [
        {"problem_id": "a", "prompt": "same", "answer": "1"},
        {"problem_id": "b", "prompt": "same", "answer": "2"},
        {"problem_id": "c", "prompt": "other", "answer": "3"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    recs = load_math_records(path)
    assert [r.problem_id for r in recs] == ["c"]
    report = last_load_report()
    assert report["n_conflict_groups"] == 1
    assert report["n_kept"] == 1
    assert sorted(report["conflicts"][0]["answers"]) == ["1", "2"]


def test_duplicate_prompt_split_conflict_raises(tmp_path: Path):
    path = tmp_path / "s.jsonl"
    rows = [
        {"problem_id": "a", "prompt": "same", "answer": "1", "split": "train"},
        {"problem_id": "b", "prompt": "same", "answer": "1", "split": "eval"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="split conflict"):
        load_math_records(path)


def test_solution_boxed_extracted(tmp_path: Path):
    path = tmp_path / "s.jsonl"
    path.write_text(json.dumps({"problem_id": "s", "prompt": "q", "solution": "work \\boxed{7}"}) + "\n", encoding="utf-8")
    assert load_math_records(path)[0].answer == "7"


def test_rho_l_one_prefix_is_one():
    from grace_gc.audit.prefix_audit import PrefixBundle, audit_bundles

    g = np.array([[1.0, 0.0], [1.01, 0.0], [0.99, 0.0], [1.0, 0.01]])
    out = audit_bundles(
        [PrefixBundle("p", 8, np.array([1.0, 0.0, 1.0, 0.0]), g)],
        np.eye(2),
        {},
        np.random.default_rng(0),
    )
    assert out["rho_l_all"] == pytest.approx(1.0)


def test_t_a_uses_var_r_not_rho_a():
    from grace_gc.audit.prefix_audit import PrefixBundle, audit_bundles
    from grace_gc.audit.stats import t_answer

    g1 = np.array([[1.0, 0.0], [1.01, 0.0], [0.99, 0.0], [1.0, 0.01]])
    g2 = np.array([[0.0, 1.0], [0.0, 1.01], [0.0, 0.99], [0.01, 1.0]])
    bundles = [
        PrefixBundle("p", 8, np.array([1.0, 0.0, 1.0, 0.0]), g1, path_id="p:0"),
        PrefixBundle("p", 16, np.array([0.0, 0.0, 0.0, 0.1]), g2, path_id="p:0"),
    ]
    out = audit_bundles(
        bundles,
        np.eye(2),
        {"answer_undecided": 0.0, "nonzero_signal_kappa": 0.0},
        np.random.default_rng(0),
    )
    times = np.asarray(out["times"], dtype=float)
    assert out["t_A"] == t_answer(np.asarray(out["var_r_curve_all"]), times, 0.05)
    assert out["t_A"] == pytest.approx(16.0)


def test_t_l_uses_detect_half_not_report():
    from grace_gc.audit.prefix_audit import PrefixBundle, audit_bundles
    from grace_gc.audit.stats import t_learn

    g_det_hi = np.array([[10.0, 0.0], [0.0, 10.0], [10.0, 10.0], [0.0, 0.0]], dtype=np.float64)
    g_rep_lo = np.tile(np.array([[1.0, 1.0]]), (4, 1)) + np.linspace(-0.01, 0.01, 4)[:, None]
    g8 = np.concatenate([g_det_hi, g_rep_lo], axis=0)
    g_det_lo = np.tile(np.array([[1.0, 1.0]]), (4, 1)) + np.linspace(-0.01, 0.01, 4)[:, None]
    g16 = np.concatenate([g_det_lo, g_rep_lo], axis=0)
    rewards = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    out = audit_bundles(
        [
            PrefixBundle("p", 8, rewards, g8, path_id="p:0"),
            PrefixBundle("p", 16, rewards, g16, path_id="p:0"),
        ],
        np.eye(2),
        {"answer_undecided": 0.0, "nonzero_signal_kappa": -1.0, "epsilon_learn": 0.2},
        np.random.default_rng(0),
    )
    times = np.asarray(out["times"], dtype=float)
    assert out["rho_l_curve"][0] < out["rho_l_detect_curve"][0]
    assert out["t_L"] == t_learn(np.asarray(out["rho_l_detect_ucb_curve"]), times, 0.2)
    assert out["t_L_point"] == t_learn(np.asarray(out["rho_l_detect_curve"]), times, 0.2)
    # One problem has no between-problem standard error; retain point detection
    # without presenting it as a confidence bound.
    assert out["t_L"] is None
    assert out["t_L_point"] == pytest.approx(16.0)
    report_t_l = t_learn(np.asarray(out["rho_l_curve"]), times, 0.2)
    assert report_t_l == pytest.approx(8.0)


def test_answer_emitted_leaves_headline_population():
    from grace_gc.audit.prefix_audit import PrefixBundle, audit_bundles

    g = np.array([[1.0, 0.0], [1.01, 0.0], [0.99, 0.0], [1.0, 0.01]])
    rewards = np.array([1.0, 0.0, 1.0, 0.0])
    out = audit_bundles(
        [
            PrefixBundle("p", 8, rewards, g, path_id="p:0", answer_emitted=False),
            PrefixBundle("p", 16, rewards, g, path_id="p:0", answer_emitted=True),
        ],
        np.eye(2),
        {"answer_undecided": 0.0, "nonzero_signal_kappa": -1.0},
        np.random.default_rng(0),
    )
    assert out["n_gated"] == 1
    assert out["n_gated_pre_emit"] == 1
    assert "elf_pre_emit" in out
    assert out["times"] == [8, 16]
    assert np.isfinite(out["rho_l_curve"][0])
    assert not np.isfinite(out["rho_l_curve"][1])
    assert np.isfinite(out["rho_l_curve_all"][1])


def test_detect_report_split_uses_second_half_for_rho():
    from grace_gc.audit.prefix_audit import PrefixBundle, _cond_var, _detect_report_rows, audit_rho_at_t

    g_det = np.tile(np.array([[10.0, 0.0]]), (4, 1)) + np.linspace(-0.1, 0.1, 4)[:, None]
    g_rep = np.tile(np.array([[0.0, 1.0]]), (4, 1)) + np.linspace(-0.01, 0.01, 4)[:, None]
    g = np.concatenate([g_det, g_rep], axis=0)
    rewards = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    det, _, rep, _ = _detect_report_rows(g, rewards)
    assert det.shape[0] == 4 and rep.shape[0] == 4
    assert _cond_var(rep) < _cond_var(det)
    part = audit_rho_at_t([PrefixBundle("p", 8, rewards, g)], prompt_var_g_by={"p": _cond_var(g)})
    assert part["rho_l"] == pytest.approx(_cond_var(rep) / _cond_var(g))
    det_part = audit_rho_at_t(
        [PrefixBundle("p", 8, rewards, g)],
        prompt_var_g_by={"p": _cond_var(g)},
        half="detect",
    )
    assert det_part["rho_l"] == pytest.approx(_cond_var(det) / _cond_var(g))


def test_audit_rho_averages_problems_not_prefixes():
    from grace_gc.audit.prefix_audit import PrefixBundle, _cond_var, audit_rho_at_t

    g_a = np.array([[1.0, 0.0], [1.01, 0.0], [0.99, 0.0], [1.0, 0.01]])
    g_b = np.array([[10.0, 0.0], [0.0, 10.0], [10.0, 10.0], [0.0, 0.0]])
    rewards = np.array([1.0, 0.0, 1.0, 0.0])
    many_a = [PrefixBundle("a", 8, rewards, g_a, path_id=f"a:{i}") for i in range(4)]
    one_b = [PrefixBundle("b", 8, rewards, g_b, path_id="b:0")]
    denom = 10.0
    part = audit_rho_at_t(many_a + one_b, prompt_var_g_by={"a": denom, "b": denom})
    rho_a = _cond_var(g_a) / denom
    rho_b = _cond_var(g_b) / denom
    prefix_mean = (4.0 * rho_a + rho_b) / 5.0
    assert part["rho_l"] == pytest.approx(0.5 * (rho_a + rho_b))
    assert part["rho_l"] != pytest.approx(prefix_mean)
    assert len(part["rho_l_vals"]) == 2


def test_t_l_uses_problem_level_ucb():
    from grace_gc.audit.stats import rho_ucb, t_learn

    vals8 = [0.01, 0.30]
    ucb8 = rho_ucb(vals8)
    ucb16 = rho_ucb([0.01, 0.02])
    times = np.array([8.0, 16.0])
    assert float(np.mean(vals8)) <= 0.2
    assert ucb8 > 0.2
    assert t_learn(np.array([float(np.mean(vals8)), 0.01]), times, 0.2) == pytest.approx(8.0)
    assert t_learn(np.array([ucb8, ucb16]), times, 0.2) == pytest.approx(16.0)


def test_algorithm_does_not_zero_truncated_grpo_short():
    import inspect

    from grace_gc.trainer.algorithm import run_algorithm1_step

    src = inspect.getsource(run_algorithm1_step)
    assert 'if state.spec.name == "grpo_short" and truncated:' not in src


def test_cond_var_uses_unbiased_factor():
    from grace_gc.audit.prefix_audit import _cond_var

    g = np.array([[1.0, 0.0], [3.0, 0.0]])
    # mean=2, sq=[1,1], ML mean=1, unbiased = 1 * 2/1 = 2
    assert _cond_var(g) == pytest.approx(2.0)


def test_wilson_and_pre_emit_filter_headline():
    from grace_gc.audit.prefix_audit import PrefixBundle, audit_bundles
    from grace_gc.audit.stats import wilson_q_inside

    rewards_open = np.array([1.0, 0.0, 1.0, 0.0])
    rewards_sure = np.array([1.0, 1.0, 1.0, 1.0])
    assert wilson_q_inside(rewards_open)
    assert not wilson_q_inside(rewards_sure)
    g = np.array([[1.0, 0.0], [1.01, 0.0], [0.99, 0.0], [1.0, 0.01]])
    out = audit_bundles(
        [
            PrefixBundle("p", 8, rewards_open, g, path_id="p:0"),
            PrefixBundle("p", 8, rewards_sure, g, path_id="p:1"),
        ],
        np.eye(2),
        {"answer_undecided": 0.0, "nonzero_signal_kappa": -1.0},
        np.random.default_rng(0),
    )
    assert out["n_gated"] == 1
    assert out["n_bundles"] == 2


def test_allocation_is_per_decision_time():
    from grace_gc.audit.prefix_audit import PrefixBundle, _p_at_decision
    from grace_gc.trainer.methods import method_spec

    g = np.array([[1.0, 0.0], [-1.0, 0.0]])
    early = PrefixBundle("a", 8, np.array([1.0, 0.0]), g, r_hat=8.0, c_hat=1.0, path_id="a:0")
    late = PrefixBundle("b", 64, np.array([0.0, 1.0]), g, r_hat=0.05, c_hat=1.0, path_id="b:0")
    analysis = {"method": "grace", "beta": 0.5, "p_min": 0.2}
    spec = method_spec("grace")
    mixed = _p_at_decision([early, late], spec, analysis, {}, {})
    per_early = _p_at_decision([early], spec, analysis, {}, {})
    per_late = _p_at_decision([late], spec, analysis, {}, {})
    assert mixed[0] != mixed[1]
    assert per_early[0] == pytest.approx(0.5, abs=1e-4)
    assert per_late[0] == pytest.approx(0.5, abs=1e-4)
    assert per_early[0] != mixed[0] or per_late[0] != mixed[1]


def test_headline_curves_use_gated_prefixes():
    from grace_gc.audit.prefix_audit import PrefixBundle, audit_bundles

    g_open = np.array([[1.0, 0.0], [1.01, 0.0], [0.99, 0.0], [1.0, 0.01]])
    g_done = np.array([[2.0, 0.0], [2.0, 0.0], [2.0, 0.0], [2.0, 0.0]])
    bundles = [
        PrefixBundle("p", 8, np.array([1.0, 0.0, 1.0, 0.0]), g_open),
        PrefixBundle("p", 8, np.array([1.0, 1.0, 1.0, 1.0]), g_done),
    ]
    out = audit_bundles(
        bundles,
        np.eye(2),
        {"answer_undecided": 0.15, "nonzero_signal_kappa": 0.0},
        np.random.default_rng(0),
    )
    assert out["n_gated"] == 1
    assert out["var_r_curve"][0] == pytest.approx(float(np.var([1.0, 0.0, 1.0, 0.0], ddof=1)))
    assert out["var_r_curve_all"][0] != out["var_r_curve"][0]


def test_prompt_var_uses_earliest_prefix_per_path():
    from grace_gc.audit.prefix_audit import PrefixBundle, _cond_var, _prompt_stats, _prompt_var_r

    g_early = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]], dtype=np.float64)
    g_late = np.array([[1.0, 1.0], [1.01, 1.0], [0.99, 1.0], [1.0, 1.01]], dtype=np.float64)
    group = [
        PrefixBundle("p", 4, np.array([1.0, 0.0, 1.0, 0.0]), g_early, path_id="p:0"),
        PrefixBundle("p", 16, np.array([1.0, 1.0, 1.0, 1.0]), g_late, path_id="p:0"),
    ]
    var_g, _norm = _prompt_stats(group)
    assert var_g == pytest.approx(_cond_var(g_early))
    assert _prompt_var_r(group) == pytest.approx(float(np.var([1.0, 0.0, 1.0, 0.0], ddof=1)))


def test_audit_jsonl_keeps_answer_emitted():
    from grace_gc.audit.prefix_audit import PrefixBundle, bundle_from_dict, bundle_to_dict
    import json

    bundle = PrefixBundle("p", 4, np.array([1.]), np.ones((1, 2)), answer_emitted=True,
                          gold="1", baseline=.5, continuation_records=[{"reward": 1., "seed": 17}])
    restored = bundle_from_dict(json.loads(json.dumps(bundle_to_dict(bundle))))
    assert restored.answer_emitted is True
    assert restored.gold == "1" and restored.baseline == .5
    assert restored.continuation_records == [{"reward": 1., "seed": 17}]


def test_independent_pass_rate_uses_eval_stream():
    pytest.importorskip("torch")
    from types import SimpleNamespace

    from grace_gc.audit.run import _independent_pass_rate
    from grace_gc.core.rng import IsolatedRNG
    from grace_gc.data.math_data import MathRecord

    rec = MathRecord("p", "q", "1")
    seen = []

    class _Eng:
        eos_id = None

        def decode(self, ids):
            return "\\boxed{1}" if ids and ids[-1] == 1 else "no"

        def generate_prefix(self, prompts, max_new, rng, stream):
            seen.append(stream)
            assert stream == "eval"
            finished = []
            fulls = []
            for i, prompt in enumerate(prompts):
                fulls.append(list(prompt) + [1 if i % 2 == 0 else 0])
                finished.append(True)
            return fulls, np.asarray(finished, dtype=bool)

    def encode(_rec, n):
        return [[7, 8]] * n, ["p"] * n, ["1"] * n

    rate = _independent_pass_rate(_Eng(), rec, encode, 4, 8, IsolatedRNG.create(0), None)
    assert seen == ["eval"]
    assert rate == pytest.approx(0.5)


def test_audit_selected_continuation_none_raises():
    pytest.importorskip("torch")

    from grace_gc.audit.run import _bundles_from_engines
    from grace_gc.core.layout import collect_lora_layout
    from grace_gc.data.math_data import MathRecord
    from grace_gc.trainer.cpu_tiny import TinyLoRAActor
    from grace_gc.trainer.tiny_engine import make_tiny_engines

    actor = TinyLoRAActor()
    engines = make_tiny_engines(actor, actor.vocab)
    engines.generate_prefix = lambda prompts, max_new, rng, stream: (
        [list(p) + [1, 2] for p in prompts],
        np.zeros(len(prompts), dtype=bool),
    )
    engines.continue_selected = lambda prefixes, selected, max_new, rng: [None] * len(prefixes)
    rec = MathRecord("p", "q add two", "1")

    def encode(_rec, n):
        return [[1, 2]] * n, ["p"] * n, ["1"] * n

    with pytest.raises(ValueError, match="continuation"):
        _bundles_from_engines(
            [rec],
            engines,
            collect_lora_layout(actor.named_lora_params()),
            n_prefixes=1,
            n_cont=1,
            decision=2,
            max_new=8,
            seed=0,
            encode_fn=encode,
        )


def test_audit_uses_each_continuation_finish_reason(monkeypatch):
    pytest.importorskip("torch")

    from grace_gc.audit import run as audit_run
    from grace_gc.audit.run import _bundles_from_engines
    from grace_gc.core.layout import collect_lora_layout
    from grace_gc.data.math_data import MathRecord
    from grace_gc.trainer.cpu_tiny import TinyLoRAActor
    from grace_gc.trainer.tiny_engine import make_tiny_engines

    seen = []

    def capture_truncated(full_ids, prompt_len, max_new, finished, eos_id, finish_reason=None):
        seen.append(finish_reason)
        return False

    monkeypatch.setattr(audit_run, "_length_truncated", capture_truncated)

    actor = TinyLoRAActor()
    engines = make_tiny_engines(actor, actor.vocab)
    engines.generate_prefix = lambda prompts, max_new, rng, stream: (
        [list(p) + [1, 2] for p in prompts],
        np.zeros(len(prompts), dtype=bool),
    )
    reasons = iter(["stop", "length"])

    def cont(prefixes, selected, max_new, rng):
        reason = next(reasons)
        engines.last_rollout = {"continue_finish_reasons": {0: reason}}
        return [list(prefixes[0]) + [3, 4]]

    engines.continue_selected = cont
    rec = MathRecord("p", "q add two", "1")

    def encode(_rec, n):
        return [[1, 2]] * n, ["p"] * n, ["1"] * n

    _bundles_from_engines(
        [rec],
        engines,
        collect_lora_layout(actor.named_lora_params()),
        n_prefixes=1,
        n_cont=2,
        decision=2,
        max_new=8,
        seed=0,
        encode_fn=encode,
    )
    assert seen[-2:] == ["stop", "length"]


def test_audit_finished_prefix_ignores_stale_continue_finish(monkeypatch):
    pytest.importorskip("torch")

    from grace_gc.audit import run as audit_run
    from grace_gc.audit.run import _bundles_from_engines
    from grace_gc.core.layout import collect_lora_layout
    from grace_gc.data.math_data import MathRecord
    from grace_gc.trainer.cpu_tiny import TinyLoRAActor
    from grace_gc.trainer.tiny_engine import make_tiny_engines

    seen = []

    def capture_truncated(full_ids, prompt_len, max_new, finished, eos_id, finish_reason=None):
        seen.append(finish_reason)
        return False

    monkeypatch.setattr(audit_run, "_length_truncated", capture_truncated)

    actor = TinyLoRAActor()
    engines = make_tiny_engines(actor, actor.vocab)
    engines.last_rollout = {"continue_finish_reasons": {0: "length"}}
    engines.generate_prefix = lambda prompts, max_new, rng, stream: (
        [list(p) + [1, actor.eos_id] for p in prompts],
        np.ones(len(prompts), dtype=bool),
    )
    rec = MathRecord("p", "q add two", "1")

    def encode(_rec, n):
        return [[1, 2]] * n, ["p"] * n, ["1"] * n

    _bundles_from_engines(
        [rec],
        engines,
        collect_lora_layout(actor.named_lora_params()),
        n_prefixes=1,
        n_cont=2,
        decision=2,
        max_new=8,
        seed=0,
        encode_fn=encode,
    )
    assert seen[-1] is None


def test_prescan_sets_baseline_to_sample_mean():
    pytest.importorskip("torch")
    import torch

    from grace_gc.core.layout import collect_lora_layout
    from grace_gc.core.rng import IsolatedRNG
    from grace_gc.predictor.reservoir import GradientReservoir
    from grace_gc.trainer.algorithm import TrainState, prescan_unseen_baselines
    from grace_gc.trainer.baseline import HistoricalBaseline
    from grace_gc.trainer.cpu_tiny import TinyLoRAActor
    from grace_gc.trainer.tiny_engine import make_tiny_engines

    actor = TinyLoRAActor()
    engines = make_tiny_engines(actor, actor.vocab)
    engines.reward_fn = lambda *a, **k: 1.0
    calls = []

    def gen(prompt_ids, max_new, rng, stream):
        calls.append((len(prompt_ids), int(max_new), stream))
        return [list(p) + [1] for p in prompt_ids], np.zeros(len(prompt_ids), dtype=bool)

    engines.generate_prefix = gen
    state = TrainState(
        spec=method_spec("grace"),
        baseline=HistoricalBaseline(),
        rng=IsolatedRNG.create(0),
        layout=collect_lora_layout(actor.named_lora_params()),
        u=np.eye(2, 2, dtype=np.float64),
        predictor=None,
        reservoir=GradientReservoir(capacity=4),
    )
    _ = torch
    token_before = dict(state.rng.counters)
    assert state.baseline.get("a") == pytest.approx(0.5)
    n = prescan_unseen_baselines(engines, state, [[1, 2], [1, 2]], ["a", "a"], ["1", "1"], 8, 2)
    assert n == 1
    assert calls == [(2, 8, "token")]
    assert state.rng.counters == token_before
    assert state.baseline.values["a"] == pytest.approx(1.0)
    assert prescan_unseen_baselines(engines, state, [[1, 2]], ["a"], ["1"], 8, 2) == 0
    assert calls == [(2, 8, "token")]


def test_eval_tiny_none_sequence_raises(monkeypatch):
    pytest.importorskip("torch")
    from grace_gc.data.math_data import MathRecord
    from grace_gc.evaluation.generate import generate_answers_tiny
    from grace_gc.trainer.tiny_engine import make_tiny_engines

    recs = [MathRecord("p", "q add two", "1")]
    orig = make_tiny_engines

    def fake_make(actor, vocab):
        engines = orig(actor, vocab)
        engines.continue_selected = lambda prefixes, selected, max_new, rng: [None] * len(prefixes)
        return engines

    monkeypatch.setattr("grace_gc.trainer.tiny_engine.make_tiny_engines", fake_make)
    with pytest.raises(ValueError, match="eval generate"):
        generate_answers_tiny(recs, n=1, max_new=4, seed=0)


def test_prescan_none_sequence_raises():
    pytest.importorskip("torch")

    from grace_gc.core.layout import collect_lora_layout
    from grace_gc.core.rng import IsolatedRNG
    from grace_gc.predictor.reservoir import GradientReservoir
    from grace_gc.trainer.algorithm import TrainState, prescan_unseen_baselines
    from grace_gc.trainer.baseline import HistoricalBaseline
    from grace_gc.trainer.cpu_tiny import TinyLoRAActor
    from grace_gc.trainer.tiny_engine import make_tiny_engines

    actor = TinyLoRAActor()
    engines = make_tiny_engines(actor, actor.vocab)
    engines.generate_prefix = lambda *a, **k: ([None], np.array([False]))
    state = TrainState(
        spec=method_spec("grace"),
        baseline=HistoricalBaseline(),
        rng=IsolatedRNG.create(0),
        layout=collect_lora_layout(actor.named_lora_params()),
        u=np.eye(2, 2, dtype=np.float64),
        predictor=None,
        reservoir=GradientReservoir(capacity=4),
    )
    with pytest.raises(ValueError, match="prescan"):
        prescan_unseen_baselines(engines, state, [[1, 2]], ["a"], ["1"], 8, 2)


def test_baseline_ema_starts_from_half():
    from grace_gc.trainer.baseline import HistoricalBaseline

    base = HistoricalBaseline(alpha=0.7)
    base.update("p", 1.0)
    assert base.get("p") == pytest.approx(0.7 * 1.0 + 0.3 * 0.5)


def test_baseline_batch_mean_not_last_start():
    from grace_gc.trainer.baseline import HistoricalBaseline

    rewards = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    batched = HistoricalBaseline(alpha=0.7)
    batched.update_mean("p", rewards)
    assert batched.get("p") == pytest.approx(0.5)
    sequential = HistoricalBaseline(alpha=0.7)
    for reward in rewards:
        sequential.update("p", reward)
    assert sequential.get("p") != pytest.approx(batched.get("p"))


def test_gate_rho_aligned_when_prompt_var_zero():
    from grace_gc.audit.prefix_audit import PrefixBundle, audit_bundles

    g_flat = np.tile(np.array([1.0, 0.0, 0.0]), (4, 1))
    g_var = np.array([[1.0, 0.0, 0.0], [1.01, 0.0, 0.0], [0.99, 0.0, 0.0], [1.0, 0.01, 0.0]])
    bundles = [
        PrefixBundle("zero", 8, np.zeros(4), g_flat),
        PrefixBundle("p", 8, np.array([1.0, 0.0, 1.0, 0.0]), g_var),
    ]
    out = audit_bundles(bundles, np.eye(3, 2), {"nonzero_signal_kappa": 0.0, "answer_undecided": 0.15}, np.random.default_rng(0))
    assert out["gates"]["rho_l"]["all"]["n"] == 2
    assert out["gates"]["rho_l"]["selected"]["n"] == out["gates"]["selected"]["n"]


def test_tiny_eval_ignores_prompt_boxed():
    pytest.importorskip("torch")
    from grace_gc.data.math_data import MathRecord
    from grace_gc.data.reward import rule_reward
    from grace_gc.evaluation.generate import generate_answers_tiny

    items = generate_answers_tiny([MathRecord("p", "answer is \\boxed{2}", "2")], n=1, max_new=0, seed=0)
    assert items[0].answers[0] == ""
    assert rule_reward(items[0].answers[0], "2") == 0.0


def test_resume_rejects_method_mismatch(tmp_path: Path):
    pytest.importorskip("torch")
    import torch

    from grace_gc.trainer.cpu_tiny import TinyLoRAActor
    from grace_gc.trainer.loop import build_run_config, run_training
    from grace_gc.trainer.state_io import restore_train_state

    data = tmp_path / "m.jsonl"
    data.write_text(json.dumps({"problem_id": "a", "prompt": "1+1", "answer": "2", "split": "train"}) + "\n", encoding="utf-8")
    cfg = build_run_config(
        [],
        {
            "seed": 1,
            "backend": "cpu_tiny",
            "data_path": str(data),
            "n_prompts": 1,
            "n_start": 2,
            "decision_tokens": 2,
            "max_new_tokens": 3,
            "num_steps": 1,
            "predictor": {"k": 2, "warmup_steps": 0, "audit_s": 1.0, "reservoir_size": 8, "epochs": 1},
        },
    )
    first = run_training(cfg, tmp_path / "run")
    actor = TinyLoRAActor()
    opt = torch.optim.SGD(actor.trainable_params(), lr=0.01)
    with pytest.raises(ValueError, match="checkpoint method"):
        restore_train_state(first["summary"]["checkpoint"], actor, opt, 8, 2, "full_pg")


def test_restore_rejects_basis_and_reward_risk_gaps(tmp_path: Path):
    pytest.importorskip("torch")
    import torch

    from grace_gc.trainer.checkpoint import load_checkpoint, save_checkpoint
    from grace_gc.trainer.cpu_tiny import TinyLoRAActor
    from grace_gc.trainer.loop import build_run_config, run_training
    from grace_gc.trainer.state_io import restore_train_state

    data = tmp_path / "m.jsonl"
    data.write_text(json.dumps({"problem_id": "a", "prompt": "1+1", "answer": "2", "split": "train"}) + "\n", encoding="utf-8")
    cfg = build_run_config(
        [],
        {
            "method": "reward_cv",
            "seed": 1,
            "backend": "cpu_tiny",
            "data_path": str(data),
            "n_prompts": 1,
            "n_start": 2,
            "decision_tokens": 2,
            "max_new_tokens": 3,
            "num_steps": 1,
            "predictor": {"k": 2, "warmup_steps": 0, "audit_s": 1.0, "reservoir_size": 8, "epochs": 1},
        },
    )
    first = run_training(cfg, tmp_path / "run")
    payload = load_checkpoint(first["summary"]["checkpoint"])
    broken_u = dict(payload)
    broken_u["basis"] = dict(payload["basis"])
    broken_u["basis"]["u"] = np.zeros((3, 2))
    save_checkpoint(tmp_path / "bad-u.npz", broken_u)
    actor = TinyLoRAActor()
    opt = torch.optim.SGD(actor.trainable_params(), lr=0.01)
    with pytest.raises(ValueError, match="basis U"):
        restore_train_state(tmp_path / "bad-u.npz", actor, opt, 8, 2, "reward_cv")
    broken_r = dict(payload)
    broken_r["predictor"] = {k: v for k, v in payload["predictor"].items() if k != "reward_risk"}
    save_checkpoint(tmp_path / "bad-r.npz", broken_r)
    actor = TinyLoRAActor()
    with pytest.raises(ValueError, match="reward_risk"):
        restore_train_state(tmp_path / "bad-r.npz", actor, opt, 8, 2, "reward_cv")
    broken_names = dict(payload)
    names = list(payload.get("layout_names") or [])
    if names:
        broken_names["layout_names"] = names[::-1]
        save_checkpoint(tmp_path / "bad-names.npz", broken_names)
        actor = TinyLoRAActor()
        opt = torch.optim.SGD(actor.trainable_params(), lr=0.01)
        with pytest.raises(ValueError, match="layout names"):
            restore_train_state(tmp_path / "bad-names.npz", actor, opt, 8, 2, "reward_cv")
    broken_s = dict(payload)
    broken_s["predictor"] = {k: v for k, v in payload["predictor"].items() if k != "success"}
    save_checkpoint(tmp_path / "bad-s.npz", broken_s)
    actor = TinyLoRAActor()
    with pytest.raises(ValueError, match="success"):
        restore_train_state(tmp_path / "bad-s.npz", actor, opt, 8, 2, "reward_cv")
    broken_p = dict(payload)
    broken_p["predictor"] = None
    save_checkpoint(tmp_path / "bad-pred.npz", broken_p)
    actor = TinyLoRAActor()
    with pytest.raises(ValueError, match="missing predictor"):
        restore_train_state(tmp_path / "bad-pred.npz", actor, opt, 8, 2, "reward_cv")


def test_grpo_short_uses_16_starts_per_prompt():
    from grace_gc.trainer.loop import resolve_start_counts
    from grace_gc.trainer.methods import apply_method_defaults, method_spec

    spec = method_spec("grpo_short")
    raw = resolve_start_counts({"n_prompts": 4, "n_start": 8}, method_spec("grpo"))
    assert raw[2] == 8
    assert raw[1] == 2
    cfg = apply_method_defaults({"n_prompts": 4, "n_start": 8, "method": "grpo_short"})
    n_prompts, starts_per, n = resolve_start_counts(cfg, spec)
    assert starts_per == 16
    assert n_prompts == 4
    assert n == 64
    with pytest.raises(ValueError, match="multiple"):
        resolve_start_counts(cfg, spec, n_start=20)
    n_g, starts_g, grown = resolve_start_counts(cfg, spec, n_start=32)
    assert starts_g == 16
    assert n_g == 2
    assert grown == 32
    pilot = apply_method_defaults({"n_prompts": 64, "n_start": 512, "method": "grpo_short"})
    n_prompts64, starts64, n64 = resolve_start_counts(pilot, spec)
    assert starts64 == 16
    assert n_prompts64 == 64
    assert n64 == 1024


def test_eval_does_not_inherit_grpo_short_train_budget():
    from grace_gc.evaluation.generate import _eval_hparams
    from grace_gc.trainer.methods import apply_method_defaults

    cfg = apply_method_defaults({"method": "grpo_short", "n_prompts": 4, "max_new_tokens": 2048})
    assert cfg["max_new_tokens"] == 1024
    assert _eval_hparams(cfg)[2] == 4096
    assert _eval_hparams({"method": "grace", "max_new_tokens": 32})[2] == 32
    assert _eval_hparams({"method": "grace", "backend": "gpu_verl", "max_new_tokens": 2048})[2] == 4096
    assert _eval_hparams({"method": "grpo_short", "max_new_tokens": 1024, "eval": {"max_new_tokens": 8}})[2] == 8
    k, n, max_new, temperature, top_p = _eval_hparams(
        {
            "eval_temperature": 0.6,
            "eval_top_p": 0.95,
            "eval": {"k": 4, "n": 4, "temperature": 0.2, "top_p": 0.7, "max_new_tokens": 4096},
        }
    )
    assert (k, n, max_new, temperature, top_p) == (4, 4, 4096, 0.2, 0.7)


def test_smoke_yaml_is_not_token_stub():
    from grace_gc.trainer.loop import build_run_config

    cfg = build_run_config(["configs/experiments/smoke.yaml"], {})
    assert cfg["max_new_tokens"] == 1024
    assert cfg["eval"]["max_new_tokens"] == 2048
    assert cfg["audit"]["max_new_tokens"] == 2048
    assert cfg["format_warmup"]["steps"] == 32
    assert cfg["predictor"]["warmup_steps"] == 0


def test_build_run_config_eval_matches_paper_avg4():
    from grace_gc.trainer.loop import build_run_config

    cfg = build_run_config(["configs/default.yaml", "configs/experiments/pilot.yaml"], {})
    assert cfg["eval"]["k"] == 4
    assert cfg["eval"]["n"] == 4
    assert cfg["eval"]["max_new_tokens"] == 4096


def test_build_run_config_writes_grpo_short_budget():
    from grace_gc.trainer.loop import build_run_config

    cfg = build_run_config(["configs/experiments/pilot.yaml"], {"method": "grpo_short"})
    assert cfg["max_new_tokens"] == 1024
    assert cfg["n_start"] == 64 * 16
    reward = build_run_config(["configs/experiments/pilot.yaml"], {"method": "reward_cv"})
    assert reward["method"] == "reward_cv"
    assert "method_note" not in reward


def test_evaluate_cli_keeps_yaml_gpu_backend(tmp_path: Path):
    data = tmp_path / "d.jsonl"
    data.write_text(
        json.dumps({"problem_id": "a", "prompt": "q", "answer": "1", "split": "eval"}) + "\n",
        encoding="utf-8",
    )
    yml = tmp_path / "g.yaml"
    yml.write_text("backend: gpu_verl\nseed: 1\neval:\n  k: 1\n  n: 1\n", encoding="utf-8")
    from scripts.evaluate import main

    with pytest.raises(ValueError, match="model_path"):
        main(["--generate", "--data-path", str(data), "--config", str(yml), "--run-dir", str(tmp_path / "e")])


def test_audit_rejects_decision_past_max_new(tmp_path: Path):
    from grace_gc.audit.run import run_audit

    with pytest.raises(ValueError, match="max_new"):
        run_audit([], {"decision_tokens": 8, "max_new_tokens": 4, "backend": "cpu_tiny"}, tmp_path / "bad")


def test_audit_does_not_inherit_grpo_short_train_budget(tmp_path: Path):
    pytest.importorskip("torch")
    from grace_gc.audit.run import _audit_max_new, run_audit
    from grace_gc.data.math_data import MathRecord

    assert (
        _audit_max_new(
            {
                "backend": "cpu_tiny",
                "method": "grpo_short",
                "max_new_tokens": 2048,
                "eval": {"max_new_tokens": 4096},
            }
        )
        == 4096
    )
    assert _audit_max_new({"backend": "gpu_verl", "method": "grpo_short", "max_new_tokens": 1024}) == 4096
    recs = [MathRecord("1", "add two", "1")]
    out = run_audit(
        recs,
        {
            "backend": "cpu_tiny",
            "method": "grpo_short",
            "max_new_tokens": 2048,
            "decision_tokens": 8,
            "eval": {"max_new_tokens": 32},
            "audit": {"n_prefixes": 1, "n_continuations": 1, "decision_grid": [8, 16]},
            "seed": 0,
            "prompt_max_tokens": 8,
        },
        tmp_path / "gs-audit",
    )
    assert 16 in out["times"]
    assert 8 in out["times"]
    assert out["max_new_tokens"] == 32
    assert "decision_grid_dropped" not in out


def test_audit_prefers_audit_max_new(tmp_path: Path):
    pytest.importorskip("torch")
    from grace_gc.audit.run import _audit_max_new, run_audit
    from grace_gc.data.math_data import MathRecord

    assert (
        _audit_max_new(
            {
                "max_new_tokens": 2048,
                "eval": {"max_new_tokens": 4096},
                "audit": {"max_new_tokens": 16},
            }
        )
        == 16
    )
    recs = [MathRecord("1", "add two", "1")]
    out = run_audit(
        recs,
        {
            "backend": "cpu_tiny",
            "method": "grpo_short",
            "max_new_tokens": 2048,
            "decision_tokens": 8,
            "eval": {"max_new_tokens": 4096},
            "audit": {
                "n_prefixes": 1,
                "n_continuations": 1,
                "decision_grid": [8, 64],
                "max_new_tokens": 16,
            },
            "seed": 0,
            "prompt_max_tokens": 8,
        },
        tmp_path / "gs-audit-cap",
    )
    assert 64 not in out["times"]
    assert 8 in out["times"]
    assert out["max_new_tokens"] == 16
    assert out["decision_grid_dropped"] == [64]


def test_tiny_eval_rejects_lora_only_checkpoint(tmp_path: Path):
    pytest.importorskip("torch")
    from grace_gc.evaluation.generate import _load_tiny_actor
    from grace_gc.trainer.checkpoint import save_checkpoint

    ckpt = tmp_path / "lora.npz"
    save_checkpoint(ckpt, {"actor": {"x": np.zeros(1)}, "actor_full": None})
    with pytest.raises(ValueError, match="actor_full"):
        _load_tiny_actor({"checkpoint": str(ckpt)})


def test_frozen_predictor_changes_actual_variance_cost():
    from grace_gc.audit.prefix_audit import PrefixBundle, audit_bundles

    g = np.array([[1.0, 0.0], [-1.0, 0.0]])
    kwargs = dict(rewards=np.array([1.0, 0.0]), grads=g, suffix_cost=np.ones(2), m_pred=np.array([10.0, 0.0]))
    high = PrefixBundle("p", 4, r_hat=8.0, c_hat=1.0, **kwargs)
    low = PrefixBundle("p", 4, r_hat=0.05, c_hat=1.0, **kwargs)
    out_high = audit_bundles([high], np.eye(2, 1), {"beta": 0.5, "p_min": 0.2}, np.random.default_rng(0))
    out_low = audit_bundles([low], np.eye(2, 1), {"beta": 0.5, "p_min": 0.2}, np.random.default_rng(0))
    assert "frozen_predictor" in str(out_high["variance_cost_note"])
    assert out_high["variance_cost"]["actual_p_var"] != out_high["variance_cost_oracle"]["actual_p_var"]
    from grace_gc.core.allocation import allocate_continuation

    # One prefix can only hit β. Different r_hat change p only in a mixed set.
    mixed = allocate_continuation(np.array([8.0, 0.05]), np.ones(2), beta=0.5, p_min=0.2)
    assert mixed.p[0] > mixed.p[1]
    _ = out_low


def test_resume_start_counts_uses_scheduled_n():
    from types import SimpleNamespace

    from grace_gc.trainer.loop import resume_start_counts
    from grace_gc.trainer.methods import method_spec

    spec = method_spec("grace")
    state = SimpleNamespace(history_costs=[0.5, 0.5], n_ref=8)
    _p, starts_per, n = resume_start_counts({"n_prompts": 4, "n_start": 8}, spec, state)
    assert n == 16
    assert starts_per == 4


def test_finished_prefix_keeps_p_one_in_variance_cost():
    from grace_gc.audit.prefix_audit import PrefixBundle, audit_bundles

    g = np.array([[1.0, 0.0], [-1.0, 0.0]])
    shared = dict(
        rewards=np.array([1.0, 0.0]),
        grads=g,
        suffix_cost=np.ones(2),
        r_hat=0.01,
        c_hat=1.0,
        m_pred=np.zeros(2),
    )
    stopped = audit_bundles(
        [PrefixBundle("p", 4, finished=True, **shared)],
        np.eye(2, 1),
        {"beta": 0.5, "p_min": 0.2},
        np.random.default_rng(0),
    )
    opened = audit_bundles(
        [PrefixBundle("p", 4, finished=False, **shared)],
        np.eye(2, 1),
        {"beta": 0.5, "p_min": 0.2},
        np.random.default_rng(0),
    )
    assert stopped["variance_cost"]["extra_var"] == pytest.approx(0.0)
    assert opened["variance_cost"]["extra_var"] > 0.0


def test_audit_residual_mean_mixed_prefix_lengths():
    from grace_gc.audit.prefix_audit import PrefixBundle, audit_bundles

    one = PrefixBundle("p", 2, np.ones(1), np.array([[1.0, 0.0]]), coords=np.zeros(1), path_id="p:a", finished=True)
    many = PrefixBundle(
        "p",
        4,
        np.ones(3),
        np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]),
        coords=np.zeros(1),
        path_id="p:b",
    )
    out = audit_bundles([one, many], np.eye(2, 1), {}, np.random.default_rng(0))
    assert out["residual_mean"] == pytest.approx(1.0)


def test_audit_residual_uses_frozen_coords():
    from grace_gc.audit.prefix_audit import PrefixBundle, audit_bundles

    g = np.array([[1.0, 0.0], [1.0, 0.0]])
    out = audit_bundles(
        [PrefixBundle("p", 4, np.ones(2), g, coords=np.zeros(1), m_pred=np.zeros(2))],
        np.eye(2, 1),
        {},
        np.random.default_rng(0),
    )
    assert out["residual_mean"] == pytest.approx(1.0)


def test_audit_features_follow_checkpoint_method():
    from grace_gc.audit.run import _predictor_audit_features
    from grace_gc.trainer.methods import method_spec

    bundle = {"features": np.array([[1.0]]), "prompt_features": np.array([[2.0]])}
    np.testing.assert_allclose(_predictor_audit_features(bundle, method_spec("grace")), [[1.0]])
    np.testing.assert_allclose(_predictor_audit_features(bundle, method_spec("prompt_cv")), [[2.0]])
    np.testing.assert_allclose(_predictor_audit_features(bundle, method_spec("reward_cv")), [[1.0]])


def test_audit_prefix_at_t_is_nested():
    from grace_gc.audit.run import _keep_prefix_at_t, _prefix_at_t

    long = [1, 2, 10, 11, 12]
    short, fin_s = _prefix_at_t(long, 2, 2, False, 99)
    longer, fin_l = _prefix_at_t(long, 2, 3, False, 99)
    assert short == [1, 2, 10, 11]
    assert longer[: len(short)] == short
    assert not fin_s and not fin_l
    done, fin = _prefix_at_t([1, 2, 10, 3], 2, 4, True, 3)
    assert done == [1, 2, 10, 3]
    assert fin
    early = [1, 2, 10, 11, 99]
    pref2, fin2 = _prefix_at_t(early, 2, 2, True, 99)
    pref8, fin8 = _prefix_at_t(early, 2, 8, True, 99)
    pref3, fin3 = _prefix_at_t(early, 2, 3, True, 99)
    assert _keep_prefix_at_t(pref2, 2, 2, fin2)
    assert not _keep_prefix_at_t(pref8, 2, 8, fin8)
    assert fin3 and _keep_prefix_at_t(pref3, 2, 3, fin3)
    truncated = [1, 2, 10, 11]
    pref_short, fin_short = _prefix_at_t(truncated, 2, 8, False, 99)
    assert not fin_short
    assert _keep_prefix_at_t(pref_short, 2, 8, fin_short)


def test_variance_cost_uses_actual_prefix_tokens():
    from grace_gc.audit.prefix_audit import PrefixBundle, audit_bundles

    g = np.array([[1.0, 0.0], [-1.0, 0.0]])
    billed = PrefixBundle(
        "p",
        512,
        np.array([1.0, 0.0]),
        g,
        suffix_cost=np.ones(2),
        r_hat=1.0,
        c_hat=1.0,
        m_pred=np.zeros(2),
        prefix_tokens=8,
        finished=True,
    )
    out = audit_bundles([billed], np.eye(2, 1), {"beta": 0.5, "p_min": 0.2}, np.random.default_rng(0))
    assert out["variance_cost"]["full_cost"] == pytest.approx(9.0)
    assert out["variance_cost"]["actual_p_cost"] == pytest.approx(9.0)


def test_fit_eval_split_keeps_path_together():
    from grace_gc.audit.variance_cost import fit_eval_split

    keys = ["a:0", "a:0", "b:1", "b:1"]
    for seed in range(16):
        fit, ev = fit_eval_split(4, np.random.default_rng(seed), keys=keys)
        fit_keys = {keys[i] for i in fit}
        ev_keys = {keys[i] for i in ev}
        assert not (fit_keys & ev_keys)
        assert fit_keys | ev_keys == {"a:0", "b:1"}


def test_audit_oracle_is_same_prefix_not_problem_mean():
    from grace_gc.audit.prefix_audit import PrefixBundle, audit_bundles

    g_fit = np.array([[4.0, 0.0], [4.0, 0.0]])
    g_eval = np.array([[0.0, 0.0], [0.0, 0.0]])
    fit_b = PrefixBundle("p", 8, np.ones(2), g_fit, path_id="p:fit", suffix_cost=np.ones(2))
    eval_b = PrefixBundle("p", 8, np.zeros(2), g_eval, path_id="p:eval", suffix_cost=np.ones(2))
    leak = PrefixBundle("p", 2, np.ones(2), np.array([[99.0, 0.0], [99.0, 0.0]]), path_id="p:fit", suffix_cost=np.ones(2))
    out = audit_bundles([fit_b, leak, eval_b], np.eye(2, 1), {"beta": 0.5, "p_min": 0.2}, np.random.default_rng(0))
    fit_idx = set(out["fit_index"])
    ev_idx = set(out["eval_index"])
    assert not (fit_idx & ev_idx)
    assert {0, 1}.issubset(fit_idx) or {0, 1}.issubset(ev_idx)
    assert "same-prefix oracle" in str(out["variance_cost_note"])
    assert "token_proxy_cost" in str(out["variance_cost_note"])


def test_checkpoint_shape_is_not_broadcast():
    import torch

    from grace_gc.trainer.state_io import load_numpy_module_state

    param = torch.zeros(2, 3)
    with pytest.raises(ValueError, match="shape"):
        load_numpy_module_state([("w", param)], {"w": np.zeros((1, 3))})


def test_snapshot_identity_checks_lora_scale():
    from grace_gc.trainer.state_io import check_snapshot_identity

    check_snapshot_identity({"model_path": "m", "lora": {"alpha": 32.0}}, {"model_path": "m", "lora": {"alpha": 32.0}})
    with pytest.raises(ValueError, match="alpha"):
        check_snapshot_identity({"lora": {"alpha": 16.0}}, {"lora": {"alpha": 32.0}})


def test_coords_null_is_not_an_array():
    raw = {"coords": None}
    coords = None if raw.get("coords") is None else np.asarray(raw["coords"], dtype=np.float64)
    assert coords is None


def test_gpu_backend_ledger_counts_a_card():
    src = Path("grace_gc/trainer/loop.py").read_text(encoding="utf-8")
    assert "n_gpu = max(n_gpu, 1)" in src
    assert 'include = ["grace_gc*", "scripts*"]' in Path("pyproject.toml").read_text(encoding="utf-8")
