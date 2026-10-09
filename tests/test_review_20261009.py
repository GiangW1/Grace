"""Regressions for the 2026-10-09 review fixes; synthetic CPU data only."""

import numpy as np
import pytest

from grace_gc.data.reward import rule_reward
from grace_gc.trainer.baseline import HistoricalBaseline
from grace_gc.trainer.cost_control import CostControl
from grace_gc.trainer.grace_step import batch_token_costs, next_start_count


def test_truncated_base_answer_scores_zero():
    assert rule_reward("work\nAnswer: 7", "7", truncated=True) == 0.0
    assert rule_reward("work\nAnswer: 7", "7", truncated=False) == 1.0


def test_ht_baseline_stays_in_reward_range():
    baseline = HistoricalBaseline(alpha=.7)
    baseline.values["p"] = .5
    baseline.update_ht_mean("p", [1., None, None, None], [.2, .2, .2, .2])
    assert 0. <= baseline.get("p") <= 1.


def test_stopped_start_is_charged_realized_suffix_not_full_budget():
    prompt_lens = [2, 2]
    prefixes = [[1, 2, 10, 11], [1, 2, 10, 11]]
    # Start 0 continued 2 tokens of a 2046-token budget; start 1 stopped.
    full_ids = [[1, 2, 10, 11, 12, 13], None]
    used, full = batch_token_costs(prompt_lens, prefixes, np.array([False, False]),
                                   np.array([1., 0.]), 2048, full_ids=full_ids)
    assert used == pytest.approx((4 + 2) / 2)
    assert full == pytest.approx((4 + 4) / 2)
    # Budget accounting would have reported (2+2044+2)/(2*2046) ~ 0.5 and doubled N.
    assert next_start_count([used / full], 8, 1.0) == 11


def test_full_pg_realized_costs_keep_reference_n():
    prefixes = [[1, 2, 10], [1, 2, 10]]
    full_ids = [[1, 2, 10, 11], [1, 2, 10, 11, 12, 13]]
    used, full = batch_token_costs([2, 2], prefixes, np.array([False, False]),
                                   np.array([1., 1.]), 64, full_ids=full_ids)
    assert used == full
    assert next_start_count([used / full], 8, 1.0) == 8


def test_self_anchored_wall_target_cannot_spiral_below_reference_n():
    controller = CostControl({"cost_control": {"enabled": True, "ema_alpha": 1.}})
    # Latency-bound generation: wall time barely depends on N.
    assert controller.observe(16, 34., 1, group=4, reference_n=16) == 16
    assert controller.observe(16, 47., 2, group=4, reference_n=16) == 16
    assert controller.observe(16, 50., 3, group=4, reference_n=16) == 16


def test_explicit_shared_target_may_still_reduce_n():
    controller = CostControl({"cost_control": {"enabled": True, "ema_alpha": 1.,
                                               "target_step_seconds": 20.}})
    assert controller.observe(16, 40., 1, group=4, reference_n=16) == 8


def test_answer_line_must_start_a_line_and_phrase_must_end_the_text():
    from grace_gc.data.reward import answer_already_emitted, extract_answer, first_parseable_index

    assert extract_answer("the answer is not 3, check.\nSo 5.\nAnswer: 5") == "5"
    assert extract_answer("the answer is 7?\nthen more work") is None
    assert extract_answer("**Answer:** 12") == "12"
    assert extract_answer("Thus the final answer is 8.") == "8"
    assert not answer_already_emitted("the answer is not obvious\nwe continue")
    pieces = ["work ", "\n", "Answer", ": ", "5"]
    assert first_parseable_index(pieces) == 4
    assert first_parseable_index(["no", " answer"]) is None


def test_informative_groups_keep_mixed_reward_groups_within_budget():
    from grace_gc.trainer.advantages import informative_groups

    rewards = [0., 0., 1., 0., 1., 1., 0., 1.]
    pids = ["a", "a", "b", "b", "c", "c", "d", "d"]
    keep, report = informative_groups(rewards, pids, max_starts=2)
    assert keep.tolist() == [False, False, True, True, False, False, False, False]
    assert report == {"n_groups": 4, "n_informative": 2, "n_kept": 1, "n_kept_starts": 2, "max_starts": 2}


def test_dynamic_sampling_draws_more_prompts_only_for_grpo():
    from grace_gc.trainer.methods import dynamic_sampling_prompts, method_spec

    cfg = {"dynamic_sampling": {"enabled": True, "oversample": 2.0}}
    assert dynamic_sampling_prompts(cfg, method_spec("grpo"), 16) == 32
    assert dynamic_sampling_prompts(cfg, method_spec("full_pg"), 16) == 16
    assert dynamic_sampling_prompts({}, method_spec("grpo"), 16) == 16


def test_grpo_dynamic_sampling_runs_with_fixed_denominator(tmp_path):
    pytest.importorskip("torch")
    import json

    from grace_gc.config import default_config, merge_configs
    from grace_gc.logging_util.ledger import ComputeLedger
    from grace_gc.logging_util.run_dir import RunDirectory
    from grace_gc.trainer.loop import run_tiny_training

    cfg = merge_configs(default_config(), {
        "method": "grpo", "num_steps": 2, "n_start": 4, "n_prompts": 2, "decision_tokens": 3,
        "max_new_tokens": 6, "cost_control": {"mode": "fixed"},
        "dynamic_sampling": {"enabled": True, "oversample": 2.0},
    })
    summary = run_tiny_training(cfg, RunDirectory(tmp_path), ComputeLedger(0, "cpu"))
    rows = [json.loads(line) for line in (tmp_path / "steps.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(row["loss_denominator"] == 4 for row in rows)
    # Two prompts x oversample 2 = 4 draws; a tiny pool can repeat a problem, which
    # GRPO groups together, so there are at most 4 groups.
    assert all(row["n"] == 8 and row["dynamic_sampling"]["n_groups"] <= 4
               and row["dynamic_sampling"]["n_kept_starts"] <= 4 for row in rows)
    assert summary["clip_trigger_rate_this_session"] is not None


def test_leave_one_out_baseline_is_unbiased_under_ht_stopping():
    import itertools

    from grace_gc.trainer.advantages import advantages_for_method
    from grace_gc.trainer.baseline import HistoricalBaseline

    # Two starts of one problem, each kept with p and rewarded with q. The HT
    # mean of Z_i (R_i - b_i)/p over all outcomes must equal q - E[b] for a b
    # independent of start i; here b_i is the other observed reward or 0.5.
    q, p = .3, .6
    baseline = HistoricalBaseline(mode="loo")
    expected = 0.
    for r0, r1, z0, z1 in itertools.product((0, 1), (0, 1), (0, 1), (0, 1)):
        prob = (q if r0 else 1 - q) * (q if r1 else 1 - q) * (p if z0 else 1 - p) * (p if z1 else 1 - p)
        rewards = [float(r0) if z0 else None, float(r1) if z1 else None]
        adv = advantages_for_method("raw_pg", rewards, ["a", "a"], baseline)
        expected += prob * z0 * adv[0] / p
    other_b = p * q + (1 - p) * .5
    assert expected == pytest.approx(q - other_b)
    assert not baseline.uses_prescan


def test_full_pg_with_loo_baseline_skips_prescan(tmp_path):
    pytest.importorskip("torch")
    import json

    from grace_gc.config import default_config, merge_configs
    from grace_gc.logging_util.ledger import ComputeLedger
    from grace_gc.logging_util.run_dir import RunDirectory
    from grace_gc.trainer.loop import run_tiny_training

    cfg = merge_configs(default_config(), {
        "method": "full_pg", "num_steps": 1, "n_start": 4, "n_prompts": 2, "decision_tokens": 3,
        "max_new_tokens": 6, "baseline": {"mode": "loo", "prescan": 4}, "cost_control": {"mode": "fixed"},
    })
    run_tiny_training(cfg, RunDirectory(tmp_path), ComputeLedger(0, "cpu"))
    rows = [json.loads(line) for line in (tmp_path / "steps.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[0]["n_prescan"] == 0


def test_problem_centered_basis_keeps_singleton_problem_labels():
    from grace_gc.predictor.basis import refresh_basis

    rng = np.random.default_rng(0)
    grads = rng.normal(size=(6, 20))
    basis = refresh_basis(grads, k=3, basis_id=1, problem_ids=["a", "b", "c", "d", "e", "f"], center="problem")
    assert basis is not None and basis.rank == 3
