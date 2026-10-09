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
