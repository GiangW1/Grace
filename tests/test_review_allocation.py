"""Explicit allocation/CV controls: fixed budget, final-p HT, and p=1."""

from itertools import product

import numpy as np
import pytest

from grace_gc.core.allocation import allocate_continuation
from grace_gc.trainer.grace_step import (
    StartRecord, assemble_ghat, control_variate_coordinates, decide_continuation, grace_batch_update,
)
from grace_gc.trainer.methods import method_spec


@pytest.mark.parametrize("shrink", [0., .3, 1.])
def test_uniform_shrink_preserves_achieved_cost_and_order(shrink):
    risk = np.array([999., .05, 2., 16., .6])
    cost = np.array([1000., 1., 4., 2., 3.])
    finished = np.array([True, False, False, False, False])
    original = allocate_continuation(risk, cost, .5, .2, finished=finished, iters=60)
    result = allocate_continuation(risk, cost, .5, .2, finished=finished, iters=60, uniform_shrink=shrink)
    assert result.p[0] == 1.
    assert np.all((result.p >= .2) & (result.p <= 1.))
    assert result.budget_expected == pytest.approx(original.budget_expected, abs=1e-12)
    assert result.budget_deviation == pytest.approx(original.budget_deviation, abs=1e-12)
    uniform = original.budget_expected / cost[~finished].sum()
    np.testing.assert_allclose(result.p[~finished], (1-shrink)*original.p[~finished] + shrink*uniform)
    order = np.argsort(original.p[~finished])
    assert np.all(np.diff(result.p[~finished][order]) >= 0.)
    if shrink == 0:
        np.testing.assert_array_equal(result.p, original.p)


def test_shrink_keeps_zero_risk_budget_deviation_and_free_finishes_defined():
    for beta in (.1, .5):
        original = allocate_continuation(np.zeros(3), np.array([1., 2., 3.]), beta, .2)
        shrunk = allocate_continuation(np.zeros(3), np.array([1., 2., 3.]), beta, .2, uniform_shrink=1.)
        np.testing.assert_allclose(shrunk.p, original.p)
        assert shrunk.budget_expected == pytest.approx(original.budget_expected)
    free = allocate_continuation(np.array([0., 2.]), np.zeros(2), .5, .2, uniform_shrink=1.)
    np.testing.assert_array_equal(free.p, np.ones(2))
    assert free.budget_expected == 0.
    finished = allocate_continuation(np.ones(2), np.ones(2), .5, .2,
                                    finished=np.ones(2, dtype=bool), uniform_shrink=.7)
    np.testing.assert_array_equal(finished.p, np.ones(2))
    assert finished.n_eligible == 0


@pytest.mark.parametrize("value", [-.1, 1.1, float("nan"), float("inf")])
def test_invalid_uniform_shrink_is_rejected(value):
    with pytest.raises(ValueError, match="uniform_shrink"):
        allocate_continuation(np.ones(2), np.ones(2), .5, .2, uniform_shrink=value)


@pytest.mark.parametrize("method,warmup,ready,beta", [
    ("grace", True, True, .5), ("grace", False, False, .5),
    ("grace", False, True, 1.), ("full_pg", False, True, .5),
    ("grpo", False, True, .5), ("grpo_short", False, True, .5),
])
def test_full_completion_paths_include_zero_risk(method, warmup, ready, beta):
    p, deviation = decide_continuation(method_spec(method), np.array([0., 1.]), np.array([1., 3.]),
                                     np.array([False, True]), beta, .2, warmup,
                                     basis_ready=ready, uniform_shrink=.5)
    np.testing.assert_array_equal(p, np.ones(2))
    assert deviation == 0.


@pytest.mark.parametrize("enabled", [False, True])
def test_final_mixed_p_preserves_full_space_ht_expectation_and_null_stoppers(enabled):
    spec = method_spec("grace")
    risk, cost = np.array([.1, 2., 10.]), np.array([1., 3., 2.])
    p, _ = decide_continuation(spec, risk, cost, np.zeros(3, dtype=bool), .6, .2, False,
                               iters=60, uniform_shrink=.4)
    u = np.eye(4, 2)
    f = np.array([[3., -1.], [.4, .2], [-2., .7]])
    g = np.array([[1., 2., 3., 4.], [-2., 1., 0., .5], [.1, .2, .3, .4]])
    expected = np.zeros(4)
    for z_tuple in product((0., 1.), repeat=3):
        z = np.asarray(z_tuple)
        probability = np.prod(np.where(z == 1., p, 1-p))
        records = [StartRecord(str(i), False, p[i], z[i], f[i].copy(), risk[i], cost[i],
                               1. if z[i] else None, .7 if z[i] else None,
                               g[i] if z[i] else None, bool(z[i]), baseline_b=.3) for i in range(3)]
        result = grace_batch_update(spec, records, u, n=3, control_variate_enabled=enabled)
        expected += probability * result["ghat"]
        assert all(rec.reward is None and rec.advantage is None for rec in records if not rec.z)
        assert all(rec.advantage == rec.reward-rec.baseline_b for rec in records if rec.z)
        if not enabled:
            np.testing.assert_array_equal(result["prediction_grad"], np.zeros(4))
            np.testing.assert_allclose(result["ghat"], np.sum(z[:, None] * g / p[:, None], axis=0) / 3)
    np.testing.assert_allclose(expected, g.mean(axis=0), atol=1e-12)
    np.testing.assert_array_equal(control_variate_coordinates(f), f)
    np.testing.assert_array_equal(control_variate_coordinates(f, False), np.zeros_like(f))


@pytest.mark.parametrize("enabled", [False, True])
def test_p_one_matches_full_pg_regardless_of_control_variate(enabled):
    u, g = np.eye(3, 1), np.array([[1., 2., 3.], [4., 0., -1.]])
    records = [StartRecord(str(i), bool(i), 1., 1., np.array([99.]), 0., 1.,
                           1., .5, g[i], False, baseline_b=.5) for i in range(2)]
    np.testing.assert_allclose(assemble_ghat(method_spec("grace"), records, u, 2,
                                           control_variate_enabled=enabled), g.mean(axis=0))
