import numpy as np
import pytest

from grace_gc.core.allocation import allocate_continuation


def test_pmin_and_budget_reported():
    risk = np.array([4.0, 1.0, 0.0])
    cost = np.array([1.0, 1.0, 1.0])
    out = allocate_continuation(risk, cost, beta=0.5, p_min=0.2)
    assert out.p.min() >= 0.2 - 1e-12
    assert out.p.max() <= 1.0 + 1e-12
    assert out.n_eligible == 3
    assert np.isfinite(out.budget_deviation)


def test_natural_finish_forced_to_one():
    risk = np.array([9.0, 9.0])
    cost = np.array([1.0, 1.0])
    out = allocate_continuation(risk, cost, beta=0.3, p_min=0.2, finished=np.array([True, False]))
    assert out.p[0] == 1.0
    assert 0.2 <= out.p[1] <= 1.0
    assert out.n_eligible == 1


def test_empty_eligible():
    out = allocate_continuation(np.array([1.0]), np.array([1.0]), beta=0.5, p_min=0.2, finished=np.array([True]))
    assert out.n_eligible == 0
    assert out.p[0] == 1.0


def test_zero_cost_and_zero_risk():
    out = allocate_continuation(np.array([2.0, 0.0]), np.array([0.0, 1.0]), beta=0.5, p_min=0.2)
    assert out.p[0] == 1.0
    assert out.p[1] >= 0.2


def test_illegal_pmin():
    with pytest.raises(ValueError):
        allocate_continuation(np.array([1.0]), np.array([1.0]), beta=0.5, p_min=0.0)
