import numpy as np
import pytest

from grace_gc.core.rng import IsolatedRNG


def test_streams_are_isolated():
    a = IsolatedRNG.create(7)
    b = IsolatedRNG.create(7)
    a.random("selection", 8)
    tok_a = a.random("token", 8)
    tok_b = b.random("token", 8)
    assert np.allclose(tok_a, tok_b)
    sel_b = b.random("selection", 8)
    assert not np.allclose(tok_a, sel_b)


def test_bernoulli_rejects_out_of_range_p():
    rng = IsolatedRNG.create(1)
    with pytest.raises(ValueError):
        rng.bernoulli("selection", np.array([-0.1, 0.5]))
    np.testing.assert_array_equal(rng.bernoulli("audit", np.zeros(3)), np.zeros(3))


def test_state_roundtrip():
    rng = IsolatedRNG.create(3)
    rng.bernoulli("audit", np.array([0.2, 0.8]))
    state = rng.state_dict()
    other = IsolatedRNG.create(99)
    other.load_state_dict(state)
    x = rng.random("continuation", 5)
    y = other.random("continuation", 5)
    assert np.allclose(x, y)


def test_restore_rejects_missing_stream():
    rng = IsolatedRNG.create(1)
    state = rng.state_dict()
    del state["bit_generators"]["eval"]
    other = IsolatedRNG.create(0)
    with pytest.raises(ValueError, match="missing streams"):
        other.load_state_dict(state)
