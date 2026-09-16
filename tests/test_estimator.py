import itertools

import numpy as np
import pytest

from grace_gc.core.estimator import (
    batch_ht_mean,
    dual_stream_mean,
    ht_estimate,
    ht_variance_trace,
    optimizer_grad_from_ghat,
    prediction_correction,
)


def test_u1_exact_mean_over_z():
    g = np.array([[1.0, -2.0, 0.5]])
    m = np.array([[0.2, 0.1, -0.3]])
    p = np.array([0.4])
    mean = p[0] * ht_estimate(g, m, p, np.array([1.0])) + (1 - p[0]) * ht_estimate(g, m, p, np.array([0.0]))
    np.testing.assert_allclose(mean, g, atol=1e-12)


def test_u2_variance_identity():
    rng = np.random.default_rng(0)
    g = rng.normal(size=(8, 3))
    m = rng.normal(size=(8, 3))
    p = np.full(8, 0.5)
    full, extra, total = ht_variance_trace(g, m, p)
    # Monte Carlo over all 2^8 is too big; check 3 rows exactly.
    g3, m3, p3 = g[:3], m[:3], p[:3]
    hats = []
    for bits in itertools.product([0.0, 1.0], repeat=3):
        z = np.array(bits)
        hats.append(ht_estimate(g3, m3, p3, z).sum(axis=0) / 3)
    hats = np.stack(hats)
    mean = hats.mean(axis=0)
    mc_var = float(np.mean(np.sum((hats - mean) ** 2, axis=1)))
    full3, extra3, total3 = ht_variance_trace(g3, m3, p3)
    assert full3 >= 0 and extra3 >= 0
    assert total3 == pytest.approx(full3 + extra3)
    # G is fixed; only Z is random, so MC variance of the batch mean is extra/N.
    assert mc_var == pytest.approx(extra3 / 3, abs=1e-12)
    assert full + extra == pytest.approx(total)


def test_u3_orthogonal_component_recovered():
    u = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=np.float64)
    f = np.array([[0.5, -0.2]])
    g = np.array([[0.5, -0.2, 3.0]])  # last coord is orthogonal
    p = np.array([0.3])
    mean = 0.3 * ht_estimate(g, (u @ f[0])[None, :], p, np.array([1.0])) + 0.7 * ht_estimate(
        g, (u @ f[0])[None, :], p, np.array([0.0])
    )
    np.testing.assert_allclose(mean[0], g[0], atol=1e-12)


def test_u4_dual_stream_and_p1_regression():
    u = np.eye(4, 2)
    f = np.array([[0.1, 0.2], [0.0, -0.3], [0.4, 0.1]])
    m = f @ u.T
    g = m + np.array([[1.0, 0, 0, 2.0], [0.5, -1, 0, 0], [0, 0, 0.2, 0]])
    p = np.array([0.5, 1.0, 0.25])
    z = np.array([1.0, 1.0, 0.0])
    vec = batch_ht_mean(g, m, p, z, n=3)
    completed = g[z >= 1.0]
    dual = dual_stream_mean(completed, p[z >= 1.0], u, f, z, p, 3)
    np.testing.assert_allclose(vec, dual, atol=1e-12)
    p1 = np.ones(3)
    z1 = np.ones(3)
    np.testing.assert_allclose(batch_ht_mean(g, m, p1, z1, n=3), g.mean(axis=0), atol=1e-12)


def test_optimizer_grad_is_negative_ghat():
    ghat = np.array([1.0, -2.0])
    np.testing.assert_allclose(optimizer_grad_from_ghat(ghat), -ghat)


def test_invalid_p_raises():
    with pytest.raises(ValueError):
        ht_estimate(np.ones((1, 2)), np.zeros((1, 2)), np.array([0.0]), np.array([1.0]))


def test_real_stream_scale_rejects_mismatch():
    from grace_gc.core.losses import real_stream_loss_scale

    with pytest.raises(ValueError):
        real_stream_loss_scale(np.ones(2), np.ones(3), np.ones(2), 2)


def test_prediction_correction_includes_completers():
    u = np.eye(2)
    f = np.array([[1.0, 0.0], [0.0, 1.0]])
    z = np.array([1.0, 0.0])
    p = np.array([0.5, 0.5])
    corr = prediction_correction(u, f, z, p, 2)
    expected = (u @ ((1 - z / p) @ f)) / 2
    np.testing.assert_allclose(corr, expected)
