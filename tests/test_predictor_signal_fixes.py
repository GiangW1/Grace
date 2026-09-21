"""Synthetic CPU regressions for predictor target and calibration consistency."""

import numpy as np
import pytest


def _case(auxiliary=False):
    pytest.importorskip("torch")
    from grace_gc.predictor.heads import PredictorHeads
    from grace_gc.predictor.reservoir import GradientReservoir, ReservoirItem

    heads = PredictorHeads(2, 1, hidden_risk=4, coord_kind="ridge",
                           feature_scaler="fixed", shrink_calibration="holdout",
                           train_auxiliary=auxiliary)
    reservoir = GradientReservoir(8)
    for pid in ("fit", "hold", "cal"):
        reservoir.add(ReservoirItem(pid, np.zeros(2), .5, 1., np.ones(2),
                                    0., 0, remaining_cost=1., prefix_finished=False))
    split = {"fit": np.array([True, False, False]),
             "hold": np.array([False, True, False]),
             "calibration": np.array([False, False, True]),
             "basis": np.array([True, False, False]), "age_weight": np.ones(3)}
    return heads, reservoir, split


def test_fixed_risk_scale_retries_zero_start_and_then_stays_fixed():
    from grace_gc.predictor.update import update_predictor_from_reservoir

    heads, reservoir, split = _case()
    u = np.eye(2, 1)
    rng = np.random.default_rng(4)
    first = update_predictor_from_reservoir(heads, reservoir, u, rng, split=split, epochs=0)
    assert heads.scaler.risk_scale == 1.
    assert not heads.risk_scale_fitted
    assert first["supervision"]["hold"]["n"] == 1  # Zero labels remain supervision.
    reservoir.items[1].g = np.array([100., 0.])
    second = update_predictor_from_reservoir(heads, reservoir, u, rng, split=split, epochs=0)
    assert heads.risk_scale_fitted and heads.scaler.risk_scale == 10000.
    assert second["risk_scale"] == 10000. and second["risk_scale_fitted"]
    np.testing.assert_array_equal(heads.scaler.scale_risk([10000.]), [1.])
    reservoir.items[1].g *= 2
    update_predictor_from_reservoir(heads, reservoir, u, rng, split=split, epochs=0)
    assert heads.scaler.risk_scale == 10000.


def test_risk_scale_empty_and_zero_targets_are_not_fitted():
    from grace_gc.predictor.scale import FeatureScaler

    scaler = FeatureScaler()
    assert not scaler.fit_risk_scale(np.array([]))
    assert not scaler.fit_risk_scale(np.zeros(4))
    assert scaler.fit_risk_scale(np.array([0., 2.]))
    assert scaler.risk_scale == 1.  # A measured scale of 1 is valid.


def test_reward_risk_training_rebuilds_current_success_features(monkeypatch):
    from grace_gc.predictor.features import reward_risk_features
    from grace_gc.predictor.update import update_predictor_from_reservoir

    heads, reservoir, split = _case(auxiliary=True)
    for item in reservoir.items:
        item.reward_feat = reward_risk_features(.2, .8, 17., .25)
    reservoir.items[1].g = np.array([2., 3.])
    seen = {}
    current_q = [.2]

    def fit_success(*args, **kwargs):
        current_q[0] = .8
        return 0.

    def fit_reward(x, y, w, **kwargs):
        seen.update(x=x.copy(), y=y.copy(), w=w.copy())
        return 0.

    monkeypatch.setattr(heads, "train_success", fit_success)
    monkeypatch.setattr(heads, "forward_success", lambda x: np.full(len(x), current_q[0]))
    monkeypatch.setattr(heads, "train_reward_risk", fit_reward)
    update_predictor_from_reservoir(heads, reservoir, np.eye(2, 1), np.random.default_rng(2),
                                     split=split, epochs=0, calibration_risk_mode="reward")
    np.testing.assert_allclose(seen["x"], [reward_risk_features(.8, .2, 17., .25)])
    np.testing.assert_allclose(seen["y"], [13.])
    np.testing.assert_allclose(seen["w"], [2.])
    assert reservoir.items[1].reward_feat[0] == .2  # Historical record stays intact.


def test_gamma_diagnostics_explain_zero_clip_and_no_design_information():
    from grace_gc.predictor.scale import design_shrink_diagnostics, design_shrink_from_projections

    coords, f, gram = np.array([[-2.], [3.]]), np.ones((2, 1)), np.ones((1, 1))
    p, weights = np.array([.5, .5]), np.array([2., 1.])
    diagnostics = design_shrink_diagnostics(coords, f, gram, p, weights)
    assert diagnostics["numerator"] == -1. and diagnostics["denominator"] == 3.
    assert diagnostics["raw_gamma"] == pytest.approx(-1. / 3)
    assert diagnostics["design_positive_n"] == 2
    assert diagnostics["design_effective_n"] == pytest.approx(9. / 5)
    assert design_shrink_from_projections(coords, f, gram, p, weights) == 0.
    no_design = design_shrink_diagnostics(coords, f, gram, np.ones(2), weights)
    assert no_design["denominator"] == 0. and no_design["raw_gamma"] is None
    assert no_design["design_positive_n"] == 0
    assert design_shrink_from_projections(coords, f, gram, np.ones(2), weights) is None


def test_cross_moment_basis_rejects_anti_aligned_crossfit_prediction_energy():
    from grace_gc.predictor.basis import refresh_predictable_basis

    # Constant features can only predict the other problem's opposite label.
    g, x, pids = np.array([[10., 0.], [-10., 0.]]), np.ones((2, 1)), ["a", "b"]
    old, old_info = refresh_predictable_basis(g, x, pids, 1, 1)
    new, info = refresh_predictable_basis(g, x, pids, 1, 1, objective="cross_moment")
    assert old.rank == 1 and old_info["predicted_energy"] == pytest.approx(200.)
    assert new is None and info["rank"] == 0
    assert info["negative_eigenvalue_n"] == 1 and info["positive_eigenvalue_n"] == 0
    assert info["cross_moment_trace"] == pytest.approx(-200.)
    assert info["oof_residual_energy"] == pytest.approx(800.)


def test_cross_moment_basis_matches_explicit_weighted_full_space_eigensystem(monkeypatch):
    from grace_gc.predictor.basis import crossfit_ridge_operator, refresh_predictable_basis

    rng = np.random.default_rng(54)
    g, x = rng.normal(size=(9, 23)), rng.normal(size=(9, 4))
    pids = [str(i // 3) for i in range(9)]
    w = np.array([.25, 1., 3., .5, 1., 2., 4., 1., 1.])
    a, _ = crossfit_ridge_operator(x, pids, w, ridge_l2=.4)
    weighted = w[:, None] * a
    full = g.T @ ((weighted + weighted.T) / 2) @ g
    vals, vecs = np.linalg.eigh(full)
    selected = np.flatnonzero(vals > np.max(np.abs(vals)) * 1e-12)[::-1][:3]
    expected = vecs[:, selected] @ vecs[:, selected].T
    dimensions = []
    original = np.linalg.eigh

    def small_eigh(matrix):
        dimensions.append(matrix.shape[0])
        return original(matrix)

    monkeypatch.setattr(np.linalg, "eigh", small_eigh)
    basis, info = refresh_predictable_basis(g, x, pids, 3, 1, weights=w, ridge_l2=.4,
                                             objective="cross_moment")
    assert max(dimensions) <= len(g)
    assert basis.rank == len(selected)
    np.testing.assert_allclose(basis.u @ basis.u.T, expected, atol=2e-12)
    np.testing.assert_allclose(basis.u[:, :basis.rank].T @ basis.u[:, :basis.rank],
                               np.eye(basis.rank), atol=1e-12)
    assert info["cross_moment_trace"] == pytest.approx(np.trace(full))
    assert info["oof_residual_energy"] == pytest.approx(np.sum(w[:, None] * (g - a @ g) ** 2))


@pytest.mark.parametrize("grads", [np.zeros((3, 7)), np.zeros((0, 7))])
def test_cross_moment_basis_empty_and_zero_normal_results(grads):
    from grace_gc.predictor.basis import refresh_predictable_basis

    basis, info = refresh_predictable_basis(grads, np.ones((len(grads), 2)),
                                             [str(i) for i in range(len(grads))], 2, 1,
                                             objective="cross_moment")
    assert basis is None and info["rank"] == 0


def test_cross_moment_basis_rank_deficiency_and_zero_weight_label_invariance():
    from grace_gc.predictor.basis import refresh_predictable_basis

    g = np.tile([3., 4., 0., 0., 0.], (5, 1))
    x, pids, w = np.ones((5, 2)), [str(i) for i in range(5)], np.array([1., 2., 3., 4., 0.])
    first, _ = refresh_predictable_basis(g, x, pids, 3, 1, weights=w, objective="cross_moment")
    assert first.rank == 1
    np.testing.assert_allclose(first.u @ first.u.T, np.outer(g[0], g[0]) / 25., atol=1e-12)
    g[-1] = np.arange(5.) * 1e12
    again, _ = refresh_predictable_basis(g, x, pids, 3, 1, weights=w, objective="cross_moment")
    np.testing.assert_allclose(first.u @ first.u.T, again.u @ again.u.T, atol=1e-12)


def test_supervision_reports_finished_context_without_filtering():
    from grace_gc.predictor.update import update_predictor_from_reservoir

    heads, reservoir, split = _case()
    reservoir.items[0].prefix_finished = True
    reservoir.items[1].prefix_finished = None
    report = update_predictor_from_reservoir(heads, reservoir, np.eye(2, 1),
                                              np.random.default_rng(1), split=split, epochs=0)
    assert report["supervision"]["fit"]["n"] == 1
    assert report["supervision"]["fit"]["n_prefix_finished"] == 1
    assert report["supervision"]["hold"]["n_prefix_finished_unknown"] == 1
    assert report["supervision"]["calibration"]["n_prefix_unfinished"] == 1
