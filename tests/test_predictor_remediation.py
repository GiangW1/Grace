"""Regression checks for predictor fixes; synthetic CPU data only."""
import copy

import numpy as np
import pytest


def test_risk_can_recover_from_negative_logit():
    torch = pytest.importorskip("torch")
    from grace_gc.predictor.heads import PredictorHeads

    heads = PredictorHeads(2, 1, hidden_coord=4, hidden_risk=4)
    with torch.no_grad():
        for parameter in heads.risk.parameters():
            parameter.zero_()
        heads.risk[-1].bias.fill_(-30)
    before = heads.risk[-1].bias.item()
    loss = heads.train_risk(np.zeros((2, 2)), np.ones(2), np.ones(2), epochs=2)
    assert np.isfinite(loss)
    assert heads.risk[-1].bias.item() > before


def test_weighted_dual_ridge_matches_unpenalized_intercept_primal(monkeypatch):
    from grace_gc.predictor.scale import fit_weighted_ridge

    rng = np.random.default_rng(31)
    x, y, w = rng.normal(size=(7, 31)), rng.normal(size=(7, 3)), np.array([0, 1, 4, 2, 8, 1, 2.])
    xb = np.column_stack([x, np.ones(7)])
    penalty = np.diag([.7] * 31 + [0.])
    expected = np.linalg.solve(xb.T @ (w[:, None] * xb) + penalty, xb.T @ (w[:, None] * y))
    solve = np.linalg.solve
    dimensions = []

    def record(a, b):
        dimensions.append(len(a))
        return solve(a, b)

    monkeypatch.setattr(np.linalg, "solve", record)
    actual = fit_weighted_ridge(x, y, w, l2=.7)
    assert max(dimensions) <= len(x)
    np.testing.assert_allclose(actual, expected, atol=1e-11)


def test_exact_weighted_basis_preserves_objective_and_singletons():
    from grace_gc.predictor.basis import refresh_basis, should_refresh_basis

    g = np.array([[1., 0, 0], [-1., 0, 0], [0, 2., 0], [0, -2., 0], [0, 0, 500.]])
    pids = ["a", "a", "b", "b", "single"]
    weights = np.array([10., 10., 1., 1., 1.])
    basis = refresh_basis(g, 2, 1, weights=weights, solver="gram", problem_ids=pids)
    assert basis.rank == 2
    np.testing.assert_allclose(basis.u.T @ basis.u, np.eye(2), atol=1e-12)
    np.testing.assert_allclose(basis.u @ basis.u.T, np.diag([1., 1., 0.]), atol=1e-12)
    assert should_refresh_basis(30, 8, 19, 1, 32, warmup_steps=20, refresh_after_warmup=True)
    assert not should_refresh_basis(30, 8, 18, 1, 32, warmup_steps=20, refresh_after_warmup=True)


def test_fixed_scaler_does_not_change_existing_raw_function():
    pytest.importorskip("torch")
    from grace_gc.predictor.heads import PredictorHeads

    heads = PredictorHeads(2, 1, hidden_coord=4, hidden_risk=4, feature_scaler="fixed")
    heads.fit_feature_scaler(np.array([[1., 2.], [3., 5.]]))
    raw = np.array([[4., 8.], [-1., 2.]])
    old = heads.forward_numpy(raw)
    heads.fit_feature_scaler(raw * 100 + 20)
    new = heads.forward_numpy(raw)
    np.testing.assert_array_equal(old.r_hat, new.r_hat)
    np.testing.assert_array_equal(old.f, new.f)


def make_reservoir():
    from grace_gc.predictor.reservoir import GradientReservoir, ReservoirItem

    res = GradientReservoir(32)
    for i in range(18):
        res.add(ReservoirItem(str(i), np.array([float(i % 3), 1., 0.]), 1., .5,
                              np.array([float(i), 1.]), float(i % 2), 0, observed_step=i))
    return res


def test_calibration_excluded_from_basis_and_fit_and_survives_restore():
    pytest.importorskip("torch")
    from grace_gc.predictor.heads import PredictorHeads
    from grace_gc.predictor.reservoir import GradientReservoir
    from grace_gc.predictor.update import prepare_predictor_split

    heads = PredictorHeads(2, 1, hidden_coord=4, hidden_risk=4, shrink_calibration="holdout")
    res = make_reservoir()
    split = prepare_predictor_split(heads, res, np.random.default_rng(8), basis_fit_only=True)
    assert split["calibration"].any()
    assert not np.any(split["calibration"] & (split["basis"] | split["fit"] | split["hold"]))
    assert np.array_equal(split["basis"], split["fit"])
    restored = GradientReservoir.from_state_dict(res.state_dict())
    again = prepare_predictor_split(heads, restored, np.random.default_rng(99), basis_fit_only=True)
    for key in split:
        np.testing.assert_array_equal(split[key], again[key])
    assert restored.items[4].observed_step == 4


def test_frozen_diagnostic_does_not_train_and_rejects_seen_problems():
    pytest.importorskip("torch")
    from grace_gc.predictor.heads import PredictorHeads
    from grace_gc.predictor.update import diagnose_frozen_predictor

    heads = PredictorHeads(2, 1, hidden_coord=4, hidden_risk=4)
    res = make_reservoir()
    heads.seen_problem_ids = {"0", "1"}
    before = copy.deepcopy(heads.state_dict())
    report = diagnose_frozen_predictor(heads, res.items[:3], np.eye(3, 1))
    assert report["n"] == 1
    assert report["n_problems"] == 1
    np.testing.assert_array_equal(before["risk"]["0.weight"], heads.state_dict()["risk"]["0.weight"])
    assert heads.seen_problem_ids == {"0", "1"}


def test_update_uses_current_design_for_holdout_not_warmup_p(monkeypatch):
    pytest.importorskip("torch")
    from grace_gc.predictor.heads import PredictorHeads
    from grace_gc.predictor.update import prepare_predictor_split, update_predictor_from_reservoir

    heads = PredictorHeads(2, 1, hidden_coord=4, hidden_risk=4, coord_kind="ridge",
                           shrink_calibration="holdout", train_auxiliary=False)
    res = make_reservoir()
    split = prepare_predictor_split(heads, res, np.random.default_rng(8))
    # Every recorded p is 1, but a current continuation design is not p=1.
    out = update_predictor_from_reservoir(heads, res, np.eye(3, 1), np.random.default_rng(9),
                                          split=split, current_step=20, calibration_beta=.5)
    assert out["calibration_n"] > 0
    assert out["calibration_p_mean"] < 1
    assert out["calibration_gamma"] is not None
    assert out["supervision"]["max_age_steps"] == 20
    assert heads.coord is None and heads.success is None and heads.reward_risk is None
    assert heads.cost is None


def test_chunked_residual_handles_cancellation_and_nonorthogonal_basis():
    from grace_gc.predictor.risk import full_space_residual

    u = np.array([[1e10, 0.], [1e10, 2.], [0., 3.]])
    f = np.array([[1., 2.], [2., -1.]])
    g = f @ u.T + np.array([[1., 2., 3.], [-1., -2., 1.]])
    actual = full_space_residual(g, f, u, block_size=1)
    np.testing.assert_allclose(actual, [14., 6.], atol=1e-12)


def test_unused_heads_preserve_retained_initialization_and_rng():
    torch = pytest.importorskip("torch")
    from grace_gc.predictor.heads import PredictorHeads

    torch.manual_seed(54)
    full = PredictorHeads(5, 2, hidden_coord=8, hidden_risk=4, constant_cost=False)
    after_full = torch.get_rng_state().clone()
    torch.manual_seed(54)
    reduced = PredictorHeads(5, 2, hidden_coord=8, hidden_risk=4,
                             coord_kind="ridge", train_auxiliary=False)
    assert torch.equal(after_full, torch.get_rng_state())
    for name, weight in full.risk.state_dict().items():
        assert torch.equal(weight, reduced.risk.state_dict()[name])
    assert all(getattr(reduced, name) is None for name in
               ("coord", "cost", "success", "reward_risk", "opt_coord", "opt_cost", "opt_success", "opt_reward_risk"))


def test_fixed_scales_and_holdout_state_roundtrip():
    pytest.importorskip("torch")
    from grace_gc.predictor.heads import predictor_from_spec
    from grace_gc.predictor.update import update_predictor_from_reservoir

    heads = predictor_from_spec(2, 1, {"coord_kind": "ridge", "feature_scaler": "fixed",
                                      "shrink_calibration": "holdout", "train_auxiliary": False})
    res = make_reservoir()
    update_predictor_from_reservoir(heads, res, np.eye(3, 1), np.random.default_rng(4), epochs=1)
    raw = heads.state_dict()
    restored = predictor_from_spec(2, 1, raw)
    restored.load_state_dict(raw)
    x = np.array([[3., 1.], [11., 1.]])
    np.testing.assert_array_equal(heads.forward_numpy(x).f, restored.forward_numpy(x).f)
    np.testing.assert_array_equal(heads.forward_numpy(x).r_hat, restored.forward_numpy(x).r_hat)
    assert restored.seen_problem_ids == set(res.problem_ids())
    assert restored.risk_scale_fitted and restored.diagnostic_history_complete
    before_scale = heads.scaler.state_dict().copy()
    for it in res.items:
        it.g *= 20
        it.features += 100
    update_predictor_from_reservoir(heads, res, np.eye(3, 1), np.random.default_rng(4), epochs=1)
    np.testing.assert_array_equal(before_scale["mean"], heads.scaler.mean)
    assert before_scale["risk_scale"] == heads.scaler.risk_scale


def test_old_predictor_checkpoint_defaults_are_explicit():
    pytest.importorskip("torch")
    from grace_gc.predictor.heads import PredictorHeads, predictor_from_spec

    old = PredictorHeads(2, 1, hidden_coord=4, hidden_risk=4).state_dict()
    for key in ("feature_scaler", "shrink_calibration", "calibration_fraction", "ridge_weight_normalization",
                "train_auxiliary", "seen_problem_ids", "diagnostic_history_complete", "risk_scale_fitted"):
        old.pop(key)
    heads = predictor_from_spec(2, 1, old)
    heads.load_state_dict(old)
    assert heads.feature_scaler == "refit" and heads.shrink_calibration == "fit"
    assert heads.ridge_weight_normalization == "sum"
    assert not heads.diagnostic_history_complete


def test_mean_ridge_is_invariant_to_common_ipw_scale():
    pytest.importorskip("torch")
    from grace_gc.predictor.heads import PredictorHeads

    rng = np.random.default_rng(7)
    x, y = rng.normal(size=(5, 20)), rng.normal(size=(5, 2))
    heads = PredictorHeads(20, 2, coord_kind="ridge", ridge_weight_normalization="mean")
    weights = np.arange(1., 6.)
    heads.train_coord(x, y, weights)
    expected = heads.predict_f(x)
    heads.train_coord(x, y, weights * 100)
    np.testing.assert_allclose(expected, heads.predict_f(x), atol=1e-13)


def test_gamma_heldout_labels_do_not_fit_coordinate_or_candidate_design(monkeypatch):
    pytest.importorskip("torch")
    from grace_gc.predictor.heads import PredictorHeads
    from grace_gc.predictor.update import prepare_predictor_split, update_predictor_from_reservoir
    import grace_gc.predictor.update as update_module

    heads = PredictorHeads(2, 1, coord_kind="ridge", shrink_calibration="holdout", train_auxiliary=False)
    res = make_reservoir()
    split = prepare_predictor_split(heads, res, np.random.default_rng(3))
    p_seen = []
    gamma_fn = update_module.design_shrink_from_projections

    def record(coords, f, gram, p, weights):
        p_seen.append(p.copy())
        return gamma_fn(coords, f, gram, p, weights)

    monkeypatch.setattr(update_module, "design_shrink_from_projections", record)
    initial = copy.deepcopy(heads.state_dict())
    update_predictor_from_reservoir(heads, res, np.eye(3, 1), np.random.default_rng(4), split=split)
    coef = heads.ridge_w.copy()
    first_gamma = heads.m_shrink
    heads.load_state_dict(initial)
    for item, selected in zip(res.items, split["calibration"]):
        if selected:
            item.g = -100 * item.g
    update_predictor_from_reservoir(heads, res, np.eye(3, 1), np.random.default_rng(4), split=split)
    np.testing.assert_array_equal(coef, heads.ridge_w)
    np.testing.assert_array_equal(p_seen[0], p_seen[1])
    assert first_gamma > 0 and heads.m_shrink == 0


def test_empty_calibration_keeps_gamma_and_reports_no_estimate():
    pytest.importorskip("torch")
    from grace_gc.predictor.heads import PredictorHeads
    from grace_gc.predictor.update import update_predictor_from_reservoir

    res = make_reservoir()
    res.items = res.items[:2]
    heads = PredictorHeads(2, 1, coord_kind="ridge", shrink_calibration="holdout", train_auxiliary=False)
    heads.m_shrink = .7
    out = update_predictor_from_reservoir(heads, res, np.eye(3, 1), np.random.default_rng(9))
    assert out["calibration_gamma"] is None and out["calibration_n"] == 0
    assert heads.m_shrink == .7


def test_basis_gram_matches_weighted_svd_and_does_not_allocate_random_projection(monkeypatch):
    from grace_gc.predictor.basis import refresh_basis
    import grace_gc.predictor.basis as basis_module

    rng = np.random.default_rng(6)
    g = rng.normal(size=(9, 81))
    weights = rng.uniform(.1, 3., 9)
    def unexpected(*args, **kwargs):
        raise AssertionError("exact basis must not allocate random projection")
    monkeypatch.setattr(basis_module, "randomized_svd", unexpected)
    gram = refresh_basis(g, 4, 1, weights=weights, center="global", solver="auto")
    svd = refresh_basis(g, 4, 1, weights=weights, center="global", solver="svd")
    np.testing.assert_allclose(gram.u @ gram.u.T, svd.u @ svd.u.T, atol=1e-13)


def test_ridge_empty_weights_and_unregularized_fit_avoid_large_gram(monkeypatch):
    from grace_gc.predictor.scale import fit_weighted_ridge

    x = np.arange(300., dtype=np.float64).reshape(3, 100)
    y = np.array([[1.], [2.], [3.]])
    def unexpected(*args, **kwargs):
        raise AssertionError("empty/zero-ridge cases do not need a squared system")
    monkeypatch.setattr(np.linalg, "solve", unexpected)
    np.testing.assert_array_equal(fit_weighted_ridge(x, y, np.zeros(3)), np.zeros((101, 1)))
    coef = fit_weighted_ridge(x, y, np.ones(3), l2=0)
    np.testing.assert_allclose(np.column_stack([x, np.ones(3)]) @ coef, y, atol=1e-12)


def test_calibration_remaining_cost_finished_and_reward_head(monkeypatch):
    pytest.importorskip("torch")
    from grace_gc.predictor.heads import PredictorHeads
    from grace_gc.predictor.update import prepare_predictor_split, update_predictor_from_reservoir
    import grace_gc.predictor.update as update_module

    heads = PredictorHeads(2, 1, coord_kind="ridge", shrink_calibration="holdout")
    res = make_reservoir()
    split = prepare_predictor_split(heads, res, np.random.default_rng(9))
    for i, item in enumerate(res.items):
        item.remaining_cost = float(10 + i)
        item.prefix_finished = i == 2
        item.reward_feat = np.array([.5, .69, .5, i + 5., .25])
    allocation = update_module.allocate_continuation
    observed = {}
    def record(risk, cost, beta, p_min, **kwargs):
        observed.update(risk=risk, cost=cost, finished=kwargs["finished"])
        result = allocation(risk, cost, beta, p_min, **kwargs)
        observed["p"] = result.p
        return result
    monkeypatch.setattr(update_module, "allocate_continuation", record)
    monkeypatch.setattr(heads, "forward_reward_risk", lambda x: np.full(len(x), 13.))
    update_predictor_from_reservoir(heads, res, np.eye(3, 1), np.random.default_rng(8),
                                    split=split, calibration_risk_mode="reward")
    np.testing.assert_array_equal(observed["risk"], np.full(split["calibration"].sum(), 13.))
    np.testing.assert_array_equal(observed["cost"], np.arange(10., 28.)[split["calibration"]])
    assert observed["finished"].sum() == 1
    np.testing.assert_array_equal(observed["p"][observed["finished"]], [1.])


def test_risk_numerics_reports_underflow_without_claiming_recovery():
    torch = pytest.importorskip("torch")
    from grace_gc.predictor.heads import PredictorHeads

    heads = PredictorHeads(2, 1, hidden_risk=4)
    with torch.no_grad():
        for parameter in heads.risk.parameters():
            parameter.zero_()
        heads.risk[-1].bias.fill_(-1000.)
    report = heads.risk_diagnostics(np.zeros((3, 2)))
    assert report["logit_min"] == -1000. and report["logit_finite_n"] == 3
    assert report["softplus_zero_n"] == 3 and report["softplus_near_floor_fraction"] == 1.


def test_predictable_basis_keeps_cross_problem_signal_not_large_irreducible_noise():
    from grace_gc.predictor.basis import refresh_basis, refresh_predictable_basis

    x = np.repeat(np.arange(-4., 4.), 2)[:, None]
    g = np.column_stack([3. * x[:, 0], np.tile([-50., 50.], 8), np.zeros(16)])
    pids = [str(i // 2) for i in range(16)]
    pca = refresh_basis(g, 1, 1, problem_ids=pids)
    predicted, metrics = refresh_predictable_basis(g, x, pids, 1, 2, ridge_l2=.1)
    np.testing.assert_allclose(pca.u @ pca.u.T, np.diag([0., 1., 0.]), atol=1e-12)
    np.testing.assert_allclose(predicted.u @ predicted.u.T, np.diag([1., 0., 0.]), atol=1e-12)
    assert metrics["crossfit_problems"] == 8 and metrics["prediction_rows"] == 16


def test_crossfit_operator_excludes_all_same_problem_labels_and_matches_ridge():
    from grace_gc.predictor.basis import crossfit_ridge_operator
    from grace_gc.predictor.scale import fit_weighted_ridge, ridge_predict

    rng = np.random.default_rng(4)
    x, g = rng.normal(size=(9, 5)), rng.normal(size=(9, 3))
    w = np.arange(1., 10.)
    pids = np.array([str(i // 3) for i in range(9)])
    operator, available = crossfit_ridge_operator(x, pids, w, ridge_l2=.3)
    assert available.all()
    for pid in np.unique(pids):
        hold, train = pids == pid, pids != pid
        assert not np.any(operator[np.ix_(hold, hold)])
        tw = w[train] / w[train].sum()
        mean = np.average(x[train], axis=0, weights=tw)
        std = np.sqrt(np.average((x[train] - mean) ** 2, axis=0, weights=tw))
        std = np.where(std > 1e-6, std, 1.)
        expected = ridge_predict((x[hold] - mean) / std,
                                  fit_weighted_ridge((x[train] - mean) / std, g[train], tw, l2=.3))
        np.testing.assert_allclose((operator @ g)[hold], expected, atol=1e-12)
        changed = g.copy()
        changed[hold] += 1000
        np.testing.assert_array_equal((operator @ changed)[hold], (operator @ g)[hold])


def test_predictable_basis_single_problem_and_zero_signal_do_not_invent_basis():
    from grace_gc.predictor.basis import refresh_predictable_basis

    basis, metrics = refresh_predictable_basis(np.ones((3, 7)), np.arange(3.)[:, None], ["one"] * 3, 2, 1)
    assert basis is None and metrics["prediction_rows"] == 0
    basis, metrics = refresh_predictable_basis(np.zeros((3, 7)), np.arange(3.)[:, None], ["a", "b", "c"], 2, 1)
    assert basis is None and metrics["rank"] == 0


def test_age_policy_applies_same_rows_and_weights_to_all_predictor_sides():
    pytest.importorskip("torch")
    from grace_gc.predictor.heads import PredictorHeads
    from grace_gc.predictor.update import prepare_predictor_split

    heads = PredictorHeads(2, 1, shrink_calibration="holdout")
    res = make_reservoir()
    res.items[16].observed_step = None
    split = prepare_predictor_split(heads, res, np.random.default_rng(3), current_step=20,
                                     max_age_steps=10, age_half_life=5, missing_step_policy="exclude")
    assert np.all(split["age_weight"][:10] == 0)
    assert split["age_weight"][10] == pytest.approx(.25)
    assert split["age_weight"][16] == 0
    for side in ("fit", "hold", "calibration", "basis"):
        assert not np.any(split[side] & (split["age_weight"] == 0))
    assert split["freshness"]["missing_step_n"] == 1
    assert split["freshness"]["expired_n"] == 10


def test_age_expiration_does_not_reassign_problem_or_fit_expired_labels(monkeypatch):
    pytest.importorskip("torch")
    from grace_gc.predictor.heads import PredictorHeads
    from grace_gc.predictor.update import prepare_predictor_split, update_predictor_from_reservoir

    heads = PredictorHeads(2, 1, coord_kind="ridge", shrink_calibration="holdout", train_auxiliary=False)
    res = make_reservoir()
    split = prepare_predictor_split(heads, res, np.random.default_rng(3), current_step=20,
                                     max_age_steps=10, age_half_life=5)
    old_ids = (res.fit_problem_ids.copy(), res.hold_problem_ids.copy(), res.calibration_problem_ids.copy())
    seen = {}
    original = heads.train_coord
    def capture(x, y, weights, **kwargs):
        seen.update(x=x, weights=weights)
        return original(x, y, weights, **kwargs)
    monkeypatch.setattr(heads, "train_coord", capture)
    report = update_predictor_from_reservoir(heads, res, np.eye(3, 1), np.random.default_rng(4),
                                             split=split, current_step=20)
    np.testing.assert_array_equal(seen["x"], np.stack([it.features for it in res.items])[split["fit"]])
    np.testing.assert_allclose(seen["weights"], 2 * split["age_weight"][split["fit"]])
    assert report["freshness"]["expired_n"] == 10
    expired = prepare_predictor_split(heads, res, np.random.default_rng(55), current_step=100, max_age_steps=1)
    empty = update_predictor_from_reservoir(heads, res, np.eye(3, 1), np.random.default_rng(4), split=expired)
    assert empty["coord_loss"] is None and empty["calibration_gamma"] is None
    assert (res.fit_problem_ids, res.hold_problem_ids, res.calibration_problem_ids) == old_ids


def test_missing_age_metadata_is_reported_with_explicit_keep_or_exclude():
    pytest.importorskip("torch")
    from grace_gc.predictor.heads import PredictorHeads
    from grace_gc.predictor.update import prepare_predictor_split

    res = make_reservoir()
    for it in res.items:
        it.observed_step = None
    heads = PredictorHeads(2, 1)
    keep = prepare_predictor_split(heads, res, np.random.default_rng(4), current_step=20,
                                    max_age_steps=5, missing_step_policy="keep")
    exclude = prepare_predictor_split(heads, res, np.random.default_rng(4), current_step=20,
                                       max_age_steps=5, missing_step_policy="exclude")
    assert np.all(keep["age_weight"] == 1) and not np.any(exclude["age_weight"])
    assert keep["freshness"]["missing_step_n"] == len(res.items)


def test_predictable_basis_matches_explicit_weighted_crossfit_svd():
    from grace_gc.predictor.basis import crossfit_ridge_operator, refresh_predictable_basis

    rng = np.random.default_rng(7)
    g, x = rng.normal(size=(8, 19)), rng.normal(size=(8, 4))
    pids = [str(i // 2) for i in range(8)]
    w = np.array([0., 2., 1., 3., .5, 1., 2., .25])
    operator, available = crossfit_ridge_operator(x, pids, w, ridge_l2=.7)
    predicted = np.sqrt(w * available)[:, None] * (operator @ g)
    _, singular, vt = np.linalg.svd(predicted, full_matrices=False)
    basis, report = refresh_predictable_basis(g, x, pids, 3, 4, weights=w, ridge_l2=.7)
    np.testing.assert_allclose(basis.u @ basis.u.T, vt[:3].T @ vt[:3], atol=1e-12)
    assert report["predicted_energy"] == pytest.approx(np.dot(singular, singular))
    changed = g.copy()
    changed[0] += 1e6
    again, _ = refresh_predictable_basis(changed, x, pids, 3, 4, weights=w, ridge_l2=.7)
    np.testing.assert_array_equal(basis.u, again.u)


def test_recency_and_fresh_policy_source_survive_checkpoint_without_reassigning_sides():
    pytest.importorskip("torch")
    from grace_gc.predictor.heads import PredictorHeads
    from grace_gc.predictor.reservoir import GradientReservoir
    from grace_gc.predictor.update import prepare_predictor_split

    res = make_reservoir()
    res.items[-1].source = "fresh_policy"
    heads = PredictorHeads(2, 1, shrink_calibration="holdout")
    before = prepare_predictor_split(heads, res, np.random.default_rng(8), current_step=20,
                                      max_age_steps=8, age_half_life=4)
    restored = GradientReservoir.from_state_dict(res.state_dict())
    after = prepare_predictor_split(heads, restored, np.random.default_rng(99), current_step=20,
                                     max_age_steps=8, age_half_life=4)
    for name in ("fit", "hold", "calibration", "basis", "age_weight"):
        np.testing.assert_array_equal(before[name], after[name])
    assert after["freshness"] == before["freshness"]
    assert after["freshness"]["sources"]["fresh_policy"] == 1


def test_no_recency_policy_preserves_legacy_unknown_dates_and_reports_them():
    from grace_gc.predictor.reservoir import age_weights

    res = make_reservoir()
    res.items[0].observed_step = None
    weights, info = age_weights(res.items, current_step=20)
    np.testing.assert_array_equal(weights, np.ones(len(res.items)))
    assert not info["enabled"] and info["missing_step_n"] == 1
