import numpy as np
import pytest

from grace_gc.predictor.basis import refresh_basis, reproject
from grace_gc.predictor.ipw import assign_new_problems, ipw_weights, split_by_problem
from grace_gc.predictor.reservoir import GradientReservoir, ReservoirItem
from grace_gc.predictor.risk import full_space_residual, risk_nll


def test_pool_last_hidden_uses_trailing_window():
    from grace_gc.predictor.features import pool_last_hidden

    seq = np.arange(10, dtype=np.float64).reshape(10, 1)
    np.testing.assert_allclose(pool_last_hidden(seq, 10, 64), [4.5])
    np.testing.assert_allclose(pool_last_hidden(seq, 10, 3), [8.0])


def test_entropy_stats_do_not_negate():
    from grace_gc.predictor.features import entropy_stats

    stats = entropy_stats(np.array([0.2, 1.0]))
    assert stats[0] == pytest.approx(0.6)
    assert stats[1] == pytest.approx(1.0)


def test_prompt_slice_matches_prefix_layout():
    from grace_gc.predictor.features import prefix_features, prompt_slice_features

    rng = np.random.default_rng(0)
    last = rng.normal(size=(2, 5, 4))
    mid = rng.normal(size=(2, 5, 4))
    token_lp = rng.normal(size=(2, 4))
    plen0, plen1 = 3, 2
    prompt = prompt_slice_features(last, mid, token_lp, [plen0, plen1], [0.1, 0.9])
    from grace_gc.predictor.features import pool_last_hidden

    pref = prefix_features(
        last[0, plen0 - 1],
        mid[0, plen0 - 1],
        pool_last_hidden(last[0], plen0),
        token_lp[0, : plen0 - 1],
        float(plen0),
        0.1,
    )
    assert prompt.shape == (2, pref.shape[0])
    np.testing.assert_allclose(prompt[0], pref)


def test_full_space_residual_keeps_gram():
    u = np.array([[1.0, 0.2], [0.0, 1.0], [0.0, 0.0]])
    g = np.array([1.0, 2.0, 3.0])
    f = np.array([0.5, -0.1])
    e = full_space_residual(g, f, u)
    m = u @ f
    assert e == pytest.approx(np.sum((g - m) ** 2), abs=1e-12)


def test_risk_nll_rejects_nonpositive():
    with pytest.raises(ValueError):
        risk_nll(np.array([1.0]), np.array([0.0]))


def test_new_problems_are_assigned_not_dumped_to_hold():
    fit, hold = set(), set()
    rng = np.random.default_rng(0)
    assign_new_problems(fit, hold, ["a", "a", "b"], rng)
    assert fit and hold
    snap = (set(fit), set(hold))
    assign_new_problems(fit, hold, ["a", "b"], rng)
    assert (fit, hold) == snap
    assign_new_problems(fit, hold, ["a", "b"] + [str(i) for i in range(16)], np.random.default_rng(1))
    assert len(fit) > len(snap[0])
    assert not (fit & hold)


def test_ipw_and_problem_split():
    w = ipw_weights(np.array([0.5, 1.0]), 0.125)
    np.testing.assert_allclose(w, np.array([16.0, 8.0]))
    rng = np.random.default_rng(0)
    pids = ["a", "a", "b", "c"]
    fit, rest = split_by_problem(pids, rng)
    assert fit.sum() + rest.sum() == 4
    assert not np.any(fit & rest)
    for pid in set(pids):
        sides = {bool(fit[i]) for i, name in enumerate(pids) if name == pid}
        assert len(sides) == 1


def test_train_coord_does_not_move_risk_weights():
    pytest.importorskip("torch")
    import torch

    from grace_gc.predictor.heads import PredictorHeads

    rng = np.random.default_rng(2)
    heads = PredictorHeads(in_dim=4, k=2, hidden_coord=8, hidden_risk=8)
    x = rng.normal(size=(6, 4))
    heads.train_risk(x, np.ones(6), np.ones(6), epochs=3)
    before = {k: v.detach().clone() for k, v in heads.risk.state_dict().items()}
    heads.train_coord(x, rng.normal(size=(6, 2)), np.ones(6), epochs=3)
    for key, tensor in heads.risk.state_dict().items():
        assert torch.equal(before[key], tensor)


def test_ipw_predictor_update_on_reservoir():
    pytest.importorskip("torch")
    from grace_gc.predictor.heads import PredictorHeads
    from grace_gc.predictor.update import update_predictor_from_reservoir

    rng = np.random.default_rng(1)
    d, k, n = 6, 2, 8
    u = refresh_basis(rng.normal(size=(n, d)), k=k, basis_id=0, seed=1).u
    res = GradientReservoir(capacity=32)
    for i in range(n):
        g = rng.normal(size=(d,))
        feat = rng.normal(size=(4,))
        res.add(ReservoirItem(str(i % 3), g, 0.4, 0.5, feat, 1.0, 0))
    heads = PredictorHeads(in_dim=4, k=k, hidden_coord=8, hidden_risk=8)
    out = update_predictor_from_reservoir(heads, res, u, rng, epochs=1)
    assert out["n"] == n
    assert out["coord_loss"] is not None


def test_evicted_fit_items_do_not_train_coord_on_hold():
    pytest.importorskip("torch")
    import torch

    from grace_gc.predictor.heads import PredictorHeads
    from grace_gc.predictor.update import update_predictor_from_reservoir

    rng = np.random.default_rng(3)
    d, k = 4, 2
    u = np.eye(d, k)
    res = GradientReservoir(capacity=2)
    res.fit_problem_ids = {"fit"}
    res.hold_problem_ids = {"hold"}
    for _ in range(3):
        res.add(ReservoirItem("hold", rng.normal(size=(d,)), 0.5, 0.125, rng.normal(size=(3,)), 1.0, 0))
    assert all(it.problem_id == "hold" for it in res.items)
    heads = PredictorHeads(in_dim=3, k=k, hidden_coord=8, hidden_risk=8)
    before = heads.coord[0].weight.detach().clone()
    out = update_predictor_from_reservoir(heads, res, u, rng, epochs=2)
    assert out["coord_loss"] is None
    assert out["risk_loss"] is not None
    assert torch.equal(before, heads.coord[0].weight)


def test_basis_refresh_and_reservoir():
    rng = np.random.default_rng(0)
    g = rng.normal(size=(12, 6))
    basis = refresh_basis(g, k=3, basis_id=1, seed=0)
    assert basis.u.shape == (6, 3)
    coords = basis.project(g)
    new = refresh_basis(g, k=3, basis_id=2, seed=1)
    mapped = reproject(coords, basis.u, new.u)
    assert mapped.shape == coords.shape
    mapped_g = reproject(coords, basis.u, new.u, g=g)
    np.testing.assert_allclose(mapped_g, g @ new.u)
    res = GradientReservoir(capacity=3)
    for i in range(5):
        res.add(ReservoirItem("p", g[i], 0.5, 0.125, np.zeros(2), 1.0, 1))
    assert len(res.items) == 3
