"""New mechanism measurements, without changing legacy rho or parse averages."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from grace_gc.audit.mechanism import full_space_decomposition, paired_geometry


@pytest.mark.parametrize("u", [np.array([[2., 4.], [0., 0.], [0., 0.]]),
                               np.array([[1., .3], [.2, 2.], [0., 0.]])])
def test_full_space_error_decomposition_handles_nonorthogonal_and_singular_basis(u):
    g = np.array([[3., 4., 5.], [-2., 1., 6.]])
    f = np.array([.4, -.7])
    result = full_space_decomposition(np.sum(g*g, axis=1), g@u, u.T@u, f)
    projected = g @ u @ np.linalg.pinv(u.T@u) @ u.T
    np.testing.assert_allclose(result["subspace_omission_energy"], np.sum((g-projected)**2, axis=1))
    np.testing.assert_allclose(result["coordinate_prediction_error_energy"], np.sum((projected-u@f)**2, axis=1), atol=1e-12)
    np.testing.assert_allclose(result["prediction_error_energy"], np.sum((g-u@f)**2, axis=1))
    np.testing.assert_allclose(np.array(result["subspace_omission_energy"])+result["coordinate_prediction_error_energy"],
                               result["prediction_error_energy"])


def test_missing_full_space_labels_are_not_reconstructed_from_sketch():
    assert full_space_decomposition(None, None, None, [1.])["status"] == "unavailable"
    result = full_space_decomposition([5.], [[1.]], [[1.]], None)
    assert result["captured_energy"] == [1.]
    assert result["coordinate_prediction_error_energy"] is None


def test_direction_geometry_zero_and_opposite_vectors():
    assert paired_geometry(np.zeros(2), np.zeros(2))["cosine"] is None
    result = paired_geometry(np.array([3., 4.]), np.array([-3., -4.]))
    assert result["cosine"] == pytest.approx(-1)
    assert result["relative_squared_error"] == pytest.approx(4)


def test_bundle_full_space_summary_uses_true_stats_and_marks_legacy_missing():
    from grace_gc.audit.prefix_audit import PrefixBundle, bundle_from_dict, bundle_to_dict, full_space_error_summary
    bundle = PrefixBundle("p", 2, np.array([0., 1.]), np.zeros((2, 1)),
                          coords=np.array([2.]), effective_coords=[1.], true_grad_norm_sq=[25., 10.],
                          true_grad_coords=[[6.], [2.]], basis_gram=[[4.]])
    restored = bundle_from_dict(json.loads(json.dumps(bundle_to_dict(bundle))))
    result = full_space_error_summary([restored])
    assert result["all"]["captured_energy"]["mean"] == pytest.approx(5)
    assert result["bundles"][0]["coordinate_prediction_error_energy"] == pytest.approx([1., 1.])
    legacy = bundle_from_dict({"problem_id": "old", "t": 2, "rewards": [0., 1.], "grads": [[2.], [3.]]})
    missing = full_space_error_summary([legacy])
    assert missing["all"]["n_unavailable_bundles"] == 1
    assert missing["all"]["captured_energy"]["mean"] is None


def test_prefix_generation_keeps_raw_coords_and_zeros_only_cv(monkeypatch):
    from grace_gc.audit import run
    from grace_gc.data.math_data import MathRecord
    from grace_gc.trainer.baseline import baseline_from_config
    from grace_gc.trainer.methods import method_spec
    class Engine:
        eos_id, last_rollout = None, None
        decode = staticmethod(lambda ids: "Answer: 1")
        def generate_prefix(self, prompts, length, rng, stream):
            assert stream == "token"  # fixed baseline pays no prescan
            return [x+[3]*length for x in prompts], np.zeros(len(prompts), bool)
        def continue_selected(self, prefixes, selected, length, rng):
            return [x+[4]*length for x in prefixes]
        def prefix_features(self, prefixes, lengths, baselines):
            assert baselines == [.3]
            return {"features": np.zeros((len(prefixes), 1))}
    predictor = SimpleNamespace(constant_cost=False, forward_numpy=lambda *a: SimpleNamespace(
        f=np.array([[2.]]), r_hat=np.array([7.]), c_hat=np.array([3.])))
    def gradient(engines, layout, full, plen, reward, baseline):
        assert baseline == .3
        return np.array([3., 4., 5.])
    monkeypatch.setattr(run, "_policy_grad_vec", gradient)
    bundles = run._bundles_from_engines([MathRecord("p", "q", "1")], Engine(), SimpleNamespace(dim=3),
        1, 2, 1, 2, 17, lambda rec,n: ([[1]]*n, [rec.problem_id]*n, [rec.answer]*n),
        predictor=predictor, u=np.array([[2.], [0.], [0.]]), spec=method_spec("grace"), jl_dim=1,
        control_variate=False, baseline_policy=baseline_from_config({"mode": "fixed", "fixed_value": .3}))
    bundle = bundles[0]
    assert bundle.coords.tolist() == [2.] and bundle.effective_coords == [0.]
    assert bundle.r_hat == 7. and np.array_equal(bundle.m_pred, [0.])
    assert bundle.true_grad_coords == [[6.], [6.]] and bundle.basis_gram == [[4.]]
    assert bundle.true_grad_norm_sq == [50., 50.]  # pre-compression


def test_audit_probability_uses_runtime_uniform_shrink_and_exact_p1():
    from grace_gc.audit.prefix_audit import PrefixBundle, _p_at_decision
    from grace_gc.trainer.methods import method_spec
    bundles = [PrefixBundle(str(i), 2, np.array([0., 1.]), np.ones((2, 2)), r_hat=r, c_hat=c)
               for i, (r,c) in enumerate(((0., 2.), (9., 1.)))]
    analysis = {"beta": .5, "p_min": .2, "uniform_shrink": 1.}
    probabilities = _p_at_decision(bundles, method_spec("grace"), analysis, {}, {})
    assert probabilities[0] == pytest.approx(probabilities[1])
    probabilities = _p_at_decision(bundles, method_spec("grace"), {**analysis, "beta": 1.}, {}, {})
    np.testing.assert_array_equal(probabilities, [1., 1.])


@pytest.mark.parametrize("mode,baseline_cfg,stored,expected", [
    ("checkpoint", {}, {"alpha": .7, "mode": "fixed", "fixed_value": .3, "values": {"p": .9}}, .3),
    ("prescan", {"mode": "fixed", "fixed_value": .25, "prescan": 0}, {}, .25),
    ("prescan", {"prescan": 2, "prescan_prior_strength": 4., "prescan_prior_mean": .5}, {}, None),
])
def test_batch_baseline_protocol_and_new_geometry(mode, baseline_cfg, stored, expected, tmp_path):
    from test_batch_audit import _fake_context
    from grace_gc.audit.batch_audit import audit_fixed_batches
    from grace_gc.data.math_data import MathRecord
    from grace_gc.logging_util.run_dir import RunDirectory
    from grace_gc.trainer.methods import method_spec
    engines, layout, encode, opt = _fake_context()
    cfg = {"seed": 17, "max_new_tokens": 4, "decision_tokens": 2, "baseline": baseline_cfg,
           "batch_audit": {"replicates": 1, "starts_per_prompt": 1, "baseline_mode": mode}}
    result = audit_fixed_batches([MathRecord("p", "q", "1")], engines, layout, encode,
        {"optimizer": opt, "baseline": stored}, None, None, method_spec("full_pg"), cfg, RunDirectory(tmp_path))
    baseline = json.loads((tmp_path / "batch_audit_baselines.json").read_text())["problems"][0]
    if expected is None:
        expected = (sum(row["reward"] for row in baseline["samples"])+2.)/6.
    else:
        assert baseline["samples"] == []
    assert baseline["baseline"] == pytest.approx(expected)
    assert set(result["paired_geometry"]) == {"raw_gradient", "clipped_gradient", "optimizer_update"}
    assert result["clipped_gradient"]["paired_mean_difference_norm"] == 0
    replicate = json.loads((tmp_path / "batch_audit_replicates.jsonl").read_text().splitlines()[0])
    assert all(row["squared_error"] == 0 for row in replicate["paired_geometry"].values())


def test_optimizer_exposes_actual_shared_clip_vector():
    from test_batch_audit import _fake_context
    from grace_gc.audit.batch_audit import OptimizerReplay
    engines, layout, _encode, opt = _fake_context()
    replay = OptimizerReplay(engines.named_lora(), layout, opt, {"grad_clip": 1.})
    _, _, clipped = replay.step(np.array([3., 4.]), with_clipped=True)
    np.testing.assert_allclose(clipped, [.6, .8])


def test_batch_cv_disable_preserves_risk_and_final_probability():
    from grace_gc.audit.batch_audit import _allocation
    from grace_gc.trainer.methods import method_spec
    engine = SimpleNamespace(prefix_features=lambda *a: {"features": np.zeros((2, 1))})
    predictor = SimpleNamespace(constant_cost=False, forward_numpy=lambda *a: SimpleNamespace(
        f=np.array([[2.], [3.]]), r_hat=np.array([1., 9.]), c_hat=np.array([2., 4.])))
    args = ([[1, 2], [1, 3]], np.array([1, 1]), np.array([False, False]), [.5, .5], engine,
            predictor, np.array([[1.], [0.]]), method_spec("grace"),
            {"basis": {"basis_id": 1, "predictor_synced_basis_id": 1}})
    cfg = {"max_new_tokens": 4, "allocation": {"beta": .5, "uniform_shrink": .4}}
    enabled = _allocation(*args, cfg)
    disabled = _allocation(*args, {**cfg, "predictor": {"control_variate": False}})
    np.testing.assert_array_equal(enabled[0], [[2.], [3.]])
    np.testing.assert_array_equal(disabled[0], [[0.], [0.]])
    for index in (1, 2, 3):
        np.testing.assert_array_equal(enabled[index], disabled[index])


def test_clipped_geometry_includes_parameter_dtype_rounding():
    torch = pytest.importorskip("torch")
    from grace_gc.audit.batch_audit import OptimizerReplay
    from grace_gc.core.layout import collect_lora_layout
    from grace_gc.trainer.state_io import optimizer_state
    parameter = torch.nn.Parameter(torch.tensor([0., 0.], dtype=torch.float16))
    named = [("q_proj.lora_A", parameter)]
    optimizer = torch.optim.SGD([parameter], lr=.1)
    replay = OptimizerReplay(named, collect_lora_layout(named), optimizer_state(optimizer), {"grad_clip": 1.})
    _, _, clipped = replay.step(np.array([3., 4.]), with_clipped=True)
    expected = torch.tensor([.6, .8], dtype=torch.float16).float().numpy()
    np.testing.assert_array_equal(clipped, expected)
    assert not np.array_equal(clipped, [.6, .8])
