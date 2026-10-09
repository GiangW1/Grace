from dataclasses import replace
import json

import numpy as np
import pytest

from grace_gc.audit.prefix_audit import PrefixBundle, audit_bundles, bundle_from_dict
from grace_gc.audit.run import _serialize_audit_bundles, _spill_trajectory_arrays
from grace_gc.logging_util.run_dir import RunDirectory


def test_audit_can_switch_existing_lag_calculations_to_full_trajectory_gradient():
    suffix = np.asarray([[1., 0.], [1., 0.], [2., 0.], [2., 0.]])
    full = suffix + np.asarray([0., 1.])
    bundle = PrefixBundle(
        "p", 1, np.asarray([0., 1., 0., 1.]), suffix,
        trajectory_grads=full,
        trajectory_grad_norm_sq=np.sum(full * full, axis=1).tolist(),
        answer_emitted=False,
    )
    result = audit_bundles([bundle], np.zeros((2, 0)),
                           {"gradient_target": "trajectory", "require_pre_emit": False},
                           np.random.default_rng(2))
    assert result["gradient_target"] == "trajectory"
    assert result["measurement_version"] == 3


def test_trajectory_target_drops_incompatible_projected_predictor():
    full = np.asarray([[1., 0., 1.], [1., 0., 1.]])
    bundle = PrefixBundle(
        "p", 1, np.asarray([0., 1.]), np.asarray([[1., 0.], [1., 0.]]),
        trajectory_grads=full, m_pred=np.asarray([1., 0.]),
        answer_emitted=False,
    )
    result = audit_bundles([bundle], np.zeros((3, 0)),
                           {"gradient_target": "trajectory", "require_pre_emit": False},
                           np.random.default_rng(3))
    assert result["gradient_target"] == "trajectory"
    assert result["full_space_error_decomposition"]["all"]["n_unavailable_bundles"] == 1
    assert result["unavailable_predictors"][0]["reason"] == (
        "projected_predictor_cannot_be_reconstructed_in_trajectory_space")


def test_trajectory_sidecars_roundtrip_without_dense_json(tmp_path):
    full = np.arange(12., dtype=np.float64).reshape(4, 3)
    bundle = PrefixBundle("p", 1, np.array([0., 1., 0., 1.]), np.zeros((4, 1)),
                          trajectory_grads=full, prefix_score_grad=np.ones(3))
    _spill_trajectory_arrays(bundle, tmp_path, 0)
    assert isinstance(bundle.trajectory_grads, np.memmap)
    assert isinstance(bundle.prefix_score_grad, np.memmap)
    run = RunDirectory(tmp_path)
    run.write_jsonl("audit_bundles.jsonl", _serialize_audit_bundles(run, [bundle]))
    raw = json.loads((tmp_path / "audit_bundles.jsonl").read_text())
    assert raw["trajectory_grads"] is None
    assert raw["prefix_score_grad"] is None
    restored = bundle_from_dict(raw, base_dir=tmp_path)
    np.testing.assert_array_equal(restored.trajectory_grads, full)
    np.testing.assert_array_equal(restored.prefix_score_grad, np.ones(3))
    assert not restored.trajectory_grads.flags.owndata
    del restored
    raw["trajectory_grads_sidecar"]["sha256"] = "wrong"
    with pytest.raises(ValueError, match="hash"):
        bundle_from_dict(raw, base_dir=tmp_path)


def test_trajectory_conversion_uses_predictions_not_true_coordinates():
    full = np.array([[2., 0., 0.], [4., 0., 0.]])
    u = np.array([[1.], [0.], [0.]])
    bundle = PrefixBundle("p", 1, np.array([0., 1.]), np.zeros((2, 1)),
                          trajectory_grads=full, trajectory_coords=[[2.], [4.]],
                          coords=np.array([.5]), basis_gram=[[1.]])
    result = audit_bundles([bundle], u,
                           {"gradient_target": "trajectory", "require_pre_emit": False},
                           np.random.default_rng(3))
    assert result["residual_mean"] == pytest.approx(7.25)
    # A compressed predictor can be reconstructed from its frozen coefficients.
    bundle = replace(bundle, m_pred=np.array([99.]), effective_coords=[.5])
    result = audit_bundles([bundle], u,
                           {"gradient_target": "trajectory", "require_pre_emit": False},
                           np.random.default_rng(3))
    assert result["residual_mean"] == pytest.approx(7.25)
    assert result["unavailable_predictors"] == []

