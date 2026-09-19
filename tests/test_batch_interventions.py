"""Frozen interventions share complete labels and never claim GPU savings."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from grace_gc.audit.batch_audit import audit_fixed_batches, resolve_batch_shape
from grace_gc.data.math_data import MathRecord
from grace_gc.logging_util.run_dir import RunDirectory
from grace_gc.trainer.methods import method_spec
from test_batch_audit import _fake_context


def test_same_labels_interventions_preserve_actual_and_p1_identity(tmp_path):
    cfg = {"seed": 17, "max_new_tokens": 4, "decision_tokens": 2,
           "allocation": {"beta": .5, "p_min": .2}, "optim": {"grad_clip": 1.},
           "batch_audit": {"replicates": 3, "starts_per_prompt": 2, "baseline_mode": "fixed"}}
    records = [MathRecord("a", "qa", "1"), MathRecord("b", "qb", "1")]
    outputs = []
    for enabled in (False, True):
        engines, layout, encode, opt = _fake_context()
        engines.prefix_features = lambda prefixes, *args: {"features": np.zeros((len(prefixes), 2))}
        predictor = SimpleNamespace(constant_cost=True,
            forward_numpy=lambda features, *args: SimpleNamespace(f=np.tile([2., -3.], (len(features), 1)),
                                                                r_hat=np.arange(1, len(features)+1, dtype=float)))
        options = copy.deepcopy(cfg)
        if enabled:
            options["batch_audit"]["interventions"] = {"p1": {"beta": 1.}, "m0": {"control_variate": False},
                                                       "uniform": {"uniform_shrink": 1.}}
        run = RunDirectory(tmp_path / str(enabled))
        result = audit_fixed_batches(records, engines, layout, encode,
            {"optimizer": opt, "basis": {"basis_id": 1, "predictor_synced_basis_id": 1}},
            predictor, np.eye(2), method_spec("grace"), options, run)
        rows = [json.loads(x) for x in (run.root/"batch_audit_samples.jsonl").read_text().splitlines()]
        outputs.append((result, rows))
    old, new = outputs
    assert old[0]["gradient"] == new[0]["gradient"]
    assert old[0]["actual_generation_tokens"] == new[0]["actual_generation_tokens"] == 48
    assert old[0]["full_gradient_calls"] == new[0]["full_gradient_calls"] == 12
    assert [r["full_token_ids"] for r in old[1]] == [r["full_token_ids"] for r in new[1]]
    arm = new[0]["interventions"]["p1"]
    assert arm["gradient"]["paired_mean_difference_norm"] < 1e-12
    assert arm["update"]["paired_mean_difference_norm"] == 0
    assert arm["token_proxy_efficiency"]["conditional_formula_raw_variance_times_expected_token_ratio"] == pytest.approx(1.)
    assert new[0]["actor_sha256_before"] == new[0]["actor_sha256_after"]
    assert new[0]["basis_head_baseline_optimizer_sha256_before"] == new[0]["basis_head_baseline_optimizer_sha256_after"]
    for row in new[1]:
        assert row["interventions"]["m0"]["p"] == row["p"]
        assert row["interventions"]["m0"]["z"] == row["z"]
        assert row["interventions"]["m0"]["m_norm_sq"] == 0
        assert row["interventions"]["p1"]["p"] == row["interventions"]["p1"]["z"] == 1
    for rep in range(3):
        assert len({r["interventions"]["uniform"]["p"] for r in new[1] if r["replicate"] == rep}) == 1


def test_batch_shape_uses_exact_step_log_then_matching_snapshot_not_next_n(tmp_path):
    payload = {"step": 40, "run_config": {"n_prompts": 4},
               "cost_control": {"last_step": 39, "last_n": 8, "next_n": 64}}
    cfg = {"n_prompts": 4, "n_start": 16, "batch_audit": {"shape": "training", "training_run": str(tmp_path)}}
    (tmp_path/"steps.jsonl").write_text(json.dumps({"step": 40, "n": 4, "n_problems": 4})+"\n")
    shape = resolve_batch_shape(cfg, payload)
    assert (shape["n_prompts"], shape["starts_per_prompt"]) == (4, 1)
    assert shape["source"] == "training_steps_at_checkpoint_step"
    (tmp_path/"steps.jsonl").unlink()
    shape = resolve_batch_shape(cfg, payload)
    assert shape["source"] == "configured_fallback_missing_matching_training_shape"
    assert shape["fixed_n"] == 16
    payload["cost_control"]["last_step"] = 40
    shape = resolve_batch_shape(cfg, payload)
    assert shape["fixed_n"] == 8 and shape["source"] == "checkpoint_cost_control_last_n"
    cfg["batch_audit"].update(shape="configured", n_prompts=2, starts_per_prompt=3)
    assert resolve_batch_shape(cfg, payload)["fixed_n"] == 6


def test_invalid_intervention_does_not_mutate_frozen_head(tmp_path):
    engines, layout, encode, opt = _fake_context()
    cfg = {"max_new_tokens": 4, "decision_tokens": 2,
           "batch_audit": {"interventions": {"bad": {"ridge_l2": 8}}}}
    with pytest.raises(ValueError, match="intervention"):
        audit_fixed_batches([MathRecord("a", "q", "1")], engines, layout, encode,
            {"optimizer": opt}, None, None, method_spec("full_pg"), cfg, RunDirectory(tmp_path))


def test_token_proxy_uses_raw_batch_variance_and_reports_undefined_denominator():
    from grace_gc.audit.batch_audit import token_proxy_efficiency, _interventions
    result = token_proxy_efficiency(2., 4., 1., 60, 100)
    assert result["empirical_raw_variance_times_expected_token_ratio"] == pytest.approx(1.2)
    assert result["conditional_formula_raw_variance_times_expected_token_ratio"] == pytest.approx(.9)
    for covariance in (0., None):
        result = token_proxy_efficiency(covariance, 4., 1., 60, 100)
        assert result["empirical_raw_variance_times_expected_token_ratio"] is None
        assert result["conditional_formula_raw_variance_times_expected_token_ratio"] is None
    with pytest.raises(ValueError, match="intervention beta"):
        _interventions({"batch_audit": {"interventions": {"invalid": {"beta": 0}}}})
