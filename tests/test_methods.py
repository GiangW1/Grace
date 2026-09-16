import numpy as np
import pytest

from grace_gc.trainer.grace_step import StartRecord, assemble_ghat, decide_continuation
from grace_gc.trainer.methods import METHOD_NAMES, method_spec, uniform_p


def test_all_methods_exist():
    for name in METHOD_NAMES:
        spec = method_spec(name)
        assert spec.name in METHOD_NAMES


def test_assemble_full_pg_uses_global_n():
    spec = method_spec("full_pg")
    recs = [
        StartRecord("a", False, 1.0, 1.0, np.zeros(2), 1, 1, 1.0, 1.0, np.array([2.0, 0.0]), True),
        StartRecord("b", False, 0.5, 0.0, np.zeros(2), 1, 1, None, None, None, False),
    ]
    ghat = assemble_ghat(spec, recs, np.eye(2), 2)
    np.testing.assert_allclose(ghat, np.array([1.0, 0.0]))


def test_full_pg_p_is_one():
    spec = method_spec("full_pg")
    p, _ = decide_continuation(spec, np.ones(4), np.ones(4), np.zeros(4, dtype=bool), 0.5, 0.2, warmup=False)
    assert np.all(p == 1.0)


def test_uniform_ht_zero_predictor():
    spec = method_spec("uniform_ht")
    u = np.eye(2)
    recs = [
        StartRecord("a", False, 0.5, 1.0, np.array([9.0, 9.0]), 1, 1, 1.0, 0.5, np.array([1.0, 0.0]), True),
        StartRecord("a", False, 0.5, 0.0, np.array([9.0, 9.0]), 1, 1, None, None, None, False),
    ]
    ghat = assemble_ghat(spec, recs, u, 2)
    # m=0, only first start contributes G/p / N
    np.testing.assert_allclose(ghat, np.array([1.0, 0.0]))


def test_grace_uses_correction():
    spec = method_spec("grace")
    u = np.eye(2)
    recs = [
        StartRecord("a", False, 1.0, 1.0, np.array([0.0, 0.0]), 1, 1, 1.0, 1.0, np.array([2.0, 0.0]), True),
        StartRecord("b", False, 0.5, 0.0, np.array([1.0, 0.0]), 1, 1, None, None, None, False),
    ]
    ghat = assemble_ghat(spec, recs, u, 2)
    np.testing.assert_allclose(ghat, np.array([1.5, 0.0]))


def test_grpo_short_overrides():
    spec = method_spec("grpo_short")
    assert spec.max_new_tokens == 1024
    assert spec.starts_per_prompt == 16
    assert spec.n_start is None
    assert spec.objective == "grpo"


def test_uniform_p_clip():
    assert uniform_p(4, 0.5, 0.2) == 0.5
    assert uniform_p(4, 0.1, 0.2) == 0.2


def test_reward_cv_uses_learned_reward_risk():
    pytest.importorskip("torch")
    import torch

    from grace_gc.predictor.features import reward_risk_features
    from grace_gc.predictor.heads import PredictorHeads

    torch.manual_seed(0)
    spec = method_spec("reward_cv")
    heads = PredictorHeads(in_dim=4, k=2, hidden_coord=8, hidden_risk=8)
    low = reward_risk_features(0.1, 0.9, 8.0, 0.1)
    high = reward_risk_features(0.9, 0.1, 8.0, 0.9)
    x = np.stack([low, high, low, high])
    heads.train_reward_risk(x, np.array([8.0, 0.2, 8.0, 0.2]), np.ones(4), epochs=40)
    risk = heads.forward_reward_risk(np.stack([low, high]))
    p, _ = decide_continuation(spec, risk, np.ones(2), np.zeros(2, dtype=bool), 0.5, 0.2, warmup=False)
    assert risk[0] > risk[1]
    assert p[0] > p[1]


def test_reward_cv_success_head_is_prefix_dependent():
    pytest.importorskip("torch")
    import torch

    from grace_gc.predictor.heads import PredictorHeads

    torch.manual_seed(1)
    heads = PredictorHeads(in_dim=4, k=2, hidden_coord=8, hidden_risk=8)
    low = np.zeros((2, 4), dtype=np.float64)
    high = np.ones((2, 4), dtype=np.float64)
    heads.train_success(np.vstack([low, high]), np.array([0.0, 0.0, 1.0, 1.0]), np.ones(4), epochs=40)
    q_low = heads.forward_success(low)
    q_high = heads.forward_success(high)
    assert float(q_high.mean()) > float(q_low.mean())
