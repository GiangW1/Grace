import numpy as np
import pytest

torch = pytest.importorskip("torch")

from grace_gc.core.layout import collect_lora_layout, layout_hash
from grace_gc.trainer.cpu_tiny import TinyLoRAActor, TinyTrainConfig, run_tiny_batch


def test_token_sum_covers_response_including_eos():
    actor = TinyLoRAActor()
    tokens = torch.tensor([[1, 2, 3, 4, actor.eos_id]])
    summed, token_lp, _ = actor.token_logprob_sum(tokens, prompt_len=2)
    # prompt=[1,2]; response logprobs are tokens 3,4,eos -> token_lp[:, 1:4]
    assert torch.allclose(summed[0], token_lp[0, 1:4].sum())


def test_token_sum_fixed_n_and_layout():
    result = run_tiny_batch(TinyTrainConfig(n=6, method="grace", seed=1, warmup=True))
    assert result["n"] == 6
    assert result["layout_dim"] > 0
    assert result["n_completed"] == 6
    actor = TinyLoRAActor()
    layout = collect_lora_layout(actor.named_lora_params())
    assert layout.dim == result["layout_dim"]
    assert len(layout_hash(layout)) == 64


def test_rng_counters_separated():
    result = run_tiny_batch(TinyTrainConfig(n=4, method="grace", seed=2))
    assert result["selection_counter"] >= 0
    assert result["token_counter"] >= 1
    assert result["audit_counter"] >= 0


def test_zero_survivors_ok():
    result = run_tiny_batch(TinyTrainConfig(n=4, method="uniform_ht", seed=3, beta=0.2, p_min=0.2))
    assert result["n"] == 4
    assert 0 <= result["n_completed"] <= 4


def test_leakage_negative_uses_suffix():
    leaked = run_tiny_batch(TinyTrainConfig(n=4, method="grace", seed=4, leak_from_suffix=True))
    clean = run_tiny_batch(TinyTrainConfig(n=4, method="grace", seed=4, leak_from_suffix=False))
    assert leaked["leak_warned"] is True
    assert clean["leak_warned"] is False
    assert not np.allclose(leaked["p"], leaked["p_prefix"])
    np.testing.assert_allclose(clean["p"], clean["p_prefix"])
    np.testing.assert_allclose(clean["p_prefix"], clean["p_prefix_again"])


def test_actor_can_move_on_p1():
    result = run_tiny_batch(TinyTrainConfig(n=8, method="full_pg", seed=5, warmup=True))
    assert result["actor_moved"] is True
    assert np.all(result["p"] == 1.0)
