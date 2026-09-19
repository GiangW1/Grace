from types import SimpleNamespace

import numpy as np
import pytest


def test_deviation_locates_absolute_response_and_prefix_suffix_positions():
    from grace_gc.backends.logprob_probe import logprob_error_summary
    result = logprob_error_summary([-.5, -1., -.2], [-.4, -.2, -.3], [3, 4, 5],
                                   prompt_len=7, prefix_tokens=1)
    assert result["signed_mean"] == pytest.approx(-.8/3)
    assert result["max_error_token"] == {"response_index": 1, "sequence_index": 8,
        "logit_index": 7, "token_id": 4, "phase": "continuation", "hf_logprob": -1.,
        "behavior_logprob": -.2, "signed_error": -.8, "absolute_error": .8}
    assert result["by_phase"]["prefix"]["n_tokens"] == 1
    assert result["by_phase"]["continuation"]["n_tokens"] == 2


def test_batch_metadata_does_not_change_scores_generate_or_mutate_records(monkeypatch):
    from grace_gc.backends import logprob_probe
    class Actor:
        training = False
        _grace_compute_dtype = "bfloat16"
        config = SimpleNamespace(_attn_implementation="sdpa")
        active_adapters = ["default"]
        peft_config = {"default": SimpleNamespace(r=16, lora_alpha=32, lora_dropout=0.)}
        def named_parameters(self):
            return iter([("q_proj.lora_A", SimpleNamespace(dtype="torch.float32", shape=(2, 2), device="cpu"))])
    monkeypatch.setattr(logprob_probe, "hf_response_logprobs", lambda *a, **k: np.array([-.5, -.8]))
    rec = SimpleNamespace(z=1, prompt_len=2, full_token_ids=[1, 2, 3, 4], problem_id="p",
        rollout_token_logprobs=[-.4, -.2], rollout_logprob_sum=-.6, prefix_tokens=1,
        request_seeds={"prefix": 17, "continuation": 18})
    out = logprob_probe.probe_behavior_batch(Actor(), None, [rec], 0,
        lora_request=SimpleNamespace(lora_name="step2", lora_int_id=2, lora_path="adapter"))
    assert out["mean_abs"] == pytest.approx(.35) and out["status"] == "compared"
    assert out["numerics"]["hf_compute_dtype"] == "bfloat16"
    assert out["numerics"]["lora_request"]["lora_int_id"] == 2
    assert out["worst_token"]["record_index"] == 0
    assert out["worst_token"]["response_index"] == 1
    assert out["details"][0]["deviation"]["by_phase"]["continuation"]["n_tokens"] == 1
    assert rec.rollout_token_logprobs == [-.4, -.2] and rec.full_token_ids == [1, 2, 3, 4]
