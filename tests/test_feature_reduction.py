"""CPU equivalence and transfer-size checks for HF prefix feature reduction."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from grace_gc.backends import hf_actor


class FeatureActor(torch.nn.Module):
    hidden_size = 5

    def __init__(self, dtype=torch.float32, uniform_binary=False):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.dtype = dtype
        self.uniform_binary = uniform_binary

    def forward(self, input_ids, attention_mask, output_hidden_states=False, use_cache=False):
        assert output_hidden_states and not use_cache and not torch.is_grad_enabled()
        pos = torch.arange(input_ids.shape[1], device=input_ids.device).float()[None, :]
        tokens = input_ids.float()
        offsets = torch.arange(self.hidden_size, device=input_ids.device).float()
        values = tokens[..., None] * (offsets + 1) + pos[..., None] * .13
        # Padded positions deliberately have nonzero hidden states and entropy.
        hiddens = tuple(h.to(self.dtype) for h in (
            values * .5, values + 3, values.cos(), values * 2, values.sin()))
        if self.uniform_binary:
            logits = torch.zeros((*input_ids.shape, 2), device=input_ids.device)
        else:
            logits = torch.stack((.2 * tokens, -.1 * pos.expand_as(tokens),
                                  (tokens + pos).sin()), dim=-1)
        return SimpleNamespace(logits=logits.to(self.dtype), hidden_states=hiddens)


def legacy_bundle(actor, prefixes, prompt_lens, baselines, pad_id, eos_id, mode, batch_size):
    """The original path: copy full hidden sequences, then reduce with NumPy."""
    feats, costs, prompt_feats = [], [], []
    with torch.no_grad():
        for start in range(0, len(prefixes), max(1, batch_size)):
            batch = prefixes[start:start + max(1, batch_size)]
            out, _, _ = hf_actor.actor_forward(actor, batch, pad_id, output_hidden_states=True)
            chunks = []
            for offset in range(0, out.logits.shape[1], 64):
                logp = out.logits[:, offset:offset + 64].float().log_softmax(-1)
                chunks.append((-(logp.exp() * logp).sum(dim=-1)).cpu().numpy())
            entropy = np.concatenate(chunks, axis=1)
            last = out.hidden_states[-1].float().cpu().numpy()
            mid = out.hidden_states[len(out.hidden_states) // 2].float().cpu().numpy()
            for row, prefix in enumerate(batch):
                i = start + row
                feat, cost, prompt_feat = hf_actor._features_from_hidden(
                    prefix, last[row], mid[row], entropy[row], prompt_lens[i], baselines[i],
                    pad_id if eos_id is None else eos_id, feature_mode=mode)
                feats.append(feat)
                costs.append(cost)
                prompt_feats.append(prompt_feat)
    return {"features": np.stack(feats), "cost_feat": np.asarray(costs, dtype=np.float64),
            "prompt_features": np.stack(prompt_feats)}


@pytest.mark.parametrize("mode", ["legacy", "response", "decision"])
@pytest.mark.parametrize("batch_size", [1, 2, 8])
@pytest.mark.parametrize("eos_id", [9, None, [9, 6]])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float64])
def test_bundle_matches_numpy_reference(mode, batch_size, eos_id, dtype):
    prefixes = [[9, 2, 3, 4, 9, 8, 7], [5] * 70 + [9, 2], [9], [2, 3],
                [2, 0, 4, 5], [2, 3, 4, 6, 5]]
    prompt_lens = [2, 67, 1, 2, 1, 0]
    baselines = [.1, .5, -.2, 1., .9, .25]
    actor = FeatureActor(dtype)
    expected = legacy_bundle(actor, prefixes, prompt_lens, baselines, 0, eos_id, mode, batch_size)
    actual = hf_actor.prefix_feature_bundle(
        actor, prefixes, prompt_lens, baselines, 0, eos_id=eos_id,
        feature_mode=mode, batch_size=batch_size)
    assert actual.keys() == expected.keys()
    for name in expected:
        assert actual[name].dtype == np.float64
        np.testing.assert_allclose(actual[name], expected[name], rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(actual["cost_feat"], expected["cost_feat"])


@pytest.mark.parametrize("mode", ["legacy", "response", "decision"])
def test_entropy_threshold_and_empty_response_match_reference(mode):
    actor = FeatureActor(uniform_binary=True)
    args = (actor, [[1], [1, 2, 3]], [1, 2], [.5, .5], 0)
    expected = legacy_bundle(*args, 9, mode, 2)
    actual = hf_actor.prefix_feature_bundle(*args, eos_id=9, feature_mode=mode, batch_size=2)
    # FP32 entropy(log(2)) lies just above the FP64 threshold used by NumPy.
    np.testing.assert_array_equal(actual["features"][:, -3], expected["features"][:, -3])
    np.testing.assert_array_equal(actual["prompt_features"][:, -3],
                                  expected["prompt_features"][:, -3])
    if mode == "response":
        np.testing.assert_array_equal(actual["features"][0, -5:-2], np.zeros(3))


def test_only_reduced_feature_values_are_copied_to_cpu(monkeypatch):
    copies = []
    original_cpu = torch.Tensor.cpu

    def record_cpu(tensor, *args, **kwargs):
        copies.append(tensor.numel())
        return original_cpu(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", record_cpu)
    prefixes = [[2] * 137, [3] * 70, [4] * 85]
    hf_actor.prefix_feature_bundle(FeatureActor(), prefixes, [7, 3, 65], [.5] * 3, 0,
                                   eos_id=9, feature_mode="decision", batch_size=2)
    feature_width = 3 * FeatureActor.hidden_size + 5
    assert sum(copies) <= 2 * len(prefixes) * feature_width
