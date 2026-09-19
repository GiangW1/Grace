"""CPU regressions for GPU-path semantics; these are not GPU performance tests."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from grace_gc.backends import hf_actor, logprob_probe
from grace_gc.trainer.actor_update import real_stream_backward_each


class RecordingActor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(4, 4, bias=False)
        self.calls = []

    def forward(self, input_ids, attention_mask, output_hidden_states=False, **kwargs):
        self.calls.append((input_ids.shape[1], torch.is_grad_enabled(), kwargs))
        x = torch.nn.functional.one_hot(input_ids % 4, 4).float()
        logits = self.q_proj(x)
        return SimpleNamespace(logits=logits, hidden_states=(x, x, x))


def test_zero_coefficients_skip_forward_without_skipping_nonzero():
    calls = []
    weight = torch.nn.Parameter(torch.tensor(2.0))

    def lp(i):
        calls.append(i)
        return weight * (i + 1)

    loss = real_stream_backward_each(lp, np.array([0., 2., 0.]), np.ones(3),
                                     np.ones(3), 3, np.ones(3, dtype=bool))
    assert calls == [1]
    assert float(loss) == pytest.approx(-8 / 3)
    assert float(weight.grad) == pytest.approx(-4 / 3)


def test_hf_probe_has_no_graph_or_unused_tail_and_disables_cache():
    actor = RecordingActor()
    ids = [1, 2] + [3] * 30
    scores = logprob_probe.hf_response_logprobs(actor, ids, 2, 0, max_n=4)
    assert len(scores) == 4
    assert actor.calls == [(6, False, {'use_cache': False})]


def test_probe_keeps_positions_when_vllm_entry_is_missing(monkeypatch):
    from grace_gc.backends import vllm_two_phase

    class SP:
        def __init__(self, **kwargs):
            pass

    raw = [None, {2: -0.1}, None, {3: -0.3}]
    llm = SimpleNamespace(generate=lambda *a, **k: [SimpleNamespace(prompt_logprobs=raw)])
    monkeypatch.setattr(vllm_two_phase, 'build_sampling_params', lambda *a, **k: SP())
    monkeypatch.setattr(vllm_two_phase, '_vllm_prompts', lambda x: x)
    scores = logprob_probe.vllm_prompt_logprobs(llm, [1, 2, 3, 3], max_n=3)
    assert len(scores) == 3
    assert scores[0] == -0.1 and np.isnan(scores[1]) and scores[2] == -0.3


def test_autocast_keeps_fp32_master_and_optimizer_state():
    actor = RecordingActor()
    actor._grace_compute_dtype = 'bfloat16'
    out, _, _ = hf_actor.actor_forward(actor, [[1, 2, 3]], 0)
    assert out.logits.dtype == torch.bfloat16
    opt = torch.optim.AdamW(actor.parameters(), lr=1e-3)
    out.logits.float().square().sum().backward()
    opt.step()
    param = next(actor.parameters())
    assert param.dtype == torch.float32 and param.grad.dtype == torch.float32
    assert opt.state[param]['exp_avg'].dtype == torch.float32


def test_feature_modes_keep_legacy_and_expose_decision_entropy():
    hidden = np.ones((6, 2))
    ent = np.array([10., 20., 30., 1., 2., 4.])
    legacy, _, _ = hf_actor._features_from_hidden(
        [1]*6, hidden, hidden, ent, 3, .5, None, feature_mode='legacy')
    response, _, _ = hf_actor._features_from_hidden(
        [1]*6, hidden, hidden, ent, 3, .5, None, feature_mode='response')
    decision, _, _ = hf_actor._features_from_hidden(
        [1]*6, hidden, hidden, ent, 3, .5, None, feature_mode='decision')
    assert legacy[-5] == pytest.approx(12.6)
    assert response[-5] == pytest.approx((30+1+2)/3)
    assert decision[-5] == 4
    assert legacy[-2] == 6 and response[-2] == decision[-2] == 3


def test_zero_real_gradient_still_applies_adam_momentum():
    from grace_gc.core.layout import collect_lora_layout
    from grace_gc.trainer.actor_update import apply_correction_clip_step

    param = torch.nn.Parameter(torch.tensor([1.]))
    named = [('q_proj.lora_A', param)]
    opt = torch.optim.AdamW([param], lr=.1, weight_decay=0.)
    param.grad = torch.tensor([-1.])
    opt.step()
    before = param.detach().clone()
    opt.zero_grad(set_to_none=True)
    stats = {}
    apply_correction_clip_step(named, collect_lora_layout(named), np.ones((1, 1)),
                               np.zeros((1, 1)), np.ones(1), np.ones(1), 1, opt,
                               use_correction=False, log_geometry=True, stats=stats)
    assert param.item() > before.item()
    assert stats['parameter_update_norm'] > 0
    assert stats['update_ascent_cosine'] is None


def test_shared_actor_initialization_loads_base_and_lora_only(tmp_path):
    from grace_gc.trainer.checkpoint import save_checkpoint
    from grace_gc.trainer.cpu_tiny import TinyLoRAActor
    from grace_gc.trainer.initialization import initialize_actor
    from grace_gc.trainer.state_io import numpy_module_state

    torch.manual_seed(31)
    original = TinyLoRAActor()
    payload = {'actor': numpy_module_state(original.named_lora_params()),
               'actor_full': numpy_module_state(original.named_all_params()), 'step': 9}
    path = tmp_path / 'shared.npz'
    save_checkpoint(path, payload)
    torch.manual_seed(78)
    target = TinyLoRAActor()
    info = initialize_actor(target, {'init_checkpoint': str(path)})
    assert info['source_step'] == 9
    for name, value in target.named_all_params():
        np.testing.assert_array_equal(value.detach().numpy(), payload['actor_full'][name])
    with pytest.raises(ValueError, match='different starting states'):
        initialize_actor(target, {'init_checkpoint': str(path), 'resume': str(path)})


def test_sampled_behavior_probe_uses_original_scores_without_generation():
    actor = RecordingActor()
    tokens = [1, 2, 3, 2, 1]
    scores = logprob_probe.hf_response_logprobs(actor, tokens, 2, 0, max_n=0).tolist()
    rec = SimpleNamespace(z=1, prompt_len=2, full_token_ids=tokens, problem_id='q',
                          rollout_token_logprobs=scores, rollout_logprob_sum=sum(scores))
    result = logprob_probe.probe_first_completed(actor, None, [rec], 0, max_n=0)
    assert result['source'] == 'sampled_behavior'
    assert result['full_response_checked']
    assert result['max_abs'] == result['hf_minus_behavior_sum'] == 0


def test_optional_probe_reports_failures_and_nonfinite(monkeypatch):
    rec = SimpleNamespace(z=1, prompt_len=1, full_token_ids=[1, 2, 3], problem_id='q',
                          rollout_token_logprobs=[float('nan'), -.4], rollout_logprob_sum=None)
    monkeypatch.setattr(logprob_probe, 'hf_response_logprobs', lambda *a, **k: np.array([-.1, -.4]))
    assert logprob_probe.probe_first_completed(None, None, [rec], 0)['status'] == 'nonfinite'
    def failed(*args, **kwargs):
        raise RuntimeError('synthetic probe failure')
    monkeypatch.setattr(logprob_probe, 'hf_response_logprobs', failed)
    assert logprob_probe.probe_first_completed(None, None, [rec], 0)['status'] == 'unavailable'


def test_nonfinite_actor_gradient_does_not_advance_optimizer():
    from grace_gc.core.layout import collect_lora_layout
    from grace_gc.trainer.actor_update import apply_correction_clip_step
    param = torch.nn.Parameter(torch.tensor([1.]))
    named = [('q_proj.lora_A', param)]
    opt = torch.optim.AdamW([param], lr=.1)
    param.grad = torch.tensor([float('nan')])
    with pytest.raises(ValueError, match='nonfinite'):
        apply_correction_clip_step(named, collect_lora_layout(named), np.eye(1), np.zeros((1, 1)),
                                   np.ones(1), np.ones(1), 1, opt, use_correction=False)
    assert param.item() == 1 and not opt.state
