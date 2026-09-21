"""Mandatory backward supplies unweighted audit G without another forward."""

import numpy as np
import pytest
import torch

from grace_gc.core.layout import collect_lora_layout, pack_grads
from grace_gc.trainer.actor_update import real_stream_backward_and_audit, real_stream_backward_each


def run_case(mask, p, z, advantage):
    theta = torch.nn.Parameter(torch.tensor([.3, -.7], dtype=torch.float64))
    unused = torch.nn.Parameter(torch.tensor([2.], dtype=torch.float64))
    named = [('q_proj.lora_A', theta), ('v_proj.lora_B', unused)]
    layout = collect_lora_layout(named)
    calls = []

    def score(i):
        calls.append(i)
        return (i + 1) * theta[0].square() + theta[1].sin()

    loss, audits = real_stream_backward_and_audit(
        score, np.array(advantage), np.array(p), np.array(z), len(p),
        np.array(z) == 1, named, layout, np.array(mask),
    )
    packed = pack_grads([(name, param.grad.numpy() if param.grad is not None
                          else np.zeros(tuple(param.shape))) for name, param in named], layout)
    return float(loss), audits, packed, calls


def test_ht_gradient_reuses_one_forward_and_keeps_audit_g_unweighted():
    advantage, p, z = [.5, -.2, .7, 0.], [.25, .8, .5, 1.], [1., 1., 0., 1.]
    loss, audits, grad, calls = run_case([True]*4, p, z, advantage)
    assert calls == [0, 1]  # no forward for stoppers or exact zero G
    assert set(audits) == {0, 1, 3}
    expected = []
    for i, a in enumerate(advantage):
        g = np.array([a * (i+1) * .6, a * np.cos(-.7), 0.])
        if i in audits:
            np.testing.assert_allclose(audits[i], g, atol=1e-14)
        expected.append(-z[i] * g / (len(p) * p[i]))
    np.testing.assert_allclose(grad, np.sum(expected, axis=0), atol=1e-14)
    target_loss = sum(-a * zz / (4 * pp) * ((i+1)*.3**2 + np.sin(-.7))
                      for i, (a, pp, zz) in enumerate(zip(advantage, p, z)))
    assert loss == pytest.approx(target_loss)


def test_audit_mask_does_not_change_actor_gradient_and_p1_is_full_pg():
    args = ([1.]*3, [1.]*3, [.5, -.5, .25])
    plain = run_case([False]*3, *args)
    audited = run_case([True]*3, *args)
    np.testing.assert_array_equal(plain[2], audited[2])
    assert plain[0] == audited[0]
    assert plain[3] == audited[3] == [0, 1, 2]
    np.testing.assert_allclose(audited[2], -np.mean(list(audited[1].values()), axis=0))


def test_zero_advantages_keep_exact_zero_labels_without_backward():
    loss, audits, grad, calls = run_case([True, True], [.2, 1.], [0., 1.], [0., 0.])
    assert loss == 0 and calls == [] and set(audits) == {1}
    np.testing.assert_array_equal(audits[1], np.zeros(3))
    np.testing.assert_array_equal(grad, np.zeros(3))


def test_fp32_reused_backward_matches_existing_weighted_loss_gradient():
    torch.manual_seed(8)
    x = torch.randn(5, 7)
    initial = torch.randn(7)
    advantage = np.array([.2, -.5, 0., .4, -.1])
    p, z = np.array([.2, .7, 1., .4, .8]), np.array([1., 1., 1., 0., 1.])
    results = []
    for reuse in (False, True):
        param = torch.nn.Parameter(initial.clone())
        named = [('q_proj.lora_A', param)]
        score = lambda i: torch.log_softmax(x[i] * param, dim=0)[i]
        if reuse:
            real_stream_backward_and_audit(score, advantage, p, z, 5, z == 1,
                                           named, collect_lora_layout(named), np.ones(5, dtype=bool))
        else:
            real_stream_backward_each(score, advantage, p, z, 5, z == 1)
        results.append(param.grad)
    torch.testing.assert_close(results[0], results[1], atol=2e-8, rtol=2e-6)
