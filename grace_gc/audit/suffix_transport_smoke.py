"""Executable correctness checks, shared by CPU tests and the real GPU smoke."""

from __future__ import annotations

import itertools

import numpy as np

from grace_gc.core.rng import IsolatedRNG
from grace_gc.core.suffix_transport import (aggregate_groups, prefix_coefficients,
                                            suffix_posterior, transport_gradient)


def math_smoke():
    """Enumerate every donor/suffix, no Monte Carlo or performance thresholds."""
    probabilities = np.array([[.75, .25], [.2, .8], [.45, .55]])
    prefix_grads = np.array([[1., 3.], [-2., .5], [.6, -1.]])
    suffix_grads = np.array([[[.2, -.4], [-.6, 1.2]], [[.8, -.3], [-.2, .075]],
                             [[.11, .22], [-.09, -.18]]])
    # Different receiver rewards catch accidental donor reward broadcasting.
    advantages = np.array([[.5, -.5], [-.5, .5], [.5, .5]])
    expected = np.mean(np.sum(probabilities[:, :, None] * advantages[:, :, None]
                            * (prefix_grads[:, None, :] + suffix_grads), axis=1), axis=0)
    errors = {}
    for strength in (0., .4, 1., -0.5):
        estimate = np.zeros(2)
        for donor, suffix in itertools.product(range(3), range(2)):
            alpha = suffix_posterior(np.log(probabilities[:, suffix]))
            gd = advantages[donor, suffix] * (prefix_grads[donor] + suffix_grads[donor, suffix])
            estimate += probabilities[donor, suffix] / 3 * transport_gradient(
                prefix_grads, gd, alpha, advantages[:, suffix], donor, strength)
        np.testing.assert_allclose(estimate, expected, atol=1e-14)
        errors[str(strength)] = float(np.max(np.abs(estimate - expected)))
    np.testing.assert_allclose(suffix_posterior([-100000., -100001., -np.inf]),
                               [1 / (1 + np.exp(-1)), 1 / (1 + np.exp(1)), 0])
    singleton = transport_gradient(prefix_grads[:1], [2., 3.], [1.], [.5], 0)
    np.testing.assert_array_equal(singleton, [2., 3.])
    # Ragged live groups + one natural finish retain the original N=4 denominator.
    np.testing.assert_allclose(aggregate_groups([(2, np.array([1., 2.])), (1, np.array([3., 4.]))],
                                                [np.array([5., 6.])], 4), [2.5, 3.5])
    left, right = IsolatedRNG.create(17), IsolatedRNG.create(17)
    left.random("selection", size=31)
    np.testing.assert_array_equal(left.integers("token", 0, 100, size=32),
                                  right.integers("token", 0, 100, size=32))
    return {"exact_enumeration_errors": errors, "checks": ["unbiased full-space expectation", "stable posterior",
            "lambda zero", "singleton", "fixed N with natural completion", "independent selection RNG"]}


def autograd_smoke(score, named, prefixes, prompt_len, suffix, *, rtol=1e-4, atol=1e-6):
    """Use unequal nonzero advantages even when the real rollout fails reward.

    Avoid rewarding correctness in a smoke: synthetic coefficients exercise all
    receiving prefixes. Uses the same loss/score implementation as training.
    """
    import torch
    from grace_gc.backends.suffix_transport import backward_group, gradient_vector

    if len(prefixes) < 2:
        raise ValueError("autograd smoke needs two teacher-forced prefixes")
    donor = 0
    advantages = np.linspace(-.3, .7, len(prefixes))
    with torch.no_grad():
        likelihoods = [float(score(prefix + suffix, len(prefix))) for prefix in prefixes]
    alpha = suffix_posterior(likelihoods)
    prefix_grads = np.stack([gradient_vector(score(prefix, prompt_len), named) for prefix in prefixes])
    gd = advantages[donor] * gradient_vector(score(prefixes[donor] + suffix, prompt_len), named)
    tests = {}
    for strength in (0., .4, 1.):
        for _, parameter in named:
            parameter.grad = None
        backward_group(score, prefixes, prompt_len, suffix, alpha, advantages, donor, .75, strength)
        actual = -np.concatenate([p.grad.detach().float().cpu().numpy().ravel() for _, p in named])
        expected = .75 * transport_gradient(prefix_grads, gd, alpha, advantages, donor, strength)
        # BF16 short/long sequence kernels may differ slightly; a relative L2
        # error is more informative than per-coordinate tolerance at zeros.
        error = float(np.linalg.norm(actual - expected))
        scale = max(float(np.linalg.norm(expected)), 1e-12)
        if error > atol + rtol * scale:
            raise AssertionError(f"ST loss/reference mismatch lambda={strength}: relative L2={error / scale}")
        tests[str(strength)] = {"relative_l2": error / scale, "ascent_dot_optimizer_negative_grad": float(actual @ expected)}
    # Same-forward decomposition checks include EOS and do not detach prefix activations.
    prefix_lp, suffix_lp = score(prefixes[0] + suffix, prompt_len, split=len(prefixes[0]))
    params = [p for _, p in named]
    gp = torch.autograd.grad(prefix_lp, params, retain_graph=True)
    gs = torch.autograd.grad(suffix_lp, params, retain_graph=True)
    gf = torch.autograd.grad(prefix_lp + suffix_lp, params)
    for a, b, full in zip(gp, gs, gf):
        torch.testing.assert_close(a + b, full, rtol=rtol, atol=atol)
    with torch.no_grad():
        combined = float(score(prefixes[0] + suffix, prompt_len))
        parts = float(prefix_lp.detach() + suffix_lp.detach())
    if not np.isclose(combined, parts, atol=atol, rtol=rtol):
        raise AssertionError("token sum differs from prefix+suffix decomposition")
    for _, parameter in named:
        parameter.grad = None
    return {"loss_checks": tests, "posterior": alpha.tolist(), "full_logprob": combined,
            "prefix_plus_suffix_logprob": parts, "n_qv_lora_tensors": len(named),
            "scope": "synthetic advantages on actual token graph; no scientific effect threshold"}


def behavior_probe(exp, prefixes, rollout, plen, stop_ids):
    from grace_gc.backends.logprob_probe import hf_response_logprobs, logprob_error_summary

    errors = []
    for i, tokens in enumerate(prefixes):
        behavior = (rollout.get("prefix_token_logprobs") or [])[i]
        hf = hf_response_logprobs(exp.actor, tokens, plen, exp.extra["pad_id"], exp.engines.eos_id, max_n=None)
        if len(behavior) != len(hf):
            raise AssertionError("sampled prefix logprobs missing/misaligned")
        # Some vLLM releases omit the named stop token and its logprob. The
        # two-phase engine restores that token; it must still be scored by HF.
        absent = [j for j, value in enumerate(behavior) if value is None]
        if absent and (absent != [len(behavior) - 1] or tokens[-1] not in stop_ids):
            raise AssertionError("non-EOS sampled logprobs are missing")
        count = len(behavior) - len(absent)
        if not np.isfinite(behavior[:count]).all():
            raise AssertionError("nonfinite sampled prefix logprobs")
        error = logprob_error_summary(hf[:count], behavior[:count], tokens[plen:plen + count], plen)
        error["restored_eos_without_behavior_logprob"] = bool(absent)
        limit = float(exp.cfg["suffix_transport"].get("smoke_logprob_mean_abs_tolerance", .5))
        if error["mean_abs"] is not None and error["mean_abs"] > limit:
            raise AssertionError(f"HF/behavior mean absolute logprob mismatch {error['mean_abs']} > {limit}; check adapter/token mapping")
        errors.append(error)
    return errors


def gpu_smoke(experiment):
    """Real rollout -> actor scoring/backward -> checkpoint and adapter probe."""
    from grace_gc.trainer.checkpoint import load_checkpoint
    from grace_gc.versions import sha256_named

    result = math_smoke()
    exp = experiment
    prefixes, finished = exp.prefixes([0], 2)
    prefix_rollout = dict(exp.engines.last_rollout)
    selected = [not ended for ended in finished]
    full = exp.continue_batch(prefixes, selected) if any(selected) else list(prefixes)
    stop_ids = as_stop_ids_for_smoke(exp.engines.eos_id)
    plen = len(exp.prompts[0])
    # Synthetic token boundaries ensure EOS and both prefix gradients are tested
    # even if this model happens to finish immediately in the real rollout.
    response = next((tokens[plen:] for tokens in full if tokens and len(tokens) > plen + 2), None)
    ordinary = [t for t in (response or prefixes[0][plen:]) if t not in stop_ids]
    if not ordinary:
        ordinary = [next(t for t in range(exp.tokenizer.vocab_size) if t not in stop_ids)]
    a = (ordinary * 4)[:2]
    b = list(reversed(a)) if a[0] != a[1] else [a[0], (a[0] + 1) % exp.tokenizer.vocab_size]
    if b[-1] in stop_ids:
        b[-1] = a[-1]
    fixture_prefixes = [exp.prompts[0] + a, exp.prompts[0] + b]
    suffix = (ordinary * 4)[:3] + [stop_ids[0]]
    # Decomposition runs at the same compute precision as training. Tight exact
    # expectation tests above are FP64; CUDA causal kernels have roundoff.
    result["autograd"] = autograd_smoke(exp.score, exp.named, fixture_prefixes, plen, suffix, rtol=.03, atol=1e-5)
    path = exp.checkpoint(0, None)
    payload = load_checkpoint(path)
    for name, parameter in exp.named:
        np.testing.assert_array_equal(payload["actor"][name], parameter.detach().float().cpu().numpy())
    if sha256_named(exp.named) != exp.initial_sha:
        raise AssertionError("smoke changed actor weights")
    result["checkpoint_roundtrip"] = True
    result["actor_unchanged"] = True
    # Compare actual sampled token scores to HF. Missing entries/large mismatch
    # can identify wrong adapter or EOS mapping before a long audit.
    result["hf_behavior_logprob_errors"] = behavior_probe(exp, prefixes, prefix_rollout, plen, stop_ids)
    result["behavior_scores_note"] = "Roundoff is reported, not an exact HF/vLLM kernel equivalence claim. Review before full-length runs."
    for i, tokens in enumerate(full):
        exp.observe(tokens if tokens is not None else prefixes[i], 0, phase="smoke")
    actual = [tokens if tokens is not None else prefixes[i] for i, tokens in enumerate(full)]
    longest = max(actual, key=len)
    full_gradient = exp.gradient(longest, plen)
    result["actual_trajectory_backward"] = {"response_tokens": len(longest) - plen,
                                             "full_gradient_norm": float(np.linalg.norm(full_gradient))}
    # Exercise a real optimizer update and subsequent adapter sync, then restore
    # the shared starting actor. Smoke owns this actor; no experiment checkpoint
    # is overwritten. A normalized SGD step avoids Adam/clipping ambiguities.
    import torch
    original = [parameter.detach().clone() for _, parameter in exp.named]
    optimizer = torch.optim.SGD([p for _, p in exp.named], lr=.001)
    old_id = exp.extra["lora_id"]
    try:
        optimizer.zero_grad(set_to_none=True)
        before = exp.score(fixture_prefixes[0] + suffix, plen)
        (-before).backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for _, p in exp.named):
            raise AssertionError("invalid smoke optimizer gradient")
        norm = torch.nn.utils.clip_grad_norm_([p for _, p in exp.named], 1.)
        if not torch.isfinite(norm) or float(norm) == 0:
            raise AssertionError("synthetic trajectory has no finite ascent direction")
        optimizer.step()
        if sha256_named(exp.named) == exp.initial_sha:
            raise AssertionError("optimizer update did not change FP32 actor masters")
        exp.timed("smoke_update_sync", exp.extra["sync"])
        if exp.extra["lora_id"] <= old_id:
            raise AssertionError("updated adapter id was reused")
        # Use actual two-phase rollout after update to exercise the loaded adapter.
        updated, _ = exp.prefixes([0], 1)
        result["updated_hf_behavior_logprob_errors"] = behavior_probe(exp, updated, exp.engines.last_rollout, plen, stop_ids)
        result["optimizer_and_updated_adapter_sync"] = True
    finally:
        with torch.no_grad():
            for (_, parameter), saved in zip(exp.named, original):
                parameter.copy_(saved)
                parameter.grad = None
        exp.timed("smoke_restore_sync", exp.extra["sync"])
    if sha256_named(exp.named) != exp.initial_sha:
        raise AssertionError("smoke failed to restore initial actor")
    result["mode"] = "smoke"
    return result


def as_stop_ids_for_smoke(ids):
    from grace_gc.data.tokenize import as_stop_ids

    return as_stop_ids(ids)
