"""Teacher forcing and detached ST losses; GPU imports stay inside functions."""

from __future__ import annotations

import numpy as np

from grace_gc.core.suffix_transport import prefix_coefficients


def range_logprob(model, token_ids, start, pad_id, *, end=None, chunk_size=64):
    """Sum probabilities of tokens [start:end], keeping the full causal graph.

    Callers supply already trimmed tokens, including the actual stop token.
    Checkpoint softmax chunks to bound FP32 temporaries on long Qwen sequences.
    No KV/prefix activations are detached for a suffix-only loss.
    """
    import torch
    from torch.utils.checkpoint import checkpoint
    from grace_gc.backends.hf_actor import actor_forward

    end = len(token_ids) if end is None else int(end)
    if not 1 <= start <= end <= len(token_ids) or chunk_size < 1:
        raise ValueError("invalid token logprob range")
    out, ids, _ = actor_forward(model, [token_ids], pad_id)

    def chunk_sum(logits, targets):
        return logits.float().log_softmax(-1).gather(-1, targets.unsqueeze(-1)).sum(dtype=torch.float64)

    result = out.logits[:, :1].sum(dtype=torch.float64) * 0 if start == end else None
    for pos in range(start - 1, end - 1, chunk_size):
        stop = min(pos + chunk_size, end - 1)
        logits, targets = out.logits[:, pos:stop], ids[:, pos + 1:stop + 1]
        value = (checkpoint(chunk_sum, logits, targets, use_reentrant=False)
                 if torch.is_grad_enabled() else chunk_sum(logits, targets))
        result = value if result is None else result + value
    return result


def gradient_vector(logprob, named):
    import torch

    grads = torch.autograd.grad(logprob, [p for _, p in named], allow_unused=False)
    result = np.concatenate([g.detach().float().cpu().numpy().ravel() for g in grads]).astype(np.float64)
    if not np.isfinite(result).all():
        raise ValueError("nonfinite q/v LoRA gradient")
    return result


def backward_group(score, prefixes, prompt_len, suffix, alpha, advantages, donor,
                   weight, strength=1.0):
    """Accumulate descent .grad=-G, one short forward per receiving prefix.

    Combine the donor's direct prefix term with its suffix in one long forward.
    score(ids, start, end=None) returns a scalar token sum, with gradients.
    """
    coefficients = prefix_coefficients(alpha, advantages, donor, strength)
    # A_D*logp(full) + (c_D-A_D)*logp(prefix) equals c_D*logp(prefix)+A_D*logp(suffix).
    # Using explicit ranges avoids an extra donor prefix backward.
    donor_full = list(prefixes[donor]) + list(suffix)
    for i, prefix in enumerate(prefixes):
        if i != donor and coefficients[i] != 0:
            (-float(weight * coefficients[i]) * score(prefix, prompt_len)).backward()
    # Two scalar ranges share one full forward by asking score for both ranges.
    prefix_lp, suffix_lp = score(donor_full, prompt_len, split=len(prefixes[donor]))
    loss = -float(weight) * (float(coefficients[donor]) * prefix_lp
                             + float(advantages[donor]) * suffix_lp)
    loss.backward()


def split_logprob(model, token_ids, start, split, pad_id, chunk_size=64):
    """Both range sums from a single HF forward, with no activation detachment."""
    import torch
    from torch.utils.checkpoint import checkpoint
    from grace_gc.backends.hf_actor import actor_forward

    if not 1 <= start <= split <= len(token_ids):
        raise ValueError("invalid prefix/suffix split")
    out, ids, _ = actor_forward(model, [token_ids], pad_id)

    def chunk_sum(logits, targets):
        return logits.float().log_softmax(-1).gather(-1, targets.unsqueeze(-1)).sum(dtype=torch.float64)

    def total(begin, end):
        result = out.logits[:, :1].sum(dtype=torch.float64) * 0
        for pos in range(begin - 1, end - 1, chunk_size):
            stop = min(pos + chunk_size, end - 1)
            logits, targets = out.logits[:, pos:stop], ids[:, pos + 1:stop + 1]
            result = result + (checkpoint(chunk_sum, logits, targets, use_reentrant=False)
                               if torch.is_grad_enabled() else chunk_sum(logits, targets))
        return result

    return total(start, split), total(split, len(token_ids))
