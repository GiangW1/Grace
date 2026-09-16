"""Optional verl DataProto adapters. Imports stay inside functions."""

from __future__ import annotations

from typing import Any


def try_import_dataproto():
    try:
        from verl import DataProto

        return DataProto
    except Exception:
        try:
            from verl.protocol import DataProto

            return DataProto
        except Exception:
            return None


def prompts_to_dataproto(prompt_token_ids: list[list[int]], pad_id: int = 0):
    DataProto = try_import_dataproto()
    if DataProto is None:
        raise ImportError("verl is not installed; cannot build DataProto")
    import torch

    width = max(len(x) for x in prompt_token_ids)
    ids = torch.full((len(prompt_token_ids), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(prompt_token_ids), width), dtype=torch.long)
    for i, seq in enumerate(prompt_token_ids):
        ids[i, : len(seq)] = torch.as_tensor(seq)
        mask[i, : len(seq)] = 1
    return DataProto.from_dict(tensors={"input_ids": ids, "attention_mask": mask})


def dataproto_to_token_lists(data, key: str = "input_ids") -> list[list[int]]:
    batch = data.batch[key]
    masks = data.batch.get("attention_mask")
    out = []
    for i in range(batch.shape[0]):
        row = batch[i].tolist()
        if masks is not None:
            keep = int(masks[i].sum().item())
            row = row[:keep]
        out.append(row)
    return out


def attach_grace_loss(worker, loss_fn) -> None:
    """Wire GRACE real-stream loss onto a verl TrainingWorker if present."""
    actor = getattr(worker, "actor", None)
    if actor is None or not hasattr(actor, "set_loss_fn"):
        raise TypeError("verl worker.actor.set_loss_fn is not available on this revision")
    actor.set_loss_fn(loss_fn)


def two_phase_with_worker(
    worker,
    prompt_token_ids: list[list[int]],
    decision_tokens: int,
    remaining_tokens: int,
    selected,
    pad_id: int = 0,
) -> dict[str, Any]:
    """Call worker.generate_sequences twice: prefixes, then selected suffixes."""
    if not hasattr(worker, "generate_sequences"):
        raise TypeError("verl worker must implement generate_sequences")
    prefixes = worker.generate_sequences(prompts_to_dataproto(prompt_token_ids, pad_id))
    prefix_ids = dataproto_to_token_lists(prefixes)
    chosen = [prefix_ids[i] for i, keep in enumerate(selected) if keep]
    continued = None
    if chosen:
        continued = worker.generate_sequences(prompts_to_dataproto(chosen, pad_id))
    _ = decision_tokens
    _ = remaining_tokens
    return {"prefixes": prefixes, "continued": continued, "prefix_ids": prefix_ids}
