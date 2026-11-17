"""Masked router summaries and bounded token outputs shared by MoE models."""

import torch
from mlops.lora.routing import routing, routing_logits

from .token_head import target_logprobs


def masked_statistics(hidden, weight, mask, *, top_k, routing_mode):
    """Use the provider's router precision/selection with fixed-size buffers.

    The fused expert operation currently exposes only all-row summaries. This
    extra router projection supplies masked statistics without repeating the
    transformer or changing expert output. Shapes depend on T, E and K, never
    on the number of valid rows or the distribution of expert assignments.
    """
    flat = hidden.reshape(-1, hidden.shape[-1])
    logits = routing_logits(flat, weight)
    _, ids = routing(logits, top_k, routing_mode)
    return masked_summaries(logits, ids, mask)


def masked_summaries(logits, ids, mask):
    """Reduce already selected routes without another projection or top-k call."""
    active = mask.reshape(-1)
    assignments = active[:, None].expand_as(ids).reshape(-1).long()
    counts = torch.zeros(logits.shape[1], dtype=torch.int64, device=logits.device)
    counts = counts.scatter_add(0, ids.reshape(-1).long(), assignments)
    probabilities = torch.where(active[:, None], logits.softmax(-1), 0.0).sum(0)
    return counts, probabilities


def token_features(
    model,
    tokens,
    sequence_lengths=None,
    *,
    router_mask=None,
    router_statistics=False,
):
    """Model-defined head inputs and optional router statistics in one pass."""
    if not router_statistics:
        hidden, _ = model.hidden(tokens, sequence_lengths)
        return {"head_hidden": hidden, "head": model.lm_head}
    hidden, _, statistics = model.hidden(
        tokens,
        sequence_lengths,
        return_metrics=True,
        router_mask=router_mask,
        detach_metrics=False,
    )
    return {
        "head_hidden": hidden,
        "head": model.lm_head,
        "router_counts": statistics["expert_counts"],
        "router_probability_sum": statistics["probability_sum"],
        "router_rows": tokens.new_tensor(tokens.numel())
        if router_mask is None
        else router_mask.sum(),
        "router_top_k": model.config.top_k,
    }


def token_outputs(
    model,
    tokens,
    targets=None,
    sequence_lengths=None,
    *,
    router_mask=None,
    head_chunk_size=None,
):
    """Sufficient router statistics plus optional target logprobs in one pass."""
    features = token_features(
        model,
        tokens,
        sequence_lengths,
        router_mask=router_mask,
        router_statistics=True,
    )
    result = {
        name: value for name, value in features.items() if name.startswith("router_")
    }
    if targets is not None:
        result["logprobs"] = target_logprobs(
            features["head_hidden"],
            features["head"],
            targets,
            chunk_size=head_chunk_size,
        )
    return result
