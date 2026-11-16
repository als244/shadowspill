"""Routed output and device-side summaries shared by the MoE architectures."""

from ..router_statistics import masked_summaries


def parallel_forward(experts, hidden, *, router_mask=None):
    from mlops.expert_parallel.quack.router import route_op
    from mlops.kernels.router import routed_logits

    c = experts.config
    flat = hidden.reshape(-1, c.model_dim)
    logits = routed_logits(
        flat,
        experts.router_weight,
        weight_grad_dtype=c.router_weight_grad_dtype,
    )
    weights, ids, counts = route_op(logits, c.top_k, c.renormalize_topk)
    routed = experts(flat, expert_ids=ids, routing_weights=weights)
    if router_mask is None:
        probabilities = logits.softmax(-1).sum(0)
    else:
        counts, probabilities = masked_summaries(logits, ids, router_mask)
    return routed.reshape_as(hidden), counts, probabilities
