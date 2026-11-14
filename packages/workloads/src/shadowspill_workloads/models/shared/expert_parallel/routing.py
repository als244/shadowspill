"""Routed output and device-side summaries shared by the MoE architectures."""


def parallel_forward(experts, hidden):
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
    return routed.reshape_as(hidden), counts, logits.softmax(-1).sum(0)
