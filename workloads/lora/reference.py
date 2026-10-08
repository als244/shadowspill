"""Independent ordinary-PyTorch expert LoRA math; no operation-library imports."""

import torch
from torch.nn import functional as F


def expert_lora(
    hidden,
    residual,
    router_weight,
    gate_up,
    down,
    a13,
    b13,
    a2,
    b2,
    *,
    top_k,
    routing_mode,
    scale,
):
    flat = hidden.reshape(-1, hidden.shape[-1])
    logits = F.linear(flat.to(router_weight.dtype), router_weight.T).float()
    probabilities = logits.softmax(-1)
    _, order = torch.sort(logits, descending=True, stable=True)
    ids = order[:, :top_k]
    weights = (
        probabilities.gather(1, ids)
        if routing_mode == "softmax_then_topk"
        else logits.gather(1, ids).softmax(-1)
    )
    counts = torch.zeros(gate_up.shape[0], dtype=torch.int64, device=hidden.device)
    counts.scatter_add_(0, ids.flatten(), torch.ones_like(ids.flatten()))
    auxiliary = (
        gate_up.shape[0]
        * ((counts.float() / ids.numel()) * probabilities.mean(0)).sum()
    )
    output = torch.zeros_like(flat, dtype=torch.float32)
    for expert in range(gate_up.shape[0]):
        h13 = flat @ gate_up[expert] + scale * (
            flat @ a13[expert].to(flat.dtype)
        ) @ b13[expert].to(flat.dtype)
        gate, up = h13.chunk(2, -1)
        active = F.silu(gate.float()).to(gate.dtype) * up
        y = active @ down[expert] + scale * (active @ a2[expert].to(active.dtype)) @ b2[
            expert
        ].to(active.dtype)
        coefficient = (weights * (ids == expert)).sum(-1)
        output = output + coefficient[:, None] * y.float()
    return (
        (residual.reshape_as(flat).float() + output)
        .to(hidden.dtype)
        .reshape_as(hidden),
        auxiliary,
        counts,
        probabilities.sum(0),
    )
