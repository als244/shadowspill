"""Routed-expert LoRA: one path for every MoE module that declares its
experts.

A module declares its routed experts by ``expert_shapes``: the gate/up and
down banks' shapes, ``[E, out, in]`` in ``nn.Linear`` orientation, from its
configuration. ``add_factors`` gives it an independent factor pair per expert
and projection, ``A [E, rank, in]`` and ``B [E, out, rank]``, sized from that
declaration rather than from the weights, which a caller that partitions
parameters as their modules are constructed may already have emptied, and
sets the module's ``lora_config``; the module's forward then runs its experts
with the factors. ``expert_lora`` is the same computation in ordinary
PyTorch, the reference the pytorch implementations run and the tests hold
the operation library to; this module imports no operation library.
"""

import torch
from torch import nn
from torch.nn import functional as F

#: The factors' attribute names: gate/up then down, A then B.
FACTORS = ("lora_gate_up_a", "lora_gate_up_b", "lora_down_a", "lora_down_b")


def add_factors(module: nn.Module, config) -> None:
    """Register the routed experts' LoRA factors on ``module``, initialized,
    beside its base weights, and set its ``lora_config``."""

    (experts, gate_up_out, gate_up_in), (_, down_out, down_in) = module.expert_shapes
    options = {"device": next(module.parameters()).device, "dtype": config.factor_dtype}
    shapes = {
        "lora_gate_up_a": (experts, config.rank, gate_up_in),
        "lora_gate_up_b": (experts, gate_up_out, config.rank),
        "lora_down_a": (experts, config.rank, down_in),
        "lora_down_b": (experts, down_out, config.rank),
    }
    for name in FACTORS:
        module.register_parameter(
            name, nn.Parameter(torch.empty(shapes[name], **options))
        )
    module.lora_config = config
    reset_factors(module)


def reset_factors(module: nn.Module) -> None:
    """Each A from a normal of standard deviation ``in ** -0.5``, each B zero,
    so the experts start at their base's output."""

    for prefix in ("lora_gate_up", "lora_down"):
        a, b = getattr(module, f"{prefix}_a"), getattr(module, f"{prefix}_b")
        nn.init.normal_(a, std=a.shape[2] ** -0.5)
        nn.init.zeros_(b)


def factors(module: nn.Module) -> tuple[torch.Tensor, ...]:
    """The module's factors in the order the expert kernels take them."""

    return tuple(getattr(module, name) for name in FACTORS)


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
    logits = F.linear(flat.to(router_weight.dtype), router_weight).float()
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
        h13 = F.linear(flat, gate_up[expert]) + scale * F.linear(
            F.linear(flat, a13[expert].to(flat.dtype)), b13[expert].to(flat.dtype)
        )
        gate, up = h13.chunk(2, -1)
        active = F.silu(gate.float()).to(gate.dtype) * up
        y = F.linear(active, down[expert]) + scale * F.linear(
            F.linear(active, a2[expert].to(active.dtype)), b2[expert].to(active.dtype)
        )
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
