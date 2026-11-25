"""Inkling expert banks reuse MLOps grouped routing and LoRA projections."""

import torch
from mlops.groupwise_experts import (
    checkpoint_cuda_available,
    native_grouped_linear,
    routed_native_grouped_linear,
)
from mlops.groupwise_experts import grouped_linear as packed_linear
from mlops.inkling_activation import routed_softcapped_swiglu
from mlops.lora.grouped import grouped_linear
from mlops.lora.routing import combine_rows, dispatch_rows, layout, routing_logits
from mlops.routing import sigmoid_topk
from torch import nn


class Experts(nn.Module):
    def __init__(self, c, layer):
        super().__init__()
        self.c = c
        self.gate = nn.Linear(c.d_model, c.n_experts, bias=False)
        self.gate.bias = nn.Parameter(
            torch.zeros(c.n_experts, dtype=torch.float32), requires_grad=False
        )
        self.expert_shapes = (
            (c.n_experts, 2 * c.d_ff_expert, c.d_model),
            (c.n_experts, c.d_model, c.d_ff_expert),
        )
        self.format = (
            c.expert_formats[layer] if c.expert_storage == "packed" else "dense"
        )
        for name, (experts, rows, cols) in zip(
            ("w13", "w2"), self.expert_shapes, strict=True
        ):
            if self.format == "dense":
                dtype = torch.bfloat16 if c.expert_storage == "bf16" else None
                value = nn.Parameter(torch.empty(experts, rows, cols, dtype=dtype))
                nn.init.normal_(value, std=c.initializer_range)
                self.register_parameter(name, value)
            else:
                fp4 = self.format == "nvfp4"

                self._frozen(
                    name,
                    (experts, rows, cols // (2 if fp4 else 1)),
                    torch.uint8 if fp4 else torch.float8_e4m3fn,
                )
                self._frozen(
                    name + "_scale",
                    (experts, rows, cols // (16 if fp4 else 32)),
                    torch.float8_e4m3fn if fp4 else torch.uint8,
                )
                if fp4:
                    self._frozen(name + "_amax", (experts,), torch.float32)
                    self._frozen(name + "_outer_scale_denominator", (1,), torch.float32)

    def _frozen(self, name, shape, dtype):
        self.register_parameter(
            name, nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)
        )

    def projection(self, x, name, prefix, offsets, routing=None):
        w = getattr(self, name)
        compute = self.c.expert_compute
        native = self.format != "dense" and (
            compute == "checkpoint"
            or (
                compute == "auto"
                and x.dtype == torch.bfloat16
                and checkpoint_cuda_available(x.device, w.dtype)
            )
        )
        adapter = hasattr(self, "lora_config")
        a = getattr(self, "lora_" + prefix + "_a").to(x.dtype) if adapter else None
        low = None
        if routing is not None and not native:
            x = dispatch_rows(x, *routing)
        if self.format == "dense":
            y = grouped_linear(x.to(w.dtype), w, offsets).to(x.dtype)
        else:
            amax = getattr(self, name + "_amax", None)
            denominator = getattr(self, name + "_outer_scale_denominator", None)
            scales = getattr(self, name + "_scale")
            if not native:
                factor = None if amax is None else amax / denominator
                y = packed_linear(x.to(torch.bfloat16), w, scales, offsets, factor).to(
                    x.dtype
                )
            elif routing is not None:
                low, y = routed_native_grouped_linear(
                    x,
                    *routing,
                    w,
                    scales,
                    offsets,
                    amax,
                    denominator,
                    adapter=a,
                )
            else:
                y = native_grouped_linear(x, w, scales, offsets, amax, denominator)
        if adapter:
            b = getattr(self, "lora_" + prefix + "_b")
            if low is None:
                low = grouped_linear(x, a, offsets)
            y = y + self.lora_config.scale * grouped_linear(low, b.to(x.dtype), offsets)
        return y

    def forward(self, hidden):
        c = self.c
        x = hidden.reshape(-1, c.d_model)
        logits = routing_logits(x, self.gate.weight)
        ids, weights = sigmoid_topk(
            logits, self.gate.bias, c.top_k, scale=c.route_scale, stable_ties=True
        )
        order, offsets, slots = layout(ids, c.n_experts)
        gu = self.projection(x, "w13", "gate_up", offsets, (order, slots))
        gamma = weights.reshape(-1).index_select(0, order.long()).unsqueeze(-1)
        active = routed_softcapped_swiglu(gu, gamma, c.gate_cap, c.up_cap)
        y = self.projection(active, "w2", "down", offsets)
        return combine_rows(
            y, order, slots, torch.ones_like(weights), torch.zeros_like(x)
        ).reshape_as(hidden)
