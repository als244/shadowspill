"""Normalized top-k experts, with optional caller-selected expert parallelism
and routed-expert LoRA factors (``workloads.models.shared.routed_lora``)."""

import mlops
import torch
from mlops.lora.experts import expert_lora
from torch import nn

from ..expert_parallel import parallel_forward
from ..routed_lora import factors, reset_factors
from .initialization import Linear


class SharedExpert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_up = Linear(
            config.d_model, 2 * config.d_ff_shared, std=config.initializer_range
        )
        self.down = Linear(
            config.d_ff_shared, config.d_model, std=config.initializer_range
        )
        self.gate = Linear(config.d_model, 1, std=config.initializer_range)

    def forward(self, hidden):
        gate, up = self.gate_up(hidden).chunk(2, dim=-1)
        return self.down(mlops.swiglu(gate, up)) * self.gate(hidden).sigmoid()


class MoE(nn.Module):
    def __init__(self, config, *, expert_factory):
        super().__init__()
        self.config = config
        self.shared = SharedExpert(config) if config.d_ff_shared else None
        self.experts = expert_factory()
        if self.experts is None:
            self.router = Linear(
                config.d_model, config.n_experts, std=config.initializer_range
            )
            gate_up, down = self.expert_shapes
            self.w13 = nn.Parameter(torch.empty(gate_up))
            self.w2 = nn.Parameter(torch.empty(down))
            self.reset_parameters()

    @property
    def expert_shapes(self):
        """The routed experts' gate/up and down banks, ``[E, out, in]``."""
        c = self.config
        return (
            (c.n_experts, 2 * c.d_ff_expert, c.d_model),
            (c.n_experts, c.d_model, c.d_ff_expert),
        )

    def reset_parameters(self):
        if self.experts is None:
            nn.init.normal_(self.w13, std=self.config.initializer_range)
            nn.init.normal_(self.w2, std=self.config.initializer_range)
            if getattr(self, "lora_config", None) is not None:
                reset_factors(self)

    def forward(self, hidden, residual):
        c = self.config
        lora = getattr(self, "lora_config", None)
        routing = {"top_k": c.top_k, "routing_mode": "topk_then_softmax"}
        if self.experts is not None:
            routed, counts, probability_sum = parallel_forward(self.experts, hidden)
            output = residual + routed
        elif lora is None:
            output, _auxiliary, counts, probability_sum = mlops.moe(
                hidden, residual, self.router.weight, self.w13, self.w2, **routing
            )
        else:
            output, _auxiliary, counts, probability_sum = expert_lora(
                hidden,
                residual,
                self.router.weight,
                self.w13,
                self.w2,
                *factors(self),
                scale=lora.scale,
                **routing,
            )
        if self.shared is not None:
            output = output + self.shared(hidden)
        return output, counts, probability_sum
