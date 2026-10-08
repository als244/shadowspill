"""Normalized top-k experts, with optional caller-selected expert parallelism."""

import mlops
import torch
from torch import nn

from .._expert_parallel import parallel_forward
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
            self.w13 = nn.Parameter(
                torch.empty(
                    config.n_experts,
                    config.d_model,
                    2 * config.d_ff_expert,
                )
            )
            self.w2 = nn.Parameter(
                torch.empty(
                    config.n_experts,
                    config.d_ff_expert,
                    config.d_model,
                )
            )
            self.reset_parameters()

    def reset_parameters(self):
        if self.experts is None:
            nn.init.normal_(self.w13, std=self.config.initializer_range)
            nn.init.normal_(self.w2, std=self.config.initializer_range)

    def forward(self, hidden, residual):
        c = self.config
        if self.experts is not None:
            routed, counts, probability_sum = parallel_forward(self.experts, hidden)
            output = residual + routed
        else:
            output, _auxiliary, counts, probability_sum = mlops.moe(
                hidden,
                residual,
                self.router.weight.T,
                self.w13,
                self.w2,
                top_k=c.top_k,
                routing_mode="topk_then_softmax",
            )
        if self.shared is not None:
            output = output + self.shared(hidden)
        return output, counts, probability_sum
