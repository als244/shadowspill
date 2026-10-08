"""Normalized top-k experts, with optional caller-selected expert parallelism."""

import mlops
import torch
from torch import nn
from torch.nn import functional as F

from .initialization import Linear


def parallel_options(
    config, group, *, dtype, compute_precision, weight_grad_dtype, activation_transport
):
    # Optional GPU dependencies are only loaded after the caller installs its
    # runtime and creates the process group, never during workload discovery.
    from mlops.expert_parallel import QuackMoEConfig

    return QuackMoEConfig(
        ep_size=torch.distributed.get_world_size(group),
        num_experts=config.n_experts,
        top_k=config.top_k,
        model_dim=config.d_model,
        expert_hidden_dim=config.d_ff_expert,
        router_dtype=dtype,
        router_weight_grad_dtype=dtype,
        weight_grad_dtype=weight_grad_dtype,
        compute_precision=compute_precision,
        activation_transport=activation_transport,
        renormalize_topk=True,
        share_expert_banks=True,
        init_std=config.initializer_range,
    )


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
    def __init__(
        self,
        config,
        *,
        group=None,
        buffer=None,
        options=None,
        device=None,
        parameter_device=None,
    ):
        super().__init__()
        self.config = config
        self.parallel = group is not None
        self.shared = SharedExpert(config) if config.d_ff_shared else None
        if self.parallel:
            from mlops.expert_parallel import QuackMoE

            self.experts = QuackMoE(options, group, buffer=buffer, device=device)
            if torch.device(parameter_device) != torch.device(device):
                # Logical parameter storage must not retain a view of the
                # shared GPU publication banks when moved to host memory.
                for name, parameter in tuple(self.experts.named_parameters()):
                    value = (
                        parameter.detach().to(parameter_device, copy=True).contiguous()
                    )
                    self.experts.register_parameter(name, nn.Parameter(value))
        else:
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
        if not self.parallel:
            nn.init.normal_(self.w13, std=self.config.initializer_range)
            nn.init.normal_(self.w2, std=self.config.initializer_range)

    def forward(self, hidden, residual):
        c = self.config
        if self.parallel:
            from mlops.expert_parallel.quack.router import route_op

            flat = hidden.reshape(-1, c.d_model)
            router = self.experts.router_weight
            logits = F.linear(flat.to(router.dtype), router).float()
            weights, ids, counts = route_op(logits, c.top_k, True)
            routed = self.experts(flat, expert_ids=ids, routing_weights=weights)
            output = residual + routed.reshape_as(hidden)
            probability_sum = logits.softmax(-1).sum(0)
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
