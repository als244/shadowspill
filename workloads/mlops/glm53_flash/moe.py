"""GLM-specific routing composed with generic sequential expert execution."""

from dataclasses import replace

import torch
from mlops import packed_swiglu
from mlops import sigmoid_topk
from mlops.sequential_moe import SequentialMoE, SequentialMoEConfig, SequentialMoELoRA
from mlops.sequential_moe.router import router
from torch import nn

from .common import linear, without_autocast


class DenseMLP(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        c = self.config = config
        self.gate_proj = linear(c.hidden_size, c.intermediate_size, c, device)
        self.up_proj = linear(c.hidden_size, c.intermediate_size, c, device)
        self.down_proj = linear(c.intermediate_size, c.hidden_size, c, device)

    def forward(self, x):
        packed = torch.cat((self.gate_proj(x), self.up_proj(x)), dim=-1)
        return self.down_proj(packed_swiglu(packed, limit=self.config.swiglu_limit))


class MoE(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        c = self.config = config
        recipe = SequentialMoEConfig(
            hidden_size=c.hidden_size,
            expert_size=c.moe_intermediate_size,
            num_experts=c.num_local_experts,
            top_k=c.num_experts_per_tok,
            rank=c.lora_rank or 32,
            alpha=c.lora_alpha,
            chunk_capacity=c.chunk_capacity,
            row_multiple=c.row_multiple,
            weight_storage=c.expert_storage,
            gemm_precision=c.gemm_precision,
            nvfp4_block_rows=1,
            activation_dtype=c.dtype,
            factor_dtype=c.lora_factor_dtype,
            gradient_dtype=torch.float32,
            router_dtype=torch.float32,
            shared_experts=0,
            swiglu_limit=c.swiglu_limit,
            initializer_range=c.initializer_range,
        )
        self.backend = (SequentialMoELoRA if c.lora_rank else SequentialMoE)(
            recipe, device=device
        )
        self.shared_experts = (
            DenseMLP(
                replace(
                    c, intermediate_size=c.moe_intermediate_size * c.n_shared_experts
                ),
                device,
            )
            if c.n_shared_experts
            else None
        )
        self.register_buffer(
            "e_score_correction_bias",
            torch.zeros(c.num_local_experts, device=device, dtype=torch.float32),
        )

    def route(self, x):
        c = self.config
        with without_autocast(x):
            logits = router(x, self.backend.router, c.chunk_capacity)
            return sigmoid_topk(
                logits,
                self.e_score_correction_bias,
                c.num_experts_per_tok,
                scale=c.routed_scaling_factor,
            )

    def forward(self, x):
        routes = self.route(x)
        initial = self.shared_experts(x) if self.shared_experts is not None else None
        return self.backend.forward_routed(x, *routes, accumulator=initial)
