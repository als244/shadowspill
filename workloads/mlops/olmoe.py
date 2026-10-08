"""OLMoE reference model using external ``mlops`` semantic operations."""

from __future__ import annotations

import mlops
import torch
import torch.nn as nn
from mlops.modules import LanguageModelHead

from workloads.common import (
    Packing,
    RotaryEmbedding,
    SequenceLengths,
    auxiliary_share,
    packed_metadata,
)
from workloads.pytorch.olmoe import OLMoEConfig

from ._expert_parallel import close_experts, expert_construction, parallel_forward
from .common import RMSNorm


class Attention(nn.Module):
    def __init__(self, config: OLMoEConfig) -> None:
        super().__init__()
        self.config = config
        self.wq = nn.Linear(config.d_model, config.query_width, bias=False)
        self.wk = nn.Linear(config.d_model, config.key_value_width, bias=False)
        self.wv = nn.Linear(config.d_model, config.key_value_width, bias=False)
        self.q_norm = RMSNorm(config.query_width)
        self.k_norm = RMSNorm(config.key_value_width)
        self.wo = nn.Linear(config.query_width, config.d_model, bias=False)

    def forward(
        self,
        hidden: torch.Tensor,
        packing: Packing,
        rotary: RotaryEmbedding,
    ) -> torch.Tensor:
        config = self.config
        batch, sequence, _width = hidden.shape
        query = self.q_norm(self.wq(hidden)).view(
            batch, sequence, config.n_heads, config.head_dim
        )
        key = self.k_norm(self.wk(hidden)).view(
            batch, sequence, config.n_kv_heads, config.head_dim
        )
        query = mlops.rope(
            query,
            packing.positions,
            config.rope_base,
            rotary.cosine,
            rotary.sine,
        )
        key = mlops.rope(
            key,
            packing.positions,
            config.rope_base,
            rotary.cosine,
            rotary.sine,
        )
        attended = mlops.flash_attention(
            query.reshape(batch * sequence, config.n_heads, config.head_dim),
            key.reshape(batch * sequence, config.n_kv_heads, config.head_dim),
            self.wv(hidden).reshape(
                batch * sequence, config.n_kv_heads, config.head_dim
            ),
            packing.cu_seqlens,
            packing.max_seqlen,
        )
        return self.wo(attended.reshape(batch, sequence, config.query_width))


class MoE(nn.Module):
    def __init__(self, config: OLMoEConfig, expert_factory=None) -> None:
        super().__init__()
        self.config = config
        self.experts = expert_factory() if expert_factory is not None else None
        if self.experts is not None:
            return
        self.router = nn.Linear(config.d_model, config.n_experts, bias=False)
        self.w13_experts = nn.Parameter(
            torch.empty(config.n_experts, config.d_model, 2 * config.d_ff_expert)
        )
        self.w2_experts = nn.Parameter(
            torch.empty(config.n_experts, config.d_ff_expert, config.d_model)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialise the parameters this module owns, in place."""

        if self.experts is not None:
            return
        config = self.config
        with torch.no_grad():
            nn.init.normal_(self.w13_experts, std=config.d_model**-0.5)
            nn.init.normal_(self.w2_experts, std=config.d_ff_expert**-0.5)

    def forward(
        self,
        hidden: torch.Tensor,
        residual: torch.Tensor,
        *,
        return_metrics: bool = False,
    ):
        # Softmax-then-top-k balances experts over the whole microbatch, so
        # where its sequences begin is no concern of the router's.
        if self.experts is not None:
            routed, counts, probability_sum = parallel_forward(self.experts, hidden)
            rows = hidden.numel() // self.config.d_model
            frequency = counts.float() / (rows * self.config.top_k)
            auxiliary = (
                self.config.n_experts * (frequency * probability_sum / rows).sum()
            )
            output = residual + routed
        else:
            output, auxiliary, counts, probability_sum = mlops.moe(
                hidden,
                residual,
                self.router.weight.T,
                self.w13_experts,
                self.w2_experts,
                top_k=self.config.top_k,
                routing_mode="softmax_then_topk",
            )
        if return_metrics:
            return output, auxiliary, counts, probability_sum
        return output, auxiliary


class Block(nn.Module):
    def __init__(self, config: OLMoEConfig, expert_factory=None) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = Attention(config)
        self.ffn_norm = RMSNorm(config.d_model)
        self.moe = MoE(config, expert_factory)

    def forward(
        self,
        hidden: torch.Tensor,
        packing: Packing,
        rotary: RotaryEmbedding,
        *,
        return_metrics: bool = False,
    ):
        hidden = hidden + self.attn(self.attn_norm(hidden), packing, rotary)
        return self.moe(self.ffn_norm(hidden), hidden, return_metrics=return_metrics)


class OLMoE(nn.Module):
    """Local MLOps experts by default; optionally shard experts over an EP group.

    Supply ``ep_group`` and either ``token_capacity`` or a caller-owned ``buffer``
    to use QuackMoE. One buffer and publication banks are shared across blocks;
    model parameters remain distinct. EP hidden activations are BF16, with
    configurable expert compute/gradient/transport precision. The EP router
    defaults to FP32. Local construction retains PyTorch-reference state keys.
    """

    SUPPORTS_PACKED = True

    def __init__(
        self,
        config: OLMoEConfig,
        *,
        ep_group=None,
        token_capacity=None,
        buffer=None,
        device=None,
        parameter_device=None,
        dtype=None,
        router_dtype=None,
        compute_precision="bf16",
        weight_grad_dtype=torch.bfloat16,
        activation_transport="bf16",
    ) -> None:
        super().__init__()
        self.config = config
        with expert_construction(
            config,
            renormalize_topk=False,
            ep_group=ep_group,
            token_capacity=token_capacity,
            buffer=buffer,
            device=device,
            parameter_device=parameter_device,
            dtype=dtype,
            router_dtype=router_dtype or torch.float32,
            compute_precision=compute_precision,
            weight_grad_dtype=weight_grad_dtype,
            activation_transport=activation_transport,
        ) as experts:
            self.embed = nn.Embedding(config.vocab_size, config.d_model)
            self.rotary = RotaryEmbedding(
                config.head_dim, base=config.rope_base, capacity=config.max_seq_len
            )
            self.blocks = nn.ModuleList()
            for _ in range(config.n_layers):
                self.blocks.append(Block(config, experts))
            self.final_norm = RMSNorm(config.d_model)
            self.lm_head = LanguageModelHead(config.d_model, config.vocab_size)
        self._owns_buffer = ep_group is not None and buffer is None

    def expert_parameters(self):
        for block in self.blocks:
            if block.moe.experts is not None:
                yield from block.moe.experts.expert_parameters()

    def close(self):
        owns_buffer, self._owns_buffer = self._owns_buffer, False
        close_experts(
            (block.moe.experts for block in self.blocks), destroy_buffer=owns_buffer
        )

    def hidden(
        self,
        tokens: torch.Tensor,
        sequence_lengths: SequenceLengths = None,
        *,
        return_metrics: bool = False,
    ):
        packing = packed_metadata(
            tokens, sequence_lengths, capacity=self.config.max_seq_len
        )
        hidden = mlops.embedding(tokens, self.embed.weight)
        auxiliary = torch.zeros((), dtype=torch.float32, device=hidden.device)
        layer_auxiliary, counts, probability_sums = [], [], []
        for block in self.blocks:
            values = block(hidden, packing, self.rotary, return_metrics=return_metrics)
            hidden, block_auxiliary = values[:2]
            auxiliary = auxiliary + block_auxiliary
            if return_metrics:
                layer_auxiliary.append(block_auxiliary.detach())
                counts.append(values[2])
                probability_sums.append(values[3].detach())
        if return_metrics:
            return (
                self.final_norm(hidden),
                auxiliary,
                {
                    "layer_auxiliary": torch.stack(layer_auxiliary),
                    "expert_counts": torch.stack(counts),
                    "probability_sum": torch.stack(probability_sums),
                },
            )
        return self.final_norm(hidden), auxiliary

    def forward(
        self, tokens: torch.Tensor, sequence_lengths: SequenceLengths = None
    ) -> torch.Tensor:
        hidden, _auxiliary = self.hidden(tokens, sequence_lengths)
        return self.lm_head(hidden)

    def loss(
        self,
        tokens: torch.Tensor,
        targets: torch.Tensor,
        *,
        seq_lens: SequenceLengths = None,
        aux_coef: float = 0.0,
        reduction: str = "mean",
        return_metrics: bool = False,
    ):
        """Loss, optionally with small detached summaries for each microbatch.

        Counts describe all router rows (including any packed padding). Loss
        sums and layer auxiliary sums use the number of trained targets, so
        callers can combine unequal microbatches without averaging averages.
        """

        values = self.hidden(tokens, seq_lens, return_metrics=return_metrics)
        hidden, auxiliary = values[:2]
        objective = self.lm_head.loss(hidden, targets, reduction=reduction)
        loss = objective + float(aux_coef) * auxiliary_share(
            auxiliary, targets, reduction
        )
        if not return_metrics:
            return loss
        if reduction not in {"sum", "mean"}:
            raise ValueError("loss metrics require sum or mean reduction")
        trained = targets.ne(-100).sum()
        auxiliary_sum = auxiliary.detach() * trained
        routing = values[2]
        return loss, {
            "ce_sum": objective.detach()
            if reduction == "sum"
            else objective.detach() * trained,
            "auxiliary_sum": auxiliary_sum,
            "weighted_auxiliary_sum": float(aux_coef) * auxiliary_sum,
            "trained_tokens": trained,
            "layer_auxiliary_sum": routing["layer_auxiliary"] * trained,
            "expert_counts": routing["expert_counts"],
            "probability_sum": routing["probability_sum"],
        }


__all__ = ["OLMoE", "OLMoEConfig"]
