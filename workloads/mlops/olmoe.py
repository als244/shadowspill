"""OLMoE reference model using external ``mlops`` semantic operations."""

from __future__ import annotations

import mlops
import torch
import torch.nn as nn

from workloads.common import (
    Packing,
    RotaryEmbedding,
    SequenceLengths,
    auxiliary_share,
    packed_metadata,
)
from workloads.pytorch.olmoe import OLMoEConfig

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
    def __init__(self, config: OLMoEConfig) -> None:
        super().__init__()
        self.config = config
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
    def __init__(self, config: OLMoEConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = Attention(config)
        self.ffn_norm = RMSNorm(config.d_model)
        self.moe = MoE(config)

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
    """State-dict-compatible optimized twin of the pure reference."""

    SUPPORTS_PACKED = True

    def __init__(self, config: OLMoEConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.rotary = RotaryEmbedding(
            config.head_dim, base=config.rope_base, capacity=config.max_seq_len
        )
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layers))
        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

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
        return hidden @ self.lm_head.weight.T

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
        objective = mlops.head_loss(
            hidden, self.lm_head.weight, targets, reduction=reduction
        )
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
