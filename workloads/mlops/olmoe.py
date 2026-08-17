"""OLMoE reference model using external ``mlops`` semantic operations."""

from __future__ import annotations

import mlops
import torch
import torch.nn as nn

from workloads.common import RotaryEmbedding, SequenceLengths, attention_metadata
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
        positions: torch.Tensor,
        lengths: tuple[int, ...],
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
            positions,
            config.rope_base,
            rotary.cosine,
            rotary.sine,
        )
        key = mlops.rope(
            key,
            positions,
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
            lengths,
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
        nn.init.normal_(self.w13_experts, std=config.d_model**-0.5)
        nn.init.normal_(self.w2_experts, std=config.d_ff_expert**-0.5)

    def forward(
        self, hidden: torch.Tensor, residual: torch.Tensor, lengths: tuple[int, ...]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output, auxiliary, _counts, _probability_sum = mlops.moe(
            hidden,
            residual,
            self.router.weight.T,
            self.w13_experts,
            self.w2_experts,
            top_k=self.config.top_k,
            routing_mode="softmax_then_topk",
            lengths=lengths,
        )
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
        positions: torch.Tensor,
        lengths: tuple[int, ...],
        rotary: RotaryEmbedding,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = hidden + self.attn(self.attn_norm(hidden), positions, lengths, rotary)
        return self.moe(self.ffn_norm(hidden), hidden, lengths)


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
        self, tokens: torch.Tensor, sequence_lengths: SequenceLengths = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions, lengths = attention_metadata(
            tokens, sequence_lengths, capacity=self.config.max_seq_len
        )
        hidden = mlops.embedding(tokens, self.embed.weight)
        auxiliary = torch.zeros((), dtype=torch.float32, device=hidden.device)
        for block in self.blocks:
            hidden, block_auxiliary = block(hidden, positions, lengths, self.rotary)
            auxiliary = auxiliary + block_auxiliary
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
    ) -> torch.Tensor:
        hidden, auxiliary = self.hidden(tokens, seq_lens)
        objective = mlops.head_loss(hidden, self.lm_head.weight, targets)
        return objective + float(aux_coef) * auxiliary


__all__ = ["OLMoE", "OLMoEConfig"]
