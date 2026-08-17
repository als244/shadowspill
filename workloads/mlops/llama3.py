"""Llama 3 reference model using external ``mlops`` semantic operations."""

from __future__ import annotations

import mlops
import torch
import torch.nn as nn

from workloads.common import RotaryEmbedding, SequenceLengths, attention_metadata
from workloads.pytorch.llama3 import Llama3Config

from .common import RMSNorm


class Attention(nn.Module):
    def __init__(self, config: Llama3Config) -> None:
        super().__init__()
        self.config = config
        self.wq = nn.Linear(config.d_model, config.d_model, bias=False)
        self.wk = nn.Linear(
            config.d_model, config.n_kv_heads * config.head_dim, bias=False
        )
        self.wv = nn.Linear(
            config.d_model, config.n_kv_heads * config.head_dim, bias=False
        )
        self.wo = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        lengths: tuple[int, ...],
        rotary: RotaryEmbedding,
    ) -> torch.Tensor:
        config = self.config
        batch, sequence, _width = hidden.shape
        query = self.wq(hidden).view(batch, sequence, config.n_heads, config.head_dim)
        key = self.wk(hidden).view(batch, sequence, config.n_kv_heads, config.head_dim)
        value = self.wv(hidden).view_as(key)
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
            value.reshape(batch * sequence, config.n_kv_heads, config.head_dim),
            lengths,
        )
        return self.wo(attended.reshape(batch, sequence, config.d_model))


class MLP(nn.Module):
    def __init__(self, config: Llama3Config) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.w3 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.w2 = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.w2(mlops.swiglu(self.w1(hidden), self.w3(hidden)))


class Block(nn.Module):
    def __init__(self, config: Llama3Config) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = Attention(config)
        self.ffn_norm = RMSNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        lengths: tuple[int, ...],
        rotary: RotaryEmbedding,
    ) -> torch.Tensor:
        hidden = hidden + self.attn(self.attn_norm(hidden), positions, lengths, rotary)
        return hidden + self.mlp(self.ffn_norm(hidden))


class Llama3(nn.Module):
    """State-dict-compatible optimized twin of the pure reference."""

    SUPPORTS_PACKED = True

    def __init__(self, config: Llama3Config) -> None:
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
    ) -> torch.Tensor:
        positions, lengths = attention_metadata(
            tokens, sequence_lengths, capacity=self.config.max_seq_len
        )
        hidden = mlops.embedding(tokens, self.embed.weight)
        for block in self.blocks:
            hidden = block(hidden, positions, lengths, self.rotary)
        return self.final_norm(hidden)

    def forward(
        self, tokens: torch.Tensor, sequence_lengths: SequenceLengths = None
    ) -> torch.Tensor:
        return self.hidden(tokens, sequence_lengths) @ self.lm_head.weight.T

    def loss(
        self,
        tokens: torch.Tensor,
        targets: torch.Tensor,
        *,
        seq_lens: SequenceLengths = None,
    ) -> torch.Tensor:
        return mlops.head_loss(
            self.hidden(tokens, seq_lens), self.lm_head.weight, targets
        )


__all__ = ["Llama3", "Llama3Config"]
