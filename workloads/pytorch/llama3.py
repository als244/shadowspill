"""Readable pure-PyTorch Llama 3 decoder used as numerical authority."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from workloads.common import (
    RMSNorm,
    RotaryEmbedding,
    SequenceLengths,
    apply_rotary,
    attention_metadata,
    causal_attention,
    language_model_loss,
    swiglu,
)


@dataclass(frozen=True, slots=True)
class Llama3Config:
    n_layers: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    d_ff: int
    vocab_size: int
    rope_base: float = 500_000.0
    max_seq_len: int = 131_072

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @classmethod
    def numerical(cls) -> Llama3Config:
        return cls(12, 2_048, 16, 4, 7_168, 128_256)

    @classmethod
    def throughput(cls) -> Llama3Config:
        return cls(32, 4_096, 32, 8, 14_336, 128_256)

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")


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
        query = apply_rotary(query, positions, rotary)
        key = apply_rotary(key, positions, rotary)
        attended = causal_attention(query, key, value, lengths)
        return self.wo(attended.reshape(batch, sequence, config.d_model))


class MLP(nn.Module):
    def __init__(self, config: Llama3Config) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.w3 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.w2 = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.w2(swiglu(self.w1(hidden), self.w3(hidden)))


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
        hidden = self.embed(tokens)
        for block in self.blocks:
            hidden = block(hidden, positions, lengths, self.rotary)
        return self.final_norm(hidden)

    def forward(
        self, tokens: torch.Tensor, sequence_lengths: SequenceLengths = None
    ) -> torch.Tensor:
        return self.lm_head(self.hidden(tokens, sequence_lengths))

    def loss(
        self,
        tokens: torch.Tensor,
        targets: torch.Tensor,
        *,
        seq_lens: SequenceLengths = None,
    ) -> torch.Tensor:
        return language_model_loss(self.hidden(tokens, seq_lens), self.lm_head, targets)


__all__ = ["Llama3", "Llama3Config"]
