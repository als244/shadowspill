"""Readable pure-PyTorch Llama 3 decoder used as numerical authority."""

from __future__ import annotations

import torch
import torch.nn as nn

from workloads.models.shared.decoder import (
    RMSNorm,
    RotaryEmbedding,
    SequenceLengths,
    apply_rotary,
    attention_metadata,
    causal_attention,
    language_model_loss,
    swiglu,
)

from ..config import Config


class Attention(nn.Module):
    def __init__(self, config: Config) -> None:
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
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.w3 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.w2 = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.w2(swiglu(self.w1(hidden), self.w3(hidden)))


class Block(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, epsilon=config.norm_epsilon)
        self.attn = Attention(config)
        self.ffn_norm = RMSNorm(config.d_model, epsilon=config.norm_epsilon)
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


class Model(nn.Module):
    SUPPORTS_PACKED = True

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.rotary = RotaryEmbedding(
            config.head_dim,
            base=config.rope_base,
            capacity=config.max_seq_len,
            scaling=config.rope_scaling,
        )
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layers))
        self.final_norm = RMSNorm(config.d_model, epsilon=config.norm_epsilon)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tied_embeddings:
            self.lm_head.weight = self.embed.weight

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
        reduction: str = "mean",
    ) -> torch.Tensor:
        return language_model_loss(
            self.hidden(tokens, seq_lens), self.lm_head, targets, reduction
        )


__all__ = ["Model"]
