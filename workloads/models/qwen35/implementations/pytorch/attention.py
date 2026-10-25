"""Gated full attention for the pure-PyTorch Qwen 3.5 decoder."""

from __future__ import annotations

import torch
from torch import nn

from workloads.models.shared.decoder import (
    RMSNorm,
    RotaryEmbedding,
    apply_rotary,
    causal_attention,
)

from ...config import Config


class GatedAttention(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.wq = nn.Linear(config.d_model, 2 * config.attention_width, bias=False)
        self.wk = nn.Linear(config.d_model, config.key_value_width, bias=False)
        self.wv = nn.Linear(config.d_model, config.key_value_width, bias=False)
        self.q_norm = RMSNorm(
            config.head_dim,
            epsilon=config.norm_epsilon,
            zero_centered=config.zero_centered_norm,
        )
        self.k_norm = RMSNorm(
            config.head_dim,
            epsilon=config.norm_epsilon,
            zero_centered=config.zero_centered_norm,
        )
        self.wo = nn.Linear(config.attention_width, config.d_model, bias=False)

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        lengths: tuple[int, ...],
        rotary: RotaryEmbedding,
    ) -> torch.Tensor:
        config = self.config
        batch, sequence, _width = hidden.shape
        query_and_gate = self.wq(hidden)
        query, gate = query_and_gate.split(config.attention_width, dim=-1)
        query = self.q_norm(
            query.view(batch, sequence, config.n_heads, config.head_dim)
        )
        key = self.k_norm(
            self.wk(hidden).view(batch, sequence, config.n_kv_heads, config.head_dim)
        )
        value = self.wv(hidden).view_as(key)
        query = apply_rotary(query, positions, rotary)
        key = apply_rotary(key, positions, rotary)
        attended = causal_attention(query, key, value, lengths).reshape(
            batch, sequence, config.attention_width
        )
        return self.wo(attended * torch.sigmoid(gate.float()).to(attended.dtype))
