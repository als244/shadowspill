"""Qwen 3.5 dense hybrid model using external ``mlops`` operations."""

from __future__ import annotations

import mlops
import torch
import torch.nn as nn
from mlops.modules import LanguageModelHead

from workloads.models.shared.decoder import (
    Packing,
    RotaryEmbedding,
    SequenceLengths,
    packed_metadata,
)
from workloads.models.shared.delta_net import GatedDeltaNet
from workloads.models.shared.norms import RMSNorm

from ..config import Config


class GatedAttention(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.wq = nn.Linear(config.d_model, 2 * config.attention_width, bias=False)
        self.wk = nn.Linear(config.d_model, config.key_value_width, bias=False)
        self.wv = nn.Linear(config.d_model, config.key_value_width, bias=False)
        self.q_norm = RMSNorm(config.head_dim)
        self.k_norm = RMSNorm(config.head_dim)
        self.wo = nn.Linear(config.attention_width, config.d_model, bias=False)

    def forward(
        self,
        hidden: torch.Tensor,
        packing: Packing,
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
        query = mlops.partial_rope(
            query,
            packing.positions,
            config.rope_base,
            config.rotary_width,
            rotary.cosine,
            rotary.sine,
        )
        key = mlops.partial_rope(
            key,
            packing.positions,
            config.rope_base,
            config.rotary_width,
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
        ).reshape(batch, sequence, config.attention_width)
        gated = attended * torch.sigmoid(gate.float()).to(attended.dtype)
        return self.wo(gated)


class MLP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.w3 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.w2 = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.w2(mlops.swiglu(self.w1(hidden), self.w3(hidden)))


class Block(nn.Module):
    def __init__(self, config: Config, index: int) -> None:
        super().__init__()
        self.kind = config.layer_kind(index)
        self.attn_norm = RMSNorm(config.d_model)
        self.mixer: nn.Module = (
            GatedAttention(config) if self.kind == "full" else GatedDeltaNet(config)
        )
        self.ffn_norm = RMSNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(
        self,
        hidden: torch.Tensor,
        packing: Packing,
        rotary: RotaryEmbedding,
        cumulative: torch.Tensor,
        chunk_indices: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self.attn_norm(hidden)
        if self.kind == "full":
            update = self.mixer(normalized, packing, rotary)
        else:
            update = self.mixer(normalized, cumulative, chunk_indices)
        hidden = hidden + update
        return hidden + self.mlp(self.ffn_norm(hidden))


class Model(nn.Module):
    """State-dict-compatible optimized twin of the pure reference."""

    SUPPORTS_PACKED = True

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.rotary = RotaryEmbedding(
            config.rotary_width, base=config.rope_base, capacity=config.max_seq_len
        )
        self.blocks = nn.ModuleList(
            Block(config, index) for index in range(config.n_layers)
        )
        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = LanguageModelHead(config.d_model, config.vocab_size)
        if config.tied_embeddings:
            self.lm_head.weight = self.embed.weight

    def hidden(
        self, tokens: torch.Tensor, sequence_lengths: SequenceLengths = None
    ) -> torch.Tensor:
        packing = packed_metadata(
            tokens, sequence_lengths, capacity=self.config.max_seq_len
        )
        # Linear attention reads its own form of the same packing.
        cumulative, chunk_indices = mlops.prepare_packed_sequence_metadata(
            sequence_lengths if packing.lengths is None else packing.lengths, tokens
        )
        hidden = mlops.embedding(tokens, self.embed.weight)
        for block in self.blocks:
            hidden = block(
                hidden,
                packing,
                self.rotary,
                cumulative,
                chunk_indices,
            )
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
        return self.lm_head.loss(
            self.hidden(tokens, seq_lens),
            targets,
            reduction=reduction,
        )


__all__ = ["Model"]
