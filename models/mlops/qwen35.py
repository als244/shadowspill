"""Qwen 3.5 dense hybrid model using external ``mlops`` operations."""

from __future__ import annotations

import mlops
import torch
import torch.nn as nn

from models.common import RotaryEmbedding, SequenceLengths, attention_metadata
from models.pytorch.qwen35 import Qwen35Config

from .common import GatedRMSNorm, RMSNorm


class GatedAttention(nn.Module):
    def __init__(self, config: Qwen35Config) -> None:
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
        query = mlops.partial_rope(
            query,
            positions,
            config.rope_base,
            config.rotary_width,
            rotary.cosine,
            rotary.sine,
        )
        key = mlops.partial_rope(
            key,
            positions,
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
            lengths,
        ).reshape(batch, sequence, config.attention_width)
        gated = attended * torch.sigmoid(gate.float()).to(attended.dtype)
        return self.wo(gated)


class GatedDeltaNet(nn.Module):
    def __init__(self, config: Qwen35Config) -> None:
        super().__init__()
        self.config = config
        self.w_qkvz = nn.Linear(config.d_model, config.qkvz_width, bias=False)
        self.w_ba = nn.Linear(config.d_model, 2 * config.lin_v_heads, bias=False)
        self.conv = nn.Conv1d(
            config.convolution_width,
            config.convolution_width,
            config.lin_conv_kernel,
            groups=config.convolution_width,
            bias=False,
        )
        self.A_log = nn.Parameter(torch.zeros(config.lin_v_heads))
        self.dt_bias = nn.Parameter(torch.zeros(config.lin_v_heads))
        self.lin_norm = GatedRMSNorm(config.lin_v_head_dim)
        self.w_out = nn.Linear(config.linear_value_width, config.d_model, bias=False)

    def forward(
        self,
        hidden: torch.Tensor,
        lengths: tuple[int, ...],
        cumulative: torch.Tensor,
        chunk_indices: torch.Tensor,
    ) -> torch.Tensor:
        config = self.config
        batch, sequence, _width = hidden.shape
        qkvz = self.w_qkvz(hidden)
        controls = self.w_ba(hidden)
        convolved = mlops.causal_conv_silu(
            qkvz[..., : config.convolution_width],
            self.conv.weight,
            lengths,
            cumulative,
            chunk_indices,
        )
        query = mlops.l2_norm(
            convolved[..., : config.linear_key_width].reshape(
                batch * sequence, config.lin_k_heads, config.lin_k_head_dim
            )
        )
        key = mlops.l2_norm(
            convolved[
                ..., config.linear_key_width : 2 * config.linear_key_width
            ].reshape(batch * sequence, config.lin_k_heads, config.lin_k_head_dim)
        )
        value = convolved[..., 2 * config.linear_key_width :].reshape(
            batch * sequence, config.lin_v_heads, config.lin_v_head_dim
        )
        beta = torch.sigmoid(controls[..., : config.lin_v_heads].float()).to(
            hidden.dtype
        )
        decay = controls[..., config.lin_v_heads :]
        mixed = mlops.linear_attention(
            query,
            key,
            value,
            beta.reshape(batch * sequence, config.lin_v_heads),
            decay.reshape(batch * sequence, config.lin_v_heads),
            self.A_log,
            self.dt_bias,
            lengths,
            cumulative,
            chunk_indices,
        ).reshape(batch, sequence, config.lin_v_heads, config.lin_v_head_dim)
        gate = qkvz[..., config.convolution_width :].view_as(mixed)
        normalized = self.lin_norm(mixed, gate)
        return self.w_out(normalized.reshape(batch, sequence, -1))


class MLP(nn.Module):
    def __init__(self, config: Qwen35Config) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.w3 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.w2 = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.w2(mlops.swiglu(self.w1(hidden), self.w3(hidden)))


class Block(nn.Module):
    def __init__(self, config: Qwen35Config, index: int) -> None:
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
        positions: torch.Tensor,
        lengths: tuple[int, ...],
        rotary: RotaryEmbedding,
        cumulative: torch.Tensor,
        chunk_indices: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self.attn_norm(hidden)
        if self.kind == "full":
            update = self.mixer(normalized, positions, lengths, rotary)
        else:
            update = self.mixer(normalized, lengths, cumulative, chunk_indices)
        hidden = hidden + update
        return hidden + self.mlp(self.ffn_norm(hidden))


class Qwen35(nn.Module):
    """State-dict-compatible optimized twin of the pure reference."""

    SUPPORTS_PACKED = True

    def __init__(self, config: Qwen35Config) -> None:
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
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tied_embeddings:
            self.lm_head.weight = self.embed.weight

    def hidden(
        self, tokens: torch.Tensor, sequence_lengths: SequenceLengths = None
    ) -> torch.Tensor:
        positions, lengths = attention_metadata(
            tokens, sequence_lengths, capacity=self.config.max_seq_len
        )
        cumulative, chunk_indices = mlops.prepare_packed_sequence_metadata(
            lengths, tokens
        )
        hidden = mlops.embedding(tokens, self.embed.weight)
        for block in self.blocks:
            hidden = block(
                hidden,
                positions,
                lengths,
                self.rotary,
                cumulative,
                chunk_indices,
            )
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


__all__ = ["Qwen35", "Qwen35Config"]
