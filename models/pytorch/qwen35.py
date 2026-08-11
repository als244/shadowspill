"""Pure-PyTorch Qwen 3.5 dense hybrid decoder."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.common import (
    GatedRMSNorm,
    RMSNorm,
    RotaryEmbedding,
    SequenceLengths,
    apply_rotary,
    attention_metadata,
    causal_attention,
    l2_normalize,
    language_model_loss,
    swiglu,
)


@dataclass(frozen=True, slots=True)
class Qwen35Config:
    n_layers: int
    d_model: int
    full_attention_interval: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    partial_rotary_factor: float
    lin_k_heads: int
    lin_v_heads: int
    lin_k_head_dim: int
    lin_v_head_dim: int
    lin_conv_kernel: int
    d_ff: int
    vocab_size: int
    rope_base: float = 10_000_000.0
    tied_embeddings: bool = False
    max_seq_len: int = 131_072

    @property
    def attention_width(self) -> int:
        return self.n_heads * self.head_dim

    @property
    def key_value_width(self) -> int:
        return self.n_kv_heads * self.head_dim

    @property
    def rotary_width(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def linear_key_width(self) -> int:
        return self.lin_k_heads * self.lin_k_head_dim

    @property
    def linear_value_width(self) -> int:
        return self.lin_v_heads * self.lin_v_head_dim

    @property
    def convolution_width(self) -> int:
        return 2 * self.linear_key_width + self.linear_value_width

    @property
    def qkvz_width(self) -> int:
        return 2 * self.linear_key_width + 2 * self.linear_value_width

    def layer_kind(self, index: int) -> str:
        return "full" if (index + 1) % self.full_attention_interval == 0 else "linear"

    @classmethod
    def numerical(cls) -> Qwen35Config:
        return cls(
            8,
            1_536,
            4,
            12,
            4,
            128,
            0.25,
            6,
            12,
            128,
            128,
            4,
            4_608,
            248_320,
        )

    @classmethod
    def throughput(cls) -> Qwen35Config:
        return cls(
            32,
            4_096,
            4,
            16,
            4,
            256,
            0.25,
            16,
            32,
            128,
            128,
            4,
            12_288,
            248_320,
        )

    def __post_init__(self) -> None:
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.lin_v_heads % self.lin_k_heads:
            raise ValueError("linear value heads must be divisible by key heads")
        if self.rotary_width <= 0 or self.rotary_width % 2:
            raise ValueError("partial rotary width must be positive and even")


def _delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
) -> torch.Tensor:
    batch, sequence, key_heads, key_width = query.shape
    value_heads, value_width = value.shape[2:]
    repeat = value_heads // key_heads
    query = query.float().repeat_interleave(repeat, dim=2)
    key = key.float().repeat_interleave(repeat, dim=2)
    state = torch.zeros(
        batch,
        value_heads,
        key_width,
        value_width,
        dtype=torch.float32,
        device=query.device,
    )
    outputs: list[torch.Tensor] = []
    for index in range(sequence):
        state = state * decay[:, index].float().exp()[:, :, None, None]
        prediction = torch.einsum("bhk,bhkv->bhv", key[:, index], state)
        update = beta[:, index].float()[:, :, None] * (
            value[:, index].float() - prediction
        )
        state = state + key[:, index, :, :, None] * update[:, :, None, :]
        outputs.append(
            torch.einsum("bhk,bhkv->bhv", query[:, index] * key_width**-0.5, state)
        )
    return torch.stack(outputs, dim=1).to(value.dtype)


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
        value = self.wv(hidden).view_as(key)
        query = apply_rotary(query, positions, rotary)
        key = apply_rotary(key, positions, rotary)
        attended = causal_attention(query, key, value, lengths).reshape(
            batch, sequence, config.attention_width
        )
        return self.wo(attended * torch.sigmoid(gate.float()).to(attended.dtype))


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

    def _segment(self, hidden: torch.Tensor) -> torch.Tensor:
        config = self.config
        batch, sequence, _width = hidden.shape
        qkvz = self.w_qkvz(hidden)
        controls = self.w_ba(hidden)
        convolution_input = qkvz[..., : config.convolution_width]
        padded = F.pad(
            convolution_input.float().transpose(1, 2),
            (config.lin_conv_kernel - 1, 0),
        )
        convolved = F.conv1d(
            padded, self.conv.weight.float(), groups=config.convolution_width
        )
        convolved = F.silu(convolved.transpose(1, 2)).to(hidden.dtype)
        query = l2_normalize(
            convolved[..., : config.linear_key_width].view(
                batch, sequence, config.lin_k_heads, config.lin_k_head_dim
            )
        )
        key = l2_normalize(
            convolved[..., config.linear_key_width : 2 * config.linear_key_width].view(
                batch, sequence, config.lin_k_heads, config.lin_k_head_dim
            )
        )
        value = convolved[..., 2 * config.linear_key_width :].view(
            batch, sequence, config.lin_v_heads, config.lin_v_head_dim
        )
        beta = torch.sigmoid(controls[..., : config.lin_v_heads].float()).to(
            hidden.dtype
        )
        raw_decay = controls[..., config.lin_v_heads :].float()
        decay = -self.A_log.float().exp() * F.softplus(raw_decay + self.dt_bias)
        mixed = _delta_rule(query, key, value, beta, decay)
        gate = qkvz[..., config.convolution_width :].view_as(mixed)
        normalized = self.lin_norm(mixed, gate)
        return self.w_out(normalized.reshape(batch, sequence, -1))

    def forward(self, hidden: torch.Tensor, lengths: tuple[int, ...]) -> torch.Tensor:
        batch, sequence, _width = hidden.shape
        if len(lengths) == batch and len(set(lengths)) == 1:
            return self._segment(hidden)
        flat = hidden.reshape(1, batch * sequence, -1)
        outputs: list[torch.Tensor] = []
        offset = 0
        for length in lengths:
            outputs.append(self._segment(flat[:, offset : offset + length]))
            offset += length
        return torch.cat(outputs, dim=1).view_as(hidden)


class MLP(nn.Module):
    def __init__(self, config: Qwen35Config) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.w3 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.w2 = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.w2(swiglu(self.w1(hidden), self.w3(hidden)))


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
    ) -> torch.Tensor:
        normalized = self.attn_norm(hidden)
        if self.kind == "full":
            update = self.mixer(normalized, positions, lengths, rotary)
        else:
            update = self.mixer(normalized, lengths)
        hidden = hidden + update
        return hidden + self.mlp(self.ffn_norm(hidden))


class Qwen35(nn.Module):
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


__all__ = ["Qwen35", "Qwen35Config"]
