"""Qwen attention, including the 3.5 text decoder's hybrid sequence mixer."""

import mlops
import torch
from torch import nn

from ..common import GatedRMSNorm
from ..qwen35 import GatedDeltaNet as _DeltaNet
from .initialization import Conv1d, Linear


class RMSNorm(nn.Module):
    def __init__(self, width, config):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(width))
        self.epsilon = config.norm_epsilon
        self.zero_centered = config.zero_centered_norm
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.weight, 0.0 if self.zero_centered else 1.0)

    def forward(self, value):
        if not self.zero_centered:
            return mlops.rms_norm(value, self.weight, self.epsilon)
        # Qwen3.5 multiplies the normalized FP32 value by (1 + weight)
        # before casting. Llama-style norm casts before that multiply.
        source = value.float()
        normalized = source * torch.rsqrt(
            source.square().mean(-1, keepdim=True) + self.epsilon
        )
        return (normalized * (1 + self.weight.float())).to(value.dtype)


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wq = Linear(
            config.d_model,
            config.attention_width * (2 if config.attention_gate else 1),
            std=config.initializer_range,
        )
        self.wk = Linear(
            config.d_model, config.key_value_width, std=config.initializer_range
        )
        self.wv = Linear(
            config.d_model, config.key_value_width, std=config.initializer_range
        )
        self.wo = Linear(
            config.attention_width, config.d_model, std=config.initializer_range
        )
        self.q_norm = RMSNorm(config.head_dim, config)
        self.k_norm = RMSNorm(config.head_dim, config)

    def forward(self, hidden, packing, rotary):
        c = self.config
        batch, sequence, _ = hidden.shape
        query = self.wq(hidden).view(batch, sequence, c.n_heads, -1)
        if c.attention_gate:
            # The published projection interleaves Q and gate per head.
            query, gate = query.chunk(2, dim=-1)
        query = self.q_norm(query)
        key = self.k_norm(
            self.wk(hidden).view(batch, sequence, c.n_kv_heads, c.head_dim)
        )
        query = mlops.partial_rope(
            query,
            packing.positions,
            c.rope_base,
            c.rotary_width,
            rotary.cosine,
            rotary.sine,
        )
        key = mlops.partial_rope(
            key,
            packing.positions,
            c.rope_base,
            c.rotary_width,
            rotary.cosine,
            rotary.sine,
        )
        attended = mlops.flash_attention(
            query.reshape(-1, c.n_heads, c.head_dim),
            key.reshape(-1, c.n_kv_heads, c.head_dim),
            self.wv(hidden).reshape(-1, c.n_kv_heads, c.head_dim),
            packing.cu_seqlens,
            packing.max_seqlen,
        ).reshape(batch, sequence, c.attention_width)
        if c.attention_gate:
            attended = attended * gate.reshape_as(attended).sigmoid()
        return self.wo(attended)


class GatedDeltaNet(_DeltaNet):
    """Reuse the packed MLOps recurrence with the published initial state/norm."""

    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        self.w_qkvz = Linear(
            config.d_model, config.qkvz_width, std=config.initializer_range
        )
        self.w_ba = Linear(
            config.d_model, 2 * config.lin_v_heads, std=config.initializer_range
        )
        self.conv = Conv1d(
            config.convolution_width,
            config.lin_conv_kernel,
            std=config.initializer_range,
        )
        # The recurrent decay parameters have FP32 storage even with BF16
        # projections. They are parameters, not a persistent inference cache.
        self.A_log = nn.Parameter(torch.empty(config.lin_v_heads, dtype=torch.float32))
        self.dt_bias = nn.Parameter(
            torch.empty(config.lin_v_heads, dtype=torch.float32)
        )
        self.lin_norm = GatedRMSNorm(config.lin_v_head_dim, epsilon=config.norm_epsilon)
        self.w_out = Linear(
            config.linear_value_width, config.d_model, std=config.initializer_range
        )
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self.A_log.uniform_(0.01, 16).log_()
            self.dt_bias.fill_(1.0)
