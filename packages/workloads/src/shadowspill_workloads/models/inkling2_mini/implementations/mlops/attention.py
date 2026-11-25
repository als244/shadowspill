"""Relative GQA and sequence-reset short convolutions for Inkling text training."""

import math

import torch
from mlops.inkling_activation import (
    PackedResidualConv,
    ResidualTapMix,
    sigmoid_gated_value,
)
from mlops.l2_norm import l2_clip_scale
from mlops.projected_relative_attention import projected_relative_gqa
from mlops.relative_attention import relative_gqa_fused, relative_gqa_query_blocks
from mlops.rms_norm import rms_gain_cast
from mlops.rms_norm import rms_normalize_fp32 as rms
from torch import nn
from torch.nn import functional as F


class Norm(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(c.d_model))
        self.eps = c.norm_epsilon

    def forward(self, x, dtype=None):
        out_dtype = x.dtype if dtype is None else dtype
        return rms_gain_cast(x, self.weight, self.eps, out_dtype)


class ShortConv(nn.Module):
    def __init__(self, width, taps, *, dynamic_input=None, heads=None):
        super().__init__()
        self.taps = taps
        self.heads = heads
        if dynamic_input is None:
            self.weight = nn.Parameter(torch.empty(width, 1, taps))
            nn.init.normal_(self.weight, std=0.02)
        else:
            self.static_w = nn.Parameter(torch.zeros(taps, width))
            self.weight_proj = nn.Parameter(torch.empty(heads * taps, dynamic_input))
            nn.init.normal_(self.weight_proj, std=0.02)

    def forward(self, x, positions, source=None):
        if self.heads is None:
            return PackedResidualConv.apply(x, self.weight, positions)
        taps = self.taps
        values = x.float()
        padded = F.pad(values, (0, 0, taps - 1, 0))
        # Each tap has a separate slice. Its backward adds overlaps in a fixed
        # order before the cotangent reaches the next microscaled expert.
        windows = torch.stack(
            [padded[:, tap : tap + x.shape[1]] for tap in range(taps)], dim=-1
        )
        lag = torch.arange(taps - 1, -1, -1, device=x.device)
        windows = windows * (lag <= positions[..., None]).unsqueeze(-2)
        dynamic = (
            F.linear(source, self.weight_proj).float().unflatten(-1, (self.heads, taps))
        )
        weight = self.static_w.T.unflatten(
            0, (self.heads, -1)
        ).float() + dynamic.unsqueeze(-2)
        windows = windows.flip(-1).unflatten(-2, (self.heads, -1))
        return ResidualTapMix.apply(
            windows, weight, values, math.sqrt(2 * taps + 1), False
        ).to(x.dtype)


def clipped(x, threshold, gain=None, length_scale=None):
    value = x.float()
    if gain is not None:
        value = value * gain.float()
    value = (value * l2_clip_scale(value, threshold)).to(x.dtype)
    if length_scale is not None:
        value = (value.float() * length_scale).to(x.dtype)
    return value


class Attention(nn.Module):
    def __init__(self, c, local):
        super().__init__()
        self.c, self.local = c, local
        q_width, kv_width = c.n_heads * c.head_dim, c.n_kv_heads * c.head_dim
        self.wq_du = nn.Linear(c.d_model, q_width, bias=False)
        self.wk_dv = nn.Linear(c.d_model, kv_width, bias=False)
        self.wv_dv = nn.Linear(c.d_model, kv_width, bias=False)
        self.wo_ud = nn.Linear(q_width, c.d_model, bias=False)
        self.wr_du = nn.Linear(c.d_model, c.n_heads * c.d_rel, bias=False)
        self.gate_attn = nn.Linear(c.d_model, q_width, bias=False)
        self.attn_sink_proj = nn.Linear(c.d_model, c.n_heads, bias=False)
        self.q_scale = nn.Parameter(torch.ones(c.n_heads, c.head_dim))
        self.rel_scale = nn.Parameter(torch.ones(c.n_heads, c.d_rel))
        self.extent = c.window if local else c.rel_extent
        self.proj = nn.Parameter(torch.empty(c.d_rel, self.extent))
        nn.init.normal_(self.proj, std=c.initializer_range)
        dynamic = {} if local else dict(dynamic_input=c.d_model, heads=c.n_kv_heads)
        self.k_sconv = ShortConv(kv_width, c.conv_kernel, **dynamic)
        self.v_sconv = ShortConv(kv_width, c.conv_kernel, **dynamic)

    def forward(self, q_input, k_input, v_input, positions, distance):
        c = self.c
        tau = (
            None
            if self.local
            else 1
            + c.log_alpha
            * ((positions.float() + 1) / c.log_floor)
            .clamp_min(1)
            .log()[..., None, None]
        )
        q = self.wq_du(q_input).unflatten(-1, (c.n_heads, c.head_dim))
        q = clipped(q, c.qk_clip, self.q_scale, tau).transpose(1, 2)
        k = self.k_sconv(self.wk_dv(k_input), positions, k_input).unflatten(
            -1, (c.n_kv_heads, c.head_dim)
        )
        v = self.v_sconv(self.wv_dv(v_input), positions, v_input).unflatten(
            -1, (c.n_kv_heads, c.head_dim)
        )
        k = clipped(k, c.qk_clip).transpose(1, 2)
        v = rms(v, c.norm_epsilon).to(v.dtype).transpose(1, 2)
        rel = self.wr_du(q_input).unflatten(-1, (c.n_heads, c.d_rel))
        rel = clipped(rel, c.rel_clip, self.rel_scale, tau)
        sink = self.attn_sink_proj(q_input).transpose(1, 2).unsqueeze(-1)
        attended = self.attend(q, k, v, rel, sink, positions, distance)
        gated = sigmoid_gated_value(
            attended.transpose(1, 2).flatten(-2), self.gate_attn(q_input)
        )
        return self.wo_ud(gated.to(q.dtype))

    def attend(self, q, k, v, relative, sink, positions, distance):
        c = self.c
        window = c.window if self.local else 0
        if c.attention_kv_heads_per_chunk:
            return projected_relative_gqa(
                q,
                k,
                v,
                relative.transpose(1, 2),
                self.proj,
                sink,
                positions,
                window,
                c.attention_kv_heads_per_chunk,
            )
        profiles = (relative @ self.proj).transpose(1, 2)
        if c.use_fused_attention(q.shape[2]):
            return relative_gqa_fused(q, k, v, profiles, sink, positions, window)
        if c.attention_query_chunk_size:
            return relative_gqa_query_blocks(
                q, k, v, profiles, sink, positions, window, c.attention_query_chunk_size
            )
        copies = c.n_heads // c.n_kv_heads
        scores = (
            q @ k.repeat_interleave(copies, dim=1).transpose(-1, -2)
        ) * c.head_dim**-0.5
        index = distance.clamp(0, self.extent - 1).expand(*scores.shape)
        bias = profiles.gather(-1, index).masked_fill(
            (distance < 0) | (distance >= self.extent), 0
        )
        allowed = (distance >= 0) & (distance <= positions[:, None, :, None])
        if window:
            allowed = allowed & (distance < window)
        scores = (scores + bias).masked_fill(~allowed, -torch.inf)
        prob = (
            torch.cat((scores, sink), dim=-1).float().softmax(-1)[..., :-1].to(q.dtype)
        )
        return prob @ v.repeat_interleave(copies, dim=1)
