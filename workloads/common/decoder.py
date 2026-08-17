"""Small semantic primitives shared by pure-PyTorch reference decoders."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

SequenceLengths = Sequence[int] | None


class RMSNorm(nn.Module):
    """Last-dimension RMS normalization with FP32 reduction."""

    def __init__(self, width: int, *, epsilon: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.epsilon = float(epsilon)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        source = value.float()
        scale = torch.rsqrt(source.square().mean(dim=-1, keepdim=True) + self.epsilon)
        return ((source * scale).to(value.dtype) * self.weight).to(value.dtype)


class GatedRMSNorm(RMSNorm):
    """RMS normalization followed by a SiLU gate."""

    def forward(self, value: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        normalized = super().forward(value)
        return (normalized.float() * F.silu(gate.float())).to(value.dtype)


class RotaryEmbedding(nn.Module):
    """Caller-owned, fixed-capacity FP32 rotary tables."""

    def __init__(self, width: int, *, base: float, capacity: int) -> None:
        super().__init__()
        if width <= 0 or width % 2:
            raise ValueError("rotary width must be a positive even integer")
        if capacity <= 0:
            raise ValueError("rotary capacity must be positive")
        frequency = 1.0 / (
            float(base)
            ** (torch.arange(0, width, 2, dtype=torch.float32) / float(width))
        )
        phase = torch.outer(torch.arange(capacity, dtype=torch.float32), frequency)
        doubled = torch.cat((phase, phase), dim=-1)
        self.register_buffer("cosine", doubled.cos(), persistent=False)
        self.register_buffer("sine", doubled.sin(), persistent=False)

    def _apply(self, function: object, recurse: bool = True) -> RotaryEmbedding:
        super()._apply(function, recurse=recurse)  # type: ignore[arg-type]
        self.cosine = self.cosine.float()
        self.sine = self.sine.float()
        return self

    def select(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cosine[positions], self.sine[positions]


def attention_metadata(
    tokens: torch.Tensor,
    sequence_lengths: SequenceLengths,
    *,
    capacity: int,
) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Return flattened positions and validated causal sequence lengths."""

    if tokens.ndim != 2:
        raise ValueError("tokens must have shape [batch, sequence]")
    batch, sequence = tokens.shape
    if sequence_lengths is None:
        lengths = (sequence,) * batch
        positions = torch.arange(sequence, device=tokens.device).repeat(batch)
    else:
        lengths = tuple(int(length) for length in sequence_lengths)
        if not lengths or any(length <= 0 for length in lengths):
            raise ValueError("sequence lengths must be positive")
        if batch != 1 or sequence != sum(lengths):
            raise ValueError("packed tokens must have shape [1, sum(sequence_lengths)]")
        positions = torch.cat(
            [torch.arange(length, device=tokens.device) for length in lengths]
        )
    if max(lengths) > capacity:
        raise ValueError(
            f"sequence length {max(lengths)} exceeds rotary capacity {capacity}"
        )
    return positions, lengths


def apply_rotary(
    value: torch.Tensor,
    positions: torch.Tensor,
    rotary: RotaryEmbedding,
) -> torch.Tensor:
    """Rotate the leading configured channels of ``[B,S,H,D]`` input."""

    cosine, sine = rotary.select(positions)
    batch, sequence, _heads, _width = value.shape
    cosine = cosine.view(batch, sequence, 1, -1)
    sine = sine.view(batch, sequence, 1, -1)
    rotary_width = cosine.shape[-1]
    rotating = value[..., :rotary_width].float()
    half = rotary_width // 2
    rotated_half = torch.cat((-rotating[..., half:], rotating[..., :half]), dim=-1)
    rotated = rotating * cosine + rotated_half * sine
    return torch.cat((rotated.to(value.dtype), value[..., rotary_width:]), dim=-1)


def causal_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    lengths: tuple[int, ...],
) -> torch.Tensor:
    """Grouped-query causal attention over flattened independent sequences."""

    batch, sequence, query_heads, head_width = query.shape
    key_heads = key.shape[2]
    if query_heads % key_heads:
        raise ValueError("query head count must be divisible by key head count")
    key = key.repeat_interleave(query_heads // key_heads, dim=2)
    value = value.repeat_interleave(query_heads // key_heads, dim=2)
    if len(lengths) == batch and len(set(lengths)) == 1 and lengths[0] == sequence:
        attended = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            is_causal=True,
        )
        return attended.transpose(1, 2)

    flat_query = query.reshape(batch * sequence, query_heads, head_width)
    flat_key = key.reshape(batch * sequence, query_heads, head_width)
    flat_value = value.reshape(batch * sequence, query_heads, head_width)
    outputs: list[torch.Tensor] = []
    offset = 0
    for length in lengths:
        stop = offset + length
        attended = F.scaled_dot_product_attention(
            flat_query[offset:stop].transpose(0, 1).unsqueeze(0),
            flat_key[offset:stop].transpose(0, 1).unsqueeze(0),
            flat_value[offset:stop].transpose(0, 1).unsqueeze(0),
            is_causal=True,
        )
        outputs.append(attended.squeeze(0).transpose(0, 1))
        offset = stop
    return torch.cat(outputs, dim=0).view(batch, sequence, query_heads, head_width)


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """SwiGLU with the BF16 rounding point used by the qualification workloads."""

    return F.silu(gate.float()).to(gate.dtype) * up


def l2_normalize(value: torch.Tensor, *, epsilon: float = 1e-6) -> torch.Tensor:
    """Last-dimension L2 normalization with FP32 reduction."""

    source = value.float()
    scale = torch.rsqrt(source.square().sum(dim=-1, keepdim=True) + epsilon)
    return (source * scale).to(value.dtype)


def language_model_loss(
    hidden: torch.Tensor, head: nn.Linear, targets: torch.Tensor
) -> torch.Tensor:
    logits = head(hidden)
    return F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1).long()
    )
