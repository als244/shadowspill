"""Qwen 3.5 MoE dimensions; the defaults are the published Qwen3.5-35B-A3B
text decoder."""

from __future__ import annotations

from dataclasses import dataclass

from workloads.models.shared.qwen_moe.config import QwenMoEConfig


@dataclass(frozen=True, kw_only=True)
class Config(QwenMoEConfig):
    n_layers: int = 40
    d_model: int = 2048
    n_heads: int = 16
    n_kv_heads: int = 2
    head_dim: int = 256
    n_experts: int = 256
    top_k: int = 8
    d_ff_expert: int = 512
    d_ff_shared: int = 512
    vocab_size: int = 248320
    max_seq_len: int = 262144
    rope_base: float = 10_000_000.0
    partial_rotary_factor: float = 0.25
    zero_centered_norm: bool = True
    attention_gate: bool = True
    full_attention_interval: int = 4
