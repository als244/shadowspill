"""Qwen 3.5 MoE dimensions; the defaults are the published Qwen3.5-35B-A3B
text decoder."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from workloads.models.shared.qwen_moe.config import QwenMoEConfig
from workloads.models.shared.qwen_moe.config import (
    from_huggingface as _from_huggingface,
)


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


#: The named shapes the catalog serves, by size.
PRESETS: Mapping[str, Config] = {
    "tiny": Config(
        n_layers=2,
        d_model=128,
        n_heads=4,
        n_kv_heads=2,
        head_dim=32,
        n_experts=8,
        top_k=2,
        d_ff_expert=128,
        d_ff_shared=128,
        vocab_size=1_024,
        max_seq_len=4_096,
        partial_rotary_factor=0.25,
        zero_centered_norm=True,
        attention_gate=True,
        full_attention_interval=2,
        lin_k_heads=2,
        lin_v_heads=4,
        lin_k_head_dim=32,
        lin_v_head_dim=32,
    ),
    "35b-a3b": Config(),
}


def from_huggingface(metadata, **overrides):
    """The ``Config`` a Hugging Face ``config.json`` of this family describes."""
    return _from_huggingface(
        Config, metadata, ("qwen3_5_moe", "qwen3_5_moe_text"), **overrides
    )
