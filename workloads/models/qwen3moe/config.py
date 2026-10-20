"""Qwen 3 MoE dimensions; the defaults are the published Qwen3-30B-A3B text
decoder."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from workloads.models.shared.qwen_moe.config import QwenMoEConfig
from workloads.models.shared.qwen_moe.config import (
    from_huggingface as _from_huggingface,
)


@dataclass(frozen=True, kw_only=True)
class Config(QwenMoEConfig):
    n_layers: int = 48
    d_model: int = 2048
    n_heads: int = 32
    n_kv_heads: int = 4
    head_dim: int = 128
    n_experts: int = 128
    top_k: int = 8
    d_ff_expert: int = 768
    d_ff_shared: int = 0
    vocab_size: int = 151936
    max_seq_len: int = 40960
    rope_base: float = 1_000_000.0
    partial_rotary_factor: float = 1.0
    zero_centered_norm: bool = False
    attention_gate: bool = False
    full_attention_interval: int = 1


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
        vocab_size=1_024,
        max_seq_len=4_096,
    ),
    "30b-a3b": Config(),
}


def from_huggingface(metadata, **overrides):
    """The ``Config`` a Hugging Face ``config.json`` of this family describes."""
    return _from_huggingface(Config, metadata, ("qwen3_moe",), **overrides)
