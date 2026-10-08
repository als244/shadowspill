"""Qwen3.5 MoE; the default text dimensions match Qwen3.5-35B-A3B."""

from dataclasses import dataclass

from ._qwen_moe.model import QwenMoE
from .qwen3_moe import Qwen3MoEConfig


@dataclass(frozen=True)
class Qwen35MoEConfig(Qwen3MoEConfig):
    n_layers: int = 40
    n_heads: int = 16
    n_kv_heads: int = 2
    head_dim: int = 256
    n_experts: int = 256
    d_ff_expert: int = 512
    d_ff_shared: int = 512
    vocab_size: int = 248320
    max_seq_len: int = 262144
    rope_base: float = 10_000_000.0
    partial_rotary_factor: float = 0.25
    zero_centered_norm: bool = True
    attention_gate: bool = True
    full_attention_interval: int = 4


class Qwen35MoE(QwenMoE):
    """Hybrid DeltaNet/GQA with routed and gated shared experts."""

    def __init__(self, config=None, **kwargs):
        super().__init__(config or Qwen35MoEConfig(), **kwargs)


__all__ = ["Qwen35MoE", "Qwen35MoEConfig"]
