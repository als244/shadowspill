"""Dense Qwen 3.5 dimensions and the named shapes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Config:
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

    def __post_init__(self) -> None:
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.lin_v_heads % self.lin_k_heads:
            raise ValueError("linear value heads must be divisible by key heads")
        if self.rotary_width <= 0 or self.rotary_width % 2:
            raise ValueError("partial rotary width must be positive and even")


#: The named shapes the catalog serves, by size. Each published size is the
#: text decoder alone; the multi-token prediction layer stays out.
PRESETS: Mapping[str, Config] = {
    "numerical": Config(
        n_layers=8,
        d_model=1_536,
        full_attention_interval=4,
        n_heads=12,
        n_kv_heads=4,
        head_dim=128,
        partial_rotary_factor=0.25,
        lin_k_heads=6,
        lin_v_heads=12,
        lin_k_head_dim=128,
        lin_v_head_dim=128,
        lin_conv_kernel=4,
        d_ff=4_608,
        vocab_size=248_320,
    ),
    "tiny": Config(
        n_layers=2,
        d_model=128,
        full_attention_interval=2,
        n_heads=4,
        n_kv_heads=2,
        head_dim=32,
        partial_rotary_factor=0.25,
        lin_k_heads=2,
        lin_v_heads=4,
        lin_k_head_dim=32,
        lin_v_head_dim=32,
        lin_conv_kernel=4,
        d_ff=256,
        vocab_size=512,
        max_seq_len=4_096,
    ),
    "2b": Config(
        n_layers=24,
        d_model=2_048,
        full_attention_interval=4,
        n_heads=8,
        n_kv_heads=2,
        head_dim=256,
        partial_rotary_factor=0.25,
        lin_k_heads=16,
        lin_v_heads=16,
        lin_k_head_dim=128,
        lin_v_head_dim=128,
        lin_conv_kernel=4,
        d_ff=6_144,
        vocab_size=248_320,
        tied_embeddings=True,
        max_seq_len=262_144,
    ),
    "9b": Config(
        n_layers=32,
        d_model=4_096,
        full_attention_interval=4,
        n_heads=16,
        n_kv_heads=4,
        head_dim=256,
        partial_rotary_factor=0.25,
        lin_k_heads=16,
        lin_v_heads=32,
        lin_k_head_dim=128,
        lin_v_head_dim=128,
        lin_conv_kernel=4,
        d_ff=12_288,
        vocab_size=248_320,
    ),
    "27b": Config(
        n_layers=64,
        d_model=5_120,
        full_attention_interval=4,
        n_heads=24,
        n_kv_heads=4,
        head_dim=256,
        partial_rotary_factor=0.25,
        lin_k_heads=16,
        lin_v_heads=48,
        lin_k_head_dim=128,
        lin_v_head_dim=128,
        lin_conv_kernel=4,
        d_ff=17_408,
        vocab_size=248_320,
        max_seq_len=262_144,
    ),
}
