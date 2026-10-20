"""Dense Qwen 3.5 dimensions and the named shapes."""

from __future__ import annotations

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

    @classmethod
    def numerical(cls) -> Config:
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
    def throughput(cls) -> Config:
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

    @classmethod
    def dense_27b(cls) -> Config:
        """The text decoder of Qwen3.5-27B, without its multi-token prediction
        layer."""
        return cls(
            64,
            5_120,
            4,
            24,
            4,
            256,
            0.25,
            16,
            48,
            128,
            128,
            4,
            17_408,
            248_320,
            max_seq_len=262_144,
        )

    def __post_init__(self) -> None:
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.lin_v_heads % self.lin_k_heads:
            raise ValueError("linear value heads must be divisible by key heads")
        if self.rotary_width <= 0 or self.rotary_width % 2:
            raise ValueError("partial rotary width must be positive and even")
