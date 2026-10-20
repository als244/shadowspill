"""Llama 3 dimensions and the named shapes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Config:
    n_layers: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    d_ff: int
    vocab_size: int
    rope_base: float = 500_000.0
    max_seq_len: int = 131_072

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @classmethod
    def numerical(cls) -> Config:
        return cls(12, 2_048, 16, 4, 7_168, 128_256)

    @classmethod
    def throughput(cls) -> Config:
        return cls(32, 4_096, 32, 8, 14_336, 128_256)

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
