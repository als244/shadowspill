"""Llama 3 dimensions and the named shapes."""

from __future__ import annotations

from collections.abc import Mapping
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
    tied_embeddings: bool = False

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")


#: The named shapes the catalog serves, by size.
PRESETS: Mapping[str, Config] = {
    "numerical": Config(12, 2_048, 16, 4, 7_168, 128_256),
    "tiny": Config(2, 128, 4, 2, 256, 512, max_seq_len=4_096),
    "1b": Config(16, 2_048, 32, 8, 8_192, 128_256, tied_embeddings=True),
    "8b": Config(32, 4_096, 32, 8, 14_336, 128_256),
}
