"""OLMoE dimensions and the named shapes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Config:
    n_layers: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    n_experts: int
    top_k: int
    d_ff_expert: int
    vocab_size: int
    rope_base: float = 10_000.0
    max_seq_len: int = 131_072

    @property
    def query_width(self) -> int:
        return self.n_heads * self.head_dim

    @property
    def key_value_width(self) -> int:
        return self.n_kv_heads * self.head_dim

    def __post_init__(self) -> None:
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if not 0 < self.top_k <= self.n_experts:
            raise ValueError("top_k must select between one and all experts")


#: The named shapes the catalog serves, by size.
PRESETS: Mapping[str, Config] = {
    "numerical": Config(16, 1_024, 8, 8, 128, 16, 4, 1_024, 50_304),
    "tiny": Config(2, 128, 4, 2, 32, 8, 2, 128, 512, max_seq_len=4_096),
    "7b-a1b": Config(16, 2_048, 16, 16, 128, 64, 8, 1_024, 50_304),
}
