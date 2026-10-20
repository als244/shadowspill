"""Qwen 3 dimensions and the named shapes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any


@dataclass(frozen=True, slots=True)
class Config:
    n_layers: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    d_ff: int
    vocab_size: int
    rope_base: float = 1_000_000.0
    max_seq_len: int = 40_960
    tied_embeddings: bool = False
    norm_epsilon: float = 1e-6

    @property
    def attention_width(self) -> int:
        return self.n_heads * self.head_dim

    @property
    def key_value_width(self) -> int:
        return self.n_kv_heads * self.head_dim

    def __post_init__(self) -> None:
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.head_dim % 2:
            raise ValueError("head_dim must be even: RoPE rotates channel pairs")


def from_huggingface(metadata: Mapping[str, Any], **overrides: Any) -> Config:
    """The ``Config`` a Hugging Face ``config.json`` describes; ``max_seq_len``
    stays this module's default unless overridden, since it is the rotary
    table's capacity, not the checkpoint's."""

    if metadata.get("model_type") != "qwen3":
        raise ValueError(f"not a Qwen 3 checkpoint: {metadata.get('model_type')!r}")
    for key, value in {
        "hidden_act": "silu",
        "attention_bias": False,
        "use_sliding_window": False,
    }.items():
        if metadata.get(key, value) != value:
            raise ValueError(f"unsupported Qwen 3 {key}: {metadata[key]!r}")
    # A rotary scaling (YaRN, for contexts past the published 40,960) is
    # not reproduced, so a checkpoint that enables one is refused.
    rope = metadata.get("rope_parameters") or {}
    if (
        metadata.get("rope_scaling") is not None
        or rope.get("rope_type", "default") != "default"
    ):
        raise ValueError(
            f"unsupported rotary scaling {metadata.get('rope_scaling') or rope!r}"
        )
    heads = int(metadata["num_attention_heads"])
    values = {
        "n_layers": int(metadata["num_hidden_layers"]),
        "d_model": int(metadata["hidden_size"]),
        "n_heads": heads,
        "n_kv_heads": int(metadata["num_key_value_heads"]),
        "head_dim": int(metadata.get("head_dim") or metadata["hidden_size"] // heads),
        "d_ff": int(metadata["intermediate_size"]),
        "vocab_size": int(metadata["vocab_size"]),
        "rope_base": float(
            rope.get("rope_theta", metadata.get("rope_theta", 1_000_000.0))
        ),
        "tied_embeddings": bool(metadata.get("tie_word_embeddings", False)),
        "norm_epsilon": float(metadata.get("rms_norm_eps", 1e-6)),
    }
    unknown = set(overrides) - {f.name for f in fields(Config)}
    if unknown:
        raise ValueError(f"unknown Qwen 3 fields {sorted(unknown)}")
    return Config(**{**values, **overrides})


#: The named shapes the catalog serves, by size. The tiny shape's attention
#: is wider than its model, as Qwen3-32B's is (8,192 against 5,120).
PRESETS: Mapping[str, Config] = {
    "tiny": Config(2, 128, 4, 2, 64, 256, 512, max_seq_len=4_096),
    "1.7b": Config(28, 2_048, 16, 8, 128, 6_144, 151_936, tied_embeddings=True),
    "8b": Config(36, 4_096, 32, 8, 128, 12_288, 151_936),
    "32b": Config(64, 5_120, 64, 8, 128, 25_600, 151_936),
}
