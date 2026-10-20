"""Llama 3 dimensions and the named shapes."""

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
    d_ff: int
    vocab_size: int
    rope_base: float = 500_000.0
    max_seq_len: int = 131_072
    tied_embeddings: bool = False
    #: Llama 3's rotary frequency scaling (``factor``, ``low_freq_factor``,
    #: ``high_freq_factor``, ``original_max_position_embeddings``), or none.
    rope_scaling: Mapping[str, float] | None = None
    norm_epsilon: float = 1e-5

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.rope_scaling is not None:
            missing = ROPE_SCALING_KEYS - set(self.rope_scaling)
            if missing:
                raise ValueError(f"rope_scaling lacks {sorted(missing)}")


ROPE_SCALING_KEYS = {
    "factor",
    "low_freq_factor",
    "high_freq_factor",
    "original_max_position_embeddings",
}

#: Llama 3.1 and 3.2 scale their rotary frequencies past the 8K context the
#: base models were trained at.
LLAMA_3_1_ROPE_SCALING: Mapping[str, float] = {
    "factor": 8.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 8192,
}
LLAMA_3_2_ROPE_SCALING: Mapping[str, float] = {**LLAMA_3_1_ROPE_SCALING, "factor": 32.0}


def from_huggingface(metadata: Mapping[str, Any], **overrides: Any) -> Config:
    """The ``Config`` a Hugging Face ``config.json`` describes; ``max_seq_len``
    stays this module's default unless overridden, since it is the rotary
    table's capacity, not the checkpoint's."""

    if metadata.get("model_type") != "llama":
        raise ValueError(f"not a Llama checkpoint: {metadata.get('model_type')!r}")
    for key, value in {
        "hidden_act": "silu",
        "attention_bias": False,
        "mlp_bias": False,
    }.items():
        if metadata.get(key, value) != value:
            raise ValueError(f"unsupported Llama {key}: {metadata[key]!r}")
    heads = int(metadata["num_attention_heads"])
    if int(metadata.get("head_dim") or metadata["hidden_size"] // heads) * heads != int(
        metadata["hidden_size"]
    ):
        raise ValueError("Llama 3's head width is the model width over its heads")
    scaling = metadata.get("rope_scaling")
    if scaling is not None:
        if scaling.get("rope_type", scaling.get("type")) != "llama3":
            raise ValueError(f"unsupported rope_scaling {scaling!r}")
        scaling = {key: float(scaling[key]) for key in ROPE_SCALING_KEYS}
    values = {
        "n_layers": int(metadata["num_hidden_layers"]),
        "d_model": int(metadata["hidden_size"]),
        "n_heads": heads,
        "n_kv_heads": int(metadata["num_key_value_heads"]),
        "d_ff": int(metadata["intermediate_size"]),
        "vocab_size": int(metadata["vocab_size"]),
        "rope_base": float(metadata.get("rope_theta", 500_000.0)),
        "tied_embeddings": bool(metadata.get("tie_word_embeddings", False)),
        "rope_scaling": scaling,
        "norm_epsilon": float(metadata.get("rms_norm_eps", 1e-5)),
    }
    unknown = set(overrides) - {f.name for f in fields(Config)}
    if unknown:
        raise ValueError(f"unknown Llama 3 fields {sorted(unknown)}")
    return Config(**{**values, **overrides})


#: The named shapes the catalog serves, by size.
PRESETS: Mapping[str, Config] = {
    "numerical": Config(12, 2_048, 16, 4, 7_168, 128_256),
    "tiny": Config(2, 128, 4, 2, 256, 512, max_seq_len=4_096),
    "1b": Config(
        16,
        2_048,
        32,
        8,
        8_192,
        128_256,
        tied_embeddings=True,
        rope_scaling=LLAMA_3_2_ROPE_SCALING,
    ),
    "8b": Config(
        32, 4_096, 32, 8, 14_336, 128_256, rope_scaling=LLAMA_3_1_ROPE_SCALING
    ),
}
