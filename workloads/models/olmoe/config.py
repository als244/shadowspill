"""OLMoE dimensions and the named shapes."""

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
    n_experts: int
    top_k: int
    d_ff_expert: int
    vocab_size: int
    rope_base: float = 10_000.0
    max_seq_len: int = 131_072
    #: The standard deviation Hugging Face initializes embeddings, projections
    #: and expert banks with.
    initializer_range: float = 0.02

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


def to_huggingface(config: Config) -> dict[str, Any]:
    """The Hugging Face ``config.json`` of a ``Config``, the inverse of
    ``from_huggingface`` apart from the rotary capacity, which it records as
    ``max_position_embeddings``; the balancing weight is the one every
    harness's objective gives a family that names none (0.01, OLMoE's
    own)."""

    if config.head_dim * config.n_heads != config.d_model:
        raise ValueError("an OLMoE configuration's heads are its model width")
    return {
        "architectures": ["OlmoeForCausalLM"],
        "model_type": "olmoe",
        "hidden_act": "silu",
        "attention_bias": False,
        "clip_qkv": None,
        "norm_topk_prob": False,
        "rms_norm_eps": 1e-5,
        "num_hidden_layers": config.n_layers,
        "hidden_size": config.d_model,
        "num_attention_heads": config.n_heads,
        "num_key_value_heads": config.n_kv_heads,
        "num_experts": config.n_experts,
        "num_experts_per_tok": config.top_k,
        "intermediate_size": config.d_ff_expert,
        "vocab_size": config.vocab_size,
        "rope_theta": config.rope_base,
        "rope_scaling": None,
        "max_position_embeddings": config.max_seq_len,
        "router_aux_loss_coef": 0.01,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
    }


def from_huggingface(metadata: Mapping[str, Any], **overrides: Any) -> Config:
    """The ``Config`` a Hugging Face OLMoE ``config.json`` describes; the
    router's softmax-then-top-k without renormalization is the only routing
    this family computes. ``max_seq_len`` stays this module's default
    unless overridden."""

    if metadata.get("model_type") != "olmoe":
        raise ValueError(f"not an OLMoE checkpoint: {metadata.get('model_type')!r}")
    expected = {"attention_bias": False, "norm_topk_prob": False, "rms_norm_eps": 1e-5}
    for key, value in expected.items():
        if metadata.get(key, value) != value:
            raise ValueError(f"unsupported OLMoE {key}: {metadata[key]!r}")
    heads = int(metadata["num_attention_heads"])
    values = {
        "n_layers": int(metadata["num_hidden_layers"]),
        "d_model": int(metadata["hidden_size"]),
        "n_heads": heads,
        "n_kv_heads": int(metadata["num_key_value_heads"]),
        "head_dim": int(metadata.get("head_dim") or metadata["hidden_size"] // heads),
        "n_experts": int(metadata["num_experts"]),
        "top_k": int(metadata["num_experts_per_tok"]),
        "d_ff_expert": int(metadata["intermediate_size"]),
        "vocab_size": int(metadata["vocab_size"]),
        "rope_base": float(metadata.get("rope_theta", 10_000.0)),
    }
    unknown = set(overrides) - {f.name for f in fields(Config)}
    if unknown:
        raise ValueError(f"unknown OLMoE fields {sorted(unknown)}")
    return Config(**{**values, **overrides})
