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


def to_huggingface(config):
    """The Hugging Face ``config.json`` of a ``Config``, the inverse of
    ``from_huggingface`` apart from the rotary capacity, which it records as
    ``max_position_embeddings``. Every layer is a mixture; the dense width a
    mixed checkpoint gives its MLP-only layers is the published model's
    (three model widths), which no layer here uses."""

    if config.d_ff_shared or config.full_attention_interval != 1:
        raise ValueError("Qwen 3 MoE has no shared experts or linear attention")
    if config.partial_rotary_factor != 1 or config.zero_centered_norm:
        raise ValueError("Qwen 3 MoE rotates whole heads and scales its norms")
    if config.attention_gate:
        raise ValueError("Qwen 3 MoE has no attention gate")
    return {
        "architectures": ["Qwen3MoeForCausalLM"],
        "model_type": "qwen3_moe",
        "hidden_act": "silu",
        "attention_bias": False,
        "norm_topk_prob": True,
        "decoder_sparse_step": 1,
        "mlp_only_layers": [],
        "num_hidden_layers": config.n_layers,
        "hidden_size": config.d_model,
        "intermediate_size": 3 * config.d_model,
        "num_attention_heads": config.n_heads,
        "num_key_value_heads": config.n_kv_heads,
        "head_dim": config.head_dim,
        "num_experts": config.n_experts,
        "num_experts_per_tok": config.top_k,
        "moe_intermediate_size": config.d_ff_expert,
        "vocab_size": config.vocab_size,
        "rope_theta": config.rope_base,
        "rope_scaling": None,
        "max_position_embeddings": config.max_seq_len,
        "rms_norm_eps": config.norm_epsilon,
        "router_aux_loss_coef": config.router_aux_loss_coef,
        "initializer_range": config.initializer_range,
        "tie_word_embeddings": False,
        "use_sliding_window": False,
        "torch_dtype": "bfloat16",
    }
