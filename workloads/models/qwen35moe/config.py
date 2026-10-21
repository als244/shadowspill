"""Qwen 3.5 MoE dimensions; the defaults are the published Qwen3.5-35B-A3B
text decoder."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from workloads.models.shared.qwen_moe.config import QwenMoEConfig
from workloads.models.shared.qwen_moe.config import (
    from_huggingface as _from_huggingface,
)


@dataclass(frozen=True, kw_only=True)
class Config(QwenMoEConfig):
    n_layers: int = 40
    d_model: int = 2048
    n_heads: int = 16
    n_kv_heads: int = 2
    head_dim: int = 256
    n_experts: int = 256
    top_k: int = 8
    d_ff_expert: int = 512
    d_ff_shared: int = 512
    vocab_size: int = 248320
    max_seq_len: int = 262144
    rope_base: float = 10_000_000.0
    partial_rotary_factor: float = 0.25
    zero_centered_norm: bool = True
    attention_gate: bool = True
    full_attention_interval: int = 4


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
        d_ff_shared=128,
        vocab_size=1_024,
        max_seq_len=4_096,
        partial_rotary_factor=0.25,
        zero_centered_norm=True,
        attention_gate=True,
        full_attention_interval=2,
        lin_k_heads=2,
        lin_v_heads=4,
        lin_k_head_dim=32,
        lin_v_head_dim=32,
    ),
    "35b-a3b": Config(),
}


def from_huggingface(metadata, **overrides):
    """The ``Config`` a Hugging Face ``config.json`` of this family describes."""
    return _from_huggingface(
        Config, metadata, ("qwen3_5_moe", "qwen3_5_moe_text"), **overrides
    )


def to_huggingface(config):
    """The Hugging Face text decoder's ``config.json`` of a ``Config``
    (``Qwen3_5MoeForCausalLM``), the inverse of ``from_huggingface`` apart
    from the rotary capacity, which it records as
    ``max_position_embeddings``: the published checkpoint's text
    configuration without its vision tower and with no multi-token
    prediction layer. Hugging Face's Qwen 3.5 MoE zero-centers its norms and
    gates its attention, so a configuration without either has no
    counterpart."""

    from workloads.models.qwen35.config import mrope_sections

    if not (config.zero_centered_norm and config.attention_gate):
        raise ValueError(
            "Hugging Face's Qwen 3.5 MoE zero-centers its norms and gates its"
            " attention; this configuration does not"
        )
    return {
        "architectures": ["Qwen3_5MoeForCausalLM"],
        "model_type": "qwen3_5_moe_text",
        "hidden_act": "silu",
        "attention_bias": False,
        "attn_output_gate": True,
        "norm_topk_prob": True,
        "num_hidden_layers": config.n_layers,
        "hidden_size": config.d_model,
        "full_attention_interval": config.full_attention_interval,
        "layer_types": [
            "full_attention"
            if config.layer_kind(index) == "full"
            else "linear_attention"
            for index in range(config.n_layers)
        ],
        "num_attention_heads": config.n_heads,
        "num_key_value_heads": config.n_kv_heads,
        "head_dim": config.head_dim,
        "linear_num_key_heads": config.lin_k_heads,
        "linear_num_value_heads": config.lin_v_heads,
        "linear_key_head_dim": config.lin_k_head_dim,
        "linear_value_head_dim": config.lin_v_head_dim,
        "linear_conv_kernel_dim": config.lin_conv_kernel,
        "mamba_ssm_dtype": "float32",
        "num_experts": config.n_experts,
        "num_experts_per_tok": config.top_k,
        "moe_intermediate_size": config.d_ff_expert,
        "shared_expert_intermediate_size": config.d_ff_shared,
        "mlp_only_layers": [],
        "mtp_num_hidden_layers": 0,
        "router_aux_loss_coef": config.router_aux_loss_coef,
        "vocab_size": config.vocab_size,
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": config.rope_base,
            "partial_rotary_factor": config.partial_rotary_factor,
            "mrope_interleaved": True,
            "mrope_section": mrope_sections(
                config.head_dim, config.partial_rotary_factor
            ),
        },
        "max_position_embeddings": config.max_seq_len,
        "rms_norm_eps": config.norm_epsilon,
        "initializer_range": config.initializer_range,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
    }
