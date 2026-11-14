"""The configuration the two Qwen mixture families share: the fields, their
validation and the derived widths. Each family's ``Config`` supplies its
published defaults."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any


@dataclass(frozen=True, kw_only=True)
class QwenMoEConfig:
    n_layers: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    n_experts: int
    top_k: int
    d_ff_expert: int
    d_ff_shared: int
    vocab_size: int
    max_seq_len: int
    rope_base: float
    partial_rotary_factor: float
    zero_centered_norm: bool
    attention_gate: bool
    full_attention_interval: int
    norm_epsilon: float = 1e-6
    lin_k_heads: int = 16
    lin_v_heads: int = 32
    lin_k_head_dim: int = 128
    lin_v_head_dim: int = 128
    lin_conv_kernel: int = 4
    initializer_range: float = 0.02
    router_aux_loss_coef: float = 0.001

    def __post_init__(self):
        for name in (
            "n_layers",
            "d_model",
            "n_heads",
            "n_kv_heads",
            "head_dim",
            "n_experts",
            "top_k",
            "d_ff_expert",
            "vocab_size",
            "max_seq_len",
            "full_attention_interval",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.n_heads % self.n_kv_heads or self.top_k > self.n_experts:
            raise ValueError("query heads must divide into KV groups; top_k <= experts")
        if not 0 < self.partial_rotary_factor <= 1 or self.rotary_width % 2:
            raise ValueError("rotary width must be positive, even, and <= head width")
        if self.rotary_width < 2 or self.d_ff_shared < 0:
            raise ValueError("invalid rotary or shared-expert width")
        if self.full_attention_interval > 1 and (
            self.lin_v_heads % self.lin_k_heads or self.lin_conv_kernel <= 0
        ):
            raise ValueError("invalid DeltaNet heads or convolution width")

    def layer_kind(self, index):
        return "full" if (index + 1) % self.full_attention_interval == 0 else "linear"

    @property
    def attention_width(self):
        return self.n_heads * self.head_dim

    @property
    def key_value_width(self):
        return self.n_kv_heads * self.head_dim

    @property
    def rotary_width(self):
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def linear_key_width(self):
        return self.lin_k_heads * self.lin_k_head_dim

    @property
    def linear_value_width(self):
        return self.lin_v_heads * self.lin_v_head_dim

    @property
    def convolution_width(self):
        return 2 * self.linear_key_width + self.linear_value_width

    @property
    def qkvz_width(self):
        return self.convolution_width + self.linear_value_width


def from_huggingface(
    config_type: type[QwenMoEConfig],
    metadata: Mapping[str, Any],
    model_types: tuple[str, ...],
    **overrides: Any,
) -> QwenMoEConfig:
    """The family's ``Config`` a Hugging Face ``config.json`` describes; the
    text settings sit under ``text_config`` for a vision-language checkpoint.
    The family's constants (zero-centered norms, the attention gate) and
    ``max_seq_len`` come from ``config_type``'s defaults."""

    text = metadata.get("text_config", metadata)
    kind = text.get("model_type", metadata.get("model_type"))
    if kind not in model_types:
        raise ValueError(f"not a {model_types[0]} checkpoint: {kind!r}")
    if text.get("attention_bias", False):
        raise ValueError("attention biases are not supported")
    if not text.get("norm_topk_prob", True):
        raise ValueError("this family renormalizes the selected experts' weights")
    rope = text.get("rope_parameters") or {}
    if rope.get("rope_type", "default") != "default":
        raise ValueError(f"unsupported rope_parameters {rope!r}")
    values: dict[str, Any] = {
        "n_layers": int(text["num_hidden_layers"]),
        "d_model": int(text["hidden_size"]),
        "n_heads": int(text["num_attention_heads"]),
        "n_kv_heads": int(text["num_key_value_heads"]),
        "head_dim": int(text["head_dim"]),
        "n_experts": int(text["num_experts"]),
        "top_k": int(text["num_experts_per_tok"]),
        "d_ff_expert": int(text["moe_intermediate_size"]),
        "d_ff_shared": int(text.get("shared_expert_intermediate_size") or 0),
        "vocab_size": int(text["vocab_size"]),
        "rope_base": float(rope.get("rope_theta", text.get("rope_theta"))),
        "partial_rotary_factor": float(
            rope.get("partial_rotary_factor", text.get("partial_rotary_factor", 1.0))
        ),
        "full_attention_interval": int(text.get("full_attention_interval", 1)),
        "norm_epsilon": float(text.get("rms_norm_eps", 1e-6)),
        "router_aux_loss_coef": float(text.get("router_aux_loss_coef", 0.001)),
    }
    for ours, theirs in (
        ("lin_k_heads", "linear_num_key_heads"),
        ("lin_v_heads", "linear_num_value_heads"),
        ("lin_k_head_dim", "linear_key_head_dim"),
        ("lin_v_head_dim", "linear_value_head_dim"),
        ("lin_conv_kernel", "linear_conv_kernel_dim"),
    ):
        if theirs in text:
            values[ours] = int(text[theirs])
    unknown = set(overrides) - {f.name for f in fields(config_type)}
    if unknown:
        raise ValueError(f"unknown fields {sorted(unknown)}")
    return config_type(**{**values, **overrides})
