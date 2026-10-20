"""The configuration the two Qwen mixture families share: the fields, their
validation and the derived widths. Each family's ``Config`` supplies its
published defaults."""

from __future__ import annotations

from dataclasses import dataclass


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
