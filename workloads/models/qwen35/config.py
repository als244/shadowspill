"""Dense Qwen 3.5 dimensions and the named shapes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any


@dataclass(frozen=True, slots=True)
class Config:
    n_layers: int
    d_model: int
    full_attention_interval: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    partial_rotary_factor: float
    lin_k_heads: int
    lin_v_heads: int
    lin_k_head_dim: int
    lin_v_head_dim: int
    lin_conv_kernel: int
    d_ff: int
    vocab_size: int
    rope_base: float = 10_000_000.0
    tied_embeddings: bool = False
    max_seq_len: int = 131_072
    norm_epsilon: float = 1e-5
    #: The published models keep every RMSNorm weight but the linear
    #: attention's as an offset from one.
    zero_centered_norm: bool = False

    @property
    def attention_width(self) -> int:
        return self.n_heads * self.head_dim

    @property
    def key_value_width(self) -> int:
        return self.n_kv_heads * self.head_dim

    @property
    def rotary_width(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def linear_key_width(self) -> int:
        return self.lin_k_heads * self.lin_k_head_dim

    @property
    def linear_value_width(self) -> int:
        return self.lin_v_heads * self.lin_v_head_dim

    @property
    def convolution_width(self) -> int:
        return 2 * self.linear_key_width + self.linear_value_width

    @property
    def qkvz_width(self) -> int:
        return 2 * self.linear_key_width + 2 * self.linear_value_width

    def layer_kind(self, index: int) -> str:
        return "full" if (index + 1) % self.full_attention_interval == 0 else "linear"

    def __post_init__(self) -> None:
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.lin_v_heads % self.lin_k_heads:
            raise ValueError("linear value heads must be divisible by key heads")
        if self.rotary_width <= 0 or self.rotary_width % 2:
            raise ValueError("partial rotary width must be positive and even")


#: The named shapes the catalog serves, by size. Each published size is the
#: text decoder alone; the multi-token prediction layer stays out.
PRESETS: Mapping[str, Config] = {
    "numerical": Config(
        n_layers=8,
        d_model=1_536,
        full_attention_interval=4,
        n_heads=12,
        n_kv_heads=4,
        head_dim=128,
        partial_rotary_factor=0.25,
        lin_k_heads=6,
        lin_v_heads=12,
        lin_k_head_dim=128,
        lin_v_head_dim=128,
        lin_conv_kernel=4,
        d_ff=4_608,
        vocab_size=248_320,
    ),
    "tiny": Config(
        n_layers=2,
        d_model=128,
        full_attention_interval=2,
        n_heads=4,
        n_kv_heads=2,
        head_dim=32,
        partial_rotary_factor=0.25,
        lin_k_heads=2,
        lin_v_heads=4,
        lin_k_head_dim=32,
        lin_v_head_dim=32,
        lin_conv_kernel=4,
        d_ff=256,
        vocab_size=512,
        max_seq_len=4_096,
    ),
    "2b": Config(
        n_layers=24,
        d_model=2_048,
        full_attention_interval=4,
        n_heads=8,
        n_kv_heads=2,
        head_dim=256,
        partial_rotary_factor=0.25,
        lin_k_heads=16,
        lin_v_heads=16,
        lin_k_head_dim=128,
        lin_v_head_dim=128,
        lin_conv_kernel=4,
        d_ff=6_144,
        vocab_size=248_320,
        tied_embeddings=True,
        max_seq_len=262_144,
        norm_epsilon=1e-6,
        zero_centered_norm=True,
    ),
    "9b": Config(
        n_layers=32,
        d_model=4_096,
        full_attention_interval=4,
        n_heads=16,
        n_kv_heads=4,
        head_dim=256,
        partial_rotary_factor=0.25,
        lin_k_heads=16,
        lin_v_heads=32,
        lin_k_head_dim=128,
        lin_v_head_dim=128,
        lin_conv_kernel=4,
        d_ff=12_288,
        vocab_size=248_320,
        norm_epsilon=1e-6,
        zero_centered_norm=True,
    ),
    "27b": Config(
        n_layers=64,
        d_model=5_120,
        full_attention_interval=4,
        n_heads=24,
        n_kv_heads=4,
        head_dim=256,
        partial_rotary_factor=0.25,
        lin_k_heads=16,
        lin_v_heads=48,
        lin_k_head_dim=128,
        lin_v_head_dim=128,
        lin_conv_kernel=4,
        d_ff=17_408,
        vocab_size=248_320,
        max_seq_len=262_144,
        norm_epsilon=1e-6,
        zero_centered_norm=True,
    ),
}


def to_huggingface(config: Config) -> dict[str, Any]:
    """The Hugging Face text decoder's ``config.json`` of a ``Config``
    (``Qwen3_5ForCausalLM``), the inverse of ``from_huggingface`` apart from
    the rotary capacity, which it records as ``max_position_embeddings``:
    the published checkpoints' text configuration without their vision
    tower and with no multi-token prediction layer, which this family
    leaves out. Hugging Face's Qwen 3.5 always keeps its norms as offsets
    from one, so a configuration with plain norms has no counterpart."""

    if not config.zero_centered_norm:
        raise ValueError(
            "Hugging Face's Qwen 3.5 zero-centers every norm; this configuration"
            " keeps plain ones (set zero_centered_norm)"
        )
    return {
        "architectures": ["Qwen3_5ForCausalLM"],
        "model_type": "qwen3_5_text",
        "hidden_act": "silu",
        "attention_bias": False,
        "attn_output_gate": True,
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
        "intermediate_size": config.d_ff,
        "mlp_only_layers": [],
        "mtp_num_hidden_layers": 0,
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
        "tie_word_embeddings": config.tied_embeddings,
        "rms_norm_eps": config.norm_epsilon,
        "torch_dtype": "bfloat16",
    }


def from_huggingface(metadata: Mapping[str, Any], **overrides: Any) -> Config:
    """The ``Config`` a Hugging Face Qwen 3.5 ``config.json`` describes: the
    text decoder of the vision-language checkpoint, whose settings sit under
    ``text_config``. ``max_seq_len`` stays this module's default unless
    overridden."""

    text = metadata.get("text_config", metadata)
    kind = text.get("model_type", metadata.get("model_type"))
    if kind not in ("qwen3_5", "qwen3_5_text"):
        raise ValueError(f"not a Qwen 3.5 checkpoint: {kind!r}")
    if text.get("attention_bias", False):
        raise ValueError("Qwen 3.5 attention biases are not supported")
    rope = text.get("rope_parameters") or {}
    if rope.get("rope_type", "default") != "default":
        raise ValueError(f"unsupported rope_parameters {rope!r}")
    values = {
        "n_layers": int(text["num_hidden_layers"]),
        "d_model": int(text["hidden_size"]),
        "full_attention_interval": int(text["full_attention_interval"]),
        "n_heads": int(text["num_attention_heads"]),
        "n_kv_heads": int(text["num_key_value_heads"]),
        "head_dim": int(text["head_dim"]),
        "partial_rotary_factor": float(
            rope.get("partial_rotary_factor", text.get("partial_rotary_factor", 1.0))
        ),
        "lin_k_heads": int(text["linear_num_key_heads"]),
        "lin_v_heads": int(text["linear_num_value_heads"]),
        "lin_k_head_dim": int(text["linear_key_head_dim"]),
        "lin_v_head_dim": int(text["linear_value_head_dim"]),
        "lin_conv_kernel": int(text["linear_conv_kernel_dim"]),
        "d_ff": int(text["intermediate_size"]),
        "vocab_size": int(text["vocab_size"]),
        "rope_base": float(
            rope.get("rope_theta", text.get("rope_theta", 10_000_000.0))
        ),
        "tied_embeddings": bool(
            text.get("tie_word_embeddings", metadata.get("tie_word_embeddings", False))
        ),
        "norm_epsilon": float(text.get("rms_norm_eps", 1e-6)),
        "zero_centered_norm": True,
    }
    unknown = set(overrides) - {f.name for f in fields(Config)}
    if unknown:
        raise ValueError(f"unknown Qwen 3.5 fields {sorted(unknown)}")
    return Config(**{**values, **overrides})


def mrope_sections(head_dim, partial_rotary_factor):
    """The rotary pairs a Qwen vision-language checkpoint splits among its
    time, height and width positions, as the published models split them
    (32 pairs as 11, 11 and 10); a text sequence's three positions are one,
    so any split rotates the same."""

    pairs = int(head_dim * partial_rotary_factor) // 2
    base, extra = divmod(pairs, 3)
    return [base + (index < extra) for index in range(3)]
