"""GLM-5.3-Flash language-backbone dimensions and precision configuration."""

from collections.abc import Mapping
from dataclasses import dataclass, fields

import torch


@dataclass(frozen=True)
class Config:
    """The dimensions, and the storage and precision choices.

    ``expert_storage`` is how the routed experts are held: ``"bf16"``,
    ``"fp8_block"`` for frozen `BlockFP8Weight`s (E4M3 once, a scale per
    128x128 block), or ``"fp8"`` for frozen `QuackFP8Weight` containers (two
    E4M3 orientations with per-row descales, as many bytes as bf16); the FP8
    forms' GEMMs run in FP8. A compressed checkpoint (FP8 or NVFP4) loads as
    ``"fp8_block"`` unless asked otherwise, an FP8 one byte for byte.
    ``gemm_precision`` is the dense projections' GEMM precision with a
    compressed checkpoint, BF16 only.
    """

    hidden_size: int = 4096
    vocab_size: int = 154880
    num_hidden_layers: int = 45
    first_k_dense_replace: int = 3
    intermediate_size: int = 12288
    moe_intermediate_size: int = 2048
    num_local_experts: int = 288
    num_experts_per_tok: int = 8
    n_shared_experts: int = 1
    num_attention_heads: int = 64
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 256
    v_head_dim: int = 256
    linear_num_heads: int = 64
    linear_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    linear_lower_bound: float = -5.0
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_kpool: int = 4
    index_topk: int = 2048
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    rms_norm_eps: float = 1e-5
    routed_scaling_factor: float = 2.5
    swiglu_limit: float = 10.0
    initializer_range: float = 0.02
    dtype: torch.dtype = torch.bfloat16
    router_weight_dtype: torch.dtype = torch.bfloat16
    expert_storage: str = "bf16"
    gemm_precision: str = "bf16"
    lora_rank: int = 0
    lora_alpha: float = 32.0
    lora_factor_dtype: torch.dtype = torch.float32
    lora_head: bool = False

    def __post_init__(self):
        if self.router_weight_dtype not in (torch.bfloat16, torch.float32):
            raise ValueError("router_weight_dtype must be BF16 or FP32")
        if self.dtype != torch.bfloat16:
            raise ValueError(
                "The GLM sparse-attention implementation currently requires BF16"
            )
        if self.num_attention_heads % 16 or self.kv_lora_rank not in (
            64,
            128,
            256,
            512,
        ):
            raise ValueError(
                "Sparse attention requires a multiple of 16 heads "
                "and a supported latent width"
            )
        if not 1 <= self.num_experts_per_tok <= self.num_local_experts:
            raise ValueError("Invalid expert/top-k configuration")
        if not -5 <= self.linear_lower_bound < 0:
            raise ValueError("The KDA safe-gate path requires a bound in [-5,0)")
        if self.lora_rank < 0:
            raise ValueError("lora_rank must be nonnegative")
        if self.expert_storage not in ("bf16", "fp8", "fp8_block"):
            raise ValueError(
                "expert_storage must be 'bf16', 'fp8' or 'fp8_block', "
                f"not {self.expert_storage!r}"
            )

    @property
    def layer_types(self):
        return tuple(
            "deepseek_sparse_attention" if i % 4 == 3 else "linear_attention"
            for i in range(self.num_hidden_layers)
        )

    @property
    def mlp_layer_types(self):
        return tuple(
            "dense" if i < self.first_k_dense_replace else "sparse"
            for i in range(self.num_hidden_layers)
        )

    @classmethod
    def tiny(cls, **kwargs):
        values = {
            "hidden_size": 64,
            "vocab_size": 128,
            "num_hidden_layers": 4,
            "first_k_dense_replace": 1,
            "intermediate_size": 128,
            "moe_intermediate_size": 128,
            "num_local_experts": 4,
            "num_experts_per_tok": 2,
            "num_attention_heads": 16,
            "q_lora_rank": 32,
            "kv_lora_rank": 64,
            "qk_nope_head_dim": 32,
            "v_head_dim": 32,
            "linear_num_heads": 2,
            "linear_head_dim": 64,
            "index_n_heads": 2,
            "index_head_dim": 32,
            "index_topk": 128,
        }
        values.update(kwargs)
        return cls(**values)


def from_huggingface(config, **overrides):
    text = config.get("text_config", config)
    fields_by_name = {field.name for field in fields(Config)}
    values = {key: value for key, value in text.items() if key in fields_by_name}
    # HF's dtype is a string; arithmetic precision is an explicit caller option.
    values.pop("dtype", None)
    values["num_local_experts"] = text["n_routed_experts"]
    linear = text.get("linear_attn_config", {})
    for name, old in {
        "linear_num_heads": "num_heads",
        "linear_head_dim": "head_dim",
        "linear_conv_kernel_dim": "short_conv_kernel_size",
        "linear_lower_bound": "gate_lower_bound",
    }.items():
        if name not in values and old in linear:
            values[name] = linear[old]
    expected = {
        "mla_use_nope": True,
        "qk_rope_head_dim": 0,
        "mhc": True,
        "scoring_func": "sigmoid",
        "norm_topk_prob": True,
        "n_group": 1,
        "topk_group": 1,
        "topk_method": "noaux_tc",
        "index_kpool_compress": True,
        "index_kpool_always_select_tail": True,
    }
    for key, value in expected.items():
        if text.get(key, value) != value:
            raise ValueError(
                f"Unsupported GLM architecture: {key}={text[key]!r}; expected {value!r}"
            )
    values.update(overrides)
    result = Config(**values)
    for key in ("layer_types", "mlp_layer_types"):
        if key in text and tuple(text[key]) != getattr(result, key):
            raise ValueError(
                f"Unsupported GLM {key}: checkpoint differs "
                "from configured architecture"
            )
    return result


#: The named shapes the catalog serves: the published architecture, and the
#: small random fixture the quickstart factory builds with ``tiny=True``.
PRESETS: Mapping[str, Config] = {"published": Config(), "tiny": Config.tiny()}
