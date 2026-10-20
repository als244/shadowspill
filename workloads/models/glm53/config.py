"""GLM-5.3's published sparse-MLA architecture and training precision."""

from collections.abc import Mapping
from dataclasses import dataclass, fields

import torch


@dataclass(frozen=True)
class Config:
    hidden_size: int = 6144
    vocab_size: int = 154880
    num_hidden_layers: int = 78
    first_k_dense_replace: int = 3
    intermediate_size: int = 12288
    moe_intermediate_size: int = 2048
    num_local_experts: int = 256
    num_experts_per_tok: int = 8
    n_shared_experts: int = 1
    num_attention_heads: int = 64
    q_lora_rank: int = 2048
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 192
    qk_rope_head_dim: int = 64
    v_head_dim: int = 256
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 2048
    indexer_types: tuple[str, ...] = ()
    rope_theta: float = 8_000_000.0
    rms_norm_eps: float = 1e-5
    routed_scaling_factor: float = 2.5
    swiglu_limit: float | None = None
    initializer_range: float = 0.02
    dtype: torch.dtype = torch.bfloat16
    expert_storage: str = "bf16"
    gemm_precision: str = "bf16"
    lora_rank: int = 0
    lora_alpha: float = 32.0
    lora_factor_dtype: torch.dtype = torch.bfloat16
    lora_head: bool = False

    def __post_init__(self):
        if self.dtype != torch.bfloat16:
            raise ValueError("GLM sparse attention requires BF16 compute")
        if self.num_attention_heads % 16 or self.kv_lora_rank not in (
            64,
            128,
            256,
            512,
        ):
            raise ValueError(
                "Sparse attention requires 16-head tiles and a supported latent width"
            )
        if self.qk_rope_head_dim != 64 or self.index_head_dim < self.qk_rope_head_dim:
            raise ValueError(
                "GLM positional sparse attention requires 64 RoPE dimensions"
            )
        if (
            self.num_hidden_layers < 1
            or not 1 <= self.num_experts_per_tok <= self.num_local_experts
        ):
            raise ValueError("Invalid layer/expert configuration")
        if self.lora_rank < 0 or self.index_topk < 1:
            raise ValueError("Invalid LoRA rank or indexer selection width")
        if self.expert_storage not in ("bf16", "fp8", "fp8_block"):
            raise ValueError("Unsupported expert storage")
        kinds = self.indexer_types or tuple(
            "full" if i < 3 or i % 4 == 2 else "shared"
            for i in range(self.num_hidden_layers)
        )
        if (
            len(kinds) < self.num_hidden_layers
            or kinds[0] != "full"
            or any(value not in ("full", "shared") for value in kinds)
        ):
            raise ValueError("Indexer types must start with full and cover every layer")
        object.__setattr__(
            self, "indexer_types", tuple(kinds[: self.num_hidden_layers])
        )

    @property
    def mlp_layer_types(self):
        return tuple(
            "dense" if i < self.first_k_dense_replace else "sparse"
            for i in range(self.num_hidden_layers)
        )

    @classmethod
    def tiny(cls, **overrides):
        values = dict(
            hidden_size=128,
            vocab_size=128,
            num_hidden_layers=4,
            first_k_dense_replace=1,
            intermediate_size=128,
            moe_intermediate_size=128,
            num_local_experts=4,
            num_experts_per_tok=2,
            num_attention_heads=16,
            q_lora_rank=32,
            kv_lora_rank=64,
            qk_nope_head_dim=32,
            v_head_dim=32,
            index_n_heads=2,
            index_head_dim=64,
            index_topk=8,
            indexer_types=("full", "shared", "full", "shared"),
        )
        values.update(overrides)
        return cls(**values)


def from_huggingface(metadata, **overrides):
    expected = {
        "model_type": "glm_moe_dsa",
        "attention_bias": False,
        "attention_dropout": 0.0,
        "hidden_act": "silu",
        "moe_layer_freq": 1,
        "scoring_func": "sigmoid",
        "norm_topk_prob": True,
        "n_group": 1,
        "topk_group": 1,
        "topk_method": "noaux_tc",
        "rope_interleave": True,
        "indexer_rope_interleave": True,
        "tie_word_embeddings": False,
    }
    for key, value in expected.items():
        if metadata.get(key, value) != value:
            raise ValueError(f"Unsupported GLM-5.3 {key}: {metadata[key]!r}")
    rope = metadata.get("rope_parameters", {})
    if rope.get("rope_type", "default") != "default":
        raise ValueError("Only the published default GLM-5.3 RoPE is supported")
    accepted = {f.name for f in fields(Config)} - {"dtype"}
    values = {k: v for k, v in metadata.items() if k in accepted}
    values["num_local_experts"] = metadata["n_routed_experts"]
    values["rope_theta"] = rope.get("rope_theta", 8_000_000.0)
    values.update(overrides)
    config = Config(**values)
    if (
        "mlp_layer_types" in metadata
        and tuple(metadata["mlp_layer_types"]) != config.mlp_layer_types
    ):
        raise ValueError(
            "Checkpoint MLP layer types disagree with configured architecture"
        )
    return config


#: The named shapes the catalog serves: the published architecture, and the
#: small random fixture the quickstart factory builds with ``tiny=True``.
PRESETS: Mapping[str, Config] = {"published": Config(), "tiny": Config.tiny()}
