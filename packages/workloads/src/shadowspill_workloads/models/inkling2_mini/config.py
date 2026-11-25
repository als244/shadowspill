"""Fixed-loop Inkling text geometry. Configuration inspection requires no framework."""

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Config:
    n_layers: int = 24
    d_model: int = 2304
    n_heads: int = 24
    n_kv_heads: int = 8
    head_dim: int = 128
    d_rel: int = 16
    rel_extent: int = 1024
    window: int = 1024
    local_layer_ids: tuple[int, ...] = tuple(i for i in range(24) if i % 6 != 5)
    conv_kernel: int = 4
    block_size: int = 6
    core_start: int = 6
    core_end: int = 18
    passes: int = 3
    gate_rank: int = 64
    n_experts: int = 128
    top_k: int = 8
    d_ff_expert: int = 1152
    embedding_vocab_size: int = 201024
    vocab_size: int = 200058
    max_seq_len: int = 1048576
    norm_epsilon: float = 1e-6
    qk_clip: float = 22.627416997969522
    rel_clip: float = 8.0
    log_floor: int = 8192
    log_alpha: float = 0.1
    route_scale: float = 8.0
    gate_cap: float = 4.0
    up_cap: float = 25.0
    logit_cap: float = 30.0
    initializer_range: float = 0.02
    expert_formats: tuple[str, ...] = ("nvfp4",) * 20 + ("mxfp8",) * 4
    expert_storage: str = "dense"
    expert_compute: str = "auto"
    attention_query_chunk_size: int = 0
    attention_kv_heads_per_chunk: int = 0
    attention_backend: str = "auto"
    head_fp32_projection: bool = True

    def use_fused_attention(self, tokens):
        return (
            bool(self.attention_kv_heads_per_chunk)
            or self.attention_backend == "fused"
            or (
                self.attention_backend == "auto"
                and not self.attention_query_chunk_size
                and tokens >= 128
                and self.head_dim in (16, 32, 64, 128)
            )
        )

    def __post_init__(self):
        if type(self.head_fp32_projection) is not bool:
            raise ValueError("head_fp32_projection must be a boolean")
        if (
            type(self.attention_kv_heads_per_chunk) is not int
            or not 0 <= self.attention_kv_heads_per_chunk <= self.n_kv_heads
        ):
            raise ValueError(
                "attention_kv_heads_per_chunk must be between zero and n_kv_heads"
            )
        if self.attention_kv_heads_per_chunk and (
            self.attention_backend == "eager" or self.attention_query_chunk_size
        ):
            raise ValueError(
                "K/V head chunking requires fused attention without query chunking"
            )
        if self.attention_backend not in ("auto", "eager", "fused"):
            raise ValueError("attention_backend must be auto, eager or fused")
        if self.attention_backend == "fused" and self.attention_query_chunk_size:
            raise ValueError(
                "explicit query blocking and fused attention select different backends"
            )
        if (
            type(self.attention_query_chunk_size) is not int
            or self.attention_query_chunk_size < 0
        ):
            raise ValueError("attention_query_chunk_size must be a nonnegative integer")
        for name in ("local_layer_ids", "expert_formats"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if not (
            0 < self.core_start < self.core_end <= self.n_layers
            and self.passes >= 1
            and self.block_size > 0
            and self.core_start % self.block_size == 0
            and self.core_end % self.block_size == 0
        ):
            raise ValueError("loop boundaries must delimit complete AttnRes blocks")
        if len(set(self.local_layer_ids)) != len(self.local_layer_ids) or any(
            i not in range(self.n_layers) for i in self.local_layer_ids
        ):
            raise ValueError("invalid local attention layer indices")
        if (
            min(
                self.n_heads,
                self.n_kv_heads,
                self.conv_kernel,
                self.log_floor,
                self.gate_rank,
                self.d_model,
                self.head_dim,
                self.d_rel,
                self.d_ff_expert,
                self.max_seq_len,
                self.rel_extent,
                self.window,
            )
            <= 0
            or self.n_heads % self.n_kv_heads
            or not 0 < self.top_k <= self.n_experts
            or not 0 < self.vocab_size <= self.embedding_vocab_size
        ):
            raise ValueError(
                "invalid Inkling head, expert, convolution or vocabulary geometry"
            )
        if (
            self.expert_storage not in {"auto", "dense", "bf16", "packed"}
            or self.expert_compute not in {"auto", "checkpoint", "bf16"}
            or len(self.expert_formats) != self.n_layers
            or set(self.expert_formats) - {"dense", "nvfp4", "mxfp8"}
        ):
            raise ValueError("expected one supported expert format per physical layer")
        if (
            self.expert_storage in {"bf16", "dense"}
            and self.expert_compute == "checkpoint"
        ):
            raise ValueError(
                "checkpoint expert arithmetic requires the frozen, packed banks"
            )
        for fmt in set(self.expert_formats) - {"dense"}:
            group = 16 if fmt == "nvfp4" else 32
            if self.d_model % group or self.d_ff_expert % group:
                raise ValueError(
                    f"{fmt} expert columns must divide into groups of {group}"
                )


PRESETS = {
    "published": Config(),
    "tiny": Config(
        n_layers=4,
        d_model=32,
        n_heads=4,
        n_kv_heads=2,
        head_dim=8,
        d_rel=4,
        rel_extent=8,
        window=4,
        local_layer_ids=(0, 1, 3),
        block_size=1,
        core_start=1,
        core_end=3,
        gate_rank=8,
        n_experts=4,
        top_k=2,
        d_ff_expert=32,
        embedding_vocab_size=96,
        vocab_size=89,
        max_seq_len=256,
        qk_clip=4.0,
        rel_clip=3.0,
        log_floor=4,
        expert_formats=("dense",) * 4,
    ),
}


def from_huggingface(metadata, **overrides):
    t = metadata.get("text_config", metadata)
    loop = t["loop_config"]
    if (
        loop.get("is_varloop", False)
        or t.get("n_shared_experts", 0)
        or t.get("q_bias", False)
        or t.get("o_bias", False)
        or not t.get("attn_res_normalize_values", True)
        or not t.get("use_value_norm_no_gamma", True)
        or not t.get("use_gate_bias", True)
    ):
        raise ValueError(
            "unsupported Inkling loop, norm, bias or shared expert settings"
        )
    if any(
        t.get("swa_" + field, t[field]) != t[field]
        for field in ("num_attention_heads", "num_key_value_heads", "head_dim")
    ):
        raise ValueError("Inkling Mini expects matching local and global head geometry")
    c = Config(
        n_layers=t["num_hidden_layers"],
        d_model=t["hidden_size"],
        n_heads=t["num_attention_heads"],
        n_kv_heads=t["num_key_value_heads"],
        head_dim=t["head_dim"],
        d_rel=t["d_rel"],
        rel_extent=t["rel_extent"],
        window=t["sliding_window_size"],
        local_layer_ids=tuple(t["local_layer_ids"]),
        conv_kernel=t["sconv_kernel_size"],
        block_size=t["n_layers_per_attn_res_block"],
        core_start=loop["core_start"],
        core_end=loop["core_end"],
        passes=loop["num_passes"],
        gate_rank=loop["gate_rank"],
        n_experts=t["n_routed_experts"],
        top_k=t["num_experts_per_tok"],
        d_ff_expert=t["intermediate_size"],
        embedding_vocab_size=t["vocab_size"],
        vocab_size=t["unpadded_vocab_size"],
        max_seq_len=t["model_max_length"],
        norm_epsilon=t["rms_norm_eps"],
        qk_clip=t["qk_l2_clip_threshold"],
        rel_clip=t["rel_l2_clip_threshold"],
        log_floor=t["log_scaling_n_floor"],
        log_alpha=t["log_scaling_alpha"],
        route_scale=t["route_scale"],
        gate_cap=t["mlp_gate_softcap"],
        up_cap=t["mlp_up_softcap"],
        logit_cap=t["final_logit_softcapping"],
        expert_storage="auto",
        expert_formats=("dense",) * t["num_hidden_layers"],
    )
    return replace(c, **overrides)


def _check(tensors, key, dtypes, shape):
    info = tensors.get(key, {})
    if info.get("dtype") not in dtypes or tuple(info.get("shape", ())) != shape:
        raise ValueError(f"invalid expert tensor {key}, expected {shape} and {dtypes}")


def storage_formats(tensors, config):
    """Identify and validate each expert payload, scale and global factor."""
    formats = []
    for layer in range(config.n_layers):
        kinds = []
        for bank, rows, cols in (
            ("w13", 2 * config.d_ff_expert, config.d_model),
            ("w2", config.d_model, config.d_ff_expert),
        ):
            key = f"model.llm.layers.{layer}.mlp.experts.{bank}_weight"

            dtype = tensors.get(key, {}).get("dtype")
            if dtype == "U8":
                kind = "nvfp4"
                _check(tensors, key, {"U8"}, (config.n_experts, rows, cols // 2))
                _check(
                    tensors,
                    key + "_scale",
                    {"F8_E4M3"},
                    (config.n_experts, rows, cols // 16),
                )
                _check(tensors, key + "_amax", {"F32"}, (config.n_experts,))
                _check(tensors, key + "_outer_scale_denominator", {"F32"}, (1,))
            elif dtype == "F8_E4M3":
                kind = "mxfp8"
                _check(tensors, key, {"F8_E4M3"}, (config.n_experts, rows, cols))
                _check(
                    tensors,
                    key + "_scale",
                    {"U8"},
                    (config.n_experts, rows, cols // 32),
                )
            else:
                kind = "dense"
                _check(tensors, key, {"BF16", "F32"}, (config.n_experts, rows, cols))
            kinds.append(kind)
        if kinds[0] != kinds[1]:
            raise ValueError(
                f"Inkling gate/up and down encodings differ in layer {layer}"
            )
        formats.append(kinds[0])
    return tuple(formats)


ACTIVATION_RECIPES = {
    "nvfp4": {
        "amax_type": "per_token",
        "block_size": 16,
        "format": "nvfp4_e2m1",
        "rounding": "nearest",
        "scale_dtype": "float8_e4m3fn",
        "scale_selection_rule": "fouroversix_mae",
    },
    "mxfp8": {
        "block_size": 32,
        "format": "mxfp8_e4m3",
        "rounding": "nearest",
        "scale_dtype": "ue8m0",
    },
}


def validate_recipe(recipe, formats, tensors):
    if set(formats) == {"dense"} and recipe is None:
        return
    if not isinstance(recipe, dict) or recipe.get("quant_method") != "inkling2":
        raise ValueError(
            "compressed Inkling experts require the Inkling hf_quant_config.json recipe"
        )
    required = {
        f"model.llm.layers.{i}.mlp.experts.{bank}_weight": fmt
        for i, fmt in enumerate(formats)
        for bank in ("w13", "w2")
        if fmt != "dense"
    }
    seen = set()
    for group in recipe["model"]["parameter_groups"]:
        for name in group["tensors"]:
            if not name.startswith("model.llm."):
                continue
            if name in seen or name not in tensors:
                raise ValueError(f"duplicate or absent Inkling recipe tensor {name}")
            seen.add(name)
            fmt = required.get(name)
            if fmt is None:
                expected_dtype = {"F32": "float32", "BF16": "bfloat16"}.get(
                    tensors[name]["dtype"]
                )
                if (
                    group.get("weight_transform") != "cast"
                    or group.get("required_activation_quantization") is not None
                    or group.get("output_dtype") != expected_dtype
                    or group.get("compute_dtype") != expected_dtype
                    or (
                        name == "model.llm.unembed.weight"
                        and group.get("fwd_output_dtype") != "float32"
                    )
                ):
                    raise ValueError(f"unsupported Inkling cast recipe for {name}")
            elif (
                group.get("compute_dtype") != fmt
                or group.get("weight_transform") != fmt + "_nearest"
                or group.get("required_activation_quantization")
                != ACTIVATION_RECIPES[fmt]
                or group.get("weight_block_shape")
                != ([1, 16] if fmt == "nvfp4" else [32, 32])
                or group.get("output_dtype")
                != ("uint8" if fmt == "nvfp4" else "float8_e4m3fn")
                or (
                    fmt == "nvfp4"
                    and group.get("weight_scale_format") != "amax_denominator"
                )
            ):
                raise ValueError(
                    f"Inkling expert recipe disagrees with stored tensor {name}"
                )
    text_weights = {
        key
        for key in tensors
        if key.startswith("model.llm.")
        and not (
            ".mlp.experts." in key
            and key.endswith(("_scale", "_amax", "_outer_scale_denominator"))
        )
    }
    if seen != text_weights:
        raise ValueError("Inkling recipe does not cover the complete text checkpoint")


def describe(metadata, tensors, architecture):
    config = Config(**architecture)
    original = from_huggingface(metadata)
    for field in (
        "local_layer_ids",
        "core_start",
        "core_end",
        "passes",
        "block_size",
        "gate_rank",
        "embedding_vocab_size",
        "d_rel",
        "rel_extent",
        "window",
        "conv_kernel",
    ):
        if getattr(original, field) != getattr(config, field):
            raise ValueError(
                f"pretrained override changes Inkling checkpoint geometry: {field}"
            )
    formats = storage_formats(tensors, config)
    validate_recipe(metadata.get("hf_quant_config.json"), formats, tensors)
    return {
        "architecture": {
            **architecture,
            "local_layer_ids": list(config.local_layer_ids),
            "expert_formats": list(formats),
        }
    }
