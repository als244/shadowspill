"""Declare the model and map its physical tensors to checkpoint entries."""

from dataclasses import replace

from mlops.lora import LoRALinear
from mlops.sequential_moe.precision.linear import FrozenLinear, FrozenLoRALinear
from mlops.sequential_moe.precision.storage import DENSE
from torch import nn

from .tensors import Copy, physical_state

LORA_FACTORS = {
    "lora_a",
    "lora_b",
    "lora_gate_up_a",
    "lora_gate_up_b",
    "lora_down_a",
    "lora_down_b",
}


def configure(
    reader,
    *,
    layer_limit,
    lora_rank,
    lora_alpha,
    lora_factor_dtype,
    lora_head,
    router_weight_dtype,
):
    c = reader.config_from_metadata(
        reader.metadata, gemm_precision=reader.gemm_precision
    )
    # The routed experts' checkpoint format chooses their default storage:
    # dense stays bf16, compressed (FP8, NVFP4) becomes one-copy block FP8.
    first = c.first_k_dense_replace
    prefix = f"{reader.source_prefix}layers.{first}.mlp.experts.0"
    _, spec = reader.weights.projection(
        prefix + ".gate_proj", (c.moe_intermediate_size, c.hidden_size), meta=True
    )
    reader.expert_format = spec.format
    storage = reader.expert_storage or ("bf16" if spec.format in DENSE else "fp8_block")
    c = reader.config_from_metadata(
        reader.metadata,
        expert_storage=storage,
        gemm_precision=reader.gemm_precision,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_factor_dtype=lora_factor_dtype,
        lora_head=lora_head,
    )
    if router_weight_dtype is not None:
        c = replace(c, router_weight_dtype=router_weight_dtype)
    if layer_limit is not None:
        if not 1 <= layer_limit <= c.num_hidden_layers:
            raise ValueError("layer_limit must select a nonempty decoder prefix")
        c = replace(c, num_hidden_layers=layer_limit)
    return c


def projection_copies(reader, destinations, prefix, spec, *, starts=None):
    keys = (
        ["weight"]
        if spec.format in DENSE
        else (
            ["weight", "weight_scale_inv"]
            if spec.format == "fp8"
            else ["weight_packed", "weight_scale", "weight_global_scale"]
        )
    )
    for i, (destination, key) in enumerate(zip(destinations, keys, strict=True)):
        src = prefix + "." + key
        shape = reader.weights.source.tensor(src, meta=True).shape
        start = starts[i] if starts is not None else None
        reader.copies.append(
            Copy(
                destination,
                src,
                start,
                start + shape[0] if start is not None else None,
                key == "weight_global_scale",
            )
        )


def replace_projections(reader, model):
    # Replace quantized dense projections with component-bearing modules.
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear | LoRALinear):
            continue
        prefix = reader.source_name(name + ".weight").removesuffix(".weight")
        parts, spec = reader.weights.projection(
            prefix, tuple(module.weight.shape), meta=True
        )
        if spec.format in DENSE:
            continue
        if name == "lm_head":
            raise ValueError(
                "GLM checkpoint output heads currently require dense storage"
            )
        parent, _, leaf = name.rpartition(".")
        replacement = FrozenLinear(parts, spec, gemm_precision=reader.gemm_precision)
        if isinstance(module, LoRALinear):
            replacement = FrozenLoRALinear(replacement, module.lora_config)
        model.get_submodule(parent).set_submodule(leaf, replacement)
        projection_copies(
            reader, [name + f".components.{i}" for i in range(spec.count)], prefix, spec
        )


def map_tensors(reader, model, c):
    targets = physical_state(model)
    bound = {copy.destination for copy in reader.copies}
    bound.update(
        destination
        for conversion in reader.conversions
        for destination in conversion.destinations
    )
    for name, value in targets.items():
        if name in bound:
            continue
        leaf = name.removesuffix(".compute").rsplit(".", 1)[-1]
        if c.lora_rank and leaf in LORA_FACTORS:
            reader.lora_initializers[name] = leaf.endswith("_a")
            continue
        if name.endswith(".self_attn.conv1d.weight"):
            width = value.shape[0] // 3
            prefix = reader.source_name(name).removesuffix("conv1d.weight")
            for i, letter in enumerate(("q", "k", "v")):
                reader.copies.append(
                    Copy(
                        name,
                        prefix + letter + "_conv1d.weight",
                        i * width,
                        (i + 1) * width,
                    )
                )
        else:
            source = reader.source_name(name.removesuffix(".compute"))
            reader.copies.append(Copy(name, source))
