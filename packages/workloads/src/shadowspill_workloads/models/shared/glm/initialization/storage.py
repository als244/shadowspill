"""Construct GLM's declared checkpoint precision without opening weight files."""

import torch
from mlops.lora import LoRALinear
from mlops.sequential_moe.precision.linear import FrozenLinear, FrozenLoRALinear
from mlops.sequential_moe.precision.storage import WeightSpec
from torch import nn


def apply(model, storage, source_name):
    if storage is None:
        return model
    dtypes = storage["tensor_dtypes"]
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear | LoRALinear):
            continue
        source = source_name(name + ".weight")
        if dtypes.get(source) != "float8_e4m3fn":
            continue
        if name == "lm_head":
            raise ValueError("GLM output heads require dense checkpoint storage")
        spec = WeightSpec("fp8", *module.weight.shape, 128, 128)
        parts = spec.empty(device=module.weight.device)
        parts[1] = parts[1].to(
            getattr(torch, dtypes[source.removesuffix(".weight") + ".weight_scale_inv"])
        )
        replacement = FrozenLinear(parts, spec)
        if isinstance(module, LoRALinear):
            replacement = FrozenLoRALinear(replacement, module.lora_config)
        parent, _, leaf = name.rpartition(".")
        model.get_submodule(parent).set_submodule(leaf, replacement)

    for name, value in (*model.named_parameters(), *model.named_buffers()):
        if name.endswith(".self_attn.conv1d.weight"):
            prefix = source_name(name).removesuffix("conv1d.weight")
            choices = {dtypes[prefix + letter + "_conv1d.weight"] for letter in "qkv"}
            if len(choices) != 1:
                raise ValueError(
                    "joined GLM convolutions require matching storage dtypes"
                )
            dtype = choices.pop()
        else:
            dtype = dtypes.get(source_name(name))
        if dtype is None or value.dtype == getattr(torch, dtype):
            continue
        target = value.to(getattr(torch, dtype))
        if isinstance(value, nn.Parameter):
            target = nn.Parameter(target, requires_grad=value.requires_grad)
        parent, _, leaf = name.rpartition(".")
        setattr(model.get_submodule(parent), leaf, target)
    return model
