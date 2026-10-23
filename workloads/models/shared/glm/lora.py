"""The LoRA switch both GLM families use: freeze the base, wrap the dense
projections and train the routed experts' factors."""

from fnmatch import fnmatchcase

from mlops.lora import LoRAConfig, LoRAHead, LoRALinear
from torch import nn

from ..routed_lora import FACTORS
from .moe import MoE


def enable_lora(model, config):
    """Freeze the base and wrap only the family's main projection targets."""
    model.requires_grad_(False)
    recipe = LoRAConfig(
        rank=config.lora_rank,
        alpha=config.lora_alpha,
        factor_dtype=config.lora_factor_dtype,
    )
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        selected = (
            config.lora_head
            if name == "lm_head"
            else any(fnmatchcase(name, pattern) for pattern in model.LORA_TARGETS)
        )
        if not selected:
            continue
        parent, _, leaf = name.rpartition(".")
        wrapper = LoRAHead if name == "lm_head" else LoRALinear
        setattr(model.get_submodule(parent), leaf, wrapper(module, recipe))
    for layer in model.layers:
        if isinstance(layer.mlp, MoE):
            for name in FACTORS:
                getattr(layer.mlp, name).requires_grad_(True)
