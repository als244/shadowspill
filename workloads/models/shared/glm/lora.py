"""The LoRA switch both GLM families use: freeze the base, wrap the dense
projections and train the routed experts' factors."""

from mlops.lora import LoRAConfig, LoRAHead, LoRALinear
from torch import nn

from .moe import FACTORS, MoE


def enable_lora(model, config):
    """Freeze the base and wrap the selected dense projections."""
    model.requires_grad_(False)
    recipe = LoRAConfig(
        rank=config.lora_rank,
        alpha=config.lora_alpha,
        factor_dtype=config.lora_factor_dtype,
    )
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear) or any(
            part in name for part in (".indexer.", ".shared_experts.")
        ):
            continue
        if name == "lm_head" and not config.lora_head:
            continue
        parent, _, leaf = name.rpartition(".")
        wrapper = LoRAHead if name == "lm_head" else LoRALinear
        setattr(model.get_submodule(parent), leaf, wrapper(module, recipe))
    for layer in model.layers:
        if isinstance(layer.mlp, MoE):
            for name in FACTORS:
                getattr(layer.mlp, name).requires_grad_(True)
