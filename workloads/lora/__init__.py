"""LoRA targeting for supplied examples; generic conversion lives in MLOps."""

import torch
from mlops.lora import LoRAConfig, LoRALinear, apply_lora, parameter_report
from torch import nn


def configure_lora(
    model,
    *,
    rank=32,
    alpha=32.0,
    factor_dtype="float32",
    targets=None,
    head="frozen",
    shared_experts=False,
    trainable_base=(),
):
    """Select attention/mixer, dense MLP and routed-expert projections by default.

    Embeddings, routers, norms, shared experts and the head remain frozen unless
    explicitly selected. head is frozen, lora or full; trainable_base contains
    parameter globs. Explicit targets replace the default module selection.
    """
    from workloads.mlops._qwen_moe.experts import MoE as QwenMoE
    from workloads.mlops.olmoe import MoE as OLMoE
    from workloads.pytorch.olmoe import MoE as PyTorchOLMoE

    from .experts import OLMoELoRA, PyTorchOLMoELoRA, QwenMoELoRA

    if head not in ("frozen", "lora", "full"):
        raise ValueError("head must be frozen, lora or full")
    config = LoRAConfig(
        rank=rank,
        alpha=alpha,
        factor_dtype=getattr(torch, factor_dtype)
        if isinstance(factor_dtype, str)
        else factor_dtype,
    )
    if targets is None:
        targets = []
        for name, module in model.named_modules():
            if not name.startswith("blocks."):
                continue
            if isinstance(module, (OLMoE, PyTorchOLMoE, QwenMoE)):
                if getattr(module, "experts", None) is not None:
                    raise ValueError(
                        "EP conversion requires a dedicated expert LoRA replacement"
                    )
                targets.append(name)
            elif isinstance(module, nn.Linear) and ".moe." not in name:
                targets.append(name)
    targets = list(targets)
    if head == "lora":
        targets.append("lm_head")
    if head == "full":
        trainable_base = [*trainable_base, "lm_head.weight"]
    model = apply_lora(
        model,
        config,
        targets=targets,
        trainable_base=trainable_base,
        converters={
            OLMoE: OLMoELoRA,
            QwenMoE: QwenMoELoRA,
            PyTorchOLMoE: PyTorchOLMoELoRA,
        },
    )
    if shared_experts:
        # These children are preserved by the routed-expert replacement. Do
        # this separately so a generic transform never selects parent+child.
        for name, module in tuple(model.named_modules()):
            if ".moe.shared." in name and isinstance(module, nn.Linear):
                parent, _, leaf = name.rpartition(".")
                setattr(model.get_submodule(parent), leaf, LoRALinear(module, config))
    return model


__all__ = ["configure_lora", "parameter_report"]
