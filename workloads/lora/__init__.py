"""LoRA targeting for supplied examples: a selected module that declares
routed experts carries their factors itself
(``workloads.models.shared.routed_lora``); every other selected module is
replaced by the generic conversion in MLOps."""

from fnmatch import fnmatchcase

import torch
from mlops.lora import LoRAConfig, apply_lora, parameter_report
from torch import nn

from workloads.models.shared.routed_lora import add_factors


def configure_lora(
    model,
    *,
    rank=32,
    alpha=32.0,
    factor_dtype="bfloat16",
    targets=None,
    head="frozen",
    shared_experts=False,
    trainable_base=(),
):
    """Use the model family's ``LORA_TARGETS`` by default. Custom models without
    a preset select their linear projections and declared routed experts.

    Inside a module that declares routed experts (``expert_shapes``) nothing
    else is selected unless ``shared_experts`` adds its linear projections
    other than its router; inside a module that declares ``takes_lora =
    False`` nothing is selected. Embeddings, norms and the head remain frozen
    unless explicitly selected. head is frozen, lora or full; trainable_base
    contains parameter globs. Explicit targets, module globs, replace the
    default module selection; a selected module that declares routed experts
    gains their factors in place.
    """
    if head not in ("frozen", "lora", "full"):
        raise ValueError("head must be frozen, lora or full")
    if any(hasattr(module, "lora_config") for module in model.modules()):
        raise ValueError("model already contains LoRA modules")
    config = LoRAConfig(
        rank=rank,
        alpha=alpha,
        factor_dtype=getattr(torch, factor_dtype)
        if isinstance(factor_dtype, str)
        else factor_dtype,
    )
    modules = dict(model.named_modules())
    owners = [
        name for name, module in modules.items() if hasattr(module, "expert_shapes")
    ]
    closed = [
        name
        for name, module in modules.items()
        if getattr(module, "takes_lora", True) is False
    ]

    def within(name, roots):
        return next((root for root in roots if name.startswith(root + ".")), None)

    if targets is None:
        preset = getattr(model, "LORA_TARGETS", None)
        targets = [
            name
            for name, module in modules.items()
            if (preset is None or any(fnmatchcase(name, pattern) for pattern in preset))
            and (
                name in owners
                or (
                    isinstance(module, nn.Linear)
                    and name != "lm_head"
                    and within(name, owners) is None
                    and within(name, closed) is None
                )
            )
        ]
    selected = []
    for pattern in targets:
        names = [name for name in modules if fnmatchcase(name, pattern)]
        if not names:
            raise ValueError(f"target {pattern!r} matched no modules")
        selected += [name for name in names if name not in selected]
    if shared_experts:
        shared_preset = getattr(model, "LORA_SHARED_TARGETS", None)
        selected += [
            name
            for name, module in modules.items()
            if isinstance(module, nn.Linear)
            and (owner := within(name, owners)) is not None
            and module is not getattr(modules[owner], "router", None)
            and (
                shared_preset is None
                or any(fnmatchcase(name, pattern) for pattern in shared_preset)
            )
            and name not in selected
        ]
    routed = [
        modules[name] for name in selected if hasattr(modules[name], "expert_shapes")
    ]
    if any(getattr(module, "experts", None) is not None for module in routed):
        raise ValueError("expert-parallel experts take no LoRA factors")
    dense = [name for name in selected if not hasattr(modules[name], "expert_shapes")]
    if head == "lora":
        dense.append("lm_head")
    if head == "full":
        trainable_base = [*trainable_base, "lm_head.weight"]
    if dense:
        model = apply_lora(model, config, targets=dense, trainable_base=trainable_base)
    else:
        _freeze(model, trainable_base)
    for module in routed:
        add_factors(module, config)
    return model


def _freeze(model, trainable_base):
    """Freeze every parameter but the ``trainable_base`` globs, as the
    generic conversion does when it replaces modules."""

    parameters = dict(model.named_parameters(remove_duplicate=False))
    trainable = set()
    for pattern in trainable_base:
        names = [name for name in parameters if fnmatchcase(name, pattern)]
        if not names:
            raise ValueError(f"trainable_base {pattern!r} matched no parameters")
        trainable.update(id(parameters[name]) for name in names)
    for parameter in parameters.values():
        parameter.requires_grad_(id(parameter) in trainable)


__all__ = ["configure_lora", "parameter_report"]
