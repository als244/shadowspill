"""Shared precision and LoRA support for supplied MLOps model components."""

from contextlib import contextmanager
from dataclasses import replace
from importlib.util import find_spec

from shadowspill_client import LoRA
from shadowspill_client._records import integer, plain


def configure(model, *, label):
    values = plain(model.config)
    overrides = dict(model.precision)
    if unknown := overrides.keys() - {"compute_dtype", "grad_dtype", "lora_dtype"}:
        raise ValueError(f"unsupported {label} precision: {sorted(unknown)}")
    if model.implementation is not None:
        overrides["implementation"] = model.implementation
    if model.lora is not None:
        overrides["lora"] = model.lora.to_dict()
    for name, value in overrides.items():
        if name in values and values[name] != value:
            raise ValueError(f"conflicting {label} setting: {name}")
        values[name] = value
    if find_spec("mlops") is None:
        raise ValueError(f"packed {label} requires the installed MLOps provider")
    return values


def validate(config, *, label):
    for name in ("compute_dtype", "grad_dtype", "lora_dtype"):
        if getattr(config, name) not in ("float32", "bfloat16"):
            raise ValueError(f"unsupported {label} {name}")
    if config.lora is not None:
        lora = LoRA.from_dict(config.lora)
        if lora.dropout:
            raise ValueError(f"the installed {label} LoRA provider requires dropout=0")
        object.__setattr__(config, "lora", lora.to_dict())
    if config.lora_seed is not None:
        integer(config.lora_seed, "lora_seed")
        if config.lora is None:
            raise ValueError("lora_seed requires LoRA")


def adapt(model, config):
    if config.lora is not None:
        from shadowspill_workloads.lora import configure_lora

        lora = LoRA.from_dict(config.lora)
        model = configure_lora(
            model,
            rank=lora.rank,
            alpha=lora.alpha,
            targets=lora.targets,
            factor_dtype=config.lora_dtype,
        )
    return model


def for_build(config, context):
    """Carry declared input bounds and optimizer precision into capture."""
    config = replace(
        config,
        sequence_length_bound=context.layout.limits.get("max_sequence_length"),
    )
    if context.gradient_dtype is None:
        return config
    return replace(
        config,
        grad_dtype=(
            config.compute_dtype
            if context.gradient_dtype == "parameter"
            else context.gradient_dtype
        ),
    )


@contextmanager
def capture_context(config):
    import torch
    from mlops.dispatch import weight_gradients_at

    from shadowspill_workloads.models.shared.sequence_geometry import (
        sequence_length_bound,
    )

    with (
        weight_gradients_at(getattr(torch, config.grad_dtype)),
        sequence_length_bound(config.sequence_length_bound),
    ):
        yield
