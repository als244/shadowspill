"""CPU Qwen settings; model and tensor providers load only in build workers."""

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from importlib import import_module
from importlib.util import find_spec
from math import isfinite
from types import ModuleType
from typing import ClassVar

from shadowspill.training.definitions import ModelBuild
from shadowspill_client import LoRA
from shadowspill_client._records import plain
from shadowspill_workloads.initialization.pretrained import describe
from shadowspill_workloads.models.qwen3moe import config as qwen3
from shadowspill_workloads.models.qwen35moe import config as qwen35

from .. import moe_provider


@dataclass(frozen=True)
class Settings:
    family: ClassVar[str]
    catalog: ClassVar[ModuleType]
    preset: str = "tiny"
    architecture: dict = field(default_factory=dict)
    implementation: str = "mlops"
    compute_dtype: str = "bfloat16"
    grad_dtype: str = "float32"
    lora_dtype: str = "bfloat16"
    lora: dict | None = None

    def __post_init__(self):
        if self.preset not in self.catalog.PRESETS:
            raise ValueError(f"unknown {self.family} preset: {self.preset}")
        values = {**asdict(self.catalog.PRESETS[self.preset]), **self.architecture}
        for name in (
            "n_layers",
            "d_model",
            "n_heads",
            "n_kv_heads",
            "head_dim",
            "n_experts",
            "top_k",
            "d_ff_expert",
            "vocab_size",
            "max_seq_len",
            "full_attention_interval",
            "lin_k_heads",
            "lin_v_heads",
            "lin_k_head_dim",
            "lin_v_head_dim",
            "lin_conv_kernel",
        ):
            if type(values[name]) is not int or values[name] <= 0:
                raise ValueError(f"Qwen {name} must be a positive integer")
        if type(values["d_ff_shared"]) is not int or values["d_ff_shared"] < 0:
            raise ValueError("Qwen d_ff_shared must be a nonnegative integer")
        for name in (
            "rope_base",
            "norm_epsilon",
            "initializer_range",
            "partial_rotary_factor",
            "router_aux_loss_coef",
        ):
            value = values[name]
            if (
                type(value) not in (int, float)
                or not isfinite(value)
                or (value < 0 if name == "router_aux_loss_coef" else value <= 0)
            ):
                raise ValueError(f"invalid Qwen {name}")
        for name in ("zero_centered_norm", "attention_gate"):
            if type(values[name]) is not bool:
                raise ValueError(f"Qwen {name} must be boolean")
        config = replace(self.catalog.PRESETS[self.preset], **self.architecture)
        if self.implementation != "mlops":
            raise ValueError("the installed packed Qwen definitions require mlops")
        for name in ("compute_dtype", "grad_dtype", "lora_dtype"):
            if getattr(self, name) not in ("float32", "bfloat16"):
                raise ValueError(f"unsupported Qwen {name}")
        if self.lora is not None:
            lora = LoRA.from_dict(self.lora)
            if lora.dropout:
                raise ValueError("the installed Qwen LoRA provider requires dropout=0")
            object.__setattr__(self, "lora", lora.to_dict())
        object.__setattr__(self, "architecture", asdict(config))


class Qwen3Settings(Settings):
    family = "qwen3moe"
    catalog = qwen3


class Qwen35Settings(Settings):
    family = "qwen35moe"
    catalog = qwen35


def configure(model):
    values = plain(model.config)
    overrides = dict(model.precision)
    if unknown := overrides.keys() - {"compute_dtype", "grad_dtype", "lora_dtype"}:
        raise ValueError(f"unsupported Qwen precision: {sorted(unknown)}")
    if model.implementation is not None:
        overrides["implementation"] = model.implementation
    if model.lora is not None:
        overrides["lora"] = model.lora.to_dict()
    for name, value in overrides.items():
        if name in values and values[name] != value:
            raise ValueError(f"conflicting Qwen setting: {name}")
        values[name] = value
    if find_spec("mlops") is None:
        raise ValueError("packed Qwen requires the installed MLOps provider")
    return values


def create(config):
    import torch

    model_class = import_module(
        f"shadowspill_workloads.models.{config.family}.implementations.mlops"
    ).Model
    previous = torch.get_default_dtype()
    try:
        # Match existing catalog construction: explicit FP32 decay parameters
        # and rotary buffers stay FP32 while ordinary projections use this dtype.
        torch.set_default_dtype(getattr(torch, config.compute_dtype))
        model = model_class(config.catalog.Config(**config.architecture))
    finally:
        torch.set_default_dtype(previous)
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


def build(config, context):
    return ModelBuild(
        asdict(config),
        grad_dtype=config.grad_dtype,
        operator_preparation=moe_provider.preparations(config.compute_dtype),
        allocation_invariants=moe_provider.allocations(config.compute_dtype),
    )


@contextmanager
def capture_context(config):
    import torch
    from mlops.dispatch import weight_gradients_at

    with weight_gradients_at(getattr(torch, config.grad_dtype)):
        yield


def initialization(config, context):
    return describe(
        config.family,
        config.catalog.Config(**config.architecture),
        context,
    )
