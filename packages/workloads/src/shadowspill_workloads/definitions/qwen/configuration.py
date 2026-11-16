"""CPU Qwen settings; model and tensor providers load only in build workers."""

from dataclasses import asdict, dataclass, field, replace
from importlib import import_module
from math import isfinite
from types import ModuleType
from typing import ClassVar

from shadowspill.training.definitions import ModelBuild
from shadowspill_workloads.initialization.pretrained import describe
from shadowspill_workloads.models.qwen3 import config as qwen3
from shadowspill_workloads.models.qwen3moe import config as qwen3moe
from shadowspill_workloads.models.qwen35 import config as qwen35
from shadowspill_workloads.models.qwen35moe import config as qwen35moe

from .. import expert_parallel, moe_provider, training_options
from ..hybrid_allocations import CONTRACTS as HYBRID_CONTRACTS
from ..text_allocations import CONTRACTS as TEXT_CONTRACTS


@dataclass(frozen=True)
class Settings:
    family: ClassVar[str]
    catalog: ClassVar[ModuleType]
    moe: ClassVar[bool] = False
    hybrid: ClassVar[bool] = False
    preset: str = "tiny"
    architecture: dict = field(default_factory=dict)
    implementation: str = "mlops"
    compute_dtype: str = "bfloat16"
    grad_dtype: str = "float32"
    lora_dtype: str = "bfloat16"
    lora: dict | None = None
    lora_seed: int | None = None
    expert_options: dict | None = None

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
            "d_ff",
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
            if name not in values:
                continue
            if type(values[name]) is not int or values[name] <= 0:
                raise ValueError(f"Qwen {name} must be a positive integer")
        if "d_ff_shared" in values and (
            type(values["d_ff_shared"]) is not int or values["d_ff_shared"] < 0
        ):
            raise ValueError("Qwen d_ff_shared must be a nonnegative integer")
        for name in (
            "rope_base",
            "norm_epsilon",
            "initializer_range",
            "partial_rotary_factor",
            "router_aux_loss_coef",
        ):
            if name not in values:
                continue
            value = values[name]
            if (
                type(value) not in (int, float)
                or not isfinite(value)
                or (value < 0 if name == "router_aux_loss_coef" else value <= 0)
            ):
                raise ValueError(f"invalid Qwen {name}")
        for name in ("zero_centered_norm", "attention_gate", "tied_embeddings"):
            if name in values and type(values[name]) is not bool:
                raise ValueError(f"Qwen {name} must be boolean")
        config = replace(self.catalog.PRESETS[self.preset], **self.architecture)
        implementations = ("mlops", "mlops_ep") if self.moe else ("mlops",)
        if self.implementation not in implementations:
            raise ValueError(f"{self.family} requires {' or '.join(implementations)}")
        training_options.validate(self, label="Qwen")
        object.__setattr__(self, "architecture", asdict(config))
        if self.moe:
            expert_parallel.validate(self)
        elif self.expert_options is not None:
            raise ValueError("dense Qwen does not use expert resource settings")


class Qwen3Settings(Settings):
    family = "qwen3"
    catalog = qwen3


class Qwen35Settings(Settings):
    family = "qwen35"
    catalog = qwen35
    hybrid = True


class Qwen3MoESettings(Settings):
    family = "qwen3moe"
    catalog = qwen3moe
    moe = True


class Qwen35MoESettings(Settings):
    family = "qwen35moe"
    catalog = qwen35moe
    moe = True
    hybrid = True


def create(config):
    import torch

    model_class = import_module(
        f"shadowspill_workloads.models.{config.family}.implementations.mlops"
    ).Model
    if config.implementation == "mlops_ep":
        from shadowspill_workloads.models.shared.expert_parallel import resources

        if config.expert_options is None:
            raise ValueError("EP model construction requires a described ModelBuild")
        return resources.construct(
            model_class,
            config.catalog.Config(**config.architecture),
            resources.configuration(config.expert_options["config"]),
            config.expert_options["layers"],
        )
    previous = torch.get_default_dtype()
    try:
        # Match existing catalog construction: explicit FP32 decay parameters
        # and rotary buffers stay FP32 while ordinary projections use this dtype.
        torch.set_default_dtype(getattr(torch, config.compute_dtype))
        model = model_class(config.catalog.Config(**config.architecture))
    finally:
        torch.set_default_dtype(previous)
    return training_options.adapt(model, config)


def build(config, context):
    if config.implementation == "mlops_ep":
        return expert_parallel.build(
            config, config.catalog.Config(**config.architecture), config.family, context
        )
    return ModelBuild(
        asdict(config),
        grad_dtype=config.grad_dtype,
        operator_preparation=moe_provider.preparations(config.compute_dtype)
        if config.moe
        else (),
        allocation_invariants=moe_provider.allocations(config.compute_dtype)
        if config.moe
        else {**TEXT_CONTRACTS, **(HYBRID_CONTRACTS if config.hybrid else {})},
    )


def initialization(config, context):
    if config.implementation == "mlops_ep":
        return expert_parallel.initialization(
            config, config.catalog.Config(**config.architecture), config.family, context
        )
    return describe(
        config.family,
        config.catalog.Config(**config.architecture),
        context,
    )
