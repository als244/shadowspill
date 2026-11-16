"""Installed packed OLMoE model; secondary objectives remain caller-selected."""

from dataclasses import asdict, dataclass, field, replace
from functools import partial
from math import isfinite

from shadowspill.training.definitions import (
    ModelBuild,
    ModelDefinition,
    PreparationDefinition,
)
from shadowspill_workloads.initialization.pretrained import describe
from shadowspill_workloads.inputs.text import preparation as inputs
from shadowspill_workloads.models.olmoe.config import PRESETS, Config

from . import expert_parallel, moe_provider, training_options
from .source import definition_source

PRESET = "shadowspill/olmoe_packed@1"


@dataclass(frozen=True)
class Settings:
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
        if self.preset not in PRESETS:
            raise ValueError(f"unknown OLMoE preset: {self.preset}")
        values = {**asdict(PRESETS[self.preset]), **self.architecture}
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
        ):
            if type(values[name]) is not int or values[name] <= 0:
                raise ValueError(f"OLMoE {name} must be a positive integer")
        for name in ("rope_base", "initializer_range"):
            value = values[name]
            if type(value) not in (int, float) or not isfinite(value) or value <= 0:
                raise ValueError(f"OLMoE {name} must be finite and positive")
        config = replace(PRESETS[self.preset], **self.architecture)
        if config.head_dim % 2:
            raise ValueError("OLMoE rotary head width must be even")
        if self.implementation not in ("mlops", "mlops_ep"):
            raise ValueError(
                "the installed OLMoE definition requires mlops or mlops_ep"
            )
        training_options.validate(self, label="OLMoE")
        object.__setattr__(self, "architecture", asdict(config))
        expert_parallel.validate(self)


def create(config):
    import torch

    from shadowspill_workloads.models.olmoe.implementations.mlops import Model

    if config.implementation == "mlops_ep":
        from shadowspill_workloads.models.shared.expert_parallel import resources

        if config.expert_options is None:
            raise ValueError("EP model construction requires a described ModelBuild")
        return resources.construct(
            Model,
            Config(**config.architecture),
            resources.configuration(config.expert_options["config"]),
            config.expert_options["layers"],
        )
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(getattr(torch, config.compute_dtype))
        model = Model(Config(**config.architecture))
    finally:
        torch.set_default_dtype(previous)
    return training_options.adapt(model, config)


def build(config, context):
    if config.implementation == "mlops_ep":
        return expert_parallel.build(
            config, Config(**config.architecture), "olmoe", context
        )
    return ModelBuild(
        asdict(config),
        grad_dtype=config.grad_dtype,
        operator_preparation=moe_provider.preparations(config.compute_dtype),
        allocation_invariants=moe_provider.allocations(config.compute_dtype),
    )


def initialization(config, context):
    if config.implementation == "mlops_ep":
        return expert_parallel.initialization(
            config, Config(**config.architecture), "olmoe", context
        )
    return describe("olmoe", Config(**config.architecture), context)


def register(registry):
    source = definition_source(
        (
            "definitions/olmoe.py",
            "definitions/training_options.py",
            "definitions/expert_parallel.py",
            "definitions/expert_allocations.py",
            "definitions/moe_allocations.py",
            "definitions/hybrid_allocations.py",
            "definitions/moe_provider.py",
            "definitions/text_allocations.py",
            "inputs/text/preparation.py",
            "initialization",
            "models/olmoe",
            "models/shared",
            "lora",
        ),
        providers=(
            "mlops",
            "flash_attn_interface",
            "flash_attn_3",
            "flash_attn_config",
            "huggingface_hub",
            "httpx",
            "httpcore",
            "httpx2",
            "httpcore2",
            *moe_provider.PROVIDERS,
            "moonep",
        ),
    )
    registry.add_preparation(
        PreparationDefinition(
            PRESET,
            partial(inputs.defaults, preset=PRESET),
            inputs.buffers,
            inputs.representative,
        ),
        source=source,
    )
    registry.add_model(
        ModelDefinition(
            "shadowspill/olmoe@1",
            Settings,
            create,
            inputs.schema,
            capabilities=("target_logprobs", "router_statistics"),
            input_adapter=inputs.adapter,
            preparation=PRESET,
            configure=partial(training_options.configure, label="OLMoE"),
            build=build,
            capture_context=training_options.capture_context,
            gradient_destination_rules=moe_provider.gradient_destination_rules,
            initialization=initialization,
            temporary_initialization=initialization,
        ),
        source=source,
    )
