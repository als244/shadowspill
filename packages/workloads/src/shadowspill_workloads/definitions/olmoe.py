"""Installed packed OLMoE model; secondary objectives remain caller-selected."""

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from functools import partial
from importlib.util import find_spec
from math import isfinite

from shadowspill.training.definitions import (
    ModelBuild,
    ModelDefinition,
    PreparationDefinition,
)
from shadowspill_client import LoRA
from shadowspill_client._records import plain
from shadowspill_workloads.initialization.pretrained import describe
from shadowspill_workloads.inputs.text import preparation as inputs
from shadowspill_workloads.models.olmoe.config import PRESETS, Config

from .moe_allocations import CONTRACTS
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
        if self.implementation != "mlops":
            raise ValueError("the installed packed OLMoE definition requires mlops")
        for name in ("compute_dtype", "grad_dtype", "lora_dtype"):
            if getattr(self, name) not in ("float32", "bfloat16"):
                raise ValueError(f"unsupported OLMoE {name}")
        if self.lora is not None:
            lora = LoRA.from_dict(self.lora)
            if lora.dropout:
                raise ValueError("the installed OLMoE LoRA provider requires dropout=0")
            object.__setattr__(self, "lora", lora.to_dict())
        object.__setattr__(self, "architecture", asdict(config))


def configure(model):
    values = plain(model.config)
    overrides = dict(model.precision)
    if unknown := overrides.keys() - {"compute_dtype", "grad_dtype", "lora_dtype"}:
        raise ValueError(f"unsupported OLMoE precision: {sorted(unknown)}")
    if model.implementation is not None:
        overrides["implementation"] = model.implementation
    if model.lora is not None:
        overrides["lora"] = model.lora.to_dict()
    for name, value in overrides.items():
        if name in values and values[name] != value:
            raise ValueError(f"conflicting OLMoE setting: {name}")
        values[name] = value
    if find_spec("mlops") is None:
        raise ValueError("packed OLMoE requires the installed MLOps provider")
    return values


def create(config):
    import torch

    from shadowspill_workloads.models.olmoe.implementations.mlops import Model

    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(getattr(torch, config.compute_dtype))
        model = Model(Config(**config.architecture))
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
        asdict(config), grad_dtype=config.grad_dtype, allocation_invariants=CONTRACTS
    )


@contextmanager
def capture_context(config):
    import torch
    from mlops.dispatch import weight_gradients_at

    with weight_gradients_at(getattr(torch, config.grad_dtype)):
        yield


def initialization(config, context):
    return describe("olmoe", Config(**config.architecture), context)


def register(registry):
    source = definition_source(
        (
            "definitions/olmoe.py",
            "definitions/moe_allocations.py",
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
            configure=configure,
            build=build,
            capture_context=capture_context,
            initialization=initialization,
        ),
        source=source,
    )
