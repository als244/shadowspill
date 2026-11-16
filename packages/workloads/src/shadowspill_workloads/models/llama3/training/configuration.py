"""CPU Llama settings and worker-only construction/precision scope."""

from dataclasses import asdict, dataclass, field, replace
from math import isfinite

from shadowspill.training.definitions import ModelBuild
from shadowspill_workloads.definitions import training_options
from shadowspill_workloads.definitions.text_allocations import CONTRACTS
from shadowspill_workloads.initialization.pretrained import describe
from shadowspill_workloads.models.llama3.config import PRESETS, Config


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

    def __post_init__(self):
        if self.preset not in PRESETS:
            raise ValueError(f"unknown Llama preset: {self.preset}")
        values = {**asdict(PRESETS[self.preset]), **self.architecture}
        for name in (
            "n_layers",
            "d_model",
            "n_heads",
            "n_kv_heads",
            "d_ff",
            "vocab_size",
            "max_seq_len",
        ):
            if type(values[name]) is not int or values[name] <= 0:
                raise ValueError(f"Llama {name} must be a positive integer")
        for name in ("rope_base", "norm_epsilon", "initializer_range"):
            value = values[name]
            if type(value) not in (int, float) or not isfinite(value) or value <= 0:
                raise ValueError(f"Llama {name} must be finite and positive")
        if type(values["tied_embeddings"]) is not bool:
            raise ValueError("Llama tied_embeddings must be boolean")
        config = replace(PRESETS[self.preset], **self.architecture)
        if config.head_dim % 2:
            raise ValueError("Llama rotary head width must be even")
        if self.implementation != "mlops":
            raise ValueError("the installed packed Llama definition requires mlops")
        training_options.validate(self, label="Llama")
        object.__setattr__(self, "architecture", asdict(config))


def create(config):
    import torch

    from shadowspill_workloads.models.llama3.implementations.mlops import Model

    model = Model(Config(**config.architecture)).to(
        getattr(torch, config.compute_dtype)
    )
    return training_options.adapt(model, config)


def build(config, context):
    config = training_options.for_build(config, context)
    return ModelBuild(
        asdict(config), grad_dtype=config.grad_dtype, allocation_invariants=CONTRACTS
    )


def initialization(config, context):
    return describe("llama3", Config(**config.architecture), context)
