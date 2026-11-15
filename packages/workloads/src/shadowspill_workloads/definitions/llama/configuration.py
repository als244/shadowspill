"""CPU Llama settings and worker-only construction/precision scope."""

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from importlib.util import find_spec
from math import isfinite

from shadowspill.training.definitions import ModelBuild, ModelInitialization
from shadowspill_client import LoRA
from shadowspill_client._records import plain
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
        for name in ("compute_dtype", "grad_dtype", "lora_dtype"):
            if getattr(self, name) not in ("float32", "bfloat16"):
                raise ValueError(f"unsupported Llama {name}")
        if self.lora is not None:
            lora = LoRA.from_dict(self.lora)
            if lora.dropout:
                raise ValueError("the installed Llama LoRA provider requires dropout=0")
            object.__setattr__(self, "lora", lora.to_dict())
        object.__setattr__(self, "architecture", asdict(config))


def configure(model):
    values = plain(model.config)
    overrides = dict(model.precision)
    if unknown := overrides.keys() - {"compute_dtype", "grad_dtype", "lora_dtype"}:
        raise ValueError(f"unsupported Llama precision: {sorted(unknown)}")
    if model.implementation is not None:
        overrides["implementation"] = model.implementation
    if model.lora is not None:
        overrides["lora"] = model.lora.to_dict()
    for name, value in overrides.items():
        if name in values and values[name] != value:
            raise ValueError(f"conflicting Llama setting: {name}")
        values[name] = value
    if find_spec("mlops") is None:
        raise ValueError("packed Llama requires the installed MLOps provider")
    return values


def create(config):
    import torch

    from shadowspill_workloads.models.llama3.implementations.mlops import Model

    model = Model(Config(**config.architecture)).to(
        getattr(torch, config.compute_dtype)
    )
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
    return ModelBuild(asdict(config), grad_dtype=config.grad_dtype)


@contextmanager
def capture_context(config):
    import torch
    from mlops.dispatch import weight_gradients_at

    with weight_gradients_at(getattr(torch, config.grad_dtype)):
        yield


def initialization(config, context):
    if context.weights.kind != "random":
        raise ValueError(
            "installed Llama currently requires random weights or a checkpoint"
        )

    @contextmanager
    def open_loader(runtime):
        from shadowspill_workloads.recipes.text.initialization import Initialize

        yield Initialize(
            "llama3", Config(**config.architecture), seed=context.weights.seed
        )

    # Existing generator uses 256K-element deterministic blocks. This covers its
    # random and rotary temporaries, the destination and frequency vector; no
    # model-sized tensor is constructed by the loader.
    width = config.architecture["d_model"] // config.architecture["n_heads"]
    return ModelInitialization(
        peak_bytes=(32 << 20) + 32 * width, chunk_bytes=1 << 20, open=open_loader
    )
