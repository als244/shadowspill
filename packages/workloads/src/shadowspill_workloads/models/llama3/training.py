"""Llama registration using shared decoder preparation and initialization."""

from shadowspill_workloads.components import training_options
from shadowspill_workloads.components.decoder import register as register_decoder
from shadowspill_workloads.components.decoder.configuration import (
    Settings as DecoderSettings,
)

from . import config


class Settings(DecoderSettings):
    family = "llama3"
    catalog = config

    def __post_init__(self):
        super().__post_init__()
        if config.Config(**self.architecture).head_dim % 2:
            raise ValueError("Llama rotary head width must be even")


def create(settings):
    import torch

    from .implementations.mlops import Model

    # Preserve Llama's construction-then-cast behavior, including rotary buffers.
    model = Model(config.Config(**settings.architecture)).to(
        getattr(torch, settings.compute_dtype)
    )
    return training_options.adapt(model, settings)


def register(registry):
    register_decoder(registry, Settings, create=create)
