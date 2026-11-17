"""OLMoE registration using shared decoder preparation and initialization."""

from shadowspill_workloads.definitions.decoder import register as register_decoder
from shadowspill_workloads.definitions.decoder.configuration import (
    Settings as DecoderSettings,
)

from . import config


class Settings(DecoderSettings):
    family = "olmoe"
    catalog = config
    moe = True

    def __post_init__(self):
        super().__post_init__()
        if self.architecture["head_dim"] % 2:
            raise ValueError("OLMoE rotary head width must be even")


def register(registry):
    register_decoder(registry, Settings)
