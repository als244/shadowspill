"""qwen35 settings using the shared decoder training hooks."""

from shadowspill_workloads.components.decoder import register as register_decoder
from shadowspill_workloads.components.decoder.configuration import (
    Settings as DecoderSettings,
)

from . import config


class Settings(DecoderSettings):
    family = "qwen35"
    catalog = config
    moe = False
    hybrid = True


def register(registry):
    register_decoder(registry, Settings)
