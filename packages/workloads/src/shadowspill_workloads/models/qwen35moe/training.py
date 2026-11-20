"""qwen35moe settings using the shared decoder training hooks."""

from shadowspill_workloads.components.decoder import register as register_decoder
from shadowspill_workloads.components.decoder.configuration import (
    Settings as DecoderSettings,
)

from . import config


class Settings(DecoderSettings):
    family = "qwen35moe"
    catalog = config
    moe = True
    hybrid = True


def register(registry):
    register_decoder(registry, Settings)
