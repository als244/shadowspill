"""qwen3moe settings using the shared Qwen training hooks."""

from shadowspill_workloads.definitions.decoder import register as register_qwen
from shadowspill_workloads.definitions.decoder.configuration import (
    Settings as QwenSettings,
)

from . import config


class Settings(QwenSettings):
    family = "qwen3moe"
    catalog = config
    moe = True
    hybrid = False


def register(registry):
    register_qwen(registry, Settings)
