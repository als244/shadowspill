"""qwen35moe settings using the shared Qwen training hooks."""

from shadowspill_workloads.definitions.decoder import register as register_qwen
from shadowspill_workloads.definitions.decoder.configuration import (
    Settings as QwenSettings,
)

from . import config


class Settings(QwenSettings):
    family = "qwen35moe"
    catalog = config
    moe = True
    hybrid = True


def register(registry):
    register_qwen(registry, Settings)
