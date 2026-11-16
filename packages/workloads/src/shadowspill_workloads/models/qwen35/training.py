"""qwen35 settings using the shared Qwen training hooks."""

from shadowspill_workloads.definitions.qwen import register as register_qwen
from shadowspill_workloads.definitions.qwen.configuration import (
    Settings as QwenSettings,
)

from . import config


class Settings(QwenSettings):
    family = "qwen35"
    catalog = config
    moe = False
    hybrid = True


def register(registry):
    register_qwen(registry, Settings)
