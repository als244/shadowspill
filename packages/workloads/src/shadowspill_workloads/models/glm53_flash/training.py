"""glm53_flash settings using shared GLM and public decoder hooks."""

from shadowspill_workloads.models.shared.glm.training import register as register_glm
from shadowspill_workloads.models.shared.glm.training.configuration import (
    Settings as GLMSettings,
)

from . import config


class Settings(GLMSettings):
    family = "glm53_flash"
    catalog = config
    moe = True
    hybrid = True


def register(registry):
    register_glm(registry, Settings)
