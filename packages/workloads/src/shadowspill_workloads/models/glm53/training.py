"""glm53 settings using shared GLM and public decoder hooks."""

from shadowspill_workloads.models.shared.glm.training import register as register_glm
from shadowspill_workloads.models.shared.glm.training.configuration import (
    Settings as GLMSettings,
)

from . import config


class Settings(GLMSettings):
    family = "glm53"
    catalog = config
    moe = True
    hybrid = False


def register(registry):
    register_glm(registry, Settings)
