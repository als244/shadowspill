"""GLM family adapters for the shared phase quickstart."""

from functools import partial

from torch._inductor import config as inductor_config

from workloads.models.glm53.checkpoint import GLMCheckpoint as GLM53Checkpoint
from workloads.models.glm53.config import Config as GLM53Config
from workloads.models.glm53.implementations.mlops import Model as GLM53Model
from workloads.models.glm53_flash.checkpoint import GLMCheckpoint as FlashCheckpoint
from workloads.models.glm53_flash.config import Config as FlashConfig
from workloads.models.glm53_flash.implementations.mlops import Model as FlashModel

from .factory import experiment

# The GLM-5.3 indexer's discrete top-k depends on intermediate BF16 rounding.
# Preserve those casts during build; later phases use the finalized task code.
glm53 = partial(
    experiment,
    "glm53",
    GLM53Config,
    GLM53Checkpoint,
    GLM53Model,
    partial(inductor_config.patch, emulate_precision_casts=True),
)
glm53_flash = partial(
    experiment, "glm53_flash", FlashConfig, FlashCheckpoint, FlashModel
)
