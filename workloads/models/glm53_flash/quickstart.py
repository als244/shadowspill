"""GLM-5.3-Flash LoRA configuration for the shared phase quickstart."""

from functools import partial

from workloads.models.shared.glm.recipe import experiment as lora_experiment

from .checkpoint import GLMCheckpoint
from .config import Config
from .implementations.mlops import Model

experiment = partial(lora_experiment, "glm53_flash", Config, GLMCheckpoint, Model)
