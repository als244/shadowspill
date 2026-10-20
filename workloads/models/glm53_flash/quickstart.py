"""GLM-5.3-Flash FP8 LoRA quickstart, including local transfer sharding."""

from functools import partial

from workloads.models.shared.glm.recipe import experiment as lora_experiment

from .checkpoint import GLMCheckpoint
from .config import Config
from .implementations.mlops import Model

experiment = partial(lora_experiment, "glm53_flash", Config, GLMCheckpoint, Model)
