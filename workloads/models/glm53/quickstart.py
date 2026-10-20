"""GLM-5.3 FP8 LoRA quickstart, sharing the Flash training configuration."""

from functools import partial

from torch._inductor import config as inductor_config

from workloads.models.shared.glm.recipe import experiment as lora_experiment

from .checkpoint import GLMCheckpoint
from .config import Config
from .implementations.mlops import Model

# The indexer's discrete top-k depends on the checkpoint's intermediate BF16
# rounding. Fusing these casts away can change selected keys, not just produce
# a small continuous rounding error. Keep this policy throughout capture,
# profiling and execution; the profiling environment includes it in cache keys.
experiment = partial(
    lora_experiment,
    "glm53",
    Config,
    GLMCheckpoint,
    Model,
    partial(inductor_config.patch, emulate_precision_casts=True),
)
