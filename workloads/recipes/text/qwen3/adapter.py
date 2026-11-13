"""Qwen3 MoE's eight-GPU preset over the shared catalog training adapter."""

from workloads.recipes.text import catalog

evaluation_metric = catalog.evaluation_metric
initializer = catalog.initializer
inputs = catalog.inputs
setup = catalog.setup
validate = catalog.validate
candidates = catalog.candidates
trainer_options = catalog.trainer_options

defaults = {
    **catalog.defaults,
    "model": "qwen3moe-30b-a3b",
    "parallelism": "dp",
    "ranks": 8,
    "sequence_length": 4096,
    "sequences_per_step": 64,
    "microbatch_sizes": [1, 2, 4, 8, 16, 32],
    "learning_rate": 3e-4,
    "schedule_steps": 10,
    "execution_gib": 77,
    "spill_gib": 96,
    "external_headroom_mib": 768,
    "eval_every": 0,
    "checkpoint_every": 5,
    "initialization": "random",
}
path_options = catalog.path_options
