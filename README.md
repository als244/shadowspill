# ShadowSpill

Run PyTorch models within a memory budget by planning which tensors to keep on
the device, spill to host or external storage, and recompute during backward.

[Install](docs/development/repository.md#setup) ·
[Train from Python](workloads/recipes/README.md) ·
[Python guide](docs/python/README.md) · [All docs](docs/README.md)

## Train a supported model

Use the shared Python runner for build, profiling, search and training:

```python
from workloads.recipes import Model, train

model = Model.from_config("llama3-1b")  # Full training from random weights.
train(model, {
    "run_dir": "runs/llama-pretrain",
    "data": "/data/fineweb-llama",      # Text tokenized for this model.
    "sequence_length": 2048,
    "sequences_per_step": 4,
    "microbatch_sizes": [1, 2, 4],     # Measure candidates and select a Plan.
    "execution_gib": 24,               # Per-GPU budget; adapt to your hardware.
    "spill_gib": 64,                   # Maximum pinned host capacity per rank.
    "steps": 100,
})
```

Run once from a configured checkout, without an outer `torchrun`. The runner
launches fresh phase workers and saves artifacts under `run_dir`. It selects the
fastest predicted Plan that fits; optional chunk and retention candidate lists
extend the search.

The catalog path covers Llama, OLMoE, dense Qwen and Qwen MoE full training.
Checkpoint-based LoRA covers Qwen3.5 MoE and GLM-5.3-Flash. Start with the
[training guide](workloads/recipes/README.md) for supported model names, data,
defaults, distributed settings and saved-phase reuse, or edit a
[commented language-model script](workloads/recipes/text/example.py).

## Custom models and direct control

Follow the [custom-model guide](docs/python/custom-models.md) and
[complete language-model phase example](docs/examples/generic-training.md).
They show where to supply your model, loss, data and initialization while using
the same compiler, profiler, planner and runtime.

The [phase APIs](docs/python/phases.md) expose preparation and search directly.
`shadowspill.Trainer` runs an already selected Plan and provides update loops,
logging and checkpoints; see its [reference](docs/python/api/training.md) for
custom execution loops. Model-specific setup stays outside the core engine.

[Quickstart](benchmarking/quickstart.md) supports model-factory experiments.
The separate [training harness](training/README.md) supports eager comparisons
and reference checks.
