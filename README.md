# ShadowSpill

Run PyTorch models within a memory budget by planning which tensors to keep on
the device, spill to host or external storage, and recompute during backward.

[Install](docs/development/repository.md#setup) · [All docs](docs/README.md)

## Choose a starting point

| Goal | Guide |
| --- | --- |
| Train a supported model | [Python training](packages/workloads/src/shadowspill_workloads/recipes/README.md): model, data, budgets and search options |
| Bring a custom model or control each phase | [Custom models](docs/python/custom-models.md), [build/profile/plan/execute](docs/python/phases.md), [complete example](docs/examples/generic-training.md) |
| Tune or extend the system | [Advanced guides](docs/python/advanced/README.md): partitioning, chunking, diagnostics and operator integration |

## Start training

Use the shared Python runner for build, profiling, search and training:

```python
from shadowspill_workloads.recipes import Model, train

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
measures candidates, selects the fastest predicted Plan that fits, and saves
artifacts under `run_dir`. See the [training guide](packages/workloads/src/shadowspill_workloads/recipes/README.md)
for supported models, data preparation, LoRA and distributed settings, or edit
the [commented training script](examples/train_text.py).
