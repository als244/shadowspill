# ShadowSpill

Run PyTorch models within a memory budget by planning which tensors to keep on
the device, spill to host or external storage, and recompute during backward.

[Install](docs/development/repository.md#setup) · [All docs](docs/README.md)

## Choose a starting point

| Goal | Guide |
| --- | --- |
| Train through the public service | [Complete client example](docs/examples/client-training.md), [client API](docs/python/client.md): settings, global inputs and results |
| Run a supplied text application | [Llama](examples/training/llama3/README.md), [Qwen3](examples/training/qwen3.md), [Qwen3.5 LoRA](examples/training/qwen35_lora.md) |
| Bring a custom model or control each phase | [Custom models](docs/python/custom-models.md), [build/profile/plan/execute](docs/python/phases.md), [complete example](docs/examples/generic-training.md) |
| Tune or extend the system | [Advanced guides](docs/python/advanced/README.md): partitioning, chunking, diagnostics and operator integration |

The service prepares and executes training through the existing engine. The
lightweight client needs Python and NumPy; model and GPU libraries stay on the
server. Migrated applications work end to end, with remaining model, precision
and deployment gaps listed in the [client guide](docs/python/client.md#current-coverage).

## Start training

After defining training settings and a global batch, the runtime API is:

```python
from shadowspill_client import SSH, ServiceClient

with ServiceClient(ssh=SSH("chicago", 8000)) as service:
    training = service.create_training_client(**setup).result()
    try:
        result = training.step(batch, optim={"lr": 0.001}).result()
        print(result.completed_updates, result.loss)
    finally:
        training.close().result()
```

The [complete example](docs/examples/client-training.md) defines `setup` and
`batch`, starts the server, and shows split updates, evaluation and checkpoint
resume. Setup and worker initialization happen before the runtime loop.

The [Python recipe runner](packages/workloads/src/shadowspill_workloads/recipes/README.md)
remains for callers still being migrated, including GLM. Direct build/profile/plan/
execute APIs remain available; the refactor is not release-complete.
