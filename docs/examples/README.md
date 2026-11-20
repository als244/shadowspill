# Examples

## Training through the service

Start with the [README training examples](../../README.md) or the standalone
client demos below. Each demo keeps its model, batch limits and memory budget
visible and saves predicted and measured HTML reports.

| Task | Example |
| --- | --- |
| Prepare data and fine-tune | [Text training](client-training.md) |
| Run full-parameter tuning, LoRA or distributed pretraining | [Standalone demos](../../examples/demos/README.md) |
| Use schedules, evaluation and checkpoint continuation from a CLI | [Training applications](../../examples/training/README.md) |
| Accumulate gradients, evaluate and resume | [Complete workflow](../python/advanced/training-workflow.md) |
| Train long contexts with computation chunks | [Long-context LoRA](../python/advanced/long-context.md) |
| Compare hypothetical budgets or transfer speeds locally | [Offline sweeps](../python/advanced/offline-sweeps.md) |
| Supply a model, objective or optimizer | [Writing extensions](../python/advanced/writing-extensions.md) |

The [catalog](../python/catalog.md) lists built-in models, optimizers and objectives.

## Direct engine examples

These examples call the build, profile, plan and execute APIs directly:

- [Forward-only execution](forward-only.md): prediction and evaluation Plans.
- [Reusable planning](reusable-planning.md): plan saved profiles without a GPU.
- [Custom partitioning](custom-partitioning.md): control capture boundaries.
- [Diagnostics](diagnostics.md): compare simulated and measured timelines.

Build and profiling need a representative GPU. Artifact inspection, planning and
reports work on CPU. Direct execution uses rank-local inputs and creates Executor
in a fresh worker before device allocations. The service manages those workers
and input delivery for public-client applications.
