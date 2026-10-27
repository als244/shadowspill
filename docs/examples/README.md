# Examples

These examples compose the public build, profiling, planning and execution APIs
from `shadowspill`. Preparation produces saved artifacts; fresh execution workers
consume Plans. See the [phase guide](../architecture/offline-pipeline.md) for the
API boundaries and the [Python reference](../python/README.md) for details.

## Start here

1. [Training and checkpointing](generic-training.md) — build/profile, CPU planning,
   bounded initialization and a Trainer loop in a fresh worker.
2. [Forward-only execution](forward-only.md) — save and execute a prediction Plan,
   or share model tensors with training for evaluation.
3. [Reusable planning and budget sweeps](reusable-planning.md) — replan saved
   profiles and produce simulated reports without a GPU.
4. [Custom stage partitioning](custom-partitioning.md) — control build-time FX
   boundaries through a small policy.

The [README recipes](../../README.md#real-model-training) cover Llama, Qwen3.5 LoRA,
Qwen3 distributed training and GLM Flash LoRA. The [model catalog](../../workloads/MODELS.md)
lists model dimensions, precisions and available implementations.

## Process and storage requirements

- Install ShadowSpill with `./scripts/setup.sh`.
- The current GPU builder and measurement phase need a supported representative
  GPU. Artifact loading, profile application, planning and reporting work on CPU.
- Create Executor/Trainer in a fresh worker before other device allocations. It owns
  the runtime pools; earlier phases do not require them.
- Runtime inputs are rank-local. Examples use CPU tensors and bounded loaders.
- Save phase archives wherever convenient. Inductor/Triton caches remain separate
  from those archives and use their normal temporary/cache locations.

## Diagnostics

[Plan and execution diagnostics](diagnostics.md) compares simulated timelines
with a saved Executor's runtime trace. The
[saved-Plan capture tool](../../benchmarking/quickstart.md#nsight-captures-of-saved-winners)
collects Nsight reports. One Executor runs its admitted Plans sequentially.
See [failures and cleanup](../python/failures.md) for error handling.
