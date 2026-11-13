# Examples

These examples compose the public build, profiling, planning and execution APIs
from `shadowspill`. Preparation produces saved artifacts; fresh execution workers
consume Plans. See the [phase guide](../python/phases.md) for the
API boundaries and the [Python reference](../python/README.md) for details.

## Start here

1. [Commented training script](../../workloads/recipes/text/example.py) — the shared
   runner with a real language model and next-token loss.
2. [Training and checkpointing](generic-training.md) — build/profile, CPU planning,
   bounded initialization and a Trainer loop in a fresh worker.
3. [Forward-only execution](forward-only.md) — save and execute a prediction Plan,
   or share model tensors with training for evaluation.
4. [Reusable planning and budget sweeps](reusable-planning.md) — replan saved
   profiles and produce simulated reports without a GPU.
5. [Custom stage partitioning](custom-partitioning.md) — control build-time FX
   boundaries through a small policy.
6. [Commented GLM LoRA script](../../workloads/recipes/text/glm53_flash_lora/example.py)
   — real checkpoint/data, chunk and retention search, pinned-host budgets,
   evaluation and checkpointing through the normal saved phases.

The [Python recipes](../../workloads/recipes/README.md) cover catalog full training
and supported checkpoint LoRA. The [model catalog](../../workloads/MODELS.md)
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
