# Examples

Start with the lightweight client for global-batch training. The service owns
preparation and workers. Direct phase examples also compose the existing build,
profiling, planning and execution APIs from `shadowspill`. See the
[Python reference](../python/README.md) for both interfaces.

## Start here

1. [Text training](client-training.md) — read a tokenized corpus, prepare a run,
   submit updates and save a checkpoint.
2. [Split updates and resume](../python/advanced/training-workflow.md) — accumulation,
   evaluation and checkpoint continuation.
3. [Qwen3 application](../../examples/qwen3.py), [Llama application](../../examples/training/llama3/README.md)
   and [Qwen3.5 LoRA](../../examples/training/qwen35_lora.md) — real token corpora
   through the same public client.
4. [Direct phases and saved-Plan Trainer](generic-training.md) — build/profile, CPU
   planning, bounded initialization and a saved-Plan loop in a fresh worker.
5. [Forward-only execution](forward-only.md) — save and execute a prediction Plan,
   or share model tensors with training for evaluation.
6. [Reusable planning and budget sweeps](reusable-planning.md) — replan saved
   profiles and produce simulated reports without a GPU.
7. [Custom stage partitioning](custom-partitioning.md) — control build-time FX
   boundaries through a small policy.
8. [GLM LoRA recipe script](../../examples/glm53_flash_lora.py)
   — real checkpoint/data, chunk and retention search, pinned-host budgets,
   evaluation and checkpointing through the normal saved phases.

The [Python recipes](../../packages/workloads/src/shadowspill_workloads/recipes/README.md) cover catalog full training
and supported checkpoint LoRA. The [model catalog](../../packages/workloads/src/shadowspill_workloads/MODELS.md)
lists model dimensions, precisions and available implementations.

## Direct phase process and storage requirements

- Install ShadowSpill with `./scripts/setup.sh`.
- The current GPU builder and measurement phase need a supported representative
  GPU. Artifact loading, profile application, planning and reporting work on CPU.
- Create Executor/Trainer in a fresh worker before other device allocations. It owns
  the runtime pools; earlier phases do not require them.
- Runtime inputs are rank-local. Examples use CPU tensors and bounded loaders.
- Save phase archives wherever convenient. Inductor/Triton caches remain separate
  from those archives and use their normal temporary/cache locations.

These requirements apply to direct phase callers. The public client submits
global `Datum` batches; the service owns worker launch and rank input delivery.

## Diagnostics

[Plan and execution diagnostics](diagnostics.md) compares simulated timelines
with a saved Executor's runtime trace. The
[saved-Plan capture tool](../../benchmarks/quickstart.md#nsight-captures-of-saved-winners)
collects Nsight reports. One Executor runs its admitted Plans sequentially.
See [failures and cleanup](../python/advanced/failures.md) for error handling.
