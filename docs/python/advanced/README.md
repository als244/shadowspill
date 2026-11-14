# Advanced guides

Start with [supported training](../../../packages/workloads/src/shadowspill_workloads/recipes/README.md) or the
[custom-model guide](../custom-models.md). These pages cover optional controls,
detailed diagnostics and extensions. Exact signatures and defaults live in the
[Python API reference](../README.md#api-reference).

## Tune training and search

| Task | Guide |
| --- | --- |
| Select microbatches, orderings, chunk sizes and retention candidates | [Training configuration](../../../packages/workloads/src/shadowspill_workloads/recipes/configuration.md) |
| Separate checkpoint regions from execution chunks | [Partitioning example](../../examples/custom-partitioning.md), [build policies](../api/frontend.md) |
| Replan saved profiles across budgets and search choices | [Reusable planning](../../examples/reusable-planning.md), [planning API](../api/planning.md) |
| Benchmark a model factory or a saved Plan | [Quickstart](../../../benchmarks/quickstart.md), [Plan runner](../../../benchmarks/plan_runner/README.md) |

## Inspect artifacts and execution

| Topic | Guide |
| --- | --- |
| Saved artifacts and cache reuse | [Artifact store](artifact-store.md) |
| Predicted performance and memory | [Plan reports](plan-report.md), [search-field reference](plan-report-fields.md) |
| Measured tasks, transfers and stalls | [Execution diagnostics](step-diagnostics.md), [diagnostics example](../../examples/diagnostics.md) |
| Pool occupancy and planner decisions | [Occupancy](occupancy.md), [PressureFit traces](pressurefit-trace.md) |
| Serialized planning objects | [Planning JSON](planning-json.md) |
| Errors and cleanup | [Failure handling](failures.md) |

## Integrate operators and runtimes

| Topic | Guide |
| --- | --- |
| Custom operators, derivatives, axes and reductions | [Operator integration](operator-integration.md) |
| Export and load external kernels | [Operator preparation](operator-preparation.md) |
| PyTorch allocation, storage and stream behavior | [Allocator integration](allocator.md) |
| Existing library adapters | [Optional integrations](../../integrations/README.md) |

Engine design belongs in [architecture](../../architecture/README.md);
native backend interfaces belong in the [C reference](../../c/README.md).
