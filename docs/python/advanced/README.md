# Advanced guides

Start with the [basic training example](../../examples/client-training.md).
These guides cover optional controls, custom definitions and detailed diagnostics.

## Training through the service

| Task | Guide |
| --- | --- |
| Accumulate gradients, evaluate and resume | [Complete workflow](training-workflow.md) |
| Pretrain, fine-tune, run GRPO/PPO actor updates or distill responses | [Objectives](objectives.md) |
| Tune preparation, reuse Plans or reconnect | [Service client reference](service-client.md) |
| Download Plans and start on other physical GPUs | [Saved Plans](saved-plans.md) |
| Supply a model, objective or optimizer | [Step-by-step guide](custom-definitions.md), [hook reference](definitions.md) |
| Work directly with build/profile/plan/execute | [Phase guide](../phases.md), [custom models](../custom-models.md) |

## Tune training and search

| Task | Guide |
| --- | --- |
| Override model tensor dtypes | [Parameter precision](precision.md) |
| Select microbatches, orderings, chunk sizes and retention candidates | [Preparation controls](service-client.md#preparation-and-local-planning), [build policies](definitions.md#preparation-and-capture) |
| Separate checkpoint regions from execution chunks | [Partitioning example](../../examples/custom-partitioning.md), [build policies](../api/frontend.md) |
| Replan saved profiles across budgets and search choices | [Reusable planning](../../examples/reusable-planning.md), [planning API](../api/planning.md) |
| Benchmark a model factory or a saved Plan | [Quickstart](../../../benchmarks/quickstart.md), [Plan runner](../../../benchmarks/plan_runner/README.md) |

## Inspect artifacts and execution

| Topic | Guide |
| --- | --- |
| Pending commands, losses and execution times | [Live command table](command-table.md) |
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
