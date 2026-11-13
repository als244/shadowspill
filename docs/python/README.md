# Python guides and API

For end-to-end training of a supported model, start with the
[shared Python `train` runner](../../workloads/recipes/README.md). It owns the
full workflow; users supply a model source, data and training/resource settings.

For custom models, follow the [integration guide](custom-models.md). For direct
control, use the [phase-API example](../examples/generic-training.md) and [phase guide](phases.md).
`Trainer` runs a selected Plan; it is the execution layer beneath the shared
runner. Public phase entry points are available from `shadowspill`, with
PyTorch-specific policies in `shadowspill.pytorch`.

## Prepare and run

| Topic | Reference |
| --- | --- |
| Capture and finalize task implementations | [Build](api/build.md) |
| Checkpoint partitioning, execution chunking, retention and lifetimes | [Frontend policies](api/frontend.md) |
| Task timing, memory observation and cache reuse | [Profiling](api/profiling.md) |
| Copy/collective measurement and overlap | [Communication](api/communication.md) |
| Budgets, traversal, save/recompute search and candidate sweeps | [Planning](api/planning.md) |
| Saved Programs, profiles and Plans | [Artifacts](api/artifacts.md), [cache guide](artifact-store.md) |
| Inputs, schedules, evaluation and checkpoints | [Trainer](api/training.md) |
| Asynchronous calls and shared state across Plans | [Executor](api/executor.md) |
| Logical ownership, calibration and multi-worker execution | [Distributed training](api/distributed.md) |
| Pool capacities, calibration and external storage | [Pool memory](api/pool-memory.md), [SSD](api/ssd.md), [allocator integration](allocator.md) |
| Direct pool/runtime access for adapter tooling | [Low-level runtime](api/runtime.md) |

## Inspect and diagnose

- [Plan reports](plan-report.md) and [search fields](plan-report-fields.md).
- [Execution diagnostics](step-diagnostics.md) and [timing an invocation](api/timing.md).
- [Pool occupancy](occupancy.md) and [PressureFit cut traces](pressurefit-trace.md).
- [Diagnostic API](api/diagnostics.md), [planning JSON](planning-json.md) and [failure handling](failures.md).

## Extend

- [Operator integration](operator-integration.md): capture, derivatives, axes and reductions.
- [Prepared operators](operator-preparation.md): external kernel export and loading.
- [Custom partitioning example](../examples/custom-partitioning.md).
- [Framework-neutral API](api/neutral.md): IR, planning, simulation and runtime tooling.
- [Optional library adapters](../integrations/README.md).

Artifact loading, profile application, planning and reports are CPU-only.
`shadowspill.profiling` imports PyTorch and performs device measurements.
The neutral IR, planner and runtime interfaces do not import a framework.
For design rather than API usage, read the [architecture index](../architecture/README.md).
