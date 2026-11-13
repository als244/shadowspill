# Python guides and API

| Goal | Guide |
| --- | --- |
| Train a supported model | [Python `train` runner](../../workloads/recipes/README.md) |
| Bring a custom model or control each phase | [Custom models](custom-models.md), [phase guide](phases.md), [complete example](../examples/generic-training.md) |
| Tune, inspect or extend the system | [Advanced guides](advanced/README.md) |

The `train` runner manages the complete workflow. `Trainer` runs an already
selected Plan. Direct phase entry points are available from `shadowspill`,
with PyTorch policies in `shadowspill.pytorch`.

## API reference

These pages define exact options and defaults. Start with a guide above for the
order of calls and a complete example.

| Topic | Reference |
| --- | --- |
| Capture and finalize task implementations | [Build](api/build.md) |
| Checkpoint partitioning, execution chunking, retention and lifetimes | [Frontend policies](api/frontend.md) |
| Task timing, memory observation and cache reuse | [Profiling](api/profiling.md) |
| Copy/collective measurement and overlap | [Communication](api/communication.md) |
| Budgets, traversal, save/recompute search and candidate sweeps | [Planning](api/planning.md) |
| Saved Programs, profiles and Plans | [Artifacts](api/artifacts.md) |
| Inputs, schedules, evaluation and checkpoints | [Trainer](api/training.md) |
| Asynchronous calls and shared state across Plans | [Executor](api/executor.md) |
| Logical ownership, calibration and multi-worker execution | [Distributed training](api/distributed.md) |
| Pool capacities, calibration and external storage | [Pool memory](api/pool-memory.md), [SSD](api/ssd.md) |
| Reports, traces and invocation timing | [Diagnostics](api/diagnostics.md), [timing](api/timing.md) |
| Direct pool/runtime access for adapter tooling | [Low-level runtime](api/runtime.md) |
| Framework-neutral tools | [IR, planning, simulation and runtime](api/neutral.md) |

Artifact loading, profile application, planning and reports are CPU-only.
`shadowspill.profiling` imports PyTorch and performs device measurements.
The neutral IR, planner and runtime interfaces do not import a framework.
