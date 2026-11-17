# Python guides and API

Start with the [training client](client.md) and the
[model, optimizer and objective catalog](catalog.md).

| Goal | Guide |
| --- | --- |
| Train through the public service | [Complete client example](../examples/client-training.md), [client API](client.md) |
| Run an editable training example | [Standalone demos](../../examples/demos/README.md) |
| Use a command-line training loop | [Training applications](../../examples/training/README.md) |
| Bring a custom model or control each phase | [Custom models](advanced/custom-definitions.md), [phase guide](phases.md) |
| Tune, inspect or extend the system | [Advanced guides](advanced/README.md) |
| Register a custom model, objective or optimizer | [Definition hooks](advanced/definitions.md) |

Setup and runtime are separate: `ServiceClient` describes and creates a run;
`TrainingClient` submits batches to ready workers. The service owns preparation,
initialization and execution. Direct phase entry points remain available from
`shadowspill`, with PyTorch policies in `shadowspill.pytorch`.

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
| Global batches, complete/split updates, evaluation and checkpoints | [Training client](client.md#runtime-operations) |
| Asynchronous calls and shared state across Plans | [Executor](api/executor.md) |
| Logical ownership, calibration and multi-worker execution | [Distributed training](api/distributed.md) |
| Pool capacities, calibration and external storage | [Pool memory](api/pool-memory.md), [SSD](api/ssd.md) |
| Reports, traces and invocation timing | [Diagnostics](api/diagnostics.md), [timing](api/timing.md) |
| Direct pool/runtime access for adapter tooling | [Low-level runtime](api/runtime.md) |
| Framework-neutral tools | [IR, planning, simulation and runtime](api/neutral.md) |

Artifact loading, profile application, planning and reports are CPU-only.
`shadowspill.profiling` imports PyTorch and performs device measurements.
The neutral IR, planner and runtime interfaces do not import a framework.
