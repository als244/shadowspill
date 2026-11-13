# ShadowSpill documentation

ShadowSpill captures a fixed-shape PyTorch computation, measures its compiled
tasks, and plans tensor residency, transfers and recomputation within memory
budgets. Execution loads a saved Plan in a fresh worker. Trainer supplies the
training loop; applications supply models, objectives and data.

## Start with a task

| I want to… | Start here |
| --- | --- |
| Install and run a small training example | [Setup](development/repository.md#setup), [training example](examples/generic-training.md) |
| Build, profile, plan and execute separately | [Phase guide](python/phases.md) |
| Train a model from a readable script | [Python recipes](../workloads/recipes/README.md) |
| Control checkpoint boundaries, chunking or retention | [Partitioning example](examples/custom-partitioning.md), [build policies](python/api/frontend.md) |
| Search budgets, microbatch sizes and policy candidates | [Quickstart](../benchmarking/quickstart.md) |
| Understand predicted or measured performance | [Plan reports](python/plan-report.md), [execution diagnostics](python/step-diagnostics.md) |
| Add an operator library | [Operator integration](python/operator-integration.md) |
| Understand or modify the engine | [Architecture](architecture/README.md), [development](development/README.md) |
| Integrate a native backend | [C guide](c/README.md) |

## Documentation map

- [Python guide and API index](python/README.md): phase controls, artifacts,
  Trainer/Executor, distributed execution, diagnostics and errors.
- [Examples](examples/README.md): complete training, forward-only execution,
  reusable planning, diagnostics and custom partitioning workflows.
- [Architecture](architecture/README.md): design and invariants, grouped by subsystem.
- [C reference](c/README.md): runtime, backends, pools, lanes, planner and simulator.
- [Development](development/README.md): setup, source ownership, naming and checks.
- [Optional integrations](integrations/README.md): adapters for specific operator libraries.

Guides show how to perform a task; API references define exact controls;
architecture pages explain why the implementation works. Experiment histories,
qualification receipts and merge-review notes live under `docs/internal/`.
