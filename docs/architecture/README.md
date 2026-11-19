# Architecture

Read the [overview](overview.md) for the phase boundaries and ownership model.
Then choose the subsystem you need; these pages are references, not a required
linear reading sequence. Previous/Next links follow the index order below.
For public API usage, use the [Python guide](../python/README.md).

The IR, planner, simulator and memory runtime are framework-neutral. PyTorch
capture and lowering form an adapter. Operator-library semantics enter through
explicit metadata and lifecycle interfaces; models and library-specific adapters
do not define core architecture.

## System and representation

- [Architecture overview](overview.md)
- [Intermediate representation](ir.md)
- [The ShadowSpillProgram](program.md)
- [Step structure and ordering](step.md)

## Service and workers

- [Training composition](training.md)
- [Service and node daemons](service/README.md)
- [Node registration and control](service/nodes.md)
- [Assignments and recovery](service/execution.md)
- [Durable service records](service-records.md)

## Capture and compilation

- [Capture, compilation and lowering](lowering.md)
- [Model and task partitioning](partitioning.md)
- [Graph-pair construction](graph-pair-construction.md)
- [Axis chunking within a stage](axis-chunking.md)
- [Custom gradient destinations](custom-gradient-destinations.md)
- [Retaining forward routing choices](routing-retention.md)
- [Model tensors and initialization](state-import.md)
- [The optimizer](optimizer.md)
- [Provider resources](resources.md)

## Planning and admission

- [The planning problem](planning-problem.md)
- [The planning pipeline](planning-pipeline.md)
- [Plan search](search.md)
- [Graph-pair selection](graph-pair-selection.md)
- [PressureFit](pressurefit.md)
- [Simulation](simulation.md)
- [Physical admission and offset handling](physical-admission.md)
- [From a resolved program to leases](admission-leases.md)
- [Fixed-offset placement](fixed-placement.md)
- [Writing a search algorithm](search-algorithm.md)

## Runtime and backends

- [Memory runtime](memory-runtime.md)
- [Plan identity](plan-identity.md)
- [Shared objects](shared-objects.md)
- [Backends](backends.md)
- [Memory pools](memory-pools.md)
- [Lanes](lanes.md)
- [Transfers](transfers.md)
- [Events](events.md)
- [Task boundaries](task-boundaries.md)
- [Step boundaries](step-boundaries.md)
- [Failure, abort, and process exit](failure-and-exit.md)
- [PyTorch adapter](adapter.md)
- [Timelines: how a step is measured](timelines.md)
