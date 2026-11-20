# Architecture

How the engine is built. Start with the overview, the first page below; the
other pages are references to read as needed. Previous/Next links follow the
order of this index. For public API usage, use the
[Python guide](../python/README.md).

The IR, planner, simulator and memory runtime are framework-neutral. PyTorch
capture and lowering form the frontend. Operator-library semantics enter
through explicit metadata and lifecycle interfaces; models and library-specific
adapters do not define core architecture.

## System and representation

The phases, the artifacts they exchange, and the program every planner and
runtime component reads.

- [Architecture overview](overview.md): the public phases, artifacts, source layout and ownership.
- [Intermediate representation](ir.md): the neutral program, schedule and plan types.
- [The ShadowSpillProgram](program.md): tasks over objects, alternatives, phases and sinks.
- [Step structure and ordering](step.md): what a built step contains and when its order is chosen.

## Service and workers

The training service: one endpoint that runs training on its own host, the
node daemons that register further machines, and the records it keeps.

- [Training composition](training.md): how preparation, worker startup and live commands use the engine.
- [Service and node daemons](service/README.md): what the service, node daemons and workers each own; runs use the service's own host.
- [Node registration and control](service/nodes.md): how a node daemon joins, heartbeats, polls for claim and stop, returns receipts, leaves and reconciles.
- [Assignments and recovery](service/execution.md): reserving the service host's resources, starting workers and recovering from failure.
- [Durable service records](service-records.md): the persistent records beside live training state.

## Capture and compilation

How a model becomes a Program: capture, partitioning, graph pairs, chunking,
gradients, retention, state and the optimizer.

- [Capture, compilation and lowering](lowering.md): from captured graphs to an untimed Program.
- [Model and task partitioning](partitioning.md): dividing a captured forward into stages.
- [Graph-pair construction](graph-pair-construction.md): building each stage's save and recompute variants.
- [Axis chunking within a stage](axis-chunking.md): running independent positions of an axis in slices.
- [Custom gradient destinations](custom-gradient-destinations.md): letting an opaque backward add into existing gradients.
- [Retaining forward routing choices](routing-retention.md): keeping discrete forward choices across recomputation.
- [Model tensors and initialization](state-import.md): declaring model state at build and filling it at execution.
- [The optimizer](optimizer.md): capturing an optimizer's update, its state and its controls.
- [Provider resources](resources.md): buffers and handles custom operators share.

## Planning and admission

How a profiled Program becomes a Plan: search, selection, simulation and
physical placement.

- [The planning problem](planning-problem.md): what `plan_program` is given and what it answers.
- [The planning pipeline](planning-pipeline.md): from a ProfiledProgram to a saved Plan.
- [Plan search](search.md): the contract between the planner and a search.
- [Graph-pair selection](graph-pair-selection.md): which complete task selections a search evaluates.
- [PressureFit](pressurefit.md): the search that ships.
- [Simulation](simulation.md): how a schedule is priced.
- [Physical admission and offset handling](physical-admission.md): proving a schedule fits real pool ranges.
- [From a resolved program to leases](admission-leases.md): deriving the leases a schedule creates.
- [Fixed-offset placement](fixed-placement.md): giving every lease a fixed address.
- [Writing a search algorithm](search-algorithm.md): implementing a search, with an example.

## Runtime and backends

The C runtime that executes a Plan: pools, leases, transfers, events, task
and step boundaries, failure, and the PyTorch adapter.

- [Memory runtime](memory-runtime.md): pools, objects, leases, the worker and the dispatcher.
- [Plan identity](plan-identity.md): naming the plan that owns a lease.
- [Shared objects](shared-objects.md): one value reached by several plans and charged once.
- [Backends](backends.md): the boundary between the runtime and an accelerator platform.
- [Memory pools](memory-pools.md): where a pool's memory comes from and how pools are built.
- [Lanes](lanes.md): the transport contract that moves bytes between two pools.
- [Transfers](transfers.md): routes, queues, calibration and dispatch.
- [Events](events.md): the runtime's synchronization objects.
- [Task boundaries](task-boundaries.md): what happens before and after each task call.
- [Step boundaries](step-boundaries.md): one invocation from initial to final residency.
- [Failure, abort, and process exit](failure-and-exit.md): how each scope of failure is handled.
- [PyTorch adapter](adapter.md): the shared object between PyTorch and the runtime.
- [Timelines: how a step is measured](timelines.md): the clocks and origins of traced timings.
