# ShadowSpill documentation

This directory describes the current ShadowSpill implementation. The
documentation is split by contract so readers do not have to infer whether a
statement applies to Python, C, or the framework-neutral design. Every page
is linked from this one, under the component it describes, with a line on
what it covers.

## Start here

Pick the entry that matches what you came to do; each path stands on its own.

- To use ShadowSpill from PyTorch, read the [Python quickstart](python/quickstart.md)
  and then the [Python API](python/README.md).
- To learn from complete workflows, use the [examples](examples/README.md).
- To inspect a plan or real step, use the [PlanReport](python/plan-report.md)
  and [StepResult diagnostics](python/step-diagnostics.md) guides.
- To understand failure propagation and cleanup, read [Errors, failures, and
  cleanup](python/failures.md).
- To consume saved planning data, use the [Program and annotated-plan JSON
  guide](python/planning-json.md).
- To understand the system, follow the ordered path in the [architecture
  overview](architecture/overview.md).
- To integrate the C library or a backend, start with the [C API
  guide](c/README.md).
- To modify the repository, use the [development guide](development/README.md).

## Architecture

The design: framework-neutral where the code is, PyTorch-specific where it
must be. The pages form one ordered reading path that follows a step from
capture to execution; the groups below are reading boundaries, not separate
ownership trees.

### Foundations

What ShadowSpill is for, and the vocabulary every later page uses.

1. [Architecture overview](architecture/overview.md) — vocabulary, artifacts,
   ownership, invariants, and supported scope.
2. [Intermediate representation](architecture/ir.md) — logical Programs,
   task alternatives, phases and sinks, schedules, and execution plans.

### PyTorch lowering

How a PyTorch model becomes a framework-neutral program: what is captured,
what is profiled, and what the planner is handed.

3. [PyTorch capture and lowering](architecture/lowering.md) — semantic roots,
   executable storage, profiling, and canonical objects.
4. [Graph-pair construction](architecture/graph-pair-construction.md) —
   structural forward/backward alternatives, saved-value accounting, and
   profiling.

### Planning

How a program becomes an executable plan: which tasks run, where every object
lives at each boundary, and what address every allocation gets.

5. [Graph-pair selection](architecture/graph-pair-selection.md) — bounded
   complete selections across occurrence-level graph-pair options.
6. [PressureFit](architecture/pressurefit.md) — mathematical formulation,
   inputs/outputs, bounded policy search, repair, and pseudocode.
7. [Physical admission and offset handling](architecture/physical-admission.md)
   — allocation lifetimes, fixed placement, dynamic scratch, and causal reuse.
8. [From a resolved program to leases](architecture/admission-leases.md) —
   what a schedule allocates and when each lease is live.
9. [Fixed-offset placement](architecture/fixed-placement.md) — how leases are
   given addresses and what the cost of doing so depends on.
10. [Planning orchestration](architecture/planning.md) — reusable artifacts,
    transfer inputs, callable publication, and PlanReport.

### Execution

How a plan is predicted, run, and measured: the simulator, the runtime and
its boundaries, the backend underneath, and the clocks a step is read on.

11. [Simulation](architecture/simulation.md) — compute, transfer, capacity, and
    causal-dependency replay.
12. [Memory runtime](architecture/memory-runtime.md) — pools, leases, worker,
    failure, and tracing.
13. [Task boundaries](architecture/task-boundaries.md) — what `before_task` and
    `after_task` each do, how allocations find their task, and what is still in
    flight when the dispatching thread returns.
14. [Failure, abort, and process exit](architecture/failure-and-exit.md) — how
    a failure is handled at each scope, and why a process that is exiting is
    abandoned rather than closed.
15. [Step boundaries](architecture/step-boundaries.md) — the recurrent
    invocation cycle: why repetition is sound, the synchronization points
    between one step and the next, the first-use order of the opening
    restore, and what step time means.
16. [Backends](architecture/backends.md) — the one component that knows a
    platform, the driver-level table it implements, and how a new provider
    plugs in.
17. [Memory pools](architecture/memory-pools.md) — pools and their arenas,
    device and pinned host, as ShadowSpill objects built on the backend.
18. [Transfers](architecture/transfers.md) — routes, the lane each owns,
    dispatch order, and calibration on those lanes.
19. [Events](architecture/events.md) — event leases and pools, sealing,
    completion tracking, and the timing pool behind traced intervals.
20. [PyTorch adapter](architecture/adapter.md) — what the compiled adapter is
    made of, how its source is laid out, what it requires of a backend, and
    what it exposes upward.
21. [Timelines](architecture/timelines.md) — the two clocks a traced step
    is measured on, the origin they share, and what an untraced step pays.

## Python

The `shadowspill` package: model-state import, capture and lowering, reusable
planning artifacts, PressureFit, diagnostics, and callable execution. The
[Python guide](python/README.md) indexes this section.

### Guides

Task-oriented pages: how to do something, and how to read what comes back.

- [Python quickstart](python/quickstart.md) — constructing the runtime,
  importing model state, planning a step, and the callable lifecycle.
- [Artifact store](python/artifact-store.md) — the content-addressed store
  through which planning artifacts are shared and reused.
- [PyTorch allocator integration](python/allocator.md) — how the ShadowSpill
  allocator sits under PyTorch and what it accounts for.
- [Interpreting a PlanReport](python/plan-report.md) — planning time, task and
  graph-pair selection, profiles, PressureFit, caches, and physical admission.
- [PlanReport field reference](python/plan-report-fields.md) — every field of
  every record the planning report carries, and what it holds.
- [Interpreting StepResult diagnostics](python/step-diagnostics.md) — task and
  transfer instants, host boundaries, allocator evidence, and simulator
  reconciliation, with a complete field reference.
- [Program and annotated-plan JSON](python/planning-json.md) — canonical Program,
  PressureFitProgram, StepProgram, and AnnotatedProgramPlan schemas.
- [Figures over a step search](python/plots.md) — the figure tree, what each
  plot represents, the conventions they share, and how `raw_data/` redraws
  them.
- [Errors, failures, and cleanup](python/failures.md) — exception taxonomy,
  structured runtime evidence, rollback, and teardown.

### API reference

One page per public surface, listing every exported name with its signature
and the contract behind it.

- [Frontend and lifecycle API](python/api/frontend.md) — `shadowspill.memory`
  and `shadowspill.pytorch`: the runtime, planning calls, planned callables,
  and state lifecycle.
- [Reusable planning artifacts](python/api/artifacts.md) — the immutable,
  content-addressed values planning is composed from.
- [Diagnostics API](python/api/diagnostics.md) — the planning and step
  diagnostics classes and how they are requested.
- [Framework-neutral Python API](python/api/neutral.md) — `shadowspill.ir`,
  `shadowspill.planner`, `shadowspill.simulator`, and `shadowspill.runtime`,
  for tooling and independent planning.

## C

The C library `libshadowspill` (simulator, planner, and runtime) and the two
pieces compiled separately: the backends and the PyTorch adapter. The
[C API guide](c/README.md) indexes this section and covers ABI use, ownership
rules, and platforms.

- [Runtime C API](c/runtime.md) — pools, objects, admitting a plan, task
  boundaries, telemetry, and admission replay.
- [Backend contract](c/backends.md) — the driver-level table a provider
  implements, and what the runtime builds on top of it.
- [Planner C API](c/planner.md) — the PressureFit problem, options, results,
  and fixed placement.
- [Simulator C API](c/simulator.md) — deterministic replay of a program under
  a plan, and the diagnostics it returns.
- [PyTorch adapter C API](c/pytorch-adapter.md) — the allocator, storage, and
  profiler bridge between PyTorch and the runtime.
- [The C tree](../csrc/README.md) — source layout and build dependencies.

## Examples

Complete, runnable workflows built from the public API, one per page. The
[examples index](examples/README.md) says what each assumes.

- [Training loop](examples/training-lifecycle.md) — create a runtime, import
  model state, train, checkpoint, and clean up.
- [Forward-only execution](examples/forward-only.md) — plan and run a forward
  pass without an optimizer.
- [Concurrent planned callables](examples/concurrent-callables.md) — dispatch
  distinct callables before either result is resolved.
- [Reusable planning and budget sweeps](examples/reusable-planning.md) — plan
  once, then re-plan across budgets from the same artifacts.
- [Diagnosing a plan and real step](examples/diagnostics.md) — join planning
  diagnostics to a traced step.
- [Custom stage partitioning](examples/custom-partitioning.md) — override the
  automatic partition when the module structure does not repeat.

## Development

How the repository is laid out, validated, and named, for anyone changing it.

- [Development guide](development/README.md) — where product code, tooling,
  and internal notes belong.
- [Repository structure and validation](development/repository.md) — the
  Python and C trees, the tests, the gates, and the tooling that runs them.
- [Naming conventions](development/naming.md) — identifier and vocabulary
  rules, including what stays generic outside a backend.

## Benchmarking and qualification

The trees that measure ShadowSpill: the planning benchmark over a program
corpus, and the release gates that check numerics and full-model throughput.
These pages live beside the code they describe, outside `docs/`.

- [Benchmarking](../benchmarking/README.md) — the self-contained,
  reproducible planning benchmark tree.
- [Quickstart script](../benchmarking/quickstart.md) — one model end to end:
  geometry search over execution budgets, figures, and a run of the winning
  plan.
- [Program collection](../benchmarking/program_collection/README.md) —
  building a corpus of pre-PressureFit step programs.
- [Planning evaluation](../benchmarking/planning_eval/README.md) —
  PressureFit, simulation, and physical admission over that corpus.
- [Qualification](../qualification/README.md) — the release-acceptance
  protocols and their launchers, including the one command that runs every
  gate in order and reports what each found.
- [Numerical qualification](../qualification/numerical/README.md) — planned
  steps checked against the eager reference.
- [Full-model performance qualification](../qualification/performance/README.md)
  — throughput floors and simulator error on the large models.
- [Workloads](../workloads/README.md) — the model and data definitions the
  benchmarks and gates consume.

## Documentation policy

Public behavior is documented here and tested against exported Python names,
public C headers, local links and heading anchors, and the Python signatures
mirrored in API examples. Installed headers remain authoritative for C layouts,
ABI constants, and exact C signatures. Source remains authoritative for cache
schema labels and internal implementation details.
