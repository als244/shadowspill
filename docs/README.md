# ShadowSpill documentation

## What ShadowSpill is

A fixed-shape PyTorch forward or training step normally needs every value it
touches resident at once, and that total is what decides whether the step runs
at all. ShadowSpill makes the total a budget instead of a limit. It captures
the step once, measures its compiled tasks, then decides for every value
whether to keep it on the device, spill it to host memory and fetch it back, or
throw it away and recompute it — and when each of those should happen, so the
copies overlap the compute that does not need them yet. It proves the answer
fits the declared pools down to the byte offset, and hands back an ordinary
Python callable that runs that plan every time it is called. PyTorch still
executes every kernel; only the memory decisions move.

Three properties shape how the rest of these pages read. The step is planned
as a framework-neutral program, so the planner, the simulator and the runtime
know nothing about PyTorch. The search that turns a program into a schedule is
a replaceable part, so pages about planning describe what any search must
answer before any page names the one that ships. And a plan is certified before
it runs rather than discovered to be wrong during it, which is why physical
admission and placement get pages of their own.

## Start here

Pick the entry that matches what you came to do; each path stands on its own.

- To use ShadowSpill from PyTorch, read the [Python quickstart](python/quickstart.md),
  then the [Python API](python/README.md), then the complete workflows in the
  [examples](examples/README.md).
- To understand the system, follow the ordered path in the [architecture
  overview](architecture/overview.md).
- To integrate the C library or a backend, start with the [C API
  guide](c/README.md).
- To modify the repository, use the [development guide](development/README.md).

Every page below is linked once, under the component it describes, with a line
on what it covers.

## Architecture

The design: framework-neutral where the code is, PyTorch-specific where it
must be. The pages form one ordered reading path that follows a step from
capture to execution; the groups below are reading boundaries, not separate
ownership trees. The overview page repeats this order as a map.

### Foundations

What ShadowSpill is for, and the vocabulary every later page uses.

1. [Architecture overview](architecture/overview.md) — vocabulary, artifacts,
   ownership, invariants, and supported scope.
2. [Intermediate representation](architecture/ir.md) — the four neutral
   types and how they relate, then schedules and execution plans.

### PyTorch lowering

How a PyTorch model becomes a framework-neutral program: what is captured,
what is profiled, and what the planner is handed.

3. [PyTorch capture and lowering](architecture/lowering.md) — semantic roots,
   executable storage, profiling, and canonical objects.
4. [Graph-pair construction](architecture/graph-pair-construction.md) —
   structural forward/backward alternatives, saved-value accounting, and
   profiling.
5. [The step artifacts](architecture/step.md) — what a captured step is: the
   walk it takes through its microbatches, the problems it lowered to, and why
   reading either needs no framework.
6. [Importing state](architecture/state-import.md) — how a caller's model
   state comes to live in the pools without a host copy of it, the contract
   that makes that possible, and how dtype is decided.
7. [The optimizer](architecture/optimizer.md) — what ShadowSpill needs from an
   optimizer and promises in return: state declared on meta and filled by the
   caller, frozen parameters, and values that change between steps.

### Planning

How a program becomes an executable plan: which tasks run, where every object
lives at each boundary, what address every allocation gets, and how a candidate
is priced before any of it is committed.

8. [The ShadowSpillProgram](architecture/program.md) — what the system plans
   for: tasks over objects, alternatives, phases and sinks, and identity.
9. [The planning problem](architecture/planning-problem.md) — the question
   asked about a program: boundaries, machine, budget, and what it omits.
10. [Plan search](architecture/search.md) — what the planner asks of a search
    and promises in return, and what a search is free to decide.
11. [Writing a search algorithm](architecture/search-algorithm.md) — the
    methods to implement, every argument, the defaults, and a worked example.
12. [Graph-pair selection](architecture/graph-pair-selection.md) — bounded
    complete selections across occurrence-level graph-pair options.
13. [PressureFit](architecture/pressurefit.md) — the search that ships:
    formulation, bounded policy search, repair, and pseudocode.
14. [Physical admission and offset handling](architecture/physical-admission.md)
    — allocation lifetimes, fixed placement, dynamic scratch, and causal reuse.
15. [From a resolved program to leases](architecture/admission-leases.md) —
    what a schedule allocates and when each lease is live.
16. [Fixed-offset placement](architecture/fixed-placement.md) — how leases are
    given addresses and what the cost of doing so depends on.
17. [Simulation](architecture/simulation.md) — the deterministic timeline a
    search prices its candidates against: compute, transfer, capacity, and
    causal-dependency replay.
18. [Planning orchestration](architecture/planning.md) — reusable artifacts,
    transfer inputs, callable publication, and PlanReport.

### Execution

How a plan is run and measured: the runtime and its boundaries, the backend
underneath, and the clocks a step is read on.

19. [Memory runtime](architecture/memory-runtime.md) — pools, leases, worker,
    failure, and tracing.
20. [Task boundaries](architecture/task-boundaries.md) — what `before_task` and
    `after_task` each do, how allocations find their task, and what is still in
    flight when the dispatching thread returns.
21. [Failure, abort, and process exit](architecture/failure-and-exit.md) — how
    a failure is handled at each scope, and why a process that is exiting is
    abandoned rather than closed.
22. [Step boundaries](architecture/step-boundaries.md) — the recurrent
    invocation cycle: why repetition is sound, the synchronization points
    between one step and the next, the first-use order of the opening
    restore, and what step time means.
23. [Backends](architecture/backends.md) — the one component that knows a
    platform, the driver-level table it implements, and how a new provider
    plugs in.
24. [Memory pools](architecture/memory-pools.md) — pools and their arenas,
    device and pinned host, as ShadowSpill objects built on the backend.
25. [Transfers](architecture/transfers.md) — routes, the lane each owns,
    dispatch order, and calibration on those lanes.
26. [Events](architecture/events.md) — event leases and pools, sealing,
    completion tracking, and the timing pool behind traced intervals.
27. [PyTorch adapter](architecture/adapter.md) — what the compiled adapter is
    made of, how its source is laid out, what it requires of a backend, and
    what it exposes upward.
28. [Timelines](architecture/timelines.md) — the two clocks a traced step
    is measured on, the origin they share, and what an untraced step pays.

## Python

The `shadowspill` package: model-state import, capture and lowering, reusable
planning artifacts, plan search, diagnostics, and callable execution. The
[Python guide](python/README.md) indexes this section.

### Guides

Task-oriented pages: how to do something, and how to read what comes back.

- [Python quickstart](python/quickstart.md) — constructing the runtime,
  importing model state, planning a step, and the callable lifecycle.
- [Artifact store](python/artifact-store.md) — the content-addressed store,
  its build and planning trees, and what each mode does with them.
- [PyTorch allocator integration](python/allocator.md) — how the ShadowSpill
  allocator sits under PyTorch and what it accounts for.
- [Interpreting a PlanReport](python/plan-report.md) — planning time, task and
  graph-pair selection, profiles, search diagnostics, store use, and physical
  admission.
- [PlanReport field reference](python/plan-report-fields.md) — every field of
  every record the planning report carries, and what it holds.
- [Interpreting StepResult diagnostics](python/step-diagnostics.md) — task and
  transfer instants, host boundaries, allocator evidence, and simulator
  reconciliation, with a complete field reference.
- [program and annotated-plan JSON](python/planning-json.md) — canonical program,
  ShadowSpillPlanningProblem, StepProgram, and AnnotatedProgramPlan schemas.
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
- [Timing: the step on the device clock](python/api/timing.md) — the events
  every invocation records, the cycle they define, and the API that reads
  it.

## C

The C library `libshadowspill` (simulator, planner, and runtime) and the two
pieces compiled separately: the backends and the PyTorch adapter. The
[C API guide](c/README.md) indexes this section and covers ABI use, ownership
rules, and platforms.

- [Runtime C API](c/runtime.md) — pools, objects, admitting a plan, task
  boundaries, telemetry, and admission replay.
- [Backend contract](c/backends.md) — the driver-level table a provider
  implements, and what the runtime builds on top of it.
- [Planner C API](c/planner.md) — the planning problem in indexed form and the
  certification a schedule passes whichever search found it, through to fixed
  placement. Names no search.
- [PressureFit C API](c/pressurefit.md) — the search that ships: its options
  and policy axes, its result and per-candidate diagnostics, and the shared
  record of the best plan placed.
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
  reproducible planning benchmark tree, and how its three entry points
  divide.
- [Quickstart script](../benchmarking/quickstart.md) — one model end to end:
  geometry search over execution budgets, figures, and a run of the winning
  plan.
- [program collection](../benchmarking/program_collection/README.md) — the
  harness that builds a corpus of step programs and runs no planner.
- [Planning evaluation](../benchmarking/planning_eval/README.md) — the
  harness that plans that corpus: search, simulation, and physical
  admission across budgets and bandwidths.
- [Qualification](../qualification/README.md) — the release-acceptance
  protocols and their launchers, including the one command that runs every
  gate in order and reports what each found.
- [Numerical qualification](../qualification/numerical/README.md) — planned
  steps checked against PyTorch alone, compiled fullgraph without ShadowSpill.
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
