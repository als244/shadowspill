# ShadowSpill documentation

## What ShadowSpill is

A fixed-shape PyTorch forward or training step normally needs every value it
touches resident at once, and that total is what decides whether the step runs
at all. ShadowSpill makes the total a budget instead of a limit. It captures
the step once, measures its compiled tasks, then decides for every value
whether to keep it on the device, move it to spill storage and fetch it back, or
throw it away and recompute it — and when each of those should happen, so the
copies overlap the compute that does not need them yet. It proves the answer
fits the declared pools down to the byte offset, and saves a Plan that Executor
runs in a fresh worker. Trainer handles the training loop over that Plan.
PyTorch still executes every kernel; only the memory decisions move.

Three properties shape how the rest of these pages read. The step is planned
as a framework-neutral program, so the planner, the simulator and the runtime
know nothing about PyTorch. The search that turns a program into a schedule is
a replaceable part, so pages about planning describe what any search must
answer before any page names the one that ships. And a plan is certified before
it runs rather than discovered to be wrong during it, which is why physical
admission and placement get pages of their own.

## Start here

Pick the entry that matches what you came to do; each path stands on its own.

- To use ShadowSpill from PyTorch, read [Training with saved plans](examples/generic-training.md),
  then the [Python API](python/README.md), then the complete workflows in the
  [examples](examples/README.md).
- To understand the system, start at the [architecture
  overview](architecture/overview.md) and follow the numbered path under
  Architecture below.
- To integrate the C library or a backend, start with the [C API
  guide](c/README.md).
- To modify the repository, use the [development guide](development/README.md).

Every page below is linked once, under the component it describes, with a line
on what it covers.

## Architecture

The design: framework-neutral where the code is, PyTorch-specific where it
must be. The numbered pages form one ordered reading path that follows a step
from capture to execution; the groups below are reading boundaries, not
separate ownership trees. The last group holds the contracts the path refers
to without passing through them.

### Foundations

What ShadowSpill is for, and the vocabulary every later page uses.

1. [Architecture overview](architecture/overview.md) — the public phases and
   artifacts, source organization, memory ownership and its invariants, and
   the native libraries.
2. [Intermediate representation](architecture/ir.md) — the four neutral
   types and how they relate, then schedules and execution plans.

### PyTorch lowering

How a PyTorch model becomes a framework-neutral program: what is captured,
what is profiled, and what the planner is handed.

3. [Capture, compilation and lowering](architecture/lowering.md) — semantic
   roots, executable storage, profiling, and canonical objects.
4. [Graph-pair construction](architecture/graph-pair-construction.md) —
   structural forward/backward alternatives, saved-value accounting, and
   profiling.
5. [Step structure and ordering](architecture/step.md) — what `build_step`
   produces, and the walk through its microbatches that planning binds to it.
6. [Model tensors and initialization](architecture/state-import.md) — how a
   caller's model state comes to live in the pools without a host copy of it,
   the contract that makes that possible, and how dtype is decided.
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
10. [Simulation](architecture/simulation.md) — the deterministic timeline a
    search prices its candidates against: compute, transfer, capacity, and
    causal-dependency replay. Read before the searches that price against it.
11. [Plan search](architecture/search.md) — what the planner asks of a search
    and promises in return, and what a search is free to decide.
12. [Writing a search algorithm](architecture/search-algorithm.md) — the
    methods to implement, every argument, the defaults, and a worked example.
13. [Graph-pair selection](architecture/graph-pair-selection.md) — bounded
    complete selections across occurrence-level graph-pair options.
14. [PressureFit](architecture/pressurefit.md) — the search that ships:
    formulation, bounded policy search, repair, and pseudocode.
15. [Physical admission and offset
    handling](architecture/physical-admission.md) — allocation lifetimes, fixed
    placement, dynamic scratch, and causal reuse.
16. [From a resolved program to leases](architecture/admission-leases.md) —
    what a schedule allocates and when each lease is live.
17. [Fixed-offset placement](architecture/fixed-placement.md) — how leases are
    given addresses and what the cost of doing so depends on.
18. [The planning pipeline](architecture/planning-pipeline.md) — reusable artifacts,
    communication inputs, physical admission and saved Plans.

### Execution

How a plan is run and measured: the runtime and its boundaries, the backend
underneath, and the clocks a step is read on.

19. [Plan identity](architecture/plan-identity.md) — what names a plan, why a
    task id cannot, what a lease records, and the registry that answers an id
    after its plan is gone.
20. [Shared objects](architecture/shared-objects.md) — one value reached by
    several plans: the object, its per-pool locations, the leases those hold,
    and why binding an object allocates nothing.
21. [Backends](architecture/backends.md) — the one component that knows a
    platform, the driver-level table it implements, and how a new provider
    plugs in. The three pages after it are built on that table.
22. [Memory pools](architecture/memory-pools.md) — pools, leases, and where a
    pool's region comes from: a kind selects an acquire/release pair, so the
    kinds are a list rather than a fixed set.
23. [Lanes](architecture/lanes.md) — what moves bytes between two pools: the
    contract a transport implements, how one is chosen from the pool kinds it
    connects, and the stream ordering it must not disturb.
24. [Transfers](architecture/transfers.md) — routes, the queue each owns,
    dispatch order, and calibration through the lane.
25. [Events](architecture/events.md) — event leases and pools, sealing,
    completion tracking, and the timing pool behind traced intervals.
26. [Memory runtime](architecture/memory-runtime.md) — leases, causal reuse,
    the worker, failure, and tracing, over the pools, lanes and events above.
27. [Task boundaries](architecture/task-boundaries.md) — what `before_task` and
    `after_task` each do, how allocations find their task, and what is still in
    flight when the dispatching thread returns.
28. [Failure, abort, and process exit](architecture/failure-and-exit.md) — how
    a failure is handled at each scope, and why a process that is exiting is
    abandoned rather than closed.
29. [Step boundaries](architecture/step-boundaries.md) — the invocation
    cycle: why repetition is sound, the synchronization points
    between one step and the next, scheduled fetches from the actual initial state,
    and what step time means.
30. [PyTorch adapter](architecture/adapter.md) — what the adapter library is
    made of, how its source is laid out, what it requires of a backend, and
    what it exposes upward.
31. [Timelines](architecture/timelines.md) — the two clocks a traced step is
    measured on, the origin they share, and what an untraced step pays.

### Beside the reading path

Contracts the path refers to rather than passes through: where stages are cut,
what provider resources a build may describe, and how training composes the
phases.

- [Model and task partitioning](architecture/partitioning.md) — the generic
  `PartitionPolicy` contract and how workloads supply custom boundaries.
- [Provider resources](architecture/resources.md) — the `resources` builder
  argument: buffers and handles a provider creates from an importable factory,
  recreated for build, profiling and execution.
- [Training composition](architecture/training.md) — the boundaries among
  caller code, recipes, Trainer, Executor and planning.

## Python

The `shadowspill` package: build, measurement, reusable profiles, CPU planning,
reports and execution through Executor or Trainer. The
[Python guide](python/README.md) indexes this section.

### Guides

Task-oriented pages: how to do something, and how to read what comes back.

- [Independent build, profile and planning phases](architecture/offline-pipeline.md)
  — the public phase APIs end to end: build, measurement and profile overrides,
  communication, CPU planning and sweeps, execution, Trainer, and distributed
  plans.
- [Artifacts and caches](python/artifact-store.md) — saved phase results,
  disposable compiler caches and optional internal stores.
- [PyTorch allocator integration](python/allocator.md) — how the ShadowSpill
  allocator sits under PyTorch and what it accounts for.
- [Interpreting a Plan](python/plan-report.md) — selected tasks, simulated
  timelines, memory layouts and search evidence.
- [Planning diagnostic fields](python/plan-report-fields.md) — inspectable
  search records, counters and candidate outcomes.
- [PressureFit cut traces](python/pressurefit-trace.md) — record and replay
  residency cuts, compare ranked alternatives, and render their trajectory.
- [Interpreting execution diagnostics](python/step-diagnostics.md) — task and
  transfer instants, host boundaries, allocator evidence, and simulator
  reconciliation, with a complete field reference.
- [Pool occupancy over a step](python/occupancy.md) — what occupies each pool
  at every moment of a planned step, attributed to objects and categories, as
  tables and pages, and on the device's clock beside a traced step.
- [program and annotated-plan JSON](python/planning-json.md) — canonical program,
  ShadowSpillPlanningProblem and AnnotatedProgramPlan schemas.
- [Errors, failures, and cleanup](python/failures.md) — exception taxonomy,
  structured runtime evidence, rollback, and teardown.

### API reference

One page per public surface, listing every exported name with its signature
and the contract behind it.

- [Frontend and lifecycle API](python/api/frontend.md) — `shadowspill.memory`
  and `shadowspill.pytorch`: pool configurations, build policies and low-level
  runtime access.
- [SSD spill storage](python/api/ssd.md) — temporary direct-I/O pools, bounded
  host staging and initialization without a complete host model.
- [Phase artifacts](python/api/artifacts.md) — Program, Profiles,
  ProfiledProgram, Plan and SweepResult: what each holds, and how each is
  saved, loaded and reported.
- [Training and forward execution over saved plans](python/api/training.md) —
  Trainer and Executor: initialization, inputs and results, evaluation,
  schedules, parameter norms, checkpoints, startup diagnostics and custom
  optimizers.
- [Distributed planning and training](python/api/distributed.md) — logical
  ownership and groups, communication calibration and CPU planning, ranks on
  different hardware, fresh execution workers, checkpoints, and the
  qualification scope.
- [Diagnostics API](python/api/diagnostics.md) — CPU report generation and
  opt-in execution diagnostics.
- [Framework-neutral Python API](python/api/neutral.md) — `shadowspill.errors`,
  `shadowspill.ir`, `shadowspill.step`, `shadowspill.task`, `shadowspill.store`,
  `shadowspill.planner`, `shadowspill.simulator`, and `shadowspill.runtime` with
  the frontend interface it is opened with, for tooling and independent
  planning.
- [Timing an invocation](python/api/timing.md) — measuring a step through its
  result handle, and the device trace that breaks its time down.

## C

The C library `libshadowspill` (simulator, planner, and runtime), plus separately
compiled backends, pool/lane extensions, and the PyTorch adapter. The
[C API guide](c/README.md) indexes this section and covers ABI use, ownership
rules, and platforms.

- [Runtime C API](c/runtime.md) — pools, objects, admitting a plan, task
  boundaries, telemetry, and admission replay.
- [Backend contract](c/backends.md) — the driver-level table a provider
  implements, and what the runtime builds on top of it.
- [Lane contract](c/lanes.md) — what moves bytes between two pools: the table a
  transport implements, and how the runtime finds one.
- [Pool memory contract](c/pool-memory.md) — where a pool's memory comes from:
  the pair a kind of memory implements, and how the runtime finds it.
- [SSD pool and lanes](c/ssd.md) — direct-I/O storage, extension registration,
  bounded host staging, synchronization and lifecycle.
- [Planner C API](c/planner.md) — the planning problem in indexed form and the
  certification a schedule passes whichever search found it, through to fixed
  placement. Names no search.
- [PressureFit C API](c/pressurefit.md) — the search that ships: its options
  and policy axes, its result and per-candidate diagnostics, and the shared
  record of the best plan placed.
- [Simulator C API](c/simulator.md) — deterministic replay of a program under
  a plan, and the diagnostics it returns.
- [PyTorch adapter C API](c/pytorch-adapter.md) — the allocator, storage, and
  allocation observer and execution adapter between PyTorch and the runtime.
- [The C tree](../csrc/README.md) — source layout and build dependencies.

## Examples

Complete, runnable workflows built from the public API, one per page. The
[examples index](examples/README.md) says what each assumes.

- [Training loop](examples/generic-training.md) — build/profile, plan on CPU,
  then train and checkpoint in a fresh worker.
- [Forward-only execution](examples/forward-only.md) — plan and run a forward
  pass without an optimizer.
- [Reusable planning and budget sweeps](examples/reusable-planning.md) — plan
  once, then re-plan across budgets from the same artifacts.
- [Diagnosing a plan and real step](examples/diagnostics.md) — saved-Plan simulation and
  optional Executor trace diagnostics.
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

## Training

Text recipes over the generic trainer, run from a JSON config or from Python.
This page lives beside its configs and commands, outside `docs/`.

- [Training recipes](../training/README.md) — a fine-tune from one config and
  two commands, on ShadowSpill or eagerly on PyTorch: data preparation, packed
  documents, precision and initialization, schedules, evaluation, resumable
  checkpoints, metrics and run comparison.

## Benchmarking and qualification

The trees that measure ShadowSpill: the planning benchmark over a program
corpus, and the release gates that check numerics and full-model throughput.
These pages live beside the code they describe, outside `docs/`.

- [Benchmarking](../benchmarking/README.md) — the self-contained,
  reproducible planning benchmark tree, and what each of its entry points is
  for.
- [Quickstart script](../benchmarking/quickstart.md) — one model end to end:
  geometry search over execution budgets, figures, and a run of the winning
  plan.
- [Program collection](../benchmarking/program_collection/README.md) — the
  harness that saves profiled-program archives and runs no planner.
- [Planning evaluation](../benchmarking/planning_eval/README.md) — the
  harness that plans that corpus: search, simulation, and physical
  admission across budgets and bandwidths.
- [Qualification](../qualification/README.md) — the release-acceptance
  protocols and their launchers, including the one command that runs the
  selected gates in a fixed order (suite, numerical and performance by
  default) and reports what each found.
- [Numerical qualification](../qualification/numerical/README.md) — planned
  steps checked against PyTorch alone, compiled fullgraph without ShadowSpill.
- [SSD numerical qualification](../qualification/numerical/ssd.md)
  — the same cases and references using the SSD pool; opt-in.
- [Full-model performance qualification](../qualification/performance/README.md)
  — throughput floors and simulator error on the large models.
- [Remote qualification](../qualification/remote/README.md) — the numerical
  matrix again, with the spill pool held by a daemon on another machine.
- [Remote performance qualification](../qualification/remote_perf/README.md) —
  the performance matrix again, with the spill pool on another machine, held
  to its simulator's prediction rather than a throughput floor.
- [Workloads](../workloads/README.md) — the model and data definitions the
  benchmarks and gates consume.
- [Example model catalog](../workloads/MODELS.md) — architecture index, dimensions,
  precision, trainable parameters, LoRA status, EP and runnable configurations.

## Documentation policy

Public behavior is documented here and tested against exported Python names,
public C headers, local links and heading anchors, and the Python signatures
mirrored in API examples. Installed headers remain authoritative for C layouts,
ABI constants, and exact C signatures. Source remains authoritative for cache
schema labels and internal implementation details.
