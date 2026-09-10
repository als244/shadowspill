# Architecture overview

## Why ShadowSpill exists

A model can exceed execution-device memory even when each individual operator
fits. Ordinary eager allocation sees one request at a time; it does not know
which values will be needed later, which values can be recomputed, or when a
transfer can overlap useful compute.

ShadowSpill turns one fixed-shape PyTorch forward or accumulated training step
into an ordered, inspectable `ShadowSpillProgram`. It then:

1. measures the compiled tasks and their memory behavior;
2. chooses which intermediate values to save or recompute;
3. schedules object residency, fetches, write-backs, evictions, and
   releases;
4. proves that the selected step fits the configured physical pools; and
5. returns a normal Python callable that repeatedly executes that admitted
   plan.

PyTorch and its compiled providers still perform every numerical operation.
ShadowSpill owns the memory policy around those operations: object identity,
residency, movement, readiness, capacity, and causal reuse.

## System at a glance

ShadowSpill has a one-time planning path and a repeated execution path:

```mermaid
flowchart LR
    subgraph plan["Plan once"]
        inputs["PyTorch model<br/>objective + optimizer<br/>fixed examples"]
        frontend["PyTorch frontend<br/>capture, partition, lower, profile"]
        program["Framework-neutral<br/>ShadowSpillProgram"]
        problem["Planning problem<br/>program + boundaries + machine"]
        search["Plan search<br/>+ simulator"]
        admission["Physical admission<br/>ranges + causal reuse"]
        materialize["Planned callable<br/>+ PlanReport"]

        inputs --> frontend --> program --> problem --> search --> admission --> materialize
    end

    runtime_config["Runtime configuration<br/>memory pools + transfer calibration"]
    runtime_config --> problem
    runtime_config --> admission

    subgraph execute["Execute repeatedly"]
        dispatcher["PyTorch dispatcher"]
        compiled["Compiled task callables"]
        runtime["C runtime<br/>objects, leases, actions"]
        worker["C worker<br/>transfers + completion"]
        pools["Configured memory pools<br/>+ directed routes"]

        dispatcher --> compiled
        dispatcher <--> runtime
        runtime <--> worker
        runtime <--> pools
        worker <--> pools
    end

    materialize --> dispatcher
```

The planning side decides what is legal and predicts its cost. The execution
side follows the admitted records; it does not rediscover graph semantics or
rerun memory-policy search.

The three planning boxes are three different things, and the pipeline reads
badly if they blur: a [program](program.md) is the work, a [planning
problem](planning-problem.md) is a question about it, and a
[search](search.md) answers that question. Which search runs is pluggable --
[PressureFit](pressurefit.md) is the one that ships, and the box says "plan
search" rather than its name because nothing upstream or downstream depends
on which it is.

## Libraries and responsibilities

The compiled code is three kinds of shared object with one direction of
dependency, and a Python package above them:

| Library | Holds | Knows about |
|---|---|---|
| `libshadowspill.so` | the neutral C library: IR digests, the simulator, physical admission, the search that ships, and the runtime with its [memory pools](memory-pools.md), [transfers](transfers.md), [events](events.md), task boundaries, and tracing. Its planner header names no search; the shipped one has a header of its own beside it | the backend contract only |
| `libshadowspill_backend_<provider>.so` | one provider's implementation of the [backend contract](backends.md): device allocation, host registration, streams, copies, events, profiler | its driver and nothing of ShadowSpill's |
| `libshadowspill_pytorch.so` | the [PyTorch adapter](adapter.md): the pluggable allocator, objects and storage views, task boundaries, tracing | PyTorch and the neutral runtime |
| `shadowspill` (Python) | capture, lowering, profiling, planning orchestration, the planned callables, diagnostics | the adapter's C API and the neutral library |

The runtime is handed a backend table at create and never links a provider;
the adapter opens the backend library by name at bootstrap. Every object with
a lifetime or a policy, pools and their arenas, routes and lanes, event pools,
calibration, is the neutral library's, built from the table's driver-level
calls, which is what lets a new provider plug in with no change above it.

## How planning responsibilities differ

The planning components answer deliberately different questions:

| Component | Question answered |
|---|---|
| Stage partitioning | Where is the captured model divided into ordered compiled tasks? |
| Graph-pair construction | What legal forward/backward implementations exist for one structural task contract? |
| Graph-pair selection | Which complete assignments of those local alternatives should be considered? |
| Plan search | For each assignment, which objects reside where and when do memory actions trigger -- and which assignment wins? |
| Simulator | What compute, transfer, dependency, and capacity timeline does that policy imply? |
| Physical admission | Can the selected task allocations and object lifetimes occupy real pool ranges without unsafe reuse? |
| Materialization | How are the admitted records and compiled callables installed into the runtime? |

Graph-pair construction is local: it builds options such as save and full
recompute for one structural contract. Graph-pair selection is global: one
selection chooses an option for every occurrence-level group. Both leave the
alternatives *open* in the program, and the search fixes them: expanding a
program into resolved programs and comparing what each answers is a
[search's own work](search.md), which is why the diagram above shows one box
and not two.

## Planning artifacts

Each boundary produces an immutable artifact that can be inspected, cached,
serialized, or passed to a lower-level API:

```mermaid
flowchart TD
    capture["Export/AOT graph"]
    pairs["TaskGraphPairs values"]
    profile["Compiled task profiles"]
    program["ShadowSpillProgram"]
    inputs["ShadowSpillPlanningProblem"]
    step["StepProgram"]
    annotated["AnnotatedProgramPlan"]
    report["PlanReport + planned callable"]

    capture --> pairs
    pairs --> profile
    profile --> program
    program -->|"+ residency, capacity, simulation inputs"| inputs
    inputs -->|"recurrent, and optionally initial"| step
    inputs -->|"graph-pair selection + policy search"| annotated
    annotated -->|"physical admission + materialization"| report
```

| Artifact | Meaning | Reusable without |
|---|---|---|
| `ShadowSpillProgram` | Logical objects, tasks, profiles, resources, and graph-pair alternatives for one schedule role | PyTorch |
| `ShadowSpillPlanningProblem` | A `ShadowSpillProgram` plus residency, capacity, admission, and simulation inputs | Capture, compilation, or profiling |
| `StepProgram` | The recurrent and optional initial `ShadowSpillPlanningProblem`, plus training-step provenance | Searching or callable materialization |
| `AnnotatedProgramPlan` | One selected schedule, physical layout, simulation result, and planning diagnostics | The model or runtime |
| `PlanReport` | The published callable's program, plan, execution mapping, profiles, artifact-store hits and misses, and diagnostics | Console logging |

`build_step_program()` stops at `StepProgram`. `plan_program()` consumes
one of its `ShadowSpillPlanningProblem` values under new budgets or transfer
bandwidths. `plan_step()` and `plan_forward()` run the complete pipeline.

## Runtime interaction

The Python dispatcher remains responsible for launching compiled PyTorch
tasks. Runtime progress belongs to the C worker:

```mermaid
sequenceDiagram
    participant D as PyTorch dispatcher
    participant R as Runtime task boundary
    participant G as Compute stream
    participant C as Compiled task
    participant W as ShadowSpill worker
    participant T as Fetch / evict lanes

    D->>R: before_task(task record)
    R-->>D: current leases + readiness events
    D->>G: insert unfinished event waits
    D->>C: invoke with rebound storages
    C->>G: enqueue numerical kernels
    D->>R: after_task(outputs and mutations)
    R->>G: record task-completion fence
    R->>W: publish ordered memory actions
    W-->>R: acknowledge eligible actions submitted
    D->>R: begin the next task
    W->>T: submit eligible transfers
    T-->>W: completion events
    W->>R: publish ready generations and releases
```

`before_task()` covers runtime acquisition, readiness waits, storage
rebinding, argument assembly, and the range-reuse waits of every allocation
the plan pinned to the task, so a task that has started is a task that only
computes. `after_task()` covers output classification, mutation publication,
releases, destination reservation, action publication, and the worker
submission acknowledgement -- which means the batch's fetches have been issued
and carry readiness events, never that any byte has moved. A transfer
dependency is placed on the compute stream instead of making the dispatcher
wait on the host when stream ordering can express the dependency.

The runtime owns explicit pool and directed-route registries. Each immutable
plan independently binds its execution pool, spill pool, fetch route, and
evict route. The worker services route submission, completion frontiers, and
deferred releases without holding a general-purpose global runtime mutex.

At the Python boundary, `submit()` returns one invocation-owned result handle.
Its `result()` method synchronizes that invocation's public completion event.
Different planned callables can therefore be host-dispatched together, while
each callable retains a simple single-outstanding-invocation invariant. A
later invocation waits only for the same plan's terminal actions and
retirements; unrelated plans continue independently.

## Component ownership

| Component | Owns | Does not own |
|---|---|---|
| PyTorch frontend | Export/AOT capture, stage partitioning, compiled callables, storage rebinding, objective and optimizer integration | Memory-policy search or transfer progress |
| IR | Objects, tasks, resources, graph-pair alternatives, schedules, and resolved task records | PyTorch tensors or provider handles |
| Planner | The question a search is asked, the contract it answers under, the helpers any search may call, certification, and the stores a plan is keyed in | How a plan is found, graph construction, or numerical execution |
| Search | Resolving a program into the alternatives it will compare, residency strategies, memory actions, and ranking what it places | The question it is handed, or whether its answer is physically admissible |
| Simulator | Deterministic compute, transfer, capacity, and dependency replay | Candidate generation or physical placement |
| Physical admission | Allocation lifetimes, task-allocation contract, fixed placements, dynamic scratch, and causal reuse dependencies | Which search produced the schedule, or its logical policy |
| Runtime | Pools and their arenas, leases, objects, routes and lanes, calibration, event and timing pools, task boundaries, failure state, and worker progress | Graph capture or model semantics |
| Backend | The driver-level table: device allocation, host memory registration, streams, copies, events, the provider's capabilities, physical memory and statistics, and profiler names and ranges | Any object lifetime or policy: pools, routes, lanes, event pooling |
| PyTorch adapter | The pluggable allocator, object and storage views, task-boundary and tracing entry points, loading the backend by name | Provider headers or planning |

The framework-neutral IR, planner, simulator, admission engine, and runtime do
not import PyTorch, and a test asserts it rather than leaving it to
convention. Producing a program -- capture, lowering, compilation, profiling
-- is the frontend's work; everything from a program onwards is neutral, so
planning a saved program pulls in no framework. Provider driver and profiler
calls remain inside concrete backends or framework adapters.

## One logical object through the system

A logical value retains one identity even as its physical address changes:

1. Capture identifies its producer, views, aliases, mutations, and stage.
2. A `TaskStorageContract` assigns a semantic storage root.
3. Compilation and profiling attach physical extents, allocation behavior,
   workspace, and timing without redefining that root.
4. `ObjectCatalog` maps the root to one canonical program object across tasks.
5. Graph-pair selection and the search decide whether the value exists,
   resides, moves, or is recreated at each boundary.
6. Physical admission assigns ranges and proves every reuse dependency.
7. Materialization registers direct task records with the runtime.
8. At execution, `before_task()` binds the current lease generation and
   `after_task()` publishes its successor generation.
9. The worker submits transfers and publishes completion; generation checks
   prevent stale work from modifying a successor.

Pointers therefore describe current placement, not semantic identity. For a
recurrent shared output, the producer updates this same logical object record;
it does not manufacture another identity. The prior lease retires behind its
causal completion fence before a successor can reuse its bytes.

## Correctness invariants

- Semantic object identity never depends on a transient pointer, allocator
  callback identity, or incidental FakeTensor storage.
- A callable is published only after logical scheduling and physical admission
  succeed for the same selected program.
- A pool range is reused only after stream order or an explicit completion
  dependency makes its predecessor inaccessible.
- Fetch and evict destinations consume capacity at their action trigger, even
  when the copy reaches its lane later.
- Every lease, event, transfer, and object publication is generation-checked;
  stale completion cannot mutate a successor.
- Execution and spill accounting never exceed their configured physical caps.
- Allocation behavior outside the admitted strict core is bounded by dynamic
  scratch; a mismatch fails before an invalid address reaches a kernel.
- Planner, simulator, admission, and runtime identities remain available in
  diagnostics so an executed step can be reconciled mechanically.

## Working vocabulary

| Term | Meaning |
|---|---|
| Stage | One ordered partition of the captured model graph. |
| Structural contract | Shape, dtype, role, alias, mutation, and executable-storage contract shared by equivalent task occurrences. |
| Task graph pairs | Every configured forward/backward alternative for one differentiated structural contract. |
| Graph-pair selection | One complete choice of graph-pair option for every occurrence-level group. |
| Storage root | One semantic allocation identity shared by all of its views. |
| Program object | One logical alias bundle with size, role, persistence, and task dependencies. |
| Action trigger | The task boundary at which a fetch, write-back, eviction, or release becomes ordered and destination capacity is reserved. |
| Physical admission | Proof that task allocations, object generations, transfers, and causal reuse fit the selected pools. |
| Memory lease | Ownership of one pool range for one residency generation. |

## Planning contract

| Capability | Contract |
|---|---|
| Graph geometry | Fixed shape and stride under captured guards |
| Capture | Strict Export/AOTAutograd/Inductor with no graph breaks |
| Custom operations | Supported when fake/meta and alias/mutation schemas are complete |
| Data-dependent outputs | Unbounded output geometry is rejected during planning |
| Execution pools | Framework-accessible memory supplied by configured device backends |
| Spill pools | Runtime-configured memory pools connected by directed transfer routes |
| Runtime ownership | Runtime-owned pools, objects, leases, and callable registrations |
| Transfer topology | Backend-provided ordered lanes serviced by the runtime worker |
| Allocation variability | Admitted strict core plus bounded optional dynamic scratch |

Unsupported behavior fails during capture, compilation, profiling, or
admission instead of selecting a heuristic semantic fallback.

## Architecture reading order

The pages form one path from capture to execution. The [documentation
index](../README.md) annotates the same order; this is the map.

**Foundations**

1. [Intermediate representation](ir.md) -- the objects, tasks, schedules and
   digests every other page is written in terms of.

**PyTorch lowering**

2. [PyTorch capture and lowering](lowering.md) -- PyTorch semantics and
   compiled storage behavior, mapped into that IR.
3. [Graph-pair construction](graph-pair-construction.md) -- local
   forward/backward alternatives for one structural contract.
4. [The step artifacts](step.md) -- what a captured step is, and why its two
   values need no framework to read.
5. [Importing state](state-import.md) -- how a model's state reaches a pool
   with no host copy of itself, and how dtype is decided.
6. [The optimizer](optimizer.md) -- state declared on meta and filled by the
   caller, and values a step may set.

**Planning**

7. [The ShadowSpillProgram](program.md) -- what the system plans for.
8. [The planning problem](planning-problem.md) -- the question asked about it.
9. [Plan search](search.md) -- what the planner asks of a search, and
   promises it.
10. [Writing a search algorithm](search-algorithm.md) -- the methods to
   implement, every argument, and a worked example.
11. [Graph-pair selection](graph-pair-selection.md) -- bounded complete
    assignments across those alternatives.
12. [PressureFit](pressurefit.md) -- the search that ships: logical residency
    and memory-action selection.
13. [Physical admission and offset handling](physical-admission.md) -- the
    selected plan proved against real pool geometry.
14. [From a resolved program to leases](admission-leases.md) -- what a schedule
    allocates, and when each lease is live.
15. [Fixed-offset placement](fixed-placement.md) -- how leases are given
    addresses, and what that costs.
16. [Simulation](simulation.md) -- the deterministic timeline a search prices
    its candidates against.
17. [Planning orchestration](planning.md) -- artifacts composed, callable and
    report published.

**Execution**

18. [Memory runtime](memory-runtime.md) -- leases, causal reuse, the worker,
    failure, and tracing.
19. [Task boundaries](task-boundaries.md) -- what `before_task` and
    `after_task` do, and what is still in flight when the dispatcher returns.
20. [Failure, abort, and process exit](failure-and-exit.md) -- what each scope
    does with a failure, and why an exiting process is abandoned.
21. [Step boundaries](step-boundaries.md) -- the recurrent invocation cycle,
    and what step time means.
22. [Backends](backends.md) -- the driver-level table a provider implements.
23. [Memory pools](memory-pools.md), [transfers](transfers.md), and
    [events](events.md) -- the runtime objects built on that table: arenas,
    routes and lanes with calibration, and event pools.
24. [PyTorch adapter](adapter.md) -- what sits between PyTorch and the runtime.
25. [Timelines](timelines.md) -- how a traced step is measured on the device
    clock.

The [Python guide](../python/README.md) and [C guide](../c/README.md) document
the corresponding public interfaces. The [examples](../examples/README.md)
apply the complete pipeline to practical workflows.

Next: [Intermediate representation](ir.md).
