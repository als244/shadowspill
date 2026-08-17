# Architecture overview

ShadowSpill turns a fixed-shape PyTorch forward or accumulated training step
into an explicit program, chooses recomputation and memory actions, proves that
the selected schedule fits its physical memory pools, and returns an ordinary
Python callable. PyTorch and the compiled provider execute the numerical work;
ShadowSpill owns memory residency, movement, readiness, and the schedule around
that work.

## Reading order

Read the architecture in dependency order:

1. This overview defines the vocabulary, artifacts, ownership boundaries, and
   correctness invariants.
2. [Intermediate representation](ir.md) defines the framework-neutral logical
   program and schedule.
3. [PyTorch capture and lowering](lowering.md) explains how PyTorch semantics
   and compiled storage behavior become that program.
4. [Graph-pair construction](graph-pair-construction.md) creates and profiles
   each legal structural forward/backward alternative.
5. [Recomputation selection](recomputation-selection.md) constructs a bounded
   portfolio of complete occurrence-level task selections.
6. [PressureFit](pressurefit.md) formulates and selects logical residency and
   memory-action policy for an arbitrary ordered Program.
7. [Physical admission and offset handling](physical-admission.md) assigns
   execution-pool ranges and proves causal reuse.
8. [Planning orchestration](planning.md) composes reusable artifacts and
   publishes the admitted callable and report.
9. [Simulation](simulation.md) predicts the selected compute, transfer, and
   memory timeline.
10. [Memory runtime](memory-runtime.md) executes the admitted plan through
   pools, leases, task boundaries, transfer lanes, and the worker.

The [Python guide](../python/README.md) and [C guide](../c/README.md) document
the corresponding public interfaces.

## End-to-end pipeline

```text
PyTorch model, objective, optimizer, and fixed examples
                         |
                         v
               Export/AOT capture
                         |
                         v
                  stage partition
                         |
                         v
              graph-pair construction
                         |
                         v
             structural task compilation
                         |
                         v
        timing, storage, and allocation profiling
                         |
                         v
          canonical Program lowering and residency
                         |
                         v
                    StepProgram
                         |
                         v
                PressureFitProgram
                         |
                         v
        recomputation and memory-policy selection
                         |
                         v
               physical admission
                         |
                         v
              AnnotatedProgramPlan
                         |
                         v
        callable compilation and materialization
                         |
                         v
          PlannedForward / PlannedTrainStep
```

Capture and graph lowering determine semantics. Compilation and profiling
determine executable geometry and cost. PressureFit chooses logical policy.
Physical admission proves that the complete selected step has a valid layout.
Materialization publishes that admitted plan to the runtime.

## Artifact ladder

| Artifact | Meaning | Reusable without |
|---|---|---|
| `Program` | Logical objects, tasks, profiles, resources, and recomputation alternatives for one schedule role | PyTorch |
| `PressureFitProgram` | A `Program` plus residency, capacity, admission, and simulation inputs | Capture, compilation, or profiling |
| `StepProgram` | Recurrent and optional initialization `PressureFitProgram` values plus training-step provenance | PressureFit or callable materialization |
| `AnnotatedProgramPlan` | One selected schedule, physical layout, simulation result, and complete planning diagnostics | The model or runtime |
| `PlanReport` | The published callable's Program, plan, execution mapping, profiles, cache lineage, and diagnostics | Console logging |

`make_step_program()` stops at `StepProgram`. `pressurefit_program()` consumes
one of its `PressureFitProgram` values under new budgets or transfer
bandwidths. `plan_step()` and `plan_forward()` run the complete pipeline and
return materialized callables.

## Working vocabulary

| Term | Meaning |
|---|---|
| Stage | One ordered partition of the captured model graph. |
| Structural ABI | The shape, dtype, role, alias, mutation, and executable-storage contract shared by equivalent task occurrences. |
| Graph-pair portfolio | Every configured forward/backward alternative for one differentiated structural ABI, commonly save and recompute. |
| Recomputation selection | One complete choice of graph-pair option for every occurrence-level group. |
| Storage root | One semantic allocation identity shared by all of its views. |
| Program object | One logical alias bundle with size, role, persistence, and task dependencies. |
| Action trigger | The task boundary at which a fetch, evict, or release becomes ordered and destination capacity is reserved. |
| Physical admission | The proof that task allocations, object generations, transfers, and causal reuse fit the selected pools. |
| Memory lease | Ownership of one pool range for one residency generation. |

## Component boundaries

| Component | Owns | Does not own |
|---|---|---|
| PyTorch frontend | Export/AOT capture, stage partitioning, compiled task callables, tensor rebinding, objective and optimizer integration | Memory-policy search or transfer progress |
| IR | Objects, tasks, resources, recomputation options, schedules, and resolved execution records | PyTorch tensors or provider handles |
| Planner | Complete-selection portfolio, residency strategies, memory actions, and candidate selection | Graph construction or numerical execution |
| Physical admission | Allocation lifetimes, task-allocation ABI, fixed placements, dynamic scratch allowance, and causal reuse dependencies | PressureFit policy |
| Simulator | Deterministic compute, transfer, memory, and dependency replay | Candidate generation |
| Runtime | Pools, leases, objects, events, transfer lanes, task boundaries, failure state, and worker progress | Graph capture or model semantics |
| Backend | Provider allocation, copy, stream, event, and profiler operations | Object or schedule policy |

Dependencies point inward through these contracts. The framework-neutral IR,
planner, simulator, and runtime do not import PyTorch. Raw CUDA driver calls
and NVTX implementation live in the CUDA backend or PyTorch allocator adapter.
The Python frontend uses PyTorch's device and stream APIs at its framework
boundary but does not expose those details to the neutral components.

## Following one object

One logical value crosses the system without making its current pointer its
identity:

1. Export and stage partitioning identify the producing node, views, aliases,
   mutations, and stage boundary.
2. Graph-pair construction exposes every configured forward/backward variant.
3. A `TaskStorageContract` assigns each variant value a semantic storage root.
4. Compilation and profiling attach executable extents, allocation behavior,
   workspace, and timing without changing that root.
5. `ObjectCatalog` maps the root to one canonical Program object across tasks.
6. Recomputation selection chooses one graph-pair option for the occurrence.
7. PressureFit selects whether the resulting value is fetched, evicted,
   retained, or released.
8. Physical admission assigns compatible pool ranges and any required causal
   reuse dependencies.
9. Materialization registers the object and direct execution records with the
   runtime.
10. `before_task()` resolves the object's current lease generation, inserts an
   unfinished readiness-event wait on the compute stream, and rebinds PyTorch
   storage.
11. `after_task()` publishes output or mutation generations, records task
   completion, and triggers ordered memory actions.
12. The worker submits transfers and publishes completion; generation checks
    prevent stale work from modifying a successor.

The [lowering contract](lowering.md) and [graph-pair construction](graph-pair-construction.md)
own steps 1–5; [recomputation selection](recomputation-selection.md),
[PressureFit](pressurefit.md), and [physical admission](physical-admission.md)
own steps 6–8; and the [memory runtime](memory-runtime.md) owns steps 9–12.

## Correctness invariants

- Semantic object identity never depends on a transient pointer, allocator
  callback identity, or incidental FakeTensor storage.
- A callable is published only after both logical scheduling and physical
  admission succeed for the same selected Program.
- A pool range is reused only after stream order or an explicit completion
  dependency makes the predecessor inaccessible.
- Fetch and evict destinations consume capacity at their action trigger, even
  when the copy reaches its lane later.
- Every lease, event, transfer, and object publication is generation-checked;
  stale completion cannot mutate a successor.
- Execution and spill accounting never exceed their configured physical caps.
- Allocation behavior outside the admitted strict core is bounded by dynamic
  scratch; a mismatch fails before an invalid address reaches a kernel.
- Planner, simulator, admission, and runtime identities are retained in
  diagnostics so the executed plan can be reconciled mechanically.

## Runtime topology

The supported runtime topology has one execution-device pool and one
pinned-host spill pool. One C worker services independent fetch and evict
lanes, completion frontiers, and deferred releases. The PyTorch dispatcher
does not perform runtime progress and does not wait for transfer completion on
the host when a stream dependency can express the same ordering.

The runtime uses central owners with narrow synchronization:

- an object-ID hash table protects membership;
- each object protects its residency generations and leases;
- each memory pool protects its range allocator and accounting;
- fetch and evict lanes have independent queues and locks;
- each recording stream has an ordered completion frontier;
- trace appends use a bounded buffer and are disabled by default.

Backend calls are not made while unrelated data-structure locks are held.
There is no general-purpose global runtime mutex.

## Planning and execution ownership

`Runtime` is initialized before planning and owns physical pools and transfer
calibration. A planned callable temporarily owns an admitted slice of those
pools and the registered execution records. Only one active callable is
supported by a runtime.

`plan_step()` and `plan_forward()` use named execution and spill pools.
Budgets default to configured capacities and may only reduce them. The selected
transfer calibration, capacities, program, physical layout, and diagnostics
are retained in the returned `PlanReport`.

At execution, each task follows one frontend skeleton:

```text
before_task -> compiled callable -> after_task
```

The frontend boundary includes runtime acquisition, stream waits, storage
rebinding, argument assembly, output classification, mutation publication,
releases, and action submission. The neutral C boundary handles object and
lease transitions only; it never manipulates PyTorch storage objects.

## Supported scope

| Capability | Current contract |
|---|---|
| Graph geometry | Fixed shape and stride under the captured guards |
| Capture | Strict Export/AOTAutograd/Inductor with no graph breaks |
| Custom operations | Supported when fake/meta and alias/mutation schemas are complete |
| Data-dependent outputs | Unbounded output geometry is rejected during planning |
| Execution backend | PyTorch on one CUDA execution device |
| Spill backend | One registered pinned-host pool |
| Runtime ownership | One process runtime and one active planned callable |
| Transfer topology | One worker with independent fetch and evict lanes |
| Allocation variability | Admitted strict core plus bounded optional dynamic scratch |
| Cross-step residency | Not represented; startup fetches and final cooldown remain visible |

The supported claim is therefore precise: fixed-shape graphs accepted by the
strict capture boundary, with correct fake/meta and alias/mutation contracts,
can be lowered without model-specific policy. Unsupported behavior fails
during capture, compilation, profiling, or admission rather than selecting a
heuristic semantic fallback.

Next: [Intermediate representation](ir.md).
