# Architecture overview

ShadowSpill turns a fixed-shape PyTorch forward or accumulated training step
into an explicit program, chooses recomputation and memory actions, admits a
physical layout, and returns an ordinary Python callable. PyTorch and the
compiled provider still execute the numerical work; ShadowSpill owns memory
residency, movement, readiness, and the schedule around that work.

```text
PyTorch model and examples
          |
          v
capture -> partition -> graph-pair construction -> lowering
          |                                      |
          |                                      v
          |                                  StepProgram
          |                                      |
          |                                      v
          +-----------------------------> PressureFit
                                                 |
                                                 v
                                      physical admission
                                                 |
                                                 v
                                      AnnotatedProgramPlan
                                                 |
                                                 v
                                  materialized runtime callable
```

## Component boundaries

| Component | Owns | Does not own |
|---|---|---|
| PyTorch frontend | Export/AOT capture, stage partitioning, compiled task callables, tensor rebinding, objective and optimizer integration | Memory-policy search or transfer progress |
| IR | Objects, tasks, resources, recomputation options, schedules, and resolved execution records | PyTorch tensors or CUDA handles |
| Planner | Recomputation portfolio, residency strategies, memory actions, and candidate selection | Numerical execution |
| Physical admission | Allocation lifetimes, task-allocation ABI, fixed placements, dynamic scratch allowance, and causal reuse dependencies | PressureFit policy |
| Simulator | Deterministic compute, transfer, memory, and dependency replay | Candidate generation |
| Runtime | Pools, leases, objects, events, transfer lanes, task boundaries, failure state, and worker progress | Graph capture or model semantics |
| Backend | Provider allocation, copy, stream, event, and profiler operations | Object or schedule policy |

Dependencies point inward through these contracts. The framework-neutral IR,
planner, simulator, and runtime do not import PyTorch. Provider-specific CUDA
code is confined to the CUDA backend and the PyTorch adapter.

## Runtime topology

The supported runtime topology has one execution-device pool and one
pinned-host spill pool. One C worker services independent fetch and evict
lanes, completion frontiers, and deferred releases. The PyTorch dispatcher
does not perform runtime progress and never waits for transfer completion on
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

The supported claim is deliberately narrow: fixed-shape graphs accepted by
ShadowSpill's strict Export/AOTAutograd/Inductor capture, with correct fake/meta
and alias/mutation contracts, can be lowered without model-specific policy.
Graph breaks, unbounded data-dependent output geometry, or custom operations
with incomplete semantic contracts fail during planning.

Cross-step cyclic residency is outside the supported schedule model. Startup
fetches and final cooldown therefore remain visible in real one-step
measurements, while the steady-state simulator focuses on the selected step
schedule.
