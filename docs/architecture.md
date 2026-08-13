# Architecture

ShadowSpill separates framework capture, planning, deterministic simulation,
and memory execution.

```text
Program + profiles ──► PressureFit ──► ExecutionPlan
       │                     │
       └──────► Simulator ◄──┘
                                ExecutionPlan
                                     │
                                     ▼
PyTorch caller ──► task boundaries ──► neutral C runtime
       │                                  │
       └──────── launches compute         ├── named MemoryPools
                                          ├── directed TransferRoutes
                                          ├── fetch/evict lanes
                                          └── one worker thread
```

## Dependency rules

- IR records contain no framework objects, pointers, or backend handles.
- The simulator accepts an explicit schedule and never invokes the planner.
- The planner may call the simulator; the reverse dependency is forbidden.
- The runtime consumes an admitted execution plan without interpreting model,
  optimizer, or operation semantics.
- A frontend captures tasks, profiles executable ABIs, and binds framework
  storage at task boundaries.
- Pool storage, route copies, events, and profiler integration come from
  backend vtables. Neutral targets build and test without an accelerator SDK.
- Models and operation libraries are clients and cannot become core
  dependencies.

## Runtime topology

`Runtime` is initialized explicitly before planning. It owns a registry of
named pools and directed routes. Each route has independent measured latency
and bandwidth; calibration publishes an immutable generation-tagged matrix.
Plans select one execution pool and one spill pool, take an exact matrix
snapshot, and record the selected fetch/evict profiles in `PlanReport`.

The initial implementation registers one accelerator execution pool and one
pinned-memory spill pool. The public pool/route representation does not assign
host or accelerator meaning to `MemoryPool` or `MemoryLease`; future peer,
remote-memory, and storage providers can implement the same contracts.

The expected first distributed deployment is one process and runtime per
execution device. Device and communication identity already exist in the IR,
so DDP, expert, pipeline, tensor, and context parallel work does not require
model-specific allocator behavior.

## PyTorch boundary

PyTorch owns compute streams and numerical dispatch. One narrow, version-pinned
adapter provides allocator callbacks and storage rebinding. The neutral runtime
owns allocation policy, object generations, readiness events, transfers,
failure propagation, and teardown.

One runtime context currently owns one worker thread and two route lanes. In
the NVIDIA provider these streams appear in NSYS as `shadowspill_fetch` and
`shadowspill_evict`; the worker appears as `shadowspill_worker`. Profiling names
and ranges use a neutral profiler vtable, with NVTX confined to the NVIDIA
implementation and a future ROCm implementation able to provide rocTX.

## PyTorch planning and lowering layout

The frontend cold path is organized around immutable artifacts rather than a
stateful planning coordinator:

```text
pytorch/planning/
├── cache.py       artifact lookup, validation, and archival policy only
├── artifacts.py   immutable values passed between planning phases
├── forward.py     forward capture → profile → Program → PressureFit composition
├── training.py    training capture → profile → Program → PressureFit composition
├── admission.py   physical-cap checks and pool sealing
├── reporting.py   PlanReport construction and lineage publication
└── common.py      validation, phase timing, and capacity calculations

pytorch/lowering/
├── catalog.py       canonical objects, aliases, and model registration
├── task_binding.py  one semantic/compiled task ABI → canonical objects
├── profiles.py      shared physical-layout and TaskProfile resolution
├── program.py       shared canonical Program publication
├── forward/
│   ├── program.py   readable partitioned-forward orchestrator
│   ├── objects.py   model and root-input registration
│   ├── tasks.py     stage binding and task emission
│   ├── residency.py initial/final residency derivation
│   └── artifacts.py immutable phase values
└── training/
    ├── program.py   readable accumulated-training orchestrator
    ├── objects.py   model/input/gradient/optimizer registration
    ├── bindings.py  stage boundaries and graph-pair variants
    ├── tasks.py     forward/backward/optimizer task emission
    ├── residency.py initial/final residency derivation
    └── artifacts.py immutable phase values
```

`plan_forward()` and `plan_step()` resolve public runtime/cache policy and then
call these phase functions in order. Every phase has a typed input and
immutable result, so tools may stop after Program construction, rerun
PressureFit for different capacities, or perform physical admission later
without reimplementing capture logic.

Semantic and physical authority remain separate. `lowering/task_binding.py`
binds offline task-storage contracts to the objects in
`lowering/catalog.py`. Both modes use the same profile/layout resolver and
Program publisher; only their real semantic differences live under
`lowering/forward/` and `lowering/training/`.
Compiler measurements supply extents, workspace, and time but never redefine
semantic identity. Cache code cannot capture, compile, lower, plan, or admit a
model—it only resolves or persists artifacts at those explicit boundaries.
