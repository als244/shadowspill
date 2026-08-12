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
