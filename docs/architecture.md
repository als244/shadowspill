# Architecture

ShadowSpill separates logical planning, deterministic simulation, memory
execution, and framework capture.

```text
Program + profiles ──► PressureFit ──► ExecutionPlan
       │                     │
       └──────► Simulator ◄──┘
                                ExecutionPlan
                                     │
                                     ▼
PyTorch caller ──► task boundaries ──► C runtime ──► backend plugin
       │                                  │
       └──────── launches compute         ├── H2D stream
                                          ├── D2H stream
                                          └── progress thread
```

## Dependency rules

- IR records contain no framework objects or backend handles.
- The simulator accepts an explicit schedule and never invokes planning.
- The planner may call the simulator, but the simulator never links the
  planner.
- The runtime consumes an admitted execution plan without interpreting model
  or optimizer semantics.
- A frontend captures tasks, profiles executable ABIs, and binds framework
  storages at task boundaries.
- Model and operation libraries are clients. They cannot become planner or
  runtime dependencies.

## Device topology

Device and resource identity are present in the IR from the start. The likely
distributed deployment is one process and one runtime context per device, but
neither the IR nor the C ownership model assumes that topology. Communication
is a task/resource kind rather than an optimizer special case.

## Runtime ownership

One process-wide framework allocator installation routes allocations to
explicit device contexts. Each context owns its slab, pinned host arena,
allocation and object tables, transfer streams, progress service, and first
failure record. PyTorch owns compute streams and numerical task dispatch.

The PyTorch adapter is intentionally narrow: allocator callbacks, storage
rebinding, and framework registration. All allocation policy, residency,
transfers, waits, and teardown remain in the C runtime.
