# ShadowSpill documentation

This directory describes the current ShadowSpill implementation. The
documentation is split by contract so readers do not have to infer whether a
statement applies to Python, C, or the framework-neutral design.

## Start here

- To use ShadowSpill from PyTorch, read the [Python quickstart](python/quickstart.md)
  and then the [Python API](python/README.md).
- To understand the system, follow the ordered path in the [architecture
  overview](architecture/overview.md).
- To integrate a compiled component or backend, start with the [C API
  guide](c/README.md).
- To modify the repository, use the [development guide](development/README.md).

## Architecture

1. [Architecture overview](architecture/overview.md) — vocabulary, artifacts,
   ownership, invariants, and supported scope.
2. [Intermediate representation](architecture/ir.md) — logical Programs,
   recomputation, schedules, and execution plans.
3. [PyTorch capture and lowering](architecture/lowering.md) — semantic roots,
   executable storage, profiling, and canonical objects.
4. [Planning and physical admission](architecture/planning.md) — PressureFit,
   fixed layout, transfer inputs, and PlanReport.
5. [Simulation](architecture/simulation.md) — compute, transfer, capacity, and
   causal-dependency replay.
6. [Memory runtime](architecture/memory-runtime.md) — pools, leases, worker,
   task boundaries, failure, and tracing.

## Historical evidence

[Engineering investigations](investigations/README.md) preserve root-cause
evidence for bugs and performance work. They describe the revision under
investigation and are not normative specifications. When an investigation and
an architecture or API page differ, the architecture or API page is the
current contract.

## Documentation policy

Public behavior is documented here and tested against exported Python names,
public C headers, local links and heading anchors, and the Python signatures
mirrored in API examples. Installed headers remain authoritative for C layouts,
ABI constants, and exact C signatures. Source remains authoritative for cache
schema labels and internal implementation details.
