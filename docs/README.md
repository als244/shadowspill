# ShadowSpill documentation

This directory describes the current ShadowSpill implementation. The
documentation is split by contract so readers do not have to infer whether a
statement applies to Python, C, or the framework-neutral design.

## Start here

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
- To integrate a compiled component or backend, start with the [C API
  guide](c/README.md).
- To modify the repository, use the [development guide](development/README.md).

## Architecture

The pages stay in one directory because they form one ordered system pipeline.
The groups below are conceptual reading boundaries, not separate ownership
trees.

### Foundations

1. [Architecture overview](architecture/overview.md) — vocabulary, artifacts,
   ownership, invariants, and supported scope.
2. [Intermediate representation](architecture/ir.md) — logical Programs,
   recomputation, schedules, and execution plans.

### PyTorch lowering

3. [PyTorch capture and lowering](architecture/lowering.md) — semantic roots,
   executable storage, profiling, and canonical objects.
4. [Graph-pair construction](architecture/graph-pair-construction.md) —
   structural forward/backward alternatives, saved-value accounting, and
   profiling.

### Planning

5. [Recomputation selection](architecture/recomputation-selection.md) — bounded
   complete selections across occurrence-level graph-pair options.
6. [PressureFit](architecture/pressurefit.md) — mathematical formulation,
   inputs/outputs, bounded policy search, repair, and pseudocode.
7. [Physical admission and offset handling](architecture/physical-admission.md)
   — allocation lifetimes, fixed placement, dynamic scratch, and causal reuse.
8. [Planning orchestration](architecture/planning.md) — reusable artifacts,
   transfer inputs, callable publication, and PlanReport.

### Execution

9. [Simulation](architecture/simulation.md) — compute, transfer, capacity, and
   causal-dependency replay.
10. [Memory runtime](architecture/memory-runtime.md) — pools, leases, worker,
    task boundaries, failure, and tracing.

## Examples

- [Training loop](examples/training-lifecycle.md)
- [Forward-only execution](examples/forward-only.md)
- [Reusable planning and budget sweeps](examples/reusable-planning.md)
- [Diagnosing a plan and real step](examples/diagnostics.md)
- [Custom stage partitioning](examples/custom-partitioning.md)

## Diagnostics, failures, and artifacts

- [Interpreting a PlanReport](python/plan-report.md) — planning time, task and
  graph-pair selection, profiles, PressureFit, caches, and physical admission.
- [Interpreting StepResult diagnostics](python/step-diagnostics.md) — seven task
  timestamps, host boundaries, allocator/transfer evidence, and simulator
  reconciliation.
- [Program and annotated-plan JSON](python/planning-json.md) — canonical Program,
  PressureFitProgram, StepProgram, and AnnotatedProgramPlan schemas.
- [Errors, failures, and cleanup](python/failures.md) — exception taxonomy,
  structured runtime evidence, rollback, and teardown.

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
