# ShadowSpill Progress

Last updated: 2026-08-10

## Current milestone

Phase 3 — deterministic PressureFit and recomputation planning.

## Status

- [x] Fresh repository and independent Conda environment created.
- [x] Repository-local commit identity configured.
- [x] PyTorch 2.13.0+cu130 and editable external `mlops` verified on the RTX
  5090.
- [x] Public naming, dependency direction, and physical-budget contracts
  documented.
- [x] Compact legacy-oracle evidence frozen with source hashes.
- [x] Clean build, wheel, test, naming, and dependency-isolation gates pass.
- [x] Canonical Program, MemorySchedule, and ExecutionPlan schemas implemented.
- [x] Stable lossless dense projections and frozen IR identity artifact added.
- [x] Standalone integer-time simulator implemented in Python and C.
- [x] Installed wheel selects the versioned compiled simulator without planner,
  framework, model, or accelerator dependencies.
- [x] Frozen compiled simulator decomposed into focused C modules without an
  ABI or behavior change.

## Completed gates

### Phase 0

- Editable and isolated wheel builds succeed through scikit-build-core.
- Python tests and 90% coverage gate pass.
- Ruff lint/format and strict mypy pass.
- C11 warnings-as-errors build and CTest canary pass.
- Production naming and old-repository dependency audits pass.
- External oracle executes only in a separate process with a sanitized Python
  environment.

### Phase 1

- Immutable, validated records cover devices, resource lanes, alias extents,
  versions, persistence, dependencies, mutations, workspace, recomputation
  choices, memory actions, entrypoints, physical admission, and predictions.
- Canonical JSON round trips byte-identically with deterministic SHA-256
  identities.
- Save/recompute alternatives resolve to an ordinary DAG by removing inactive
  tasks and dependency edges; simultaneous non-exclusive writers are rejected.
- Dense projections preserve every compiled-component input without framework
  objects or backend handles.
- 51 Python tests pass with 98% branch coverage, including Hypothesis-generated
  programs and field-specific invalid-record cases.
- Strict mypy, Ruff, isolated wheel installation, C warnings-as-errors, CTest,
  naming, and dependency-isolation gates pass.

### Phase 2 behavior gate

- Task dependencies, compute/communication/control resource lanes, H2D and D2H
  lanes, workspace intervals, residency, ordered memory actions, physical
  capacities, stalls, and detailed failures are replayed with integer bytes and
  nanoseconds.
- Python and C results match on focused overlap/concurrency cases, structured
  capacity failures, large integer transfers, and Hypothesis-generated programs.
- 64 randomized schedules exactly match the prior simulator in a separate
  process for task/transfer intervals, host/device peaks, and makespan.
- Compiled replay is deterministic across 1, 2, and 8 concurrent caller workers.
- 83 Python tests pass; full source branch coverage is 94%.
- C11 warnings-as-errors, ASan, UBSan, native canaries, Ruff, strict mypy,
  naming, isolated wheel build/install, and compiled public-API smoke gates pass.

### Phase 2 structural gate

- The compiled simulator is split into validation, memory/state, task,
  transfer, event, diagnostics, status, and orchestration modules behind one
  private internal header.
- Normal, external-oracle, compiled differential, deterministic-concurrency,
  packaging, naming, and public-symbol results are unchanged.
- Fresh separate Clang ASan and UBSan builds pass both native canaries and the
  compiled Python differential suite. This avoids an intermittent GCC 13 ASan
  signal-handler loop observed even around a trivial canary on this host.
- Static schedule validation and both simulators now model simultaneous
  retained-host/device copies and invalidate retained host contents after
  output writes. Stale host generations cannot source a prefetch or satisfy
  final residency; the behavior has dedicated Python and public-C canaries.

## Gate rule

No later phase is declared active until the current phase passes all of its
tests and is committed. Unexpected findings and design changes are recorded in
the ignored internal progress log before this tracked summary is updated.
