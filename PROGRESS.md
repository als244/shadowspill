# ShadowSpill Progress

Last updated: 2026-08-10

## Current milestone

Phase 2 — standalone deterministic simulator.

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

## Gate rule

No later phase is declared active until the current phase passes all of its
tests and is committed. Unexpected findings and design changes are recorded in
the ignored internal progress log before this tracked summary is updated.
