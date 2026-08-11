# ShadowSpill Progress

Last updated: 2026-08-10

## Current milestone

Phase 5 — CUDA slab backend and PyTorch allocator/storage adapter.

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
- [x] Deterministic PressureFit, recomputation portfolio, simulator-verified
  candidate selection, and canonical ExecutionPlan construction implemented.
- [x] Public planner C ABI, header, compiled selector, documentation, and
  frozen schedule artifacts added.
- [x] Framework-neutral slab runtime, deterministic mock backend, stream-safe
  allocation retirement, object residency/version state machine, and public C
  API implemented.
- [x] CUDA Driver API backend performs one conventional slab allocation, one
  pinned-host arena allocation, asynchronous copies/events/waits, and exposes
  capability and operation ledgers without VMM.
- [x] PyTorch's supported pluggable allocator is connected to the neutral slab
  with exact address lookup, full record-stream retirement, structured OOM
  propagation, and no Python monkey-patching.
- [x] The narrow version-pinned storage adapter preserves Parameter, storage,
  view, and alias identity while a planned object is dematerialized and later
  rebound to a different slab address and allocation generation.
- [x] Real CUDA task boundaries insert one wait per unfinished input and allow
  simultaneous H2D, D2H, and unrelated compute without a steady-state transfer
  stream synchronization.

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

### Phase 3

- PressureFit is operation-, framework-, optimizer-, and model-independent. It
  plans stable alias/task identities, workspace admission, tentative initial
  placement, residency cuts, dirty writeback, packed H2D, and bounded
  recomputation alternatives entirely from canonical IR.
- Every returned schedule is accepted by the standalone simulator. Candidate
  workers preserve identical schedules, digests, diagnostics, and tie breaks.
- The retained 10-layer pressure artifact matches 326 us, 94 ordered actions,
  and the A0/A1/A2 activation offloads. The 5-layer artifact is also logically
  identical. A simulator-valid earlier head prefetch improves the 2-layer
  artifact from the retained 114 us to 110 us; the dominant result is frozen
  and documented rather than intentionally regressed.
- `libshadowspill_planner.so` provides a stable compiled candidate-verification
  and selection boundary over the public simulator ABI. It contains no Python,
  framework, backend, model, or operation dependency.
- 101 Python tests pass with 92.10% branch coverage. C warnings-as-errors,
  native canaries, Clang ASan/UBSan, strict mypy, Ruff, naming, external-oracle,
  isolated wheel, RPATH, public-header, and exported-symbol gates pass.

### Phase 4

- `libshadowspill_runtime.so` owns one coalescing device slab, one bounded host
  arena, allocation/object tables, one progress thread, H2D/D2H queues,
  readiness events, failures, and deterministic teardown without framework or
  vendor types.
- Ordinary allocations support synchronous lease, logical free, multiple
  recorded streams, delayed physical reuse, allocator blocking/wakeup, and
  diagnostic no-progress OOM. Telemetry reports requested physical occupancy,
  largest free range, external fragmentation, blocked allocators, transfers,
  and waits.
- Alias-group objects preserve identity while their device addresses and
  generations change. Exact release/offload/prefetch actions are submitted at
  task boundaries; every unfinished input prefetch inserts its own compute
  stream wait.
- The deterministic mock backend covers delay, transfer ordering, stream
  retirement, changed addresses, failure injection, and first-cause
  propagation. A worker failure wakes an allocator blocked on pending release.
- A stream-timeline race found by the new regression is fixed: H2D completion
  changes readiness but never rolls back a device version advanced after the
  compute stream has waited on that transfer.
- Seven C canaries pass with warnings as errors, GCC 11 ASan, Clang UBSan, and
  GCC 11 ThreadSanitizer. TSan is run with ASLR disabled because all instrumented
  binaries otherwise fail at startup on this host with an unexpected-shadow
  mapping before executing application code. Valgrind Memcheck and Helgrind
  also pass; the only suppression is a scoped Valgrind 3.18/glibc 2.34
  `pthread_cond_timedwait` false positive.
- The isolated wheel contains the runtime/mock shared libraries and public
  headers. A clean virtual environment loads ABI version 1 and compiles/runs a
  consumer against only the installed artifacts. The full 101-test Python
  suite remains at 92.10% branch coverage.

## Gate rule

No later phase is declared active until the current phase passes all of its
tests and is committed. Unexpected findings and design changes are recorded in
the ignored internal progress log before this tracked summary is updated.
