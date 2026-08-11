# ShadowSpill Progress

Last updated: 2026-08-10

## Current milestone

Phase 8 — public forward and training callables.

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
- [x] CUDA admission telemetry reports per-process physical bytes through NVML,
  separately from logical slab occupancy, plus whole-device used/total bytes.
- [x] Neutral physical admission computes explicit provider/workspace/host
  leeway and replays complete allocation lifetimes through the production slab
  policy without moving planner directives.
- [x] PyTorch allocator bootstrap accepts a physical cap, measures context
  bytes before slab creation, derives the slab internally, and rejects any
  bootstrap whose per-process NVML reading exceeds the cap.
- [x] Physical sealing rejects provider requirements larger than the reserved
  headroom; later per-process growth beyond either headroom or the total cap is
  latched as a non-callback plan violation.
- [x] Ordinary frees match stream-ordered allocator behavior: same-stream
  ranges are immediately reusable, while any distinct recorded stream retains
  event-gated retirement.
- [x] Bounded task-scoped allocation telemetry records requested/charged
  lifetimes, slab extents, output promotion, and exact sequential workspace
  reuse without allocating inside allocator callbacks.
- [x] Fixed-shape tensor/static guards, storage-free CUDA model replicas, strict
  Export, objective schemas, and save/recompute AOTAutograd graph pairs work
  for ordinary ATen and unrelated registered custom operations.
- [x] Model-independent automatic partitioning finds outer repeated blocks,
  retains nested experts inside each block, differentiates stages independently,
  and keys profiling by structural ABI rather than layer/task position.
- [x] Atomic content-addressed profile caching scatters one unique measurement
  to all matching task positions; a warm cache invokes no measurement kernels.
- [x] Optimizer capture is allowlist-free: lazy state discovery, first/recurrent
  task distinction, tensor-state lifting, and bounded opaque fallback work for
  AdamW, Adam, SGD, and an unrelated custom optimizer without changing the
  caller's optimizer.
- [x] Structural task profiling now compiles real CUDA tasks once, warms outside
  timing, measures calibrated CUDA-event samples, and derives exact anonymous
  workspace live peaks from the production slab allocator.
- [x] Forward task positions lower deterministically into canonical Program
  objects, storage alias groups, profiles, dependencies, initial/final
  residency, and narrow framework entrypoint bindings accepted by PressureFit.
- [x] Initial CPU payloads can populate retained pinned backing directly, and
  final device objects can transfer to ordinary caller allocator ownership
  without copying, changing addresses, or leaking logical object identities.
- [x] Public `forward_pass()` now runs captured pure-ATen stages through the
  production allocator, PressureFit schedule, storage rebinding, and ordinary
  PyTorch task dispatch across repeated fixed-shape calls.
- [x] Forward checkpoint snapshots/reloads, caller-owned output lifetime,
  original Parameter identity, ties, model mode, deterministic close, and CPU
  restoration pass in a fresh allocator process.
- [x] Captured objective graph pairs now have a small explicit executor that
  reconstructs loss/metrics and returns input-aligned gradients; both save and
  recompute ABIs match ordinary autograd on the same parameters and inputs.
- [x] Capture inventories retain their FakeTensor storages until lowering ends,
  preventing reclaimed StorageImpl identities from aliasing unrelated residual
  extents across save/recompute variants.
- [x] Two heterogeneous microbatch objective pairs now lower into one canonical
  accumulated Program with mutually exclusive save/recompute task pairs,
  persistent gradient identities, functional accumulation mutations, and one
  final optimizer task accepted by PressureFit and the simulator.

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

### Phase 5

- The CUDA backend owns one conventional `cuMemAlloc` slab and one bounded
  `cuMemHostAlloc` arena per context. Its progress service uses independent H2D
  and D2H streams, events, queries, and stream waits without VMM or a
  steady-state device allocation.
- PyTorch's supported pluggable allocator routes ordinary tensor, compiler, and
  workspace allocations through the neutral slab. Logical free, exact address
  lookup, and `record_stream` prevent early range reuse; impossible OOM retains
  complete no-progress diagnostics even though PyTorch 2.13 may construct a
  null-data tensor before the public boundary checks the failure latch.
- The private version-pinned C++ operation only swaps StorageImpl DataPtrs. A
  real CUDA canary proves one Parameter and its view retain object/storage
  identity across D2H, null dematerialization, and H2D into a different address
  and generation; stale generations fail before mutation.
- A three-stream CUDA canary proves two unfinished inputs insert two waits and
  that H2D and D2H transfers complete while an independent compute kernel is
  still running. No transfer-stream synchronization occurs at steady task
  boundaries.
- Nsight Systems records the documented task, storage-rebind, allocation,
  event-wait, H2D, and D2H NVTX ranges. The automated trace audit finds one
  slab allocation, one pinned arena, real bidirectional overlap, no VMM entry
  point, and no device/context-wide synchronization.
- The private adapter exports only its versioned `shadowspill_pytorch_*` ABI.
  All eleven CTest cases, all 105 Python tests (91.00% coverage), Ruff, strict
  mypy, naming, installed-wheel relocation, and symbol gates pass.

### Phase 8 forward vertical slice

- The public planning session validates CPU model state and guarded inputs,
  installs or reuses compatible process-global physical admission, captures
  FakeTensor/Export stages, profiles each structural ABI, runs PressureFit, and
  materializes the original model incrementally through retained host backing.
- The executor acquires and rebinds every task input before invoking the
  compiled PyTorch stage, promotes outputs, submits PressureFit actions in
  order, dematerializes released storage, and transfers final output leaves to
  ordinary caller ownership.
- A release-completion race discovered by the public canary is fixed without
  changing action timing: one retired binding token permits the frontend to
  null a pointer whose allocation was already reclaimed by progress.
- Read-only parameters correctly require H2D on each non-cyclic invocation but
  no D2H writeback, because their retained host copies remain current. A real
  D2H forward gate therefore requires pressured activation eviction rather
  than an unchanged-weight assertion.
- Fifteen CTest canaries and 169 Python tests pass. The CUDA public test covers
  repeated execution, exact metadata guards, state reload, caller-retained
  output lifetime, and idempotent close; full branch coverage is 90.42%.

### Phase 8 training integration (active)

- Canonical accumulated lowering now carries explicit fixed loss-tangent
  bindings and a deterministic provisional model/input layout. This permits
  incremental CUDA placeholder materialization before the one allowed optimizer
  factory invocation, without requiring the complete model to be resident.
- The selected-task executor implements two forward/backward contributions,
  planned gradient storage, in-place accumulation, one optimizer mutation, and
  caller ownership for detached loss/metric tensors. Public checkpoint and
  close lifecycle plumbing is present but not yet qualified.
- A fresh-process gate found that AOT may save a parameter *view* as a forward
  residual. Excluding only identical input object IDs falsely declared the view
  a produced allocation. Lowering now excludes every output whose alias group
  is already an input; a multi-linear regression covers this case.
- Promoted outputs now surrender PyTorch allocator ownership immediately while
  retaining their current address as a non-owning task binding. Caller results
  receive an explicit owning slab lease; this prevents address-reuse races with
  PyTorch's private pluggable-allocator pointer map.
- Fixed non-tensor inputs are guarded publicly but specialized out of compiled
  graph ABIs. Original tensor-primal positions remain recorded so AOT backward
  gradients still map correctly when static inputs contributed `None` leaves.
- The fresh-process training gate passes five steps against eager PyTorch with
  two heterogeneous microbatches, exact optimizer-call count, string metadata,
  tensor/static metrics, bitwise checkpoint replay, Parameter identity, CPU
  restoration, one slab, one pinned arena, and zero allocator callback errors.
- Lazy optimizer state now has two independently simulated immutable schedules:
  the initial update produces state objects, while recurrent updates consume
  and mutate those same dense identities. The executor selects the initial
  schedule only while state is absent and never moves either schedule's
  PressureFit directives.
- Spillable optimizer tensor state is promoted into planned object ownership at
  first creation, follows ordinary offload/prefetch generations thereafter,
  and is synchronously exposed as CPU storage only for checkpoint or close.
  Loading an empty checkpoint restores the initial phase; loading populated
  state adopts newly constructed CUDA tensors and writes them into retained
  host backing before the next recurrent invocation.
- Scalar optimizer control tensors remain bounded ordinary task allocations.
  This covers both PyTorch AdamW's intentionally CPU-resident step counters and
  custom/capturable device counters without pretending a host scalar is a
  spillable accelerator object. Large tensor state remains fully planned.
- An optimizer-profiling defect was isolated and committed separately: Inductor
  examples for intrinsically no-grad optimizer mutations must be detached.
  Otherwise AOTAutograd constructs an invalid mutation epilogue for
  heterogeneous parameter shapes.
- The fresh-process canary now uses ordinary lazy-state AdamW for five steps,
  two heterogeneous microbatches, step-three bitwise checkpoint replay, and
  CPU optimizer restoration. The Python public suite retains the state-free
  SGD gate and adds the same initial/recurrent AdamW lifecycle.
- Sixteen CTests and 179 Python tests pass; the complete coverage gate is above
  its required 90% threshold after exercising both optimizer phases. ASan,
  UBSan, and ThreadSanitizer continue to pass all neutral-runtime canaries.
- Planning now runs PressureFit against the public host cap, computes the larger
  of the initial/recurrent predicted host peaks, adds documented leeway, and
  grows the pinned arena exactly once before sealing when optimizer state makes
  the provisional model/input estimate insufficient. Existing host offsets and
  payloads survive replacement.
- Runtime ABI v6 and PyTorch adapter ABI v10 expose this planning-only growth.
  Shrinkage and post-seal growth fail; Python also requires old plus replacement
  arenas to fit simultaneously under `host_budget`, so reconciliation cannot
  create a transient hidden pinned-memory overage.
- The stateful CUDA canary now uses a 1024-square AdamW parameter, proves two
  pinned allocations (bootstrap plus reconciliation), then passes five steps,
  bitwise checkpoint replay, and CPU restoration. Native ASan, UBSan, and TSan
  pass after the arena-preservation change. All 181 Python tests pass the
  90.02% branch-coverage gate.
- Automatic training partitioning is now connected to the public path. Every
  microbatch has one immutable save/recompute choice per authored stage,
  explicit boundary activations and cotangents, reverse-stage VJPs, global
  gradient accumulation, and one optimizer task. Ordinary PyTorch SGD and lazy
  AdamW pass repeated public execution and checkpoint replay through this path.
- AOT can return one allocation both as a public stage boundary and as a saved
  residual. Associating the separately evaluated AOT storage with the
  canonical boundary alias prevents double promotion and premature release.
  Ephemeral objects released at a non-cyclic step boundary are now forgotten
  by the framework object store, so their next-step production is not mistaken
  for an in-place contribution into a dematerialized prior generation.
- The milestone gate passes all 16 native/CUDA canaries and 183 Python tests at
  90.04% branch coverage, together with Ruff, strict mypy, and the naming audit.
- Remaining qualification work is bounded opaque optimizer task admission,
  external mlops AdamW, stressed recomputation/offload tests, and the
  approximately-1B/full-model gates.
- CUDA-only registered optimizer operations no longer require a real duplicate
  model during discovery. If the copied optimizer creates its lazy state and
  then rejects the CPU sandbox, ShadowSpill converts the standard optimizer
  parameter/state inventory to FakeTensor CUDA values and captures through the
  operation's registered fake/meta contract. This is optimizer-agnostic and
  introduces no production dependency on `mlops`.
- External `mlops.optim.AdamW` now passes both isolated capture and a real
  three-step public `plan()` smoke test, including lazy BF16 moment/master
  state, two accumulated microbatches, checkpoint serialization, reload, and
  CPU restoration. The original optimizer and model remain unmodified by the
  discovery pass.
- Valid eager optimizer updates that Dynamo cannot graph are admitted as
  deterministic bounded task artifacts. A private real-CUDA sandbox is warmed
  and measured once through the same event and allocation telemetry used for
  compiled graphs; PressureFit receives the measured runtime and workspace,
  while steady-state execution calls the user's ordinary `Optimizer.step()`
  inside the exact annotated boundary.
- A Dynamo failure inside a `@no_grad` optimizer exposed a PyTorch compiler-state
  leak that disabled gradients for later AOT captures in the same thread.
  Optimizer capture now restores the caller's grad-mode bit on every success or
  failure exit. A public forced-opaque optimizer passes numerical parity and CPU
  lifecycle restoration.
- The complete milestone gate passes 190 Python tests at 90.08% branch
  coverage, all 16 production native/CUDA canaries, Ruff, strict mypy, and the
  naming audit. Neutral ASan and UBSan canaries pass; GCC TSan canaries pass
  with ASLR disabled so its fixed shadow address is available on this host.

### Phase 9 reference model families (active)

- Fresh `models/pytorch` implementations now cover Llama 3, Qwen 3.5 dense,
  and OLMoE without importing ShadowSpill or `mlops`. Their approximately-1B
  configurations contain exactly 1,179,699,200, 1,006,955,408, and 975,766,528
  parameters, respectively; compact variants exercise eager forward and
  backward semantics.
- Matching `models/mlops` implementations use the separately installed
  semantic operation package and retain byte-for-byte compatible state-dict
  keys and tensor geometries. Tiny reference/optimized pairs match forward
  results, objectives, and every parameter gradient for all three families.
- Qwen uses the intended `LLLF` hybrid schedule, partial rotary attention, and
  Gated DeltaNet recurrence. OLMoE functionally returns each layer's router
  auxiliary loss instead of storing a live tensor on a module, preserving a
  capture-friendly graph while retaining the pure reference semantics.
- The complete gate passes 199 Python tests at 90.08% branch coverage, all 16
  native/CUDA canaries, Ruff, strict mypy, and the production naming audit.
  Compiled simulator and planner paths must be supplied during coverage so
  their differential tests run rather than appearing as intentionally
  uncovered Python projections.
- The first exact 1.180B planning attempt found that structural profiling kept
  every compiled ABI's real CUDA example arguments alive until the end of the
  phase. Profiling now retains only compiled functions after each measurement,
  so physical demand is the largest isolated ABI rather than the sum of all
  previously visited ABIs. A separate remaining blocker is the monolithic
  all-parameter optimizer ABI; it is not being hidden by this fix.
- CUDA-only lazy-state discovery now inventories every gradient-bearing
  parameter, rather than stopping after the first registered operation rejects
  the CPU sandbox. Discovery retries only parameters with empty state and
  rejects any failed boundary that already mutated parameter values. A
  two-parameter external `mlops.optim.AdamW` regression proves that both update
  operations and both complete state bundles enter the recurrent graph.

## Gate rule

No later phase is declared active until the current phase passes all of its
tests and is committed. Unexpected findings and design changes are recorded in
the ignored internal progress log before this tracked summary is updated.
