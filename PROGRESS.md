# ShadowSpill Progress

## 2026-08-12 — Task-boundary component extraction

- Began the accepted runtime concurrency redesign from clean commit
  `6783eda` after reconfirming all 16 native/CUDA canaries and the complete
  Python suite.
- Added a dedicated neutral `task_boundaries.c` component and private
  interface. Public `shadowspill_before_task` and `shadowspill_after_task`
  now enter through that component while the original locked algorithms remain
  byte-for-byte in legacy implementation functions.
- This is a mechanical ownership extraction only: the global lock, statement
  order, actions, synchronization, and runtime behavior are unchanged. All 16
  canaries and focused runtime/training tests pass.
- Extracted the public create/wait/resize/close/destroy entrypoints through a
  dedicated lifecycle component using the same compatibility technique. The
  legacy bodies and resource teardown order remain unchanged; all focused
  runtime, allocator, overlap, OOM, and training canaries pass.
- Moved failure latching, status observation, and failure snapshots out of the
  status-string module into a dedicated failure-state component with a private
  interface. This is a literal move under the existing global lock; focused
  runtime failure and transition tests pass.

## 2026-08-12 — Stable object lifetime foundation

- Added atomic reference and detached state plus a future per-object mutex to
  every neutral alias-bundle record. The table owns one reference and queued
  actions retain their objects independently, so detaching identity no longer
  implies immediate destruction.
- Allocation records now carry a stable reference counter in preparation for
  completion payloads. The counter does not alter current allocation reuse or
  teardown behavior yet.
- The global runtime lock remains authoritative during this milestone. All
  focused runtime, allocator, overlap, OOM, and training canaries pass.
- Added generation-tagged, atomically reference-counted event leases around
  opaque backend events. Task fences, retirement records, transfer
  completions, and object readiness now retain leases explicitly. The CUDA
  backend's already sealed physical-event pool remains the underlying owner,
  so steady-state driver event creation stays prohibited.
- Fresh correctly linked ASan and UBSan builds pass all five neutral runtime
  canaries. GCC ThreadSanitizer on this host aborts before user code with an
  `unexpected memory mapping` runtime error; this toolchain limitation is
  recorded and will be revisited before fine-grained locking is accepted.

Last updated: 2026-08-11

## Current milestone

Phase 9 — model-scale numerical qualification and allocator diagnosis.

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
- Recurrent optimizer graphs are now partitioned into dependency-closed tensor
  components before Program construction. Independent per-parameter updates
  become ordered tasks with bounded working sets; operations sharing mutable
  state remain together, and opaque optimizers remain one eager task. A real
  two-component external `mlops.optim.AdamW` public run passes three steps and
  bitwise checkpoint replay without introducing an `mlops` core dependency.
- Successful CPU discovery is probed first to preserve data-dependent opaque
  classification, then recurrent graph capture is normalized to FakeTensor
  CUDA. This fixed a stale CPU-kernel/CUDA-storage fault found by the first
  component execution. Device-side scalar optimizer state is planned like any
  other tensor; CPU-restored checkpoints are incrementally rematerialized.
- Optimizers with registered step pre/post hooks remain bounded opaque tasks.
  This preserves their Python-visible once-per-step semantics; tensor-only
  component execution is used only when no such side-effect contract exists.
- The first exact-scale retry after optimizer partitioning exposed a missing
  dependency for long-lived forward boundaries. A skip value produced by an
  earlier stage could pass through an intermediate stage and become an input
  to a later stage, while lowering depended only on the immediately preceding
  stage. Forward tasks now include every actual input object's known producer
  in addition to the sequential stage edge. This is a DAG-correctness fix; it
  does not alter task order, PressureFit choices, or annotated action triggers.
- An exact 1.180B Llama retry is now progressing beyond Program validation, but
  its planning process remains CPU-bound beyond the 90-second cold-planning
  gate. Phase attribution and structural-key counts are required before that
  scale run can qualify; the latency is not being accepted as expected cost.
- Exact-scale geometry explains the first latency component: 24 independent
  recomputation groups yield about 98 bounded selections, and the fixed
  40-policy portfolio evaluates roughly 3,920 simulator-verified candidates.
  Threads do not accelerate its Python schedule construction. Instead, each
  recomputation selection now projects immutable task/object/resource geometry
  to the compiled simulator once; candidates bind only action and residency
  arrays. A controlled serial portfolio improved 3.7x with identical schedule
  digest and makespan. Explicit worker modes remain unchanged.
- A qualification-only stack watchdog identified a separate exact-scale
  optimizer-discovery cost: after each CUDA-only per-parameter failure, capture
  compared every parameter tensor against its snapshot. Discovery now enables
  and audits one still-uninitialized parameter at a time, then performs one
  final complete audit to catch illegal gradless mutation. This reduces
  unchanged verification from quadratic to linear tensor bytes without
  specializing the optimizer or operation.
- The first complete exact 1.180B timing report after these fixes is 429.99 s:
  capture/lowering 13.07 s, optimizer capture 3.51 s, compilation 4.82 s, and
  PressureFit/simulation 403.90 s. Dense projection, pressure sweeps, cached
  initial placement/residency/interval plans, incremental packed-fit admission,
  and cached cut scores preserve all frozen schedules but do not meet the
  latency gate. The remaining dense candidate/repair loop must execute in the
  compiled planner; reducing the portfolio or changing directives is rejected.
- PressureFit's analytic residency reducer now has a framework-neutral dense C
  implementation in `libshadowspill_planner.so`. It preserves the Python
  policy's boundary pressure, output reservation, cut legality, score ordering,
  repair-pressure, and infeasibility semantics; schedule emission and every
  selected transfer trigger remain unchanged. A permanent 192-case differential
  matrix plus frozen schedule artifacts gates production use.
- The first dense decode dropped greedy-placement anchors while retaining the
  same visible spans. That representation difference could have changed later
  interval extension, so it was rejected and fixed before production wiring;
  decoded plans now retain the exact seed anchors. At 901 tasks and 1,504 alias
  groups, the compiled reduction is 5.1x faster than the Python reference with
  an identical plan. Exact-scale end-to-end replanning remains the required
  latency measurement.
- The compiled-residency milestone gate passes all 16 native/CUDA/PyTorch
  canaries and 398 Python tests with five expected skips, plus Ruff, strict
  mypy, and the production naming audit.
- The first exact 1.180B rerun with the compiled reducer still required 415.51
  seconds to plan, versus 429.99 seconds before it. The translation preserved
  policy but retained an alias-by-boundary scan inside every cut search, so the
  latency gate remains failed. The next optimization indexes fixed legal-cut
  geometry and performs only constant-time eligibility checks per alias.
- Legal prefix, gap, and split cut geometry is now preindexed once per dense
  residency problem. Semantic residency, interval, emitted-schedule, and
  simulator-outcome caches eliminate repeated work across policy candidates
  without changing the candidate portfolio or any directive. On a synthetic
  451-task program, unique simulator calls fell from 184 to 13 and planning
  fell from about 4.7 seconds to 1.5 seconds.
- The resulting exact 1.180B Llama plan took 187.65 seconds: 12.99 seconds for
  capture/lowering, 4.80 seconds for compilation, and 161.76 seconds for
  PressureFit/simulation. It retained 24 recomputation selections, 159 tasks,
  1,447 memory actions, the 6.25-GiB predicted device peak, and a 0.385943-second
  simulated makespan. This is a 2.2x planner improvement over the preceding
  exact run, but still fails the 90-second total and 60-second planning gates.
- The first exact steady-state call then exposed an independent runtime issue
  in the large vocabulary-head task: an anonymous roughly 501-MiB `zeros_like`
  allocation produced a tensor with no storage. The allocator already latches
  no-progress diagnostics, but the pluggable callback's null return is surfaced
  by PyTorch only at the later tensor access. The next behavior-only milestone
  will translate that latched failure at the task boundary and distinguish
  insufficient total capacity from contiguous-range fragmentation before any
  admission or allocator policy is changed.
- Cold-path allocator-failure translation now reports the failing task and the
  neutral runtime's requested, free, and largest-range evidence without adding
  work to successful task dispatch. The exact rerun proved spatial
  fragmentation: task 71 requested 525,336,576 bytes with 682,739,712 bytes
  free, but its largest range was 489,988,096 bytes. The simulator's byte plan
  is valid; production training admission failed to replay/prove slab geometry.
- The diagnostic rerun also found that `close()` queries a runtime after its
  first fatal allocation failure and can mask the primary exception. Failure
  cleanup must preserve the first cause while restoring model ownership.
- Save and recomputation capture now receive distinct detached Tensor views of
  the same example storages. This isolates custom-autograd saved-tensor lifetime
  without changing aliases or graph ABI; a newly registered, unrelated custom
  operation with saved-tensor autograd passes both AOT alternatives.
- The resulting one-layer full-vocabulary diagnostic completed capture and
  planning in 34.2 seconds, then exposed a separate invalid initial-action
  transition before task execution. That reduced-plan transition remains a
  separate runtime bug and is not being conflated with exact-scale slab
  fragmentation.
- Device ranges now use a general two-ended allocation policy: planned
  prefetches pack from low addresses and ordinary framework allocations pack
  from high addresses. This introduces no operation/size special case, extra
  copy, or schedule change. The exact task-71 allocation then succeeded, and a
  complete 1.18B two-microbatch training step executed in 0.439 seconds
  including an explicit final stream synchronization.
- Parallelizing complete policy candidates was corrected to keep each
  selection context and its caches single-owner. Automatic parallel selection
  nevertheless regressed exact PressureFit from 161.18 to 176.39 seconds and
  total planning to 202.46 seconds because remaining Python work is GIL-bound;
  public training therefore remains serial until the complete hot loop moves
  into the compiled planner.
- The second exact invocation reached an `after_task` plan violation. Thus the
  spatial allocation fix passes one full mathematical step, but multi-step
  correctness remains failed pending terminal-state/reset diagnosis.
- The reduced program's initial-action rejection was a lowering bug: graph
  values may use several object IDs for views of one storage, but external
  inputs were classified by object ID rather than alias bundle. Both training
  lowering paths now classify produced and external storage by alias identity;
  no produced alias is admitted as an initial host input.
- The next reduced failure exposed a legitimate zero-copy output case. AOT
  backward returned an existing input allocation as a new logical cotangent at
  the same task boundary that released the old logical value. The runtime now
  performs a validated allocation-ownership handoff: the exact release action
  remains in place, but it retires the old object identity without freeing the
  range newly owned by the output. A native transition canary covers this
  general allocator behavior.
- Direct task-boundary failures now report the neutral runtime's first-cause
  object and allocation, rather than only failures previously seen by an
  allocator callback. This added no successful-path CUDA operation or device
  synchronization.
- The one-layer width-2,048/full-vocabulary model completes three accumulated
  training steps and deterministic close at an 8-GiB diagnostic cap. Step one
  took 0.210 seconds and recurrent steps took 0.105 seconds each; objectives
  decreased across all three steps. Its 115 actions were preserved.
- Tighter 4- and 5-GiB diagnostic plans still expose the unfinished physical
  admission gate. The 4-GiB run missed a 525,336,576-byte contiguous workspace
  by four bytes despite 1.39 GiB total free; the 5-GiB plan changed residency
  and left only a 400,515,072-byte largest range. The session currently does
  not feed its annotated allocation timeline through `replay_slab_timeline`.
  This remains a separate admission bug; no PressureFit action was moved.
- Slab admission now replays initial device placement, task-output allocation,
  exact profiled workspace extents, releases, D2H completion, and H2D admission
  in simulator-time order with the production allocator policy. Both training
  optimizer phases and forward-only plans must pass before the callable is
  returned; predicted external fragmentation is recorded in `PlanReport`.
- Ordinary allocations now use a coalescing best-fit/high-address selection,
  while planned prefetches retain low-address placement. This is the documented
  segregated-fit policy: small allocations consume the smallest suitable hole
  instead of needlessly splitting a larger future workspace range. Python and
  C replays have matching permanent regressions.
- The original 25% workspace admission allowance is unchanged. The 4-GiB
  reduced plan is now rejected during planning with the same four-byte spatial
  deficit later observed in CUDA, rather than poisoning the process-global
  runtime. The admitted 8-GiB plan retains 115 actions, completes three
  two-microbatch steps in 0.209/0.105/0.105 seconds, and closes cleanly.
- At the exact 1.180B Llama configuration and a 6.25-GiB physical cap, spatial
  admission rejected generation zero of a 525,336,576-byte alias: aggregate
  free capacity was 720,492,548 bytes but the largest range was 321,692,928
  bytes. This is an admission result, not a simulator schedule rewrite; exact
  numerical qualification will use the next admitted cap. The replay remains
  deliberately conservative about zero-copy task-output handoffs until those
  structural relationships are represented in profiling evidence.
- The admission milestone passes the complete Python suite, all 16 CTest
  canaries (including CUDA forward/training/overlap), Ruff, strict mypy, the
  production naming audit, and `git diff --check`.
- Exact 1.180B Llama admission at 8.0 and 8.5 GiB selected different logical
  residency schedules but both left the same 624,623,616-byte largest range
  for a 634,400,768-byte task workspace. A 10-GiB run with shorter token
  geometry also fragmented before the same workspace. Extra physical capacity
  is not monotonic because PressureFit legitimately retains more objects; this
  cannot be debugged by repeatedly increasing the cap.
- Complete PressureFit selections now use a content-addressed atomic cache
  independent of the structural task-profile cache. Hits revalidate the exact
  immutable schedule, rerun the standalone simulator, restore full candidate
  diagnostics, and report hit/miss counts. This makes repeated physical
  admission experiments inexpensive without changing recomputation choices or
  transfer triggers.
- A fresh-process forward cache canary reduced the PressureFit phase from
  8.86 ms cold to 0.84 ms warm with identical scheduling inputs. The complete
  Python suite, all 16 CTest canaries, Ruff, strict mypy, naming, and whitespace
  gates pass.
- The recurring exact-scale 634,400,768-byte fragmentation failure was a
  physical-replay bug, not a real allocator request. Training lowering charges
  the sum of per-parameter gradient contributions as task workspace for the
  simulator; admission had reconstructed that scalar sum as one contiguous
  extent. It now preserves the measured task extents plus each mutated
  gradient object's actual extent. No simulator byte, PressureFit selection,
  or action changed.
- With corrected geometry, exact 1.180B Llama at a strict 10-GiB cap admitted
  159 tasks and all 965 selected actions, including 7.08 GB D2H, 1.17 GB H2D,
  and eleven selected recomputation stages. Five two-microbatch steps completed
  in 0.482/0.337/0.337/0.350/0.350 seconds. Restoring the step-three checkpoint
  reproduced steps four and five bitwise for losses and the complete
  model/optimizer digest, then closed cleanly.
- Exact checkpoint replay exposed that non-persistent PyTorch buffers were
  incorrectly required by `load_state_dict()`. Forward and training now retain
  those runtime buffers while loading only ordinary state-dict names; lowering
  marks them `RUN`, not `CHECKPOINT`. CUDA lifecycle regressions cover this.
- The matching eager external-mlops run used identical CPU-generated inputs.
  Step-one objectives were bitwise equal; later objectives stayed within
  sub-percent BF16 relative drift except one 9.5e-4 step-five loss whose
  1.3e-5 absolute difference is 1.36% relative. Full per-tensor parameter and
  optimizer-state metrics remain the next numerical gate.

## Gate rule

No later phase is declared active until the current phase passes all of its
tests and is committed. Unexpected findings and design changes are recorded in
the ignored internal progress log before this tracked summary is updated.

## 2026-08-11 — Stream-safe anonymous reuse and exact Llama qualification

- Task profiling now preserves the ordered allocator callback trace and maps
  returned tensor storage to output leaves. Spatial admission replays those
  concrete lifetimes rather than a sorted peak-extent multiset. Logical free
  has its own telemetry transition; asynchronous physical release remains
  separate and is not mistaken for the tensor lifetime.
- The progress worker had leased every queued H2D destination as soon as Python
  submitted the actions. The simulator leases at transfer start, after the
  annotated trigger and earlier H2D work. The worker now preserves prefetch
  queue order, waits for the trigger event before admission, and has at most one
  H2D allocation/copy in flight. No directive or trigger moved.
- A random AdamW scalar-step mismatch exposed a cross-stream use-after-free:
  ordinary same-stream logical free returned the range to the global slab, so
  the H2D worker could overwrite it before compute reached the free. Every
  ordinary free now records retirement events. Only an allocation on the sole
  recorded stream may reuse a pending block; background and other-stream reuse
  wait for physical retirement.
- Whole-block cached reuse was safe but caused severe internal fragmentation:
  task 72 reused a 525,336,576-byte block for an 8,192-byte request and carried
  that charge through a chain of small temporaries. A later 525,336,576-byte
  gradient then failed despite adequate aggregate capacity. Cached reuse now
  splits the requested aligned prefix; the unused suffix retains the original
  retirement events and remains unavailable to transfer streams until safe.
- Exact 1.180B Llama passed twice at a strict 10-GiB cap with two heterogeneous
  microbatches, five steps, real D2H/H2D, eleven recomputation selections, and
  two optimizer parameter groups. All 111 AdamW counters were exactly 3 at the
  checkpoint and 5 after uninterrupted and replayed execution. Replay was
  bitwise; minimum cosine was 0.999749, maximum relative L2 was 0.01263, and
  minimum sign agreement was 0.99463.
- Cold planning was 117.45 seconds: 68.31 seconds for 30 cold ABI profiles and
  27.44 seconds for uncached recomputation selection. Warm planning was 26.65
  seconds with a 3.9-ms profile-cache phase and 26.7-ms selection-cache phase.
  Warm steps were 0.392–0.445 seconds against a 0.375-second simulator. The
  numerical gate passes; cold profiling and the remaining runtime/simulator
  gap remain explicit performance work.
- All 16 native/CUDA CTest canaries, the complete Python suite, Ruff, and strict
  mypy pass after the correction. Runtime ABI is 7, adapter ABI is 11, and the
  structural-profile schema is v4 so unsafe cached traces cannot be consumed.

## 2026-08-11 — Bounded provider retention and exact Qwen qualification

- Qwen's first linear-attention forward ABI retained one 24-byte tensor and
  one 32-byte tensor on every isolated call. Exact allocation-ID, output-leaf,
  dispatch, and Python-reference tracing proved these were not graph outputs,
  Triton scratch, or hidden autograd state. FLA's bounded identity-based tensor
  cache retains derived sequence lengths and indices while newly materialized
  packed metadata rotates through its four-entry cache.
- Profiling no longer assumes that a second call must retain nothing. It audits
  the allocator's logical live-byte baseline for up to sixteen isolated calls,
  requires two stable observations, reserves the observed high-water as fixed
  slab use, and still rejects retention whose live baseline keeps growing. The
  rule is operation- and provider-independent and changes no task or transfer
  directive. Profile schema v7 separates it from prior semantics.
- Exact 1.007B Qwen 3.5 passed five two-microbatch steps at a strict 10-GiB cap
  with real D2H/H2D, selected recomputation, and bitwise step-three checkpoint
  replay. Worst objective relative error was 0.002316; minimum per-tensor cosine
  was 0.999649, maximum relative L2 was 0.016754, and minimum sign agreement was
  0.994792.
- The numerical gate passes, but performance does not: cold planning took
  148.60 seconds, including 115.31 seconds profiling 43 unique ABIs. Runtime
  steps were 0.916--1.168 seconds against a 0.404-second simulator prediction.
  These latency and fidelity gaps remain explicit Phase 10 work.

## 2026-08-11 — Identity cotangents and OLMoE authority audit

- OLMoE spatial replay found a backward leaf that was literally one of the
  backward graph's tangent inputs. Lowering had assigned that pass-through
  value a fresh output alias, so admission waited for an allocation that the
  graph correctly never made. Tensor inventory now merges exact input/output
  views, the task does not claim a fresh output, and accumulation recognizes
  the already-bound view without adding it to itself. A repeated auxiliary-loss
  block regression covers the identity-cotangent form.
- After that correction, the exact 0.976B optimized OLMoE run admitted and
  completed five accumulated steps plus bitwise checkpoint replay. It did not
  pass the numerical qualification: minimum cosine was 0.993619 and maximum
  relative L2 was 0.11308 after five steps. An independent eager repeat was
  bitwise identical, so the difference is not eager kernel nondeterminism.
- A reduced real OLMoE produced identical results with automatic stage
  partitioning and a single whole-graph stage. A frozen-optimizer gradient
  comparison localized the difference below scheduling: the explicit compiled
  AOT backward differed slightly from eager after one call, but remained inside
  the stated one-step thresholds (minimum cosine 0.999964, maximum relative L2
  0.00866, minimum sign agreement 0.99219). Discrete expert routing amplified
  that small compiled/eager difference across later updates.
- This audit also found that the first qualification harness used the external
  `models.mlops` implementations as its numerical authority. The project plan
  instead designates `models/pytorch` as the authority. The optimized Llama,
  Qwen, and OLMoE measurements remain valuable supplemental evidence; the
  formal family gate will use pure PyTorch models with `mlops.optim.AdamW`.
- The complete Python suite and all 16 native/CUDA CTest canaries pass after
  both behavior corrections. Ruff, production mypy, the naming audit, and
  whitespace checks also pass.

## 2026-08-11 — Best-fit planned placement and pure-PyTorch Llama

- Pure-PyTorch Llama exposed a 1.051-GB task workspace that failed spatial
  admission at 10 and 10.25 GiB despite about 1.95 GB aggregate slab space.
  Planned prefetches used first-fit-low, allowing smaller objects to split the
  earliest large range while tighter compatible holes remained elsewhere.
- Planned allocations now choose the smallest compatible range and place at
  its low end; anonymous allocations retain smallest-compatible/high-end
  placement. Admission and the neutral C runtime implement the same rule. This
  preserves large workspace ranges without changing any PressureFit directive,
  recomputation choice, or transfer trigger.
- The original strict 10-GiB pure-PyTorch Llama plan then admitted and completed
  all five accumulated steps, 7.08 GB D2H, 1.90 GB H2D, selected recomputation,
  and bitwise checkpoint replay. Warm-profile planning still took 58.09 seconds
  (32.22 seconds capture/lowering and 14.71 seconds compilation), demonstrating
  that profile-cache hits alone do not meet the warm-plan latency target.
- The only numerical threshold miss was ten AdamW second-moment tensors at a
  maximum 0.023905 relative L2; global minimum cosine was 0.999721 and minimum
  sign agreement 0.993844. Under the requested modest additional leeway, the
  formal relative-L2 limit becomes 0.025 globally. No model-, tensor-, or
  optimizer-state exception is introduced.

## 2026-08-11 — Structural compilation and profiling lifetime audit

- The pure-PyTorch 1.007B Qwen eager authority completed, but its first planned
  construction remained active for more than twenty minutes. A second
  instrumented attempt reached structural profiling and was interrupted inside
  `gc.collect()`, not inside an accelerator kernel or Inductor.
- Planning retained the complete AOT artifact forest while invoking global
  collection after individual measurements and executable transfer. Each
  collection repeatedly traversed very large, still-live FX graphs and reclaimed
  no corresponding graph storage, making the phase effectively quadratic.
- ShadowSpill now relies on explicit reference release throughout profiling and
  performs no global garbage collection in the compiler/profiler. The existing
  weak-reference regression proves representative CUDA arguments are released
  between structural ABIs.
- A 2,000-node eager-FX fallback was evaluated, not accepted. It made progress
  after the GC correction but measured one 9,013-node derivative at 8.59 seconds
  per invocation, which would destroy runtime throughput. The fallback has been
  removed; task graphs remain compiled and provider plus node-count provenance
  is retained for diagnosis and cache identity.
- This is a planning-latency correction, not a schedule change: PressureFit
  actions, transfer triggers, task ordering, and runtime synchronization remain
  untouched. Qwen then exposed a separate upstream lowering defect, recorded in
  the internal log and addressed as an independent change.

## 2026-08-11 — Remove discarded whole-objective AOT graphs

- Public training capture exported each complete objective and immediately
  constructed save-all and recompute whole-model AOT graph pairs. Automatic
  partitioning then ignored both pairs and differentiated every stage again.
  Qwen's fixed-length reference recurrence made those discarded graphs enormous.
- Objective export and whole-graph differentiation are now separate internal
  operations. The public planning session uses export-only capture, partitions
  the functional graph, and constructs only graph pairs that become executable
  stage alternatives. Whole-graph pairs remain available to focused oracle
  tests and the non-partitioned internal lowering path.
- A regression replaces the whole-graph pair builder with a failing sentinel and
  proves export-only capture never invokes it. Focused AOT, partition, training,
  and training-lowering suites pass without changing runtime task semantics.

## 2026-08-11 — Complete tensor-argument structural identity

- Auditing safe graph-pair reuse found that the structural artifact identity
  described tensor shape and stride but omitted storage offset and cross-input
  alias relationships. Two call ABIs with the same geometry but different view
  semantics could therefore share a profile or executable incorrectly.
- Tensor geometry now includes storage offset, and each artifact records a dense
  alias-group vector for its tensor arguments. Both enter the compatibility
  digest. A focused regression distinguishes overlapping views from equal-shaped
  independent tensors before any graph-pair reuse is enabled.

## 2026-08-11 — Structural AOT graph-pair reuse

- Automatic training partitioning now captures each unique pre-AOT stage/root
  ABI once. Repeated occurrences share graph code but receive newly constructed
  forward/residual/tangent examples bound to their own FakeTensor storages. Both
  rebound artifact digests must equal the representative before reuse.
- The exact pure Qwen frontend produced 16 stage occurrences, 8 unique stage
  ABIs, and 8 cache hits across its two heterogeneous shapes. PlanReport exposes
  these counts. Fresh-process public CUDA training and focused partition/lowering
  tests pass, including occurrence-specific parameter storage.
- Frontend-only timing improved enough to expose the next limit but remains over
  budget: 56.81 seconds for two objective exports, 68.82 seconds for stage AOT,
  and 125.75 seconds total. Pure Qwen's Python token recurrence expands one
  linear stage to 1,914--2,714 Export nodes and up to 9,013 backward nodes.
  Deduplication cannot make that unique ABI cheap; a compact bounded operation
  contract is required before repeating full qualification.

## 2026-08-11 — Bounded pure-PyTorch Qwen recurrence

- The pure Qwen reference now registers its ATen delta-rule recurrence as one
  bounded PyTorch operation with fake and first-order autograd contracts. Its
  backend remains the readable PyTorch recurrence; its explicit reverse
  recurrence is differentially tested against eager Autograd for every input.
- No ShadowSpill core code recognizes Qwen or linear attention. This exercises
  the same arbitrary registered-operation contract available to user models and
  keeps the pure model independent of `mlops`.
- Exact two-shape frontend time fell from 125.75 to 16.68 seconds: Export fell
  from 56.81 to 8.57 seconds and stage AOT from 68.82 to 8.00 seconds. The
  largest backward graph fell from 9,013 to 722 nodes while structural reuse
  remained 8 unique ABIs and 8 hits.
- The first full compiled profile rejected the backward fake kernel because it
  promised input-preserving strides while the explicit VJP returned contiguous
  gradients. Real and fake outputs now declare contiguous layouts explicitly,
  and `torch.library.opcheck` guards the complete operation contract.

## 2026-08-11 — Causal destination-reservation incident

- Exact optimized-Llama qualification exposed a timing-dependent admission
  failure at task 71. The simulator placed a 525,336,576-byte optimizer-state
  H2D at 53.137849 ms, after task 71's predicted 31.931168--35.025216 ms
  interval. The real FIFO H2D lane reached and leased that destination before
  task 71, leaving only 76,484,104 bytes free for task 71's second required
  525,336,576-byte allocation. No pending event could make progress, so the
  allocator correctly raised a diagnostic no-progress OOM.
- The root cause is a semantic mismatch: the directive trigger queued work, but
  simulator and spatial admission charged destination capacity only at the
  predicted transfer start while runtime charged it at actual dispatch. Correct
  admission therefore depended on exact relative compute/transfer timing.
- H2D speculative runahead, Python lifetime leakage, missing logical free,
  size-class incompatibility, and out-of-order dispatch were each disproved.
  The complete two-sided ledger, timeline diagrams, and corrected causal
  contract are documented in `docs/deadlock_example.md`.
- Decision: destination capacity is reserved at the annotated trigger boundary;
  binding/copy still occurs only at actual transfer-lane head. Simulator, exact
  spatial replay, and runtime must share this lifetime. Timing may affect
  overlap and makespan but never physical feasibility.

## 2026-08-11 — Causal reservation implemented and replayed

- The C and Python simulators now charge H2D device and D2H host destinations at
  trigger completion. Exact slab admission places the same claims at trigger
  task end, and the neutral C worker reserves ranges before dispatch while
  retaining one physical transfer per direction. Dispatch consumes an existing
  reservation; it creates no additional capacity demand.
- Added a four-object isolated regression reproducing the false old admission:
  120 bytes appears feasible when a queued destination is charged at transfer
  start but requires 150 bytes under the true causal lifetime. Both simulator
  implementations reject 120 with `used=100`, `requested=50`, and
  `capacity=120`; at 150, compute still overlaps the first H2D and the second
  H2D remains FIFO.
- The full incident report now identifies the initial failing call exactly:
  head/objective recomputation task `task_000071`, profile allocation ordinal
  28, a 525,336,576-byte anonymous BF16 result of
  `grad_logits.T @ hidden_chunk`. The preceding same-sized persistent gradient
  allocation succeeded; the failing `malloc` had no object/allocation identity
  because no range was returned.
- Replaying the old 10-GiB optimized-Llama plan under corrected semantics
  reconstructs 10,093,654,520 bytes of demand, exactly matching the runtime
  ledger. The old schedule is rejected; a fresh selected plan has 1,039 actions,
  7,078,196,088 D2H bytes, 1,197,678,816 H2D bytes, and completes a real
  two-microbatch training step in 0.410 seconds.
- The correctness rerun isolated planning latency: warm structural profiling
  used 0.003 seconds and all capture/lowering/compilation phases about 12
  seconds, while Python PressureFit used 297.478 of 315.327 total seconds. The
  next performance milestone is therefore the fresh complete C PressureFit
  implementation, not weaker graph partitioning or fewer transfer choices.
- All 16 compiled gates, Ruff, MyPy, and 423 ordinary Python tests pass. Five
  public CUDA tests conflict only when included after CUDA-initializing tests in
  one process; all five pass through their required fresh-process CTest gates.

## 2026-08-11 — Direct PressureFit goldens and benchmark

- Added deterministic fixtures at the exact public `pressurefit()` boundary.
  Their request is Program + initial/final residency + simulator configuration
  + normalized options; their expected result is selections + schedule + full
  simulation + all candidate diagnostics. PyTorch capture/profiling and outer
  `plan()` artifacts are explicitly not part of these goldens.
- Added a cache-free direct replay benchmark. It measures only
  `pressurefit()`, stores raw/median nanosecond samples keyed by request-suite
  digest, and requires the complete expected-result digest to match. This is
  the baseline for the fresh C implementation and prevents AOT, profiling, or
  cache behavior from contaminating the reported PressureFit speedup.
- The current six-case matrix passes both Llama implementations and mlops Qwen.
  Pure Qwen has a frontend workspace/capacity mismatch; pure OLMoE needs a
  bounded contract for data-dependent `aten.bincount`; mlops OLMoE has a
  numerical-state failure despite bitwise checkpoint replay. These are tracked
  separately from PressureFit wall time.

## 2026-08-11 — Compiled-reference numerical authority

- The apparent mlops OLMoE numerical discrepancy was isolated to Inductor, not
  ShadowSpill. The uncompiled and compiled forwards are bitwise identical. In
  backward, the shared BF16 attention input receives three BF16 projection
  gradients. Eager executes two separate BF16 additions, whereas Inductor
  fuses them into one Triton kernel, loads all three operands into FP32
  registers, and rounds once at the final BF16 store. `aot_eager` is bitwise
  eager, and Inductor's `emulate_precision_casts=True` also restores eager
  rounding, confirming the cause.
- The performance-preserving decision is to retain normal Inductor fusion. The
  numerical reference is now the complete objective compiled with
  `torch.compile(fullgraph=True, dynamic=False)` under PyTorch's standard
  allocator. Reference identity includes this execution convention, and
  qualification artifacts use versioned compiled-reference schemas rather than
  describing that process as eager.
- A direct five-step comparison of whole-model compiled execution under the
  standard allocator and ShadowSpill produced the same complete model and
  optimizer-state digest:
  `91782bf1ca1808fef1eb10df31c1a616c247935bebc9706b29b7b10f54086b00`.
  All ten objectives were bitwise identical.
- Fresh 8-GiB mlops OLMoE qualification passed every gate. It transferred
  5,854,600,600 bytes D2H and 1,112,609,528 bytes H2D, selected recomputation,
  peaked at 8,170,504,192 physical bytes under the 8,589,934,592-byte cap, and
  reproduced the step-three checkpoint replay bitwise.
- The run attributed 7.847 seconds to 30 cold structural profiles and 450.506
  seconds to Python PressureFit out of 477.342 seconds of planning. PressureFit,
  not AOT or profiling, is the dominant planning-latency target for this case.

## 2026-08-11 — Task-lifetime workspace and compiled-view reconciliation

- Pure Qwen's earlier task-64 rejection was caused by profiler lifetime
  accounting, not its AdamW geometry. Disposable AOT return leaves remained
  referenced until after `after_task`, so the profiler classified several
  gigabytes of dead task output as anonymous workspace. The profiler and
  executor now delete undeclared raw/output leaves before closing the task
  boundary, synchronize the profiling stream, drain allocator retirements, and
  only then sample the persistent baseline.
- With corrected accounting, the exact 10-GiB pure-Qwen Program has
  9,260,655,616 bytes of task capacity, a true maximum anonymous workspace of
  2,354,838,832 bytes, and therefore 6,905,816,784 bytes of object capacity.
  Its largest mandatory boundary is the token-embedding/output-head AdamW task:
  a 762,839,040-byte parameter, gradient, and two moments plus an 8-byte step,
  totaling 3,051,356,168 bytes. It now passes the required-capacity floor.
- The next admission check exposed a distinct FakeTensor/compiler storage
  mismatch. One task returned a 39,936-byte view at byte offset 39,936 into an
  otherwise inaccessible 79,872-byte temporary. Fake/AOT metadata retained the
  unused prefix, while Inductor correctly returned a compact 39,936-byte,
  offset-zero allocation. The IR had consequently expected twice the physical
  allocation.
- Task-output lowering now compacts a newly produced backing storage only when
  exactly one returned view exposes it and it aliases no existing input.
  Multiple returned views retain the full storage bundle. The same Qwen task
  contains a 196,608-byte storage exposed by three views at byte offsets 0,
  39,936, and 98,304; that bundle remains shared and unchanged. Focused
  lowering, training-lowering, and spatial-admission tests pass.

## 2026-08-11 — AOT semantic aliases separated from compiled physical outputs

- A second pure-Qwen admission failure disproved the preceding
  "multiple FakeTensor views imply one physical bundle" rule. Forward task 35
  exposed three Program objects in a 104,448-byte fake storage, but its compiled
  allocation trace contained four independent 26,112-byte output allocations
  for leaves 10, 13, 12, and 15. The first conflicting binding was leaf 10:
  runtime allocation ordinal 7 had 26,112 bytes while `alias_001770` incorrectly
  expected the entire 104,448-byte fake backing.
- AOTAutograd exposes user-visible alias semantics through
  `ViewAndMutationMeta.output_info`, including aliases of inputs and
  intermediates. It does not expose a stable final physical layout for
  compiler-private saved tensors. PyTorch explicitly permits Inductor to change
  output layouts, and differentiable multi-output views may intentionally be
  classified as non-aliasing. FakeTensor storage identity was therefore the
  wrong physical contract.
- Decision: semantic aliases that reuse existing graph inputs remain governed
  by AOT/FakeTensor identity. A newly allocated compiled output is grouped by
  the allocation-to-output-leaf map observed once when validating the compiled
  structural ABI. This observation is independent of timed samples: lowering
  does not use task duration or workspace estimates to invent graph semantics.
- The inventory now has explicit per-invocation compiled-output scopes. Leaves
  from one actual allocation share a residency bundle; leaves from distinct
  allocations are distinct even if fake execution reused a storage. This is a
  graph- and operator-independent rule. Focused forward lowering, partitioned
  training lowering, identity-cotangent aliasing, and spatial-admission tests
  pass; the full pure-Qwen gate is next.

## 2026-08-11 — Interrupted pure-Qwen run: phase correction and allocator hot path

- The long pure-PyTorch Qwen qualification was stopped rather than allowed to
  continue silently. No qualification process remains. Its traceback was in
  the CUDA custom-op implementation
  `qwen35_delta_rule_backward -> _delta_rule_backward_reference`, during an
  actual ShadowSpill training step; it was not still searching PressureFit.
- Cache timestamps bound the cold recurrent PressureFit/finalization interval
  to about 31.1 seconds: the last structural profile was written at
  17:31:18.620 and the only recomputation result at 17:31:49.693. There was no
  second optimizer-phase result. The prior inference that 100% host CPU meant
  a second PressureFit search was wrong.
- The compiled standard-allocator reference took 51.775 seconds for its first
  compile-and-run step, then 0.300, 0.296, 0.297, and 0.297 seconds. Therefore
  the pure recurrence and its CUDA arithmetic are not intrinsically slow at
  this geometry.
- A warm-cache diagnostic rebuilt the callable in 125.197 seconds
  (`capture_lowering=17.328`, cached profiling=0.349, cached
  PressureFit=0.030) and was stopped at the first training step. The residual
  planning time is compiled-entrypoint reconstruction/loading and now has live
  phase output; it is separate from PressureFit.
- The decisive runtime evidence is in the cold task profiles. The 43 unique
  task medians sum to 39.199 seconds under ShadowSpill. The slowest two tasks
  take 4.936 and 4.913 seconds and produce 5,029 and 5,035 allocator events.
  Several other recurrence tasks produce 1,500--3,400 events each and take
  1.3--4.1 seconds. GPU utilization was near idle while one host thread was
  busy dispatching this work.
- Root cause: every anonymous temporary free currently creates and records a
  CUDA retirement event, the progress thread repeatedly scans all allocation
  records/events, and pointer lookup is a linear scan. The token recurrence
  creates thousands of short-lived tensors, turning correct cross-stream
  retirement machinery into the dominant task cost. PyTorch's caching
  allocator reuses same-stream temporaries without one event per free.
- Decision: retain stream-safe semantics but add a task-scoped anonymous fast
  path. Same-stream frees are immediately reusable in stream order; allocations
  still awaiting task exit share the task-completion fence. Cross-stream
  `record_stream` remains explicitly fenced. This is allocator-generic and
  does not change model arithmetic, PressureFit directives, or task boundaries.
- Qualification now prints every completed reference/planned step plus the
  planning phase summary, so a long execution cannot again be mistaken for a
  planner search.
- Added the standalone incident report `docs/allocator_storm.md`. It presents
  the standard-allocator and original-ShadowSpill timelines side by side,
  includes the 43-profile/50,028-event ledger and the six slowest structural
  ABIs, and leaves an explicit validation section for post-fix measurements.

## 2026-08-11 — Allocator storm corrected and pure-Qwen gate passes

- Implemented task-local same-stream reuse in the neutral C runtime. An
  anonymous allocation freed on its active task stream can be leased again on
  that stream without an event or wait. Unreused task-local ranges share one
  task-completion fence; explicit `record_stream` dependencies retain the
  conservative cross-stream retirement path.
- Added direct hash indexes for allocation ID, pointer, and exact-size
  reusable extents. Normal allocation/free no longer scans the growing
  historical telemetry list. The progress thread is not awakened by
  eventless task-local frees, and shared task fences are queried at most once
  per progress epoch.
- Added a reusable CUDA event pool. Planning now sizes it generically from the
  selected task count, planned object count, and 64 service events, then seals
  it with the physical budget. No steady-state `cuEventCreate` fallback is
  permitted. The PyTorch adapter ABI is now v12 and exposes event-pool
  capacity, peak, driver-create, rejection, and sealed telemetry.
- A native canary performs 2,500 same-stream allocation/free pairs inside one
  task. It observes pointer reuse, no backend event operation inside the task,
  fewer than 32 backend operations after task exit, and zero pending
  retirements. Full native/CUDA/PyTorch ctest passes 16/16.
- Found and fixed a separate measurement-contract bug: workspace telemetry ran
  inside `before_task`/`after_task`, but timed and warmed executions did not.
  Profiling therefore forced the allocator's conservative out-of-task path,
  unlike production execution. Every warmup, CUDA-event sample, and cache-hit
  executable warmup now uses a task boundary. Profile schema v8 invalidates
  the inflated legacy measurements.
- On the same 43 pure-Qwen structural ABIs, the allocation trace remains
  essentially unchanged (50,028 original events versus 50,020 corrected), but
  summed medians fall from 39.199 seconds to 0.195 seconds (201x). The
  5,029-event 96-token backward falls from 4.936 seconds to 18.19 ms (271x).
  PressureFit predicted makespan falls from 42.567 seconds to 0.501 seconds.
- The complete 10-GiB pure-Qwen qualification now passes: five training steps,
  two heterogeneous microbatches, real D2H/H2D, recomputation, and bitwise
  checkpoint replay. Worst loss relative error is 0.003739; minimum cosine is
  0.999650; maximum relative L2 is 0.021320; minimum sign agreement is
  0.994792. Peak process physical use is 9.613 GiB, with zero steady-state
  device or pinned-host allocations.
- Median planned step time is 1.317 seconds versus 0.297 seconds for the
  compiled standard-allocator reference. This remaining 4.44x gap is not
  accepted as a throughput result. The simulator predicts 0.501 seconds, so
  task dispatch and runtime/transfer overhead require NSYS decomposition.
- Cold planning remains too slow at 134.788 seconds: 17.290 seconds in
  capture/lowering, 41.895 seconds in compilation/profiling, and 63.848 seconds
  in Python PressureFit. These are now cleanly separated from the fixed GPU
  task samples and remain explicit optimization gates.
- Updated `docs/allocator_storm.md` with the implemented fixed timeline,
  before/after ABI table, corrected ledger, numerical evidence, budget
  evidence, and unresolved throughput work.

## 2026-08-11 — Warm-cache queue preservation and bounded active scans

- The first warm-cache Qwen execution exposed a timing-dependent runtime bug
  hidden by cold profiling latency. Before backward task 17, the required
  4-byte objective seed (`alias_000507`) was host-only with no allocation even
  though the schedule declared it initially device-resident. The slab had
  5,020,345,940 free bytes and a 4,988,206,336-byte largest range, ruling out
  capacity pressure.
- Root cause: `after_task` correctly created a shared fence for task-local
  retirements when a task had zero memory actions, but then assigned the
  transfer queue tail to the empty action list. The next action append replaced
  the queue head and discarded remaining initial prefetches. Cold execution
  happened to drain the initial queue first; warm execution exposed the race.
- Fix: action head/tail state changes only for a non-empty action list. Added a
  deterministic mock-backend regression that holds two H2D requests, closes a
  retirement-only task, appends a third request, and requires all three
  objects to become device-ready.
- A subsequent run revealed a second hot-list issue: steps increased from
  0.792 to 1.110 seconds as task closure and progress scanned every historical
  allocation record. Added an intrusive active-allocation list; allocation ID,
  pointer, and telemetry history remain intact, but hot retirement, handoff,
  fallback, abort, and progress scans cover only live/pending records.
- Preserved the complete active-list successor before releasing an allocation.
  Without that detail, progress retired only one completed range before trying
  a large prefetch and produced a false 762,839,040-byte admission failure with
  more completed retirements still available. Added a regression requiring all
  completed retirements to run before action admission.
- The corrected warm Qwen steps are 0.533/0.615/0.619/0.523/0.616 seconds and
  replay steps are 0.533/0.618 seconds, with no monotonic history growth.
  Median is 0.615 seconds versus 0.297 seconds for standard allocation and
  0.501 seconds simulated. All numerical, checkpoint, transfer, recomputation,
  and physical-cap gates still pass.
- Event demand peaked at 76 leases. Replaced object-count sizing (which
  unnecessarily created 2,955 events) with selected-task upper bound + twice
  observed profiling peak + 64 service events, with a minimum of 256. The pool
  remains sealed before execution and growth remains forbidden.
- Final qualification after event-reserve retuning passes with a 303-event
  pool, 76 peak leases, zero steady-state driver event creations, and zero
  growth rejections. Steps are 0.521/0.612/0.611/0.517/0.615 seconds (0.611
  median), retaining the same numerical and physical-budget evidence.

## 2026-08-11 — 30-GiB Qwen control exposes non-cyclic reset accounting

- Repeated the corrected pure-PyTorch Qwen qualification with a 30-GiB
  physical cap. Five measured steps were 0.560/0.606/0.635/0.563/0.641
  seconds (0.606-second median), versus the compiled standard-allocator
  steady-state median of 0.297 seconds. Increasing the cap therefore did not
  remove the remaining wall-time gap.
- This was not a transfer-free allocator control. The recurrent Program still
  declares all 395 initial aliases host-resident and all but two terminal
  aliases host-resident. Its selected schedule contains 388 offloads totaling
  6,041,733,224 bytes D2H, 1,028 releases, no in-schedule H2D, and selected
  recomputation. The executor waits for the prior invocation to become idle
  and submits the next invocation's initial prefetches outside the annotated
  MemorySchedule.
- Consequently, the reported 0.501-second simulated makespan and zero H2D
  bytes do not describe all work on the measured recurrent call boundary. The
  0.606-second result cannot be attributed to allocator callbacks alone: it
  includes the intentionally non-cyclic terminal writeback/startup-prefetch
  protocol, 129 staged task dispatches, storage rebinding, and runtime task
  boundaries. This simulator/reporting contract gap must be corrected before
  using simulator error to optimize the remaining runtime.
- The 30-GiB process peak was 31,797,018,624 bytes under the
  32,212,254,720-byte cap, with a 31,153,192,960-byte slab and a
  10,546,162,360-byte slab allocation peak. Numerical tolerances and bitwise
  checkpoint replay passed. The qualification artifact's overall `passed`
  field is false only because the stressed gate intentionally requires both
  D2H and H2D schedule traffic; this diagnostic schedule reported no
  in-schedule H2D.

## 2026-08-11 — Compute-only timing separates dispatch from writeback

- Added qualification-only CUDA events that begin immediately before the
  first compiled task launch, after its readiness waits, and end immediately
  after the final optimizer launch. They exclude initial placement and
  terminal writeback while retaining kernels, inter-task GPU-idle gaps,
  mid-program waits, and staged frontend/runtime dispatch overhead. Ordinary
  planned calls create or record no timing event.
- The fresh-process accumulated-training canary exercises the bracket and the
  compiled standard-PyTorch reference records the equivalent event interval.
  Focused qualification/verification tests and the CUDA training canary pass.
- For the 30-GiB pure-Qwen control, the standard-allocator compute-event steady
  median is 292.141 ms. The simulator's final compute task ends at 275.975 ms,
  followed by a 224.546-ms simulated D2H tail to the 500.521-ms makespan. Real
  ShadowSpill compute-only events are
  488.773/490.257/493.626/493.874/495.860 ms (493.626-ms median).
- Thus the non-cyclic handoff explains why public wall time exceeds the
  simulator's complete-schedule clock, but it does not explain the remaining
  compute-path gap: staged ShadowSpill is 201.485 ms slower than the standard
  compute event and 217.651 ms slower than the simulated compute lane. NSYS
  must attribute Python task dispatch, task-boundary calls, storage rebinding,
  allocator/lock work, action submission, and D2H/compute contention before a
  behavior-preserving optimization is selected.
- Decision: report startup wait, startup transfer, compute interval, cooldown,
  reset-inclusive steady cycle, and public-call wall independently. Measure
  startup/cooldown but leave cyclic optimization outside the current agenda.
  Before the 100-step real-data approximately-1B correctness and full-model
  throughput gates, freeze the five supported cold model/provider cells, then
  make the complete C PressureFit exactly match and outperform those direct
  PressureFit goldens.

## 2026-08-11 — Phase-1 task timing and trace extraction

- Moved execution attribution ahead of the five-cell golden matrix as a hard
  gate. The installed internal agenda now prohibits matrix collection until at
  least 95% of the 201.485-ms Qwen compute discrepancy is explained.
- Extended qualification timing from one whole-program event pair to one
  precreated CUDA-event pair per selected task. Results identify forward,
  backward, and optimizer tasks, exact GPU intervals, host time in
  `before_task`, rebinding/input lookup, compiled dispatch, output processing,
  and `after_task`, plus the real first-to-last optimizer span.
- Added an optional native debug clock using `cuLaunchHostFunc`. `before_task`
  enqueues a readiness callback after all input waits; `after_task` enqueues a
  completion callback after the task's compute. Callbacks write only a
  `CLOCK_MONOTONIC` timestamp and sequence into preallocated records: they make
  no CUDA calls, allocate no memory, take no lock, and never enter Python.
- Decision: callback timing is a reusable task/simulator ordering cross-check,
  not the sole duration authority. Host-callback scheduling can delay a stream,
  so CUDA events and NSYS remain authoritative and measured runs must quantify
  callback perturbation. The facility is disabled outside explicitly armed
  qualification calls.
- Added `qualification.extract_execution_trace`, a deterministic NSYS SQLite
  extractor for task kernels, compute idle gaps, optimizer span, host launch
  gaps, task-boundary ranges, transfer overlap/dispatch, and event/synchronize
  API counts.
- Adapter ABI v13 advertises the optional debug facility. All 16 CMake tests,
  the fresh-process CUDA training canary, focused ABI/extractor tests, and Ruff
  pass.

## 2026-08-11 — Structured plan and real-step diagnostics

- Split diagnostic ownership cleanly. `PlanReport.diagnostics` is now
  mandatory and fully resolved when `plan()` or `forward_pass()` returns.
  `StepResult.diagnostics` exists only for a real call made with `trace=True`;
  resolving its handle is explicitly synchronizing.
- Execution tracing is armed only with `train_step(inputs, trace=True)`. The
  first explicit trace request prepares bounded CPU buffers and timing events
  before `trace_begin`, reports that setup separately, and reuses them later.
  Planning/profile activity cannot enter a step trace, and untraced calls
  allocate no trace resources.
- Every traced task exposes exactly seven `CLOCK_MONOTONIC` boundaries: four
  host enter/exit timestamps and the stream-ordered
  `before_readiness_waits`, `before_task_compute`, and `after_task_compute`
  callbacks. The same task record includes its expected structural-profile
  runtime and a CUDA-event duration cross-check.
- Plan diagnostics now retain mutually exclusive phase timings, structural
  profile/cache counts, direct task-to-unique-stage mappings, candidate and
  chosen graph-pair variants, and all legal save/recompute graph pairs for
  each unique stage. Each forward/backward graph record includes input,
  mutation, output, alias, workspace, persistent-provider, sample/runtime, and
  allocation-lifetime geometry. The inventory is always constructed; the
  measured `diagnostic_inventory` phase makes its one-time cost visible.
- The first 30-GiB Qwen detailed run measured a 494.6–502.9-ms compute bracket
  over its stable steps. Per-task CUDA-event durations sum to 285.0–289.5 ms,
  leaving 205.0–216.7 ms as inter-task gaps. Median host-category sums were
  approximately 170 ms in `before_task`, 72 ms in storage rebinding/input
  lookup, 275 ms inside compiled dispatch, 33 ms in output postprocessing, and
  11 ms in `after_task`; these categories overlap GPU execution and therefore
  are not added to the gap. The evidence localizes the discrepancy to staged
  task-boundary/launch behavior rather than optimizer kernels, but the
  trace-on/trace-off control and staged standard-allocator control are still
  required before declaring a root cause.
- The same run planned in 86.220 seconds: 17.581 seconds capture/lowering,
  37.488 seconds compiled-entrypoint construction, 0.358 seconds structural
  profiling from 43 cache hits, and 19.964 seconds PressureFit. Five numerical
  steps and checkpoint replay stayed within the configured loss/vector
  tolerances; the qualification aggregate remained false because this
  30-GiB diagnostic schedule intentionally had terminal D2H but no
  in-schedule H2D, while the stressed gate requires both directions.

## 2026-08-11 — Default-off runtime trace integration

- Finalized the public execution switch as
  `train_step(inputs, trace=False)`. There is no planning-time trace option.
  `PlanReport.diagnostics` remains unconditional, while real-step tracing is
  requested only for the individual invocation being investigated.
- Added runtime trace ABI v1 to neutral runtime ABI v8 and exposed it through
  PyTorch adapter ABI v14. The API has explicit prepare, begin, end, and read
  operations. Current preparation owns reusable CPU arrays but does not enable
  tracing; this storage policy is private and may later become chunked without
  changing the ABI or Python result.
- The native event ledger records session bounds, task boundaries, readiness
  waits, action queueing, destination reservation, H2D/D2H dispatch and
  observed completion, allocation-pressure waits, stream retirement, and
  first failure. Allocation lifetimes are copied alongside the causal runtime
  ledger. Capacity exhaustion is explicit in diagnostics and cannot silently
  change numerical execution.
- `StepDiagnostics` separates timing, per-task boundaries, allocator
  lifetimes, transfer events, runtime events/counters, and simulator
  comparisons. Resolving the asynchronous diagnostics handle drains the
  transfer service before ending the trace, so terminal D2H is included. This
  synchronization occurs only after the caller explicitly requests
  `diagnostics.result()`.
- The first trace request precreates and initializes reusable per-task CUDA
  timing events, compute-bracket events, native trace storage, and callback
  records before `trace_begin`; `trace_setup_seconds` reports that one-time
  work. Later traces reuse those resources. The normal path creates no trace
  resources and launches no trace callbacks.
- A mock-runtime canary proves tracing is opt-in and validates ordered
  queue/reservation/dispatch/completion events. The real CUDA accumulated
  training canary validates all seven task timestamps, native trace
  completeness, no overflow, reuse across steps, and numerical/checkpoint
  behavior. All native/CUDA canaries, the full Python suite, strict mypy, and
  Ruff pass.

## 2026-08-11 — Readiness serialization root cause

- Froze the unchanged linear-object-lookup Qwen baseline before modifying
  runtime behavior. The full trace is
  `/tmp/shadowspill_qwen35_linear_lookup_trace_baseline.json`; compact internal
  and NSYS evidence lives under `qualification/results/phase1/`.
- Root cause: the progress service admitted only one transfer per direction to
  its CUDA stream. Although all 395 startup destinations were already
  physically reserved, a later object's event did not exist until earlier
  H2Ds completed. `before_task` therefore host-blocked instead of inserting a
  stream wait and returning.
- Removed the redundant software one-in-flight gates. CUDA streams retain FIFO
  copy order; the complete admitted window now receives pooled events and is
  enqueued. A delayed mock regression proves that acquiring the second object
  returns through a stream wait while the first copy remains in flight.
- Corrected 30-GiB Qwen evidence: host-condition readiness waits fell 311 ->
  0, stream-event waits rose 2 -> 27, total host `before_task` time fell
  165.209 -> 15.571 ms, and median untraced execution fell 642.580 -> 542.809
  ms. Task CUDA time remained essentially unchanged (286.281 versus 290.923
  ms); no PressureFit action or recomputation choice moved.
- All 129 selected tasks record `before_readiness_waits` and
  `before_task_compute`. Only three tasks inserted actual event dependencies.
  No-wait intervals up to 0.458 ms are diagnostic `cuLaunchHostFunc`
  overhead/jitter, not residency stalls.
- Surprise: terminal backward task 17 retained a real 44.910-ms dependency on
  a 4-byte FP32 input. Direct FX inspection identifies input slot 58 as
  `tangents_1`, the AOTAutograd `torch.ones_like(loss)` seed consumed by
  `aten.nll_loss_backward`. Lowering incorrectly classifies this
  compiler-generated constant as a host-backed step input, placing it last in
  the 6.0418-GB non-cyclic startup queue. This is a separate generic lowering
  bug; the terminal unit cotangent should be specialized out of the backward
  ABI, while internal activation cotangents remain ordinary planned objects.
- Added `docs/runtime_overheads.md` with the standalone clocks, causal
  timeline, exact task/object evidence, before/after measurements, and
  remaining attribution controls.

## 2026-08-11 — Terminal objective cotangent specialization

- Direct FX inspection proved that task 17's 4-byte FP32 input was the
  AOTAutograd terminal cotangent `d(loss)/d(loss) = 1`, not model state,
  activation data, or anonymous workspace. It had been materialized as a
  host-backed fixed input and placed last in the 6.0418-GB startup H2D queue.
- Added a structural terminal-only specialization in the PyTorch frontend.
  When the differentiated root is scalar and capture supplied the known unit
  cotangent, the backward graph constructs a scalar one from an existing
  tensor input with `aten.new_ones`; the cotangent is removed from the public
  backward ABI and Program. No operation, model, or objective name is used.
  Internal stage cotangents remain ordinary planned activation-gradient
  objects.
- Added coverage for native losses, a composite MSE plus weighted auxiliary
  objective, custom autograd, terminal-versus-internal graph pairs, and the
  lowered fixed-object inventory. CPU capture uses the same device-relative
  rule and no CUDA literal.
- The fresh 30-GiB Qwen plan reduced startup H2D from 395 objects and
  6,041,784,940 bytes to 394 objects and 6,041,784,936 bytes. The preceding
  task's compute completion to task 17 compute start fell from 44.922 ms to
  0.288 ms; task 17 no longer records a readiness dependency.
- This did not close the overall execution discrepancy. The fresh traced
  first-to-last compute interval was 439.745 ms, compared with a noisy
  312.561-ms sum of task CUDA intervals and the 292.141-ms compiled PyTorch
  authority. The next isolated cause under test is the linear object-table
  lookup used repeatedly during input acquisition and storage validation;
  the same trace attributes 73.885 ms to rebinding/validation.

## 2026-08-11 — Lifecycle-safe object lookup index

- Replaced linear object-ID lookup with an internal `ShadowSpillObjectTable`
  that owns records in one intrusive lifetime list and indexes them in hash
  buckets. Insert and removal update both structures under the existing
  runtime lock; no synchronization or state-transition behavior changed.
- Centralized removal for both explicit unregister and caller-output handoff.
  Regression coverage removes and re-registers the same ID through both paths,
  rejects duplicates, and proves that no stale hash entry survives object
  destruction. This specifically covers the lifecycle omission that caused an
  earlier experimental hash index to dereference a freed record.
- All 16 native and CUDA canaries pass with warnings treated as errors. Lock
  decomposition remains a separate milestone after the unchanged-lock lookup
  result is measured.

## 2026-08-12 — Warm Qwen runtime discrepancy fully reconciled

- Captured a fresh warm standard-allocator NSYS control and a warm
  `trace=False` ShadowSpill control. Both execute essentially identical
  numerical work: 40,681 versus 40,528 kernels and 76.420 versus 77.220 ms of
  summed kernel-active time. Five prior warm iterations precede capture, so
  the long dispatch intervals are not JIT compilation.
- The pure-PyTorch delta-rule reference is the source of the kernel storm. Its
  explicit recurrence launches 728 kernels for a 64-token forward component,
  1,080 for a 96-token component, and approximately 28 kernels per token in
  backward. ShadowSpill does not introduce these operations.
- Reconciled the non-cyclic steady cycle: approximately 38.881 ms of residual
  prior-step D2H, 59.538 ms of startup readiness, and 384.361 ms from first
  task compute to final optimizer compute yield 482.780 ms, consistent with
  the 476.734-ms untraced median. The 384.361-ms compute span contains
  299.281 ms of task CUDA intervals and 85.079 ms of inter-task idle. Standard
  compiled PyTorch is 292.141 ms, so numerical task work differs by only
  7.140 ms.
- Found three generic causes of the inter-task/runtime regression. First,
  startup and terminal transfer queues cause CUDA launch backpressure: warm
  NSYS observes 25.245-ms and 52.004-ms individual launch calls, while the
  standard control's largest launch is 0.076 ms. Second, all 37,563 allocator
  callbacks contend with the progress service's 223,208 `cuEventQuery` calls
  under one global mutex; slow mutex acquisitions total approximately 54 ms.
  Third, each of 97 optimizer components rebuilds the complete model and
  optimizer binding inventory, accounting for 56.765 ms of the 61.686-ms
  input/rebind bucket. Actual C++ storage-rebind ranges total only 8.086 ms.
- The current three-host-callback task trace is materially invasive. One
  `cuLaunchHostFunc` submission blocked for 50.961 ms, and traced wall time was
  593.992 ms versus a 476.734-ms untraced median. Decision: retain four host
  timestamps but replace the three stream callbacks with preallocated CUDA
  events resolved by the explicitly synchronizing diagnostics handle.
- Audited all execution work outside the currently named native boundaries.
  Logical `before_task` will include static input selection, acquire/waits,
  rebinding, argument assembly, and the compute-start marker. Logical
  `after_task` will include the compute-end marker, output/gradient handling,
  dematerialization, native action submission, released-object cleanup, and
  terminal optimizer cleanup. Input guards/staging, prior cooldown, initial
  placement, caller-output ownership, result construction, and asynchronous
  progress remain explicit step/component ledgers rather than hidden task
  time.
- Expanded `docs/runtime_overheads.md` into a standalone end-to-end account
  with exact ledgers, standard-versus-ShadowSpill controls, allocator and lock
  distributions, optimizer fragmentation, trace observer effect, boundary
  definitions, and the ordered corrections. Baseline reports are
  `qualification/results/phase1/qwen35_standard_allocator.nsys-rep` and
  `qualification/results/phase1/qwen35_untraced_current.nsys-rep`.

## 2026-08-12 — Completion-query contention and runtime ownership design

- Correlated the user-visible NSYS mutex bands with
  `execution_000004.microbatch_0000.stage_0004.forward.recompute` (canonical
  `task_000009`). During its 16.733-ms profiled host range, 794 allocator
  callbacks accumulated 4.850 ms, 64 slow mutex waits accumulated 7.067 ms,
  and the progress worker issued 17,557 event queries. Across the step the
  worker issued 223,208 `cuEventQuery` calls while sharing the allocator's
  global runtime mutex.
- Confirmed that completion events are required for H2D readiness, D2H
  authority/range release, annotated task-fence actions, and `record_stream`
  retirement. The defect is full-population repeated polling under the global
  lock, not event synchronization itself. The selected correction is
  per-stream FIFO completion frontiers, head-only `cuEventQuery` outside state
  locks, shared-fence deduplication, a precreated event pool, and adaptive
  backoff/pressure wakeups. `cuEventSynchronize` and stream host callbacks are
  explicitly excluded from the dispatch-thread progress path.
- Verified the current thread topology from initialization: one ShadowSpill C
  progress thread, one H2D CUDA stream, and one D2H CUDA stream per device
  runtime, in addition to the caller/PyTorch dispatcher. CUDA streams are not
  CPU threads. One worker remains the intended default because the problem is
  lock scope/query complexity rather than insufficient worker parallelism.
- Documented the proposed central ownership model in
  `docs/runtime_overheads.md`: an object hash table for membership/lifetime;
  per-alias-bundle object records and locks; an allocation pool/condition;
  immutable execution records with direct retained object pointers; separate
  H2D/D2H lanes; per-stream completion queues; event pool; atomic trace; and
  separate failure/lifecycle state. Hot transitions use retain,
  snapshot-and-commit and normally hold at most one mutex.
- Bumped the private PyTorch adapter ABI to v15 and added cold-path task-label
  registration. C and Python NVTX task ranges now use dense chronological
  `execution_XXXXXX` identities plus semantic stage names; canonical IR task
  IDs remain fallback/correlation metadata.

## 2026-08-12 — Concurrency-redesign starting point frozen

- Clarified the measured improvement boundary. Optimizer binding caching and
  replacing stream host callbacks with preallocated CUDA events reduced the
  less-invasive first-selected-task to final-optimizer span from 384.361 ms to
  325.128 ms, with a 297.512-ms sum of task intervals and 27.616 ms of gaps.
  The global runtime lock and full-population completion scan are not yet
  fixed; 325.128 ms has therefore not yet reached the approximately 300-ms
  target.
- Captured a new warm NSYS report from commit `418a930` at
  `qualification/results/phase1/qwen35_runtime_overheads_updated.nsys-rep`,
  with matching JSON and SQLite artifacts. The Program digest, schedule
  digest, 129 selected tasks, 1,415 ordered actions, 30-GiB predicted peak,
  and 503.781231-ms simulator prediction are unchanged.
- The instrumented capture reports a 505.5-ms selected-task span and 267,736
  `cuEventQuery` calls. NSYS heavily perturbs this small-kernel workload, so
  the report is the attribution authority, while untraced and reusable CUDA
  event measurements remain the performance authority.
- Updated `qualification/extract_execution_trace.py` to recognize semantic
  `execution_XXXXXX.<stage-name>` NVTX identities while retaining legacy trace
  compatibility. The updated extraction identifies all 129 tasks and 97
  optimizer components rather than depending on canonical IR task labels.
- Created the ignored live implementation plan and log under
  `docs/internal/plans/concurrency_fix/`. The concurrency redesign is now the
  sole prerequisite; the five-cell cold golden matrix in Phase 2 of
  `remaining_agenda.md` stays paused until performance, runtime equivalence,
  and multi-budget numerical/checkpoint gates pass.

## 2026-08-12 — Duplicate task NVTX ownership corrected

- The updated NSYS capture exposed two identical semantic task rows. The C
  adapter already owns the failure-safe outer range from native `before_task`
  through native `after_task`; debug execution timing also pushed the same
  range from Python around the full logical task.
- Removed only the Python outer task push/pop. The C range remains the sole
  owner, while nested `before_task`, storage-rebind, `compiled_call`, and
  `after_task` ranges and all seven diagnostic timestamps remain unchanged.
- Tightened the NSYS extractor to reject duplicate chronological semantic
  task ranges instead of silently choosing one. Duplicate tracing is now a
  qualification failure.
- A fresh warm Qwen validation at
  `qualification/results/phase1/qwen35_nvtx_deduplicated.nsys-rep` contains
  exactly 129 outer semantic ranges for 129 unique execution IDs, zero
  duplicates, 129 nested `compiled_call` ranges, and 97 optimizer tasks. Its
  258,781 event queries confirm that the separate concurrency defect remains
  visible for the next milestone.

## 2026-08-12 — FIFO completion frontiers implemented

- Replaced retirement, task-fence, H2D, and D2H full-population event queries
  with a central per-recording-stream FIFO completion tracker. The worker
  snapshots only each stream head, issues the nonblocking backend query with
  no runtime-state or completion lock held, and atomically publishes the
  completion before existing transition code consumes it.
- Completion records retain generation-tagged event leases. This separates
  backend completion lifetime from task fences, transfer actions, objects,
  and allocation retirement owners without changing action or residency
  order. Query failures retain their causal object/allocation identity; the
  first implementation omitted this attribution and caused the existing
  failure canary to reject the change.
- Added adaptive idle polling from 10 microseconds through 10 milliseconds.
  Stream order allows a completed head to drain already-completed successors
  immediately and proves that an incomplete head's successors need not be
  queried.
- Added a deterministic 64-retirement completion canary and mock-backend event
  telemetry. It observes 72 backend queries for 64 completions, or 1.125
  queries per completion, while reclaiming every allocation and event. The
  acceptance ceiling is four queries per completion.
- Normal warnings-as-errors CTest (17 tests), the complete Python suite, Ruff,
  strict mypy, and UBSan pass. Both GCC and Clang ASan binaries pass direct
  runs, but this host intermittently segfaults during ASan process startup even
  for the empty `shadowspill_build_canary`; GDB suppresses the failure. This is
  therefore classified as sanitizer-host infrastructure rather than a runtime
  lifetime failure. Repeated normal canaries and UBSan remain clean.
- Re-ran the exact frozen 30-GiB Qwen case. Program digest
  `65300023e849...` and schedule digest `e349ce5f7c2a...` remain unchanged,
  with 129 execution tasks and 1,415 actions. The selected-task span improved
  from 325.128 ms to 311.976 ms, passing the 312.4-ms gate; inter-task gaps
  fell from 27.616 ms to 23.080 ms and the untraced recurrent interval from
  480.857 ms to 469.041 ms. Allocation requests (37,563), frees (37,443),
  zero-byte requests (120), and readiness waits (26) are identical.

## 2026-08-12 — Allocation-pool ownership isolated

- Introduced an allocation-pool mutex and capacity condition that exclusively
  own slab geometry, allocation hashes, reusable records, accounting, and
  retirement membership. PyTorch `malloc`, logical `free`, pointer lookup,
  `record_stream`, and allocation-pressure waits no longer acquire the legacy
  runtime/action mutex.
- Added atomic pending-retirement and pending-capacity-action counters so a
  blocked allocator can distinguish possible progress from an impossible OOM
  without scanning the transfer queue under a second lock. Device free and
  largest-range snapshots likewise let failure reporting remain independent
  of allocation ownership.
- Moved worker retirement processing behind the allocation owner, but narrowed
  transfer dispatch and completion commits so CUDA copy/event submission never
  holds the allocation-pool lock. Allocation geometry is locked only while a
  prefetch range is reserved/adopted or an offloaded/released range is returned.
- Made first-failure publication independent through a dedicated cold-path
  lock plus an atomic status latch. Trace and allocation-telemetry append slots
  now use bounded atomic cursors, so diagnostics do not require either runtime
  state lock.
- The CUDA backend's sealed physical event pool remains the source of event
  handles. Runtime event-lease creation in steady state only borrows a pooled
  handle and creates a generation-tagged metadata owner; it does not call
  `cuEventCreate` after admission.
- All 17 warnings-as-errors CTest canaries, the complete Python suite, Ruff,
  and strict mypy pass. Backend calls used to record ordinary non-task
  retirement fences are the remaining allocation-path snapshot/commit work;
  this checkpoint intentionally does not claim that the full lock redesign is
  complete.

## 2026-08-12 — Ordinary free uses retirement snapshot/commit

- Converted the non-task logical-free path into two phases: snapshot recorded
  streams and mark the allocation retirement as preparing under the allocation
  owner; record and submit completion events with no allocation lock held;
  then publish the event list only if allocation ID and generation still match.
- Excluded preparing records from reuse. This preserves immediate logical-free
  semantics while preventing another allocation from observing a partially
  built retirement set.
- Removed redundant same-stream waits during pending-block reuse. Candidate
  selection already proves that all old uses and the new allocation are on the
  same CUDA stream, whose ordering is itself the retirement dependency.
- All native, CUDA, PyTorch, Python, lint, and type-check gates continue to pass.

## 2026-08-12 — Object table gains explicit concurrent ownership

- Added a read/write lock to the central object-ID hash table and an acquire
  operation that retains a stable object reference before releasing the table
  read lock. Insert and detach now use the write side, while final destruction
  remains governed by the object's atomic reference count.
- Preserved the per-object mutex introduced with stable records. The next
  transition step can therefore hold table membership only long enough to
  acquire an object and use the object's semantic lock for residency/version
  state, rather than protecting the whole object population globally.
- Hardened partial initialization so a failed table allocation never destroys
  an uninitialized POSIX read/write lock. All validation gates remain green.

## 2026-08-12 — Training task boundaries are real orchestrators

- Replaced the duplicated graph-task and optimizer-task dispatch bodies with
  one `_execute_task()` skeleton around `_before_task()`,
  `_run_compiled_task()`, `_after_task()`, and `_abort_task()`.
- `_before_task()` now owns the complete frontend input boundary: runtime
  acquisition, readiness, batched alias rebinding, and argument assembly.
  `_after_task()` owns output flattening/classification, declared-output and
  gradient publication, dematerialization, native action publication, released
  binding cleanup, and terminal optimizer cleanup.
- Graph and optimizer modes select only their small argument/output behaviors;
  they no longer maintain independent boundary implementations. Existing
  diagnostic subfields continue to measure native acquisition, rebinding,
  dispatch, postprocessing, native after-task, and cleanup.
- Full native/CUDA and Python validation, Ruff, and strict mypy pass without
  changing task IDs, actions, graph-pair selection, or arithmetic.

## 2026-08-12 — Ordered action queue receives a dedicated owner

- Replaced the action-list fields embedded in miscellaneous runtime state with
  a central `ShadowSpillActionQueue`: its own mutex, ordered head/tail, and
  atomic count. Queue membership no longer depends implicitly on whichever
  caller happens to hold the lifecycle mutex.
- Queue publication, progress traversal, object-detach checks, teardown, and
  diagnostics now use the queue owner. The atomic count lets lifecycle and
  allocator code test quiescence/progress without taking the queue lock.
- Preserved the legacy lifecycle lock around object transition bodies for this
  mechanical checkpoint. Removing that final overlap requires per-object
  snapshot/commit and is deliberately a separate behavioral gate.
- Full native/CUDA, Python, Ruff, and strict-mypy validation passes.

## 2026-08-12 — Forward tasks use the same explicit boundary shape

- Reworked each forward stage into `_before_task()`,
  `_run_compiled_task()`, `_after_task()`, and `_abort_task()` around a small
  prepared-task record. Input rebinding is now wholly before-task work, while
  output promotion, dematerialization, and action publication are wholly
  after-task work.
- The forward-only public behavior, output pytree, alias semantics, and
  no-autograd execution are unchanged. Forward/training CUDA canaries and the
  full PyTorch unit suite pass.

## 2026-08-12 — Immutable execution topology admitted natively

- Added a framework-neutral execution table keyed by dense task ID. Admission
  copies input, mutation, and ordered-action topology; resolves each object ID
  once through the object hash; and retains direct stable object references.
- Added predecoded `before_execution`/`after_execution` C and PyTorch-adapter
  entrypoints. Recurrent training plans and forward plans now adopt their task
  topology once and stop rebuilding ctypes identifier/update/action arrays on
  every task invocation.
- Admission revealed that produced objects were previously registered only on
  first allocation. Plan adoption now creates zero-residency placeholder object
  records for every referenced alias bundle; ordinary graph output allocation
  is still promoted into the existing record at production time.
- Identical duplicate admission is idempotent and a conflicting definition for
  one task identity is rejected. A native canary covers both cases and the hot
  admitted boundary. All full validation gates pass.
- This checkpoint still projects the retained records through the proven legacy
  transition helpers. The next step removes those residual per-call object
  hashes and consumes direct references inside the boundary orchestrators.

## 2026-08-12 — Admitted input acquisition consumes direct records

- The hot admitted `before_execution` path now consumes predecoded object
  pointers and the first-occurrence unique-input list built during admission.
  It performs no object-ID hash lookup, alias deduplication scan, or
  allocation-ID hash lookup during task execution.
- Each object directly names its current allocation lease. The boundary checks
  both allocation and generation identity before publishing a pointer, so a
  stale recycled lease cannot satisfy an input binding.
- The legacy boundary remains available for non-admitted construction paths.
  This checkpoint intentionally preserves its readiness waiting and global
  transition serialization; moving those transitions to per-object ownership
  is a separate gate.

## 2026-08-12 — Admitted after-task path consumes direct topology

- The admitted after-task boundary now executes explicit mutation publication,
  allocation-handoff validation, task-fence creation, ordered-action
  instantiation, retirement attachment, and queue publication phases.
- Every phase consumes stable object references from the immutable execution
  record. Recurrent tasks no longer hash mutation/action object IDs or retain
  duplicate legacy array projections.
- The exported boundary is a short ordered orchestrator; the named helpers keep
  failure attribution and ownership cleanup local to the phase that can fail.
  The global transition lock is still intentionally retained at this checkpoint
  so this lookup refactor cannot change worker/dispatcher ordering.

## 2026-08-12 — Hot task boundaries adopt object-local readiness ownership

- Admitted before/after boundaries no longer acquire the legacy runtime mutex.
  Residency, generation, version, current lease, and readiness state are read
  or mutated under each alias-bundle object's mutex; backend compute-stream
  waits occur after retaining the event and releasing that lock.
- Added an object-local condition variable and explicit `prefetch_pending`
  invariant. A consumer can wait for queued transfer dispatch without scanning
  the global action population, and dispatch publication wakes only consumers
  of the affected object.
- The first forward canary exposed that initial placement uses the generic,
  non-admitted action boundary. Pending-prefetch publication was therefore
  centralized as a runtime action invariant shared by both boundaries rather
  than patched into the forward executor.
- Transfer/statistics counters accessed by the worker and dispatcher are now
  atomic. Focused forward, recurrent training, overlap, transition, and failure
  gates pass. Worker transfer submission still holds the action/object locks
  and is the next snapshot/commit conversion.

## 2026-08-12 — Generic memory pools and independent transfer lanes

- Introduced one neutral `ShadowSpillMemoryPool` abstraction and instantiated
  it twice as `device_pool` and `host_pool`. Both own a bounded backing base,
  coalescing range geometry, alignment, accounting, mutex, and capacity
  condition, and both use the same reserve/release interface. PyTorch logical
  allocation records and stream retirement remain device-side clients rather
  than contaminating the generic pool contract.
- Split H2D and D2H queue membership into two `ShadowSpillTransferLane`
  instances with independent pending and in-flight FIFO ownership. Backend
  wait/copy/event submission now happens after releasing action, lane, object,
  and memory-pool locks, followed by a generation-checked object commit.
- A first implementation incorrectly limited each transfer lane to one
  submitted copy. The transition canary caught this: a second transfer failed
  to obtain its readiness event early enough for nonblocking dispatcher
  run-ahead. The corrected lane admits every FIFO-ready copy onto the CUDA
  stream; CUDA stream order serializes on-wire work while each object receives
  its own event immediately.
- Claimed action records initially remained marked `processing` after a
  successful dispatch, preventing the worker from revisiting their completion.
  The action state now distinguishes a temporary worker claim from durable
  in-flight transfer membership, and every nonterminal progress result releases
  the claim.
- Named the pure-C progress pthread `shadowspill.wkr`. It never enters Python
  or acquires the GIL; NSYS can now distinguish it from the PyTorch dispatcher.
- All 17 warnings-as-errors C/CUDA canaries, the complete Python suite, Ruff,
  and strict mypy pass. The Qwen performance/NSYS gate remains pending until
  the remaining allocation-record and batched-storage-boundary work lands.

## 2026-08-12 — Lost allocator-idle wakeup found after `29e5f56`

- The first full Qwen control after commit `29e5f56` stopped with an idle GPU
  in `CudaTaskProfiler.take_functions()` at
  `shadowspill_pytorch_allocator_wait_idle()`. The PyTorch main thread and the
  newly named `shadowspill.wkr` thread were both in condition waits, so this
  was a real runtime quiescence deadlock rather than slow AOT compilation.
- Added diagnostic-only retirement categories to the allocator statistics and
  temporarily observed planning quiescence from Python. Every affected ABI
  reported `pending == fenced`, with zero evented, preparing, or unfenced
  retirements. Representative batches were 133, 242, 309, and 307 allocation
  records; the worker cleared each batch almost immediately once the caller
  polled instead of entering the blocking idle wait.
- Root cause: the idle waiter checks `pending_retirements` while holding the
  lifecycle mutex and then calls `pthread_cond_wait`. Retirement completion
  changes that predicate under the device-pool lock and broadcasts the
  lifecycle condition without holding its associated mutex. A completion can
  therefore clear the final retirement and broadcast after the predicate check
  but before the waiter sleeps. The notification is lost and no later work is
  guaranteed to wake the already-satisfied waiter.
- This finding is intentionally recorded before changing synchronization. The
  resulting Qwen control retained exact frozen Program/schedule identities and
  completed planning in 70.58 seconds with a 471.22-ms median untraced compute
  bracket. A pre-polling NSYS capture is being collected first; the isolated
  lost-wakeup fix and later low-latency polling policy will be separate commits
  and evidence entries.

## 2026-08-12 — Broad wakeup experiment rejected; final correction unresolved

- A first candidate introduced one wakeup epoch used by both
  `runtime_wait_idle` and the progress worker, then notified it on completion
  submissions and several queue transitions. It eliminated the quiescence
  deadlock, but it did not preserve warmed Qwen execution behavior and is not
  an acceptable fix.
- The exact frozen Qwen Program and schedule were unchanged, yet three initial
  traced selected-task spans were 410.746, 412.342, and 418.478 ms versus the
  312.102-ms pre-change control. Two further samples were 411.401 and
  436.268 ms. Summed CUDA task intervals increased from 294.635 ms to
  393.750--418.501 ms, and summed synchronous compiled-call host time increased
  from 332.899 ms to 460.346 ms.
- PyTorch allocator callback counts did not grow: both paths reported 37,563
  nonzero allocations, 37,443 logical frees, and 120 separately classified
  zero-byte requests. Physical retirement completions did change from 3,208 to
  3,677. Transfer timing did not explain the earlier forward/backward slowdown:
  the old H2D completion window was 26.854--168.953 ms and the broad-wakeup
  window was 26.225--166.479 ms.
- The current causal hypothesis is allocator-side dispatch inflation. A
  task-local allocation freed on its sole compute stream is deliberately left
  as a reusable pending record, allowing a later compatible `malloc()` to reuse
  it without an event query. Excess worker wakeups can both contend with the
  synchronous PyTorch dispatcher for the device-pool lock and retire such
  records before the cheap reuse path claims them. The aggregate evidence is
  consistent with this mechanism, but it does not yet split lock-wait latency
  from reuse loss at individual callbacks. This hypothesis is therefore not
  recorded as resolved.
- Existing standard-allocator evidence covers the whole compiled step, not the
  identical selected staged executable. It shows comparable aggregate numerical
  kernel work (40,681 kernels and 76.420 ms kernel-active with the standard
  allocator versus 40,528 kernels and 77.220 ms in the historical ShadowSpill
  capture), but cannot by itself measure per-stage dispatcher overhead. A fresh
  same-stage standard-allocator control remains required.
- The replacement candidate gives `runtime_wait_idle` a dedicated predicate
  epoch and notifies it only after the final action or retirement counter has
  actually transitioned to zero. The progress worker retains its prior
  condition, cadence, and notifications. This candidate has passed 5,120
  focused serialized final-retirement/wait-idle race boundaries and all 17
  native/CUDA canaries. It remains explicitly unaccepted until the repeated
  full Qwen control reproduces the prior timing and exact plan identity.

## 2026-08-12 — Narrow idle-wakeup correction passes the Qwen gate

- Fix commit: `ad2f4ef` (`Fix allocator idle-wait notification race`).
- The accepted correction separates quiescence notification from progress
  scheduling. `runtime_wait_idle()` waits on a dedicated epoch/condition.
  `complete_action`, physical retirement, and pending-allocation reuse first
  decrement their counters and notify the idle waiter only when the previous
  value was one. Failure and close transitions also notify it. The progress
  worker continues to use its original condition and timed-poll cadence.
- This ordering closes the demonstrated race mechanically: the waiter holds the
  idle-wakeup mutex while testing the counters and entering `cond_wait`, while
  the final counter transition acquires the same mutex before advancing the
  epoch and signaling. The signal therefore cannot occur between the predicate
  check and sleep. Unlike the rejected broad correction, this mechanism never
  wakes the worker or changes retirement timing.
- The repeated 30-GiB Qwen control produced selected-task spans of 304.287,
  304.859, and 304.511 ms and summed task CUDA intervals of 287.851, 289.017,
  and 288.979 ms. These improve on the pre-change 312.102-ms span and
  294.635-ms task sum and satisfy the 312.4-ms gate. The Program digest remains
  `65300023e849c757c4d5d663ce7161c6fe7edb9b1b95170ebec3a7aa112cd7e3` and
  the schedule digest remains
  `e349ce5f7c2a7132cec4a8f24b082ffb1124b0691df210ce6a0914083ffff3ed`:
  129 execution tasks and 1,415 actions are unchanged.
- Planning completed in 71.381 seconds. Allocation behavior remained stable at
  37,563 nonzero requests, 37,443 materialized requests/frees, 120 separately
  classified zero-byte requests, 26 inserted readiness waits, and no terminal
  pending retirement or queued action.
- All 17 warnings-as-errors native/CUDA canaries pass, including 5,120 focused
  final-retirement/wait-idle race boundaries. The default Python suite, Ruff,
  and strict mypy pass. An additional cold, individually invoked public SGD
  pytest exposed the pre-existing/nondeterministic lowering assertion “one fake
  tensor view maps to distinct compiled allocations”; the dedicated fresh-
  process public training canary passes. That lowering issue is tracked
  separately and is not combined with this synchronization correction.

## 2026-08-12 — Post-fix Qwen NSYS artifact captured

- Captured the exact 30-GiB Qwen control after fix commit `ad2f4ef` at
  `qualification/results/phase1/qwen35_idle_wakeup_fixed.nsys-rep`; its SQLite
  export and semantic summary use the matching basename. These generated
  qualification artifacts remain intentionally ignored by Git.
- The capture retained the frozen Program and schedule digests, 129 semantic
  execution ranges, 97 optimizer tasks, and 1,415 memory actions. The semantic
  extractor rejected no duplicate task ranges, and NSYS records the pure-C
  progress thread as `shadowspill.wkr` rather than `python`.
- NSYS perturbed the selected-task span to 460.325 ms and the summed CUDA task
  intervals to 433.488 ms; these are diagnostic capture measurements, not the
  performance gate. The immediately preceding non-NSYS `trace=True` control is
  authoritative at 304.287--304.859 ms.
- The extracted trace reports a 460.310-ms compute-stream span, 74.064 ms of
  kernel-union time, and 386.246 ms between kernel intervals. It contains 394
  H2D and 388 D2H copies, matching the plan's startup/terminal action shape.
  The trace is now available for the planned completion-frontier and allocator
  lock analysis before any polling-policy change.

## 2026-08-12 — Post-fix allocator and task-gap root cause

- The apparent allocator anomalies have two measured causes. Of 37,563
  callbacks, only seven exceed 50 us: five are dominated by device-pool mutex
  contention and two are explicit NSYS profiler overhead. The four callbacks
  above 100 us divide evenly between those causes; no intrinsic allocator work
  above 50 us remains unexplained.
- The screenshot's 23.708-us allocation waited 22.427 us for the progress
  worker. Its timed poll had expired, it queried one event, then walked roughly
  766 active allocation records under `device_pool.lock`; no record completed.
  Larger examples waited 173.663 us while 197 records/1.590 GB were retired and
  114.042 us while 30 records/872.0 MB were retired. FIFO event polling is
  already effective (1,317 queries versus 223,208 historically), but completed
  allocations are still rediscovered by a full active-list scan under the
  allocator-visible lock.
- The unified executor skeleton exists, but its diagnostic scopes and batching
  milestone are incomplete. The ranges called `before_task`/`after_task` wrap
  only native bridge calls; output processing and per-object storage operations
  remain outside. The displayed execution-14-to-15 gap is fully accounted:
  326.335 us postprocessing, 33.252 us native after-task, 76.930 us transition,
  73.726 us native before-task, 2.137 us transition, 333.443 us rebind/argument
  assembly, and 13.763 us dispatch transition = 859.586 us.
- The worker does issue copies: 394 `cuMemcpyHtoDAsync_v2` calls were enqueued
  at 29.440--31.222 ms and 388 `cuMemcpyDtoHAsync_v2` calls at
  524.786--542.685 ms, along with 782 stream waits and 782 event records. Local
  mid-compute windows show only event queries because copies were already
  enqueued asynchronously.
- The dispatcher's 390 `cudaStreamIsCapturing` calls pair one-for-one with 390
  diagnostic event records: three task timestamps for 129 tasks plus three
  step-level timestamps. They are expected only for `trace=True`, cost 0.110 ms
  in aggregate, and are unrelated to the timing-disabled C runtime event pool.
- All 40,528 dispatcher-launched kernels correlate to compiled-call ranges;
  none occur in the inter-task windows. The measured gap is host-side work,
  not omitted gradient-accumulation CUDA execution.
- Detailed standalone evidence and the remaining corrective order are in
  `docs/runtime_overheads.md`. No polling/condition semantics have been changed
  during this investigation.

## 2026-08-12 — Direct retirement queue removes allocator starvation

- Replaced the progress worker's full `active_allocations` retirement scan
  with a direct queue populated when an allocation receives its stream events
  or task fence. Completion checks read already-published event state outside
  the device pool; the pool lock is entered only for a generation-validated
  range release. Program digest, schedule digest, 129 tasks, and 1,415 actions
  remain exact.
- Two rejected intermediates are retained as design evidence. Releasing only
  one completed range per worker pass inflated reset/cooldown to roughly
  0.914 seconds because thousands of terminal retirements became serialized by
  poll intervals. An always-on foreground-waiter atomic restored priority but
  touched a contended cache line on every allocator callback and increased the
  selected-task span to 314.350--316.299 ms. Neither design was accepted.
- The accepted policy belongs to the generic `MemoryPool`, not to retirement:
  foreground callers first take an uncontended `trylock` fast path and publish
  waiter intent only after real contention; background reclamation uses a
  nonblocking pool acquisition that refuses to reacquire while a foreground
  waiter exists. A foreground `malloc` can therefore wait behind at most the
  already-running range mutation, while background terminal draining remains
  batched when no foreground caller is waiting.
- Production-like controls recorded selected spans of
  307.783--309.521 ms in the first run and 311.384--314.125 ms in the repeated
  abstraction-equivalent run. Corresponding task sums were
  287.913--290.724 and 292.806--293.228 ms. The reset-inclusive untraced
  medians were 467.205 and 469.027 ms. Both preserve the frozen plan; the
  repeated spread is recorded rather than selecting only the faster sample.
- The updated NSYS artifact is
  `qualification/results/phase1/qwen35_retirement_queue_dispatcher_priority.nsys-rep`.
  Genuine allocator callbacks above 50 us fell from five before the change and
  one in the first direct-queue candidate to zero. The three apparent callbacks
  above 100 us are 103.5--115.2-us NSYS `Chunk Allocation` observer overhead.
  Nested allocator mutex wait fell from 0.895 ms to 0.200 ms.
- NSYS selected span changed from 460.310 to 471.577 ms: kernel union increased
  only 0.553 ms, summed task intervals increased 6.603 ms, and inter-task idle
  increased 4.668 ms. The allocator NVTX aggregate increased 5.699 ms under
  instrumentation because every callback now includes the priority `trylock`;
  this accounts for most of the within-task launch-spacing change. Unprofiled
  task sums return to the prior range and remain the performance authority.
- All 17 warnings-as-errors native/CUDA/PyTorch canaries, the default Python
  suite, Ruff, and strict package mypy pass. The first manual mypy invocation
  incorrectly included generated Inductor cache trees and reported duplicate
  generated module names; the configured package-only strict invocation passes.

## 2026-08-12 — Commit-specific retirement NSYS confirmation

- Accepted implementation commit: `18584ef` (`Prioritize allocator clients
  over retirement work`). The commit contains the direct retirement queue and
  generic `MemoryPool` priority policy; it contains none of the transitional
  runtime waiter helpers or bitmask result handling.
- Captured that exact binary as
  `qualification/results/phase1/qwen35_memory_pool_priority_18584ef.nsys-rep`
  with SQLite, extracted semantic JSON, and StepDiagnostics siblings. Frozen
  Program/schedule identity, 129 tasks, and 1,415 actions remain unchanged.
- Of 37,563 allocator callbacks, none has more than 50 us of real work after
  subtracting same-thread profiler overhead. The only displayed callback above
  50 us is 119.605 us with 115.499 us of NSYS `Chunk Allocation` overhead.
  Median/p95/p99 are 0.499/2.120/2.991 us. Allocator-nested mutex wait is
  0.286 ms versus 0.895 ms before and 0.338 ms for the direct-queue-only
  candidate.
- Commit-specific NSYS selected span/task-event sum are 471.637/442.490 ms
  versus 460.325/433.488 ms before. Compute kernel union is 75.678 ms versus
  74.064 ms. Thus 1.614 ms is changed kernel occupancy, 9.002 ms is inside task
  event intervals, and 2.310 ms is additional between-task spacing. The
  allocator NVTX aggregate grows by 4.896 ms because the traced foreground
  path includes a priority `trylock`; this accounts for much of the intra-task
  launch-spacing change without recreating a long mutex tail.
- The largest interval increases occur in recurrent backward execution tasks.
  Their kernels are generally unchanged within tens of microseconds while
  instrumented host calls grow by 0.4--1.6 ms. This is host launch/profiler
  variance, not extra plan work, transfer directives, or retirement scanning.

## 2026-08-12 — Full frontend task-boundary scopes

- Made training `_before_task()` and `_after_task()` the complete frontend
  orchestration boundaries. `before_task` now contains native acquisition,
  readiness, storage rebinding, and argument assembly. `after_task` contains
  output/gradient processing, dematerialization, native action submission, and
  cleanup. Nested ranges retain component attribution; `compiled_call` remains
  a disjoint numerical range.
- The first instrumentation ordering was rejected. It recorded the
  `before_task_compute` CUDA event before popping the outer `before_task` NVTX
  range, putting one host NVTX operation into every measured task interval.
  Qwen selected spans rose to 316.009--319.892 ms and task sums to
  298.680--302.128 ms despite unchanged compiled graphs.
- Corrected ordering: full preparation and its NVTX scope close first;
  `_before_task()` then records `before_task_compute` and returns. The repeated
  frozen control measured 308.941, 309.116, and 313.543 ms selected spans
  (309.116-ms median), with 291.723--295.456-ms task sums and a 467.562-ms
  reset-inclusive untraced median. Program/schedule digests, 129 tasks, and
  1,415 actions remain exact.
- Focused lowering/training tests, the CUDA training canary, Ruff, and strict
  mypy pass. No storage batching or task arithmetic changed in this milestone.

## 2026-08-12 — Transactional batched storage boundaries

- Added one private PyTorch adapter operation that validates an entire storage
  rebinding request before changing any Tensor storage, then applies the batch.
  A stale object generation therefore cannot partially mutate a task boundary.
  The scalar operation remains available for isolated lifecycle operations and
  compatibility canaries.
- Training and forward execution now issue one batch for distinct task inputs
  and one batch for release/offload dematerialization. Declared forward
  outputs, first gradient objects, and lazily created optimizer state are also
  rebound in batches after their allocator records have been promoted.
- The adapter, allocator, forward, and training canaries pass, including a new
  all-or-nothing stale-generation test. Ruff and strict mypy pass on the
  affected Python modules.
- Frozen Qwen control
  `qualification/results/phase1/qwen35_batched_storage_control.json`
  preserves Program `65300023...d7e3`, schedule `e349ce5f...f3ed`, 129
  tasks, and 1,415 actions. Across the final traced sample, total frontend
  rebinding fell from 6.956 to 5.401 ms and postprocessing from 15.082 to
  13.746 ms relative to the prior full-boundary control.
- The same run measured 315.518--317.826 ms selected spans and
  299.443--299.696 ms task sums. These are roughly 6 ms above the prior
  three-sample distribution despite byte-identical numerical graphs and
  schedule; no semantic conclusion is drawn from this intermediate run. The
  final repeated control after predecoded task records remains the acceptance
  measurement.

## 2026-08-12 — Predecoded Python records and native execution handles

- Added immutable frontend execution records containing the selected task,
  semantic execution identity, distinct input aliases, exact ordered actions,
  compiled callable, and graph-argument template. Repeated execution no longer
  resolves those fields from task/action dictionaries or recomputes trace
  identities.
- Added a versioned neutral-runtime opaque execution handle. It is resolved
  once after admission, is borrowed until runtime teardown, and points to the
  immutable record that already retains direct input-object, mutation, and
  action references. Handle-based `before_execution`/`after_execution` bypass
  execution-table hashing and its RW lock; ID-based entrypoints remain as
  compatibility wrappers.
- Runtime ABI is now 11 and the private PyTorch adapter ABI is 18. A native
  transition canary executes the same admitted task through both compatibility
  and direct-handle paths. All 17 native/CUDA/PyTorch canaries pass.
- Final frozen control
  `qualification/results/phase1/qwen35_predecoded_execution_control.json`
  preserves the exact Program/schedule digests, 129 tasks, and 1,415 actions.
  Selected spans are 310.061, 312.179, and 310.928 ms (310.928-ms median,
  within the 312.4-ms gate); task sums are 294.188--295.807 ms. Summed
  before/after frontend boundary work is 27.634 ms versus 29.436 ms before
  batching/predecoding. The reset-inclusive untraced median is 463.518 ms.

## 2026-08-12 — Task-scoped storage and predecoded boundary fast paths

- Added task-scoped batch storage acquisition and dematerialization operations.
  Arbitrary lifecycle rebinding retains complete runtime generation validation;
  the repeated task path consumes bindings already validated by the neutral
  runtime boundary and still rejects stale non-null addresses transactionally.
- Added direct allocation-owner identity to output adoption and a batched
  storage-adoption adapter. Forward output publication no longer performs an
  objects-by-outputs scan. The frozen Qwen control reduced forward output
  publication from 8.344 to 2.566 ms per step without changing the Program,
  schedule, task, or action identities.
- The admitted frontend record now predecodes forward output aliases, gradient
  groups, recurrent optimizer arguments, dematerialization targets, and
  ephemeral cleanup. The first optimizer component no longer rebuilds the
  complete optimizer tensor inventory. Lazy or custom first-step optimizer
  paths retain the generic dynamic fallback.
- Added detailed output-classification/adoption/state/accumulation timers and a
  separate reusable two-event selected-task bracket. The latter enables no
  native trace, callback, NVTX range, per-task event, allocator snapshot, or
  Python component timestamp and therefore measures the production path.
- Frozen evidence is
  `qualification/results/phase1/qwen35_predecoded_boundary_control.json`.
  Program `65300023...d7e3`, schedule `e349ce5f...f3ed`, 129 tasks, and 1,415
  actions remain exact. Its production-like selected-task samples are 291.914,
  295.083, and 293.029 ms (293.029-ms median), only 0.30% above the 292.141-ms
  standard-allocator authority. Detailed tracing measures 296.652--297.291 ms;
  the roughly 4-ms difference is observer overhead from three timing events per
  task and detailed host/native telemetry, not production execution.

## 2026-08-12 — Fused frontend/native task boundary checkpoint

- Added private C++ adapter operations that combine current-stream resolution,
  neutral-runtime acquisition, generation validation, and transactional storage
  rebinding into one frontend crossing. The matching completion operation
  combines planned-output adoption, dematerialization, and neutral-runtime
  publication. Generic lifecycle and first-step fallback paths remain intact.
- Admission now stores each input position's predecoded unique-object index and
  each unique object's first position. The neutral runtime fills bindings in
  two linear passes instead of rescanning every input position for every unique
  object.
- The exact frozen Qwen control is
  `qualification/results/phase1/qwen35_fused_boundary_control.json`. Program
  `65300023...d7e3`, schedule `e349ce5f...f3ed`, 129 tasks, and 1,415 actions
  remain unchanged. Production selected spans are 290.669, 291.453, and
  291.481 ms (291.453-ms median), versus the 292.141-ms standard-allocator
  authority. Detailed tracing measures 294.541--298.909 ms.
- In the detailed sample, median fused native-before cost is 26.8 us forward,
  77.4 us backward, and 10.2 us optimizer. Median native-after cost is still
  143.0 us forward, 93.4 us backward, and 18.2 us optimizer. Inspection shows
  that `after_task` still performs action allocation/validation and repeated
  allocation-population scans on the dispatcher. The next isolated change moves
  causally deferrable action work to a non-blocking progress-worker submission
  path while preserving exact action ordering and transfer triggers.

## 2026-08-12 — Generic memory leases and spill-role terminology

- Replaced the separate device-allocation and host-storage record concepts with
  one `ShadowSpillMemoryLease` owned by one generic `ShadowSpillMemoryPool`.
  A runtime currently instantiates an execution pool and a spill pool, while an
  object owns a runtime-sized array of per-pool locations. This representation
  does not assume that the spill pool is host memory.
- Unified prefetch destination reservation and allocation identity. A transfer
  action now creates one lease at causal reservation time and advances that
  same record through reserved, transferring, active, retiring, and free
  states. The former destination-offset plus later allocation-record split has
  been removed.
- Found a generic pool-relocation bug while growing the provisional pinned
  arena: existing spill leases retained pointers relative to the old arena
  base. Added an intrusive active-lease registry to every pool and made pool
  rebasing update every lease address atomically with the range geometry.
- Found an H2D metadata-completion race in the new location-array path. A later
  annotated release could clear an execution location before the worker
  processed the earlier copy completion; completion then dereferenced the
  cleared lease and could overwrite the later residency state. Completion now
  commits only when the object location still names the same lease and
  generation.
- Renamed the semantic secondary-memory role from `backing` to `spill`
  throughout current IR, planner, simulator, runtime, adapter, tests, and
  diagnostics. The canonical field is `retain_spill_copy`; physical host and
  CUDA terminology is confined to concrete backend edges. Runtime ABI is 12
  and the private PyTorch adapter ABI is 22.
- All 17 compiled/native/CUDA/PyTorch canaries pass with warnings as errors.
  The full Python suite passes after the intentional schema artifact update.
  Its Program and ExecutionPlan digests changed solely because the canonical
  serialized key changed; the schedule digest remains
  `332f8ada15b358d303efc2f3589c6646acb7e841093e421cde3d885740410bae`.
- Design update: transfer semantics belong to a directed pool-pair route. The
  runtime will calibrate every supported route during initialization and
  expose a dense latency/bandwidth matrix to the planner. CUDA device and
  pinned-host pools are merely the first concrete pair; CUDA peer, ROCm,
  remote-memory, and storage routes can provide the same interface later.

## 2026-08-12 — Explicit Runtime, route calibration, and generic profiler boundary

- Separated process-lifetime runtime initialization from planning. The PyTorch
  API now requires an explicit `Runtime`, an execution-pool name, and a
  spill-pool name. Pool budgets default to initialized capacities and reject
  overrides above those capacities before capture or model mutation.
- Added the optional `execution_device` keyword to `plan()` and
  `forward_pass()`. The default resolves PyTorch's current accelerator device;
  an explicit ordinal/device selects it. Either path rejects a mismatch with
  the selected execution pool before planning. `PlanReport` records the
  resolved ordinal.
- Added a dense, generation-tagged transfer-capability matrix. Runtime startup
  calibrates fetch and evict routes; users can recalibrate all or selected
  routes while locally idle, including concurrently across independently
  synchronized processes. A plan retains the exact immutable matrix snapshot,
  digest, provenance, timestamp, and selected fetch/evict profiles it consumed.
- Split physical arena ownership and directed transfer behavior into
  `ShadowSpillMemoryPoolBackend` and `ShadowSpillTransferRoute`. Pools remain
  generic; transfer direction and implementation belong to a source/destination
  route. The current two-pool adapter is a compatibility instantiation, not a
  planner assumption.
- Added a generic profiler vtable with a no-op neutral path. All NVTX use now
  lives in the NVIDIA backend profiler implementation; the neutral runtime and
  PyTorch storage adapter invoke opaque profiler callbacks. The worker is named
  `shadowspill_worker`, and the transfer streams are named
  `shadowspill_fetch` and `shadowspill_evict` for NSYS.
- Updated public canaries and qualification entrypoints so none relies on the
  removed behavior where `plan()` implicitly installed/configured the
  allocator. A training-canary expectation changed from two pinned allocations
  to one because the spill pool is now created once by `Runtime` and is not
  resized by planning.
- Validation: the CUDA/PyTorch build passes all 18 compiled canaries; the
  explicitly CUDA-disabled build compiles the neutral runtime, mock backend,
  simulator, and planner and passes all 10 neutral canaries. Ruff, strict mypy,
  and the complete Python suite pass. Runtime ABI is 13 and the private PyTorch
  adapter ABI is 23.

## 2026-08-12 — Public constructor and transfer-vocabulary closure

- Renamed the public constructors to `plan_step()` and `plan_forward()` and
  removed the old aliases. This makes the operation being planned explicit and
  leaves room for future non-step planning entrypoints without overloading a
  generic `plan` name.
- Closed the transfer terminology boundary: plans, runtime actions, lanes,
  counters, traces, simulator records, and qualification summaries use `fetch`
  and `evict`; configured pool roles use `execution` and `spill`. Physical copy
  direction names are retained only where a provider trace or driver API
  requires its literal vocabulary.
- Added an executable naming audit. It rejects the old secondary-pool and
  physical-direction terms in production code and rejects CUDA/ROCm/HIP names
  in the neutral IR, simulator, planner, and runtime. Historical evidence and
  concrete provider adapters remain deliberately outside that policy check.
- Removed direct Python NVTX calls from the training executor. Python task
  ranges now cross the private adapter through the generic profiler interface;
  the only NVTX implementation is the NVIDIA backend provider. The worker and
  transfer lanes remain named `shadowspill_worker`, `shadowspill_fetch`, and
  `shadowspill_evict` through the same provider abstraction.
- Runtime traces now expose `fetch`/`evict`, and the NSYS SQLite extractor maps
  provider copy labels back to those semantic directions before producing a
  ShadowSpill report.
- Runtime ABI is now 14 and the private PyTorch adapter ABI is 24. Both the
  CUDA-enabled warnings-as-errors build (18/18 compiled canaries) and the
  CUDA-disabled neutral build (10/10 compiled canaries) pass. Ruff and strict
  mypy pass, and the complete Python suite passes with five expected skips for
  fresh-process-only public accelerator tests.

## 2026-08-12 — Semantic transfer annotations and active worker passes

- Finding: transfer NVTX ranges were emitted inside the concrete CUDA copy
  function, where only physical direction and byte pointers were available.
  Consequently every range was the generic
  `shadowspill.runtime.transfer.fetch` or `.evict`, even though the immutable
  `ExecutionPlan` knew the alias, graph relationship, and semantic execution
  identities.
- Fix: admission now copies one preformatted semantic label into each ordered
  execution action. FETCH labels include alias bundle, object role, bytes,
  trigger execution, and next input consumer. EVICT labels include the same
  object metadata plus the latest output, mutation, last-input, or persistent
  source. The worker reads the immutable label directly around route
  submission; it performs no string formatting or object/task lookup. The
  backend's old generic nested range was removed.
- Isolated NSYS evidence is
  `qualification/results/semantic_transfer_annotations/overlap_semantic.nsys-rep`.
  It contains five distinct semantic transfer ranges (three EVICT, two FETCH)
  and zero exact generic transfer ranges.
- API decision: bounded diagnostics and provider annotations are orthogonal.
  Planned training calls now accept `runtime_trace=False` and
  `profiler_annotations=False`; forward calls expose the latter. Runtime trace
  controls `StepResult.diagnostics` and the seven task timestamps. Profiler
  annotations control the generic provider, including allocator, task, and
  transfer ranges. Both are disabled by default.
- Worker finding: the worker still entered `pthread_cond_timedwait` after every
  non-progress pass, including while actions or retirements were outstanding.
  That retained sleep/wake latency despite the completion and action handlers
  themselves being nonblocking. The hot loop now immediately repeats while
  work exists. It uses a one-millisecond condition wait only when both queues
  are truly idle, which avoids burning a core before work exists and provides
  a lost-wakeup safety bound. CUDA-event queries remain FIFO and adaptively
  throttled; this change does not restore the former query storm.
- Validation before model-scale qualification: the CUDA/Torch build passes all
  18 compiled canaries, all Python tests pass, and Ruff plus strict mypy pass.
  Runtime ABI is 15, profiler ABI is 2, and private PyTorch adapter ABI is 25.

## 2026-08-12 — Qwen control and completion-polling correction

- The post-change 30 GiB Qwen control retained Program digest
  `a83f56768b5de19ef2162844acb492fbb83e819693726e512ff83203d4ed4044`
  and schedule digest
  `6eec901b7293a1237496d8496d32f95cd8059a8750e6137537ae1c40e8a2d7e6`.
  Its production-like first-selected-task through final-optimizer span was
  290.328 ms median (296.686 ms in the detailed traced call), versus the
  312.4 ms gate. It executed 129 tasks and all 1,549 actions with 388 evicts,
  394 fetches, no callback failure, no trace overflow, and no allocator left
  blocked.
- The matching NSYS capture contains 782 distinct semantic transfer ranges and
  zero exact generic transfer ranges. It observed 1,351 ShadowSpill driver
  `cuEventQuery` calls totaling 0.749 ms, or about 0.55 us per query. This
  established that the former 223,208-query problem was harmful because those
  queries occurred beneath a shared runtime lock, not because a lock-free
  FIFO-head query is intrinsically expensive.
- Finding: the completion tracker still exponentially delayed an incomplete
  FIFO head. With the public 100 us base, successive checks occurred after
  200 us, 400 us, 800 us, 1.6 ms, 3.2 ms, 6.4 ms, and then as much as 10 ms.
  That can delay publication and dependent actions by milliseconds merely to
  save sub-microsecond queries.
- Fix under qualification: preserve FIFO head-only queries outside all object,
  pool, and action locks, but replace the exponential policy with a fixed 1 us
  default cadence. Runtime diagnostics now report the exact per-step event
  query delta. This is not considered accepted until the Qwen control, NSYS,
  and tight-budget correctness/checkpoint gates are repeated.
- A 2 us calibration control was retained as an intermediate measurement: it
  made 49,988 queries during the detailed call and reduced the selected-task
  median to 285.747 ms without changing either plan digest. The requested
  production candidate is 1 us; the interrupted 2 us NSYS attempt is not an
  accepted artifact.

## 2026-08-12 — Robust offline lowering and no-copy state replacement

- Added an immutable offline `TaskStorageContract` that derives task-output
  roots, views, aliases, and mutations solely from FX provenance, Export graph
  signatures, dispatcher schemas, and fresh symbolic geometry evaluation.
  FakeTensor storage IDs and allocator observations no longer determine task
  semantics.
- Added a separate `CompiledTaskLayout` that reconciles observed physical
  output allocations, offsets, workspace lifetimes, and provider growth with
  the already-fixed semantic contract. Physical evidence may reject a
  mismatch, but cannot merge or split semantic roots.
- Replaced duplicated forward/backward output binding with `ObjectCatalog` and
  `TaskBindingResolver`. Split-root FX topology now carries cross-stage value
  provenance directly, including explicit stage-local user-output and mutation
  projections.
- Export-functionalized state changes remain fresh compiled results. The
  runtime installs the result allocation as the canonical object's next
  generation, retires the prior generation behind the same task fence, and the
  private PyTorch adapter rebinds registered views. No `aten.copy_` or other
  numerical copy is inserted.
- Simulator and slab admission charge the exact physical replacement extent
  during the task's old/new overlap and replay the corresponding atomic alias
  generation change. `PlanReport.diagnostics` exposes the semantic contract,
  physical layout, replacement-transition bytes, all legal graph pairs, and
  the selected execution-task variant.
- A native delayed-transfer test proves that replacement remains causally
  correct while the old generation's fetch is still in flight, and that its
  storage cannot be reused before task completion. The fresh CUDA mutation
  canary executes three recurrent replacements, matches eager PyTorch, and
  restores final CPU state through checkpointing and close.
- Validation: all 19 compiled native/CUDA/PyTorch canaries pass; Ruff and
  strict mypy pass; and the complete Python suite passes. The runtime ABI is
  16 and the private PyTorch adapter ABI is 26.

## 2026-08-13 — Composable PyTorch planning and lowering

- Replaced stateful planning coordinators with explicit capture, profile,
  Program, PressureFit, admission, reporting, and cache artifact boundaries.
  `PlanReport` exposes the canonical Program and PressureFit result for
  direct budget sweeps.
- Reorganized PyTorch lowering around shared object-catalog, task-binding,
  physical-profile, compiled-layout, and Program-publication modules. Forward
  and training have symmetric partitioned-lowering packages whose
  `program.py` functions read as phase orchestrators and retain only their
  real mode-specific behavior.
- Removed obsolete whole-graph and mixed-authority lowering paths. Ruff
  complexity checks, strict mypy, the complete Python suite, and fresh-process
  CUDA forward/training canaries pass after the structural change.

## 2026-08-13 — Stage partitioning and graph-pair portfolio boundary

- Replaced the monolithic partition module with a focused `partition/`
  package. Its 43-line public orchestrator now only resolves a built-in or
  custom contiguous policy, splits Export, and constructs ordered `Stage`
  occurrences with boundary provenance. Forward task-ABI capture was moved
  out of partitioning, and the package has no AOTAutograd, graph-pair,
  profiling, cache, recomputation, or planner dependency.
- Defined the semantic levels explicitly: `Stage` is one topological partition
  occurrence; `StageExample` adds occurrence-local representative values; a
  structural task ABI additionally includes fixed geometry, layouts, aliases,
  mutations, static arguments, and differentiated roots. Equivalent repeated
  stage occurrences may therefore share compiled graph choices without
  conflating their initialized values.
- Added a separate `graph_pairs/` package and immutable
  `GraphPairPortfolio`/`GraphPairVariant` records. The current builder emits
  the established default-partitioner `save` choice and PyTorch's
  runtime-optimized min-cut `recompute` choice at an explicitly fixed `1.0`
  activation-memory budget. Persistence, diagnostics, lowering, task emission,
  and canonical Program construction now accept any ordered portfolio. A
  synthetic third `0.5` option proves downstream selection is no longer
  binary-coded.
- Pre-commit review found that the old min-cut call inherited Functorch's
  ambient `activation_memory_budget`, which is currently `1.0`; it was not a
  `0.0` full-recompute endpoint. The builder now fixes that existing budget
  explicitly so external configuration cannot change cache identity, and the
  default-partitioner choice correctly records no min-cut budget. This is a
  metadata/determinism correction, not a graph or schedule change.
- Consolidated generic cache policy and the artifact ledger in the sole
  `pytorch/cache.py`. Graph-pair, profile, and PressureFit persistence remain
  explicit typed repositories with their own schemas; no partitioning or
  lowering package contains a generic `cache.py`. Graph-pair cache schema v4
  persists every option ID, memory-budget fraction, reference option, and
  structural ABI.
- Validation: Ruff and strict mypy pass over all 104 source files; the complete
  Python suite passes; and all 20 compiled neutral/CUDA/PyTorch CTest canaries
  pass. The first CTest invocation exposed a stale build-tree configuration
  that embedded the base-Conda Python interpreter and failed before tests on an
  incompatible Torch private API. Reconfiguring that same tree explicitly with
  the pinned `shadowspill` interpreter made all eight PyTorch canaries pass.

## 2026-08-13 — PyTorch frontend package ownership cleanup

- Replaced the flat 36-module PyTorch package root with explicit capture,
  compilation, optimizer, materialization, execution, diagnostics, admission,
  and runtime-adapter packages. Public imports remain stable; private flat
  implementation paths were intentionally removed.
- `plan_forward()`/`plan_step()`, planned callable lifecycle, plan diagnostics,
  and step diagnostics now have separate focused modules. Runtime telemetry and
  compiler dependencies are explicit, with a side-effect-free compilation
  package initializer preventing an import cycle.
- Optimizer capture now reads as validation, isolated state discovery, and
  recurrent graph/opaque-task publication. Training execution predecodes an
  immutable `PlanRun` outside the repeated path. Parameter stage ownership now
  belongs to optimizer staging rather than graph-pair construction.
- Pruned the installed test-only objective-pair executor and the obsolete
  monolithic public module. Ruff, strict mypy, and the full PyTorch test suite
  pass; no planner, simulator, schedule, arithmetic, or runtime behavior was
  changed.
- Final caller audit removed three unused qualification-only timing forwarding
  methods from `PlannedTrainStep` and two dead executor collectors beneath
  them. All remaining private top-level functions and methods have at least one
  concrete source, test, or qualification caller; package `__all__` exports
  were also mechanically verified to resolve.

## 2026-08-13 — Readable PyTorch orchestration and profiling ownership

- Decomposed the remaining mixed-responsibility frontend implementations into
  focused capture, compiled-layout, diagnostics, execution-boundary,
  materialization, optimizer, admission, and runtime-adapter helpers. The
  public planning functions remain short linear orchestrators over immutable
  artifacts; forward and training share the same capture/profile/Program/
  PressureFit/admission shape without hiding mode-specific semantics.
- Created a dedicated `pytorch/profiling/` package. Stateless explicit-task
  compilation remains in `pytorch/compilation/`; representative values,
  profiling metadata, structural keys, device conditioning, CUDA-event timing,
  workspace telemetry, warmed executable ownership, manifest reconciliation,
  and measurement persistence now have explicit profiling owners.
- Finding: re-exporting the CUDA profiler from `profiling/__init__.py` caused a
  real import cycle because neutral runtime telemetry imports only immutable
  allocation-event records. Fix: the package initializer exports lightweight
  records/repository APIs only, while planning imports the profiler
  implementation explicitly. Runtime telemetry has no dependency on profiler
  execution.
- Preserved the earlier single-cache-policy decision. `pytorch/cache.py` is the
  only `cache.py`; profiling and compiled-manifest persistence are named typed
  repositories and receive their read/write/overwrite policy from planning.
- Split process-local executable ownership into `ProfileExecutableStore`.
  Compilation no longer owns representative CUDA values or profiling cache
  lifecycle, and the store eagerly drops occurrence-local values after each
  isolated measurement while retaining selected compiled code.
- Removed the obsolete compilation-owned profiling, representative-input, and
  profiling-metadata modules after updating every production, qualification,
  and test import. Architecture and frontend documentation now state the
  dependency direction explicitly.
- Validation: Ruff passes across source, tests, and qualification tools; strict
  mypy passes over 127 source files; the complete Python suite passes with only
  expected CUDA skips; and all 20 native/CUDA/PyTorch CTest canaries pass.
  No Program topology, PressureFit policy, transfer trigger, arithmetic, or
  runtime synchronization behavior was intentionally changed.

## 2026-08-14 — Explicit PyTorch state relocation

- Added a dedicated persistent-state frontend with model/optimizer relocation,
  externalization, runtime object IDs, exact spill-pointer validation, and
  zero-copy adoption into resolved execution plans.
- `relocate_model_state()` now returns a distinct module hierarchy whose state
  directly views runtime spill leases. `release_source=False` retains the input
  model; `release_source=True` retains no ShadowSpill reference, so assigning
  the result back to the same variable permits immediate source collection.
- Planning now requires the returned relocated model and never copies or
  releases model storage. Callable close restores spill-backed CPU bindings;
  `externalize_model_state(..., release_runtime=True)` is the explicit inverse.
- A 64 MiB RSS/ledger probe and fresh-process forward, mutation, training,
  transition, and relocation canaries confirm one spill object per unique
  storage, no duplicate spill bytes during planning, preserved ties/views and
  numerical values, source-lifetime semantics, and complete runtime release.

## 2026-08-14 — Source-releasing relocation is the default

- `relocate_model_state()` and `relocate_optimizer_state()` now default
  `release_source=True`. The opt-out remains available for callers that
  deliberately need to retain an independent anonymous source allocation.
- Public examples continue to spell `release_source=True` explicitly so the
  canonical ownership transfer remains visible at the call site.

## 2026-08-14 — Qualification tooling adopts relocated model ownership

- Audit finding: planned launchers relocated `case.model` into a local
  variable, but the case container still retained the original anonymous CPU
  model. That defeated `release_source=True` and duplicated model payloads.
- Added one qualification helper that returns a replacement case whose
  `.model` is the spill-backed result. All training-plan launchers now replace
  their case before planning and use the matching centralized externalization
  helper during teardown.
- Standard-allocator numerical references and offline FakeTensor-only compiler
  tools remain intentionally unrelocated. Focused ownership tests prove the
  old case-held model becomes collectible and teardown externalizes the model
  with `release_runtime=True` by default. A real CUDA helper round trip also
  confirms source collection, externalization, and clean runtime teardown.

## 2026-08-14 — Checkpoint compatibility documented explicitly

- The PyTorch frontend guide now documents the exact training checkpoint
  schema, active-callable restore path, and ordinary PyTorch model/optimizer
  restore path. It distinguishes the complete checkpoint from its conventional
  `checkpoint["model"]` state mapping.
- Public callable docstrings and the README now state that exported state is
  synchronizing ordinary CPU state and explain when direct model loading
  requires prior close/externalization. Application-owned RNG, scheduler,
  scaler, and data-loader state are explicitly outside the three-key schema.
- The frontend guide now also demonstrates background filesystem serialization
  while subsequent steps execute. It explicitly states that overlap begins
  only after the synchronous runtime-to-anonymous-memory snapshot finishes;
  only the later anonymous-memory-to-filesystem I/O is asynchronous. Snapshot
  tensors use pageable CPU memory outside ShadowSpill pool budgets and
  telemetry.

## 2026-08-14 — Planning failures are typed and rollback cannot orphan retirement

- Added distinct public planning failures for capture, compilation, profiling,
  physical admission, and PressureFit infeasibility. Compiler/profile failures
  retain structural ABI, task kind, operator inventory, and the original
  PyTorch exception through exception chaining. Allocator no-progress failures
  retain requested/free/largest-range accounting and active execution-task
  identity. Genuine CUDA faults remain their original provider exceptions.
- A realistic profiling OOM uncovered a teardown deadlock. The failed callback
  requested 2,147,483,648 bytes with 1,054,834,684 bytes free and a
  1,029,684,992-byte largest range. Tensor destruction after the latch created
  one task-local logical free. Immediately before plan clearing the runtime had
  one pending retirement, zero actions, and exactly one unfenced retirement.
- Root cause: `after_task` observed the existing `NO_PROGRESS` status, skipped
  normal fence publication, and then cleared the thread-local task scope. The
  subsequent abort call could no longer identify the owning task. Recovery
  cleared the latch, after which `wait_idle` waited for a retirement that had
  no event, fence, or queue record and therefore no possible progress source.
- Fix: failed task boundaries now publish a compute-stream fence for every
  task-local retirement before leaving scope. The explicit abort path publishes
  known stream events even when another failure is already latched. Recovery
  refuses to clear `NO_PROGRESS` unless each pending retirement has a complete
  queued record, converting any future invariant break into an immediate
  `INVALID_STATE` rather than an unbounded wait.
- A native canary reproduces the exact latch/free/failed-after-task sequence,
  proves premature recovery is rejected, then proves fenced recovery drains
  and a subsequent allocation succeeds. The CUDA planning-failure canary now
  raises the intended OOM, rolls back, and successfully plans/executes a later
  model on the same runtime; the complete failure canary finishes in 5.6 s.
- Pure-PyTorch OLMoE exposed a separate capture-boundary hole. Its FakeTensor
  objective-schema probe raises `DynamicOutputShapeException` at
  `aten.bincount.default` before strict Export begins, so the raw PyTorch type
  escaped. The probe now raises `CaptureError` from that exact exception,
  preserving the complete PyTorch model traceback and both planning-phase
  notes without synthesizing model-specific context.
- Failure-taxonomy correction: an irreducible task live set that exceeds
  execution capacity is a `PlanInfeasibleError`, but it is not a PressureFit
  search failure. Added a framework-neutral feasibility preflight over every
  legal recomputation selection. Public forward/training planning now runs it
  in a distinct `feasibility_preflight` phase before cache lookup or
  `pressurefit_simulation`; PressureFit retains the same floor assertion only
  as a defensive invariant for direct callers. The CUDA failure canary proves
  the constrained 520 MiB case carries the preflight note and never enters the
  PressureFit phase. Physical spill reservation, workspace/provider headroom,
  spatial replay, and pool sealing failures remain plain `AdmissionError`s.

## 2026-08-14 — Null representative allocation rejected before CUDA use

- The committed failure-handling baseline was exercised with the unchanged
  16 GiB execution / 112 GiB spill mlops Llama 3 8B launcher. Structural
  profile 13/16 reached optimizer ABI `3e5469863361...`; its next
  representative owner requested 117,440,512 bytes with 39,845,512 bytes free
  and a 39,744,768-byte largest free range. The allocator correctly latched a
  no-progress OOM.
- Remaining bug: `CUDAPluggableAllocator` returned a null pointer to
  `torch.empty`, but PyTorch did not raise at that construction boundary. The
  representative builder immediately launched `target.copy_(reference)` using
  the invalid target. The first visible error was `CUDA_ERROR_INVALID_VALUE`,
  followed asynchronously by an illegal-address fault that poisoned cleanup
  and aborted process finalization.
- Fix: representative-input materialization now checks the allocator's latched
  failure immediately after creating each alias-group owner, before building
  views or issuing any fill/copy/kernel. The common check preserves the precise
  allocator OOM diagnostic and is reusable by other explicit frontend
  allocation boundaries.
- The exact 8B rerun now raises `RuntimeExecutionError: ShadowSpill no-progress
  OOM` at the same 117,440,512-byte request and exits normally with status 1.
  It emits no dependent `copy_`, invalid-address error, cleanup warning, or
  process abort. This confirms failure reporting and teardown before optimizer
  task partitioning changes begin.

## 2026-08-14 — Lazy optimizer state preinitialized at exact step zero

- The clean no-progress OOM above was the expected pre-fix 8B result. Its
  failing structural ABI was a monolithic recurrent optimizer graph with 1,164
  fresh output roots. It existed only to discover lazy state on the first
  optimizer step; the recurrent Program already had 32 dependency-closed,
  stage-interleaved optimizer tasks.
- Lazy optimizer tensor state is now initialized exactly at step zero before
  structural compilation. Optimizers with an explicit per-parameter state
  initializer use it directly. Ordinary traceable PyTorch optimizers run their
  Python state-creation preamble through a one-use compiler boundary that stops
  before the numerical update. Parameter version counters prove that no update
  occurred. State that is itself a numerical graph output, such as SGD's first
  momentum buffer, keeps the existing distinct initial-plan fallback.
- This implements the previously selected single-recurrent-plan contract:
  `state_dict()` before step one exposes normal zero-initialized optimizer
  state, checkpoint/restore uses that same state, and the optimizer factory is
  still invoked exactly once. The change is optimizer-capability based rather
  than optimizer- or model-name based.
- The unchanged 8B Llama run no longer profiles the monolithic ABI. Structural
  optimizer ABIs fell from four to three, total profiles from 16 to 15, while
  the intended 32 stage optimizer tasks remain. All 15 profiles completed in
  9.005 seconds, proving the expected no-progress OOM was removed without
  weakening the 16 GiB physical cap.
- A separate planner-memory defect then surfaced: the native PressureFit path
  retained all 1,020 full compiled selection contexts concurrently. With the
  configured 112 GiB pinned spill arena, initialized workload state, and those
  projections, Linux killed the process at about 170.1 GiB RSS. This is not the
  optimizer fix and will be corrected separately with bounded context
  production/consumption while preserving portfolio order and tie-breaking.

## 2026-08-14 — Correction: duplicate optimizer ownership caused the host OOM

- The earlier attribution of the 170.1 GiB Linux OOM kill to retained
  PressureFit contexts was disproved by direct measurement. Reconstructing all
  1,020 compiled contexts from the saved 8B Program increased RSS by only
  1,010.3 MiB (145.9 MiB at 100 contexts, 521.6 MiB at 500, and 991.3 MiB at
  1,000). Context retention is not the dominant host-memory defect.
- The canonical Program contains 14.958 GiB of parameter storage, 29.915 GiB
  of optimizer state, and 30.458 GiB of gradients. The configured 112 GiB
  registered pinned spill arena already owned the authoritative model bytes,
  but freshly initialized optimizer state remained simultaneously resident in
  ordinary anonymous CPU storage while its spill copy was materialized later.
  That extra 29.915 GiB, together with the spill arena, compiler state, and the
  roughly 1 GiB context portfolio, mechanically explains the prior host OOM.
- The generic ownership fix relocates initialized optimizer storages directly
  into persistent spill objects immediately after optimizer capture and
  releases their anonymous CPU sources. Training materialization adopts those
  exact leases into plan aliases instead of copying them again. Close first
  exposes ordinary CPU optimizer state and then releases its persistent spill
  ownership; rollback abandons both representations deterministically.
- In the first unchanged 8B rerun, optimizer relocation took 2.012 seconds and
  process RSS immediately afterward was about 115 GiB. The run completed all
  15 profiles and all 1,020 context compilations while remaining near
  128.5 GiB RSS, confirming that the duplicate optimizer payload—not the
  expected pre-fix no-progress OOM and not a planner deadlock—was the dominant
  host-memory failure.
- The complete run finished rather than being killed. Native PressureFit
  evaluated 40,800 candidates across 1,020 contexts in 1,768.061 seconds;
  complete `pressurefit_simulation` took 1,770.407 seconds. Decoding and cache
  publication raised peak RSS to 150,589,696 KiB, still below the machine
  limit, and deterministic rollback released the registered arena and worker.
- The selected byte-feasible schedule was then rejected by exact slab replay.
  Forward/recompute `task_000191` first allocated a persistent
  1,050,673,152-byte output through the ordinary high-address path, then held
  two 525,336,576-byte anonymous extents before requesting a second
  1,050,673,152-byte anonymous extent. The slab had 1,644,592,960 bytes free,
  but its largest hole was 795,983,872 bytes. The compiled layout identifies
  the persistent output as allocation ordinal 21, so the next isolated fix is
  to admit immutable per-task output-placement hints and keep that output on
  the planned low-address side. This requires no copy and changes no
  PressureFit action or trigger.
- The optimizer spill-ownership implementation passes Ruff, strict mypy, all
  23 native/CUDA canaries, and the complete Python suite with five expected
  skips. It is therefore ready to commit independently of the spatial fix.

## 2026-08-14 — Task output placement made causal and allocation-exact

- Root cause of the post-optimizer `task_000191` spatial failure was a
  planner/runtime layout mismatch, not insufficient aggregate capacity. Exact
  profiling already identified the 1,050,673,152-byte persistent output as
  allocator callback ordinal 21, but both real execution and slab replay
  treated every task-created allocation as anonymous until output binding
  after the callable returned. The output was therefore placed from the high
  workspace side and split the free space needed by callback ordinal 28.
- Admission now records immutable allocation-placement hints per execution
  task. Each hint contains the exact nonzero allocator callback ordinal,
  requested byte count, and low/high spatial side. The dispatcher keeps the
  ordinal counter in its task-local C scope. A mismatch or missing callback is
  a plan violation; a matched hint changes only range selection. The lease
  remains anonymous until normal output promotion, so semantic ownership and
  failure cleanup are unchanged.
- Fresh persistent outputs use low-side placement while anonymous workspace
  remains high-side. Outputs that were measured reusing a task-local extent
  keep that exact reuse path and receive no fresh-placement hint. This avoids
  copies, changes no arithmetic or PressureFit directive, and makes exact slab
  replay use the same placement rule as steady-state allocation.
- Added a native execution canary proving callback ordinal 1 is placed below
  surrounding workspace, plus a 300-byte isolated replay where all-high
  placement fragments two 100/160-byte workspace lifetimes but output-low
  placement succeeds with a 210-byte peak. Ruff, strict mypy, all 23
  native/CUDA canaries, and the complete Python suite pass with five expected
  skips. The cached 8B selection is the next validation gate.

## 2026-08-14 — Correction: two-ended placement is not a completeness proof

- The low-output/high-workspace rule above was a useful diagnosis but not a
  sufficient admission contract. The authoritative 8B schedule has a
  15,263,695,044-byte aggregate device peak in a 15,315,501,056-byte slab,
  leaving roughly 51.8 MiB of byte capacity. Online best-fit nevertheless
  rejected a 1,050,673,152-byte request with 1,644,592,960 bytes free because
  its largest range was only 795,983,872 bytes. A rule choosing one of two
  ends can still make a locally plausible placement that fragments a later
  lifetime.
- Admission now derives complete allocation lifetimes from the selected
  schedule and the isolated per-task allocation traces, then deterministically
  packs those intervals offline. Every nonzero admitted allocator callback is
  identified by task-local ordinal and requested bytes and receives an exact
  static offset. Fetch destinations are reserved at their causal directive
  trigger, not at predicted on-wire time. The runtime fails closed on missing,
  extra, reordered, or resized callbacks; it never changes a PressureFit
  action or trigger to repair placement.
- Caller-owned forward/output allocations are the intentional exception to a
  fixed address. Admission reserves a non-overlapping high suffix for their
  maximum simultaneous footprint, while static plan lifetimes occupy the low
  prefix. Runtime allocates caller outputs dynamically within that suffix, so
  retained outputs remain valid across calls and eventually produce the
  documented no-progress OOM rather than colliding with the next call's static
  layout.
- Strict callback validation exposed a separate profiler defect in bounded
  eager optimizers. `deepcopy(Parameter)` drops `.grad`, so opaque AdamW
  profiling previously executed a no-op and recorded zero workspace while the
  real task allocated two sequential 4 MiB temporaries. Profiling now restores
  representative captured gradients on the copied Parameters; the corrected
  trace reports an 8 MiB high-water with two 4 MiB extents. Its cache identity
  is versioned only for opaque optimizers, preserving unrelated compiled-graph
  measurements.
- Hooked AdamW also exposed that side-effect-free lazy state initialization
  had been disabled merely because step hooks existed. Discovery now invokes
  the unwrapped step implementation, so hooks remain reserved for real
  optimizer calls while structurally initializable state is created at exact
  step zero. The real optimizer remains opaque and its hooks still execute
  exactly once per public step.
- All 23 native/CUDA canaries, the complete Python suite (five expected skips),
  Ruff, and strict mypy pass. The original pre-fix full-model fixture remains
  a required negative regression: it must raise the structured ShadowSpill
  no-progress OOM for the 117,440,512-byte request with 39,845,512 bytes free
  and a 39,744,768-byte largest range; it must not surface a generic CUDA or
  admission error.

## 2026-08-14 — Exact C PressureFit replay reduced from 1,768 s to 96 s

- Preserved the original 16 GiB mlops Llama 3 8B profiling failure as a
  negative fixture. Before step-zero optimizer-state initialization, the
  117,440,512-byte representative allocation had 39,845,512 bytes free and a
  39,744,768-byte largest range. Its required public result remains a
  task-attributed `RuntimeExecutionError: ShadowSpill no-progress OOM`.
- The positive post-optimizer fixture contains 1,056 Program tasks, 9,233
  aliases, 256 recomputation groups, 1,020 selection contexts, and 40
  candidates per context. The frozen C planner required 1,768.061 seconds.
  A representative context required 27.083 seconds before this pass.
- Profiling identified four mechanical costs rather than a change in planning
  policy: simulation scanned every task after every event; residency reduction
  repeatedly rediscovered the same spans and cut candidates; schedule creation
  repeatedly searched whole action populations; and each candidate reserved
  and copied arrays sized for the theoretical Cartesian action maximum rather
  than its actual actions.
- The implementation now maintains task-lane and active-task bit frontiers,
  indexes residency cuts while preserving the original score and tie order,
  carries a one-pass next-span frontier, uses trigger/rank masks for schedule
  placement, grows schedule/transfer storage to actual populations, and caches
  packed residency matrices. These are representation and lookup changes only;
  no PressureFit directive, trigger, recomputation alternative, or feasibility
  rule moved.
- The selected context now takes 0.969 seconds and exactly reproduces all 40
  candidate diagnostics plus the complete selected schedule. The complete
  1,020-context replay takes 96.004 seconds. It reproduces all 40,800 candidate
  diagnostics, the exact recomputation selection, all 23,648 actions, and the
  31,656,981,184-ns makespan. The schedule digest remains
  `dd1e52b8d82ac6006d262a7f3fa9b033f42d48a85a6f0a65551a86fc2b9ac8e8`.
- Peak replay RSS fell from 10,346,120 KiB before packed caching to between
  8,525,224 and 9,851,496 KiB in complete exact reruns. ASan passed. UBSan
  exposed a zero-count null `memset` in the new cut-active bitmap; explicit
  zero-count guards fixed it and the sanitized planner/simulator canaries now
  pass.

## 2026-08-14 — Full-model plan passes; checkpoint exposes a separate host peak

- The optimizer-fixed mlops Llama 3 8B program successfully completed exact
  planning and slab admission with a 16 GiB execution cap and a 96 GiB spill
  pool. All 1,020 recomputation contexts were valid, all 40,800 candidates were
  evaluated, exact PressureFit took 486.841 seconds in the memory-loaded live
  process, and the full `plan_step()` call took 585.542 seconds. The isolated
  direct PressureFit authority remains 96.004 seconds; this distinction is
  retained rather than describing the integrated memory-contention wall as an
  algorithmic regression.
- The selected plan predicts an 85.4 GiB spill peak and passed exact spatial
  admission. The process was subsequently killed while the qualification
  harness entered `training.state_dict()`, before its warm execution step.
  Peak RSS was 175,399,564 KiB. The 96 GiB fully committed pinned arena plus
  live compiler/runtime state and the ordinary CPU model-and-optimizer
  checkpoint copy exceeded practical host capacity.
- This is not the historical negative-control failure. The pre-fix 8B fixture
  must still surface the structured, task-attributed ShadowSpill no-progress
  OOM for a 117,440,512-byte execution-pool request with 39,845,512 bytes free
  and a 39,744,768-byte largest range. A Linux host OOM kill, admission error,
  or raw CUDA invalid-address error does not satisfy that regression.
- Qualification will use the smallest spill-pool cap that admits the unchanged
  plan with explicit margin, leaving host capacity for the documented ordinary
  CPU checkpoint. This changes neither the 16 GiB physical execution cap nor
  any PressureFit directive, recomputation choice, or task ordering.

## 2026-08-14 — Checkpoint restoration no longer manufactures device storage

- Repeating the full-model checkpoint with an 88 GiB spill pool avoided the
  Linux kill and exposed a second, deterministic defect after the CPU
  optimizer snapshot had been copied. `_restore_optimizer_host_only()` raised
  `Attempted to access the data pointer on an invalid python storage` while
  binding a newly allocated placeholder.
- The method was unnecessarily manufacturing a full-size CUDA allocation for
  every optimizer-state alias, binding it, dematerializing it, and queuing a
  release merely to restore CUDA device identity with a null pointer. The 8B
  optimizer inventory is about 29.9 GiB while the execution slab is 16 GiB;
  the loop could outrun asynchronous retirements and eventually receive the
  allocator's null no-progress result. The subsequent `data_ptr()` access hid
  that allocator cause behind the generic PyTorch storage error.
- CPU exposure never changes the neutral runtime object. It now retains each
  tensor's existing dematerialized CUDA view before temporarily assigning the
  CPU snapshot view, then restores that exact view in `finally`. This performs
  no device allocation, binding, action submission, or generation change.
- The fresh-process training canary now asserts that `state_dict()` changes
  neither CUDA allocation-callback nor free-callback counts. Numerical
  checkpoint/replay remains bitwise, and the focused public training tests and
  fresh-process canary pass.
