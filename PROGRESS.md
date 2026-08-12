# ShadowSpill Progress

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
