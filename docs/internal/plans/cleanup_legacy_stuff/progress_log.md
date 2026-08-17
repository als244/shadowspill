# Canonical Runtime and Legacy-Path Removal — Progress Log

This is an append-only implementation record. Findings, design corrections,
tests, measurements, regressions, fixes, and commits are added chronologically.

## 2026-08-17 — Accepted baseline

- Accepted plan copied to `implementation_plan.md`.
- Base revision: `519b6555584309209abd5f30f7a0eb75f31e70bc`.
- Isolated branch: `cleanup/canonical-runtime`.
- Isolated worktree: `/home/shein/Documents/grad_school/research/shadowspill-cleanup`.
- The active 2,520-point sweep remains in the original worktree on `main` and
  continues to use the `shadowspill` conda environment.
- The cleanup implementation will use the separate `shadowspill-cleanup`
  environment and separate build/cache/result directories.
- No production changes have been made yet.

## 2026-08-17 — Isolation and baseline gate

- Created worktree `/home/shein/Documents/grad_school/research/shadowspill-cleanup`
  on branch `cleanup/canonical-runtime` at the accepted base revision.
- Cloned the active environment to `/home/shein/miniconda3/envs/shadowspill-cleanup`
  and installed the cleanup worktree editable only into that clone.
- Verified environment isolation:
  - `shadowspill` imports from the original `shadowspill` worktree.
  - `shadowspill-cleanup` imports from the cleanup worktree.
- Verified the active planning sweep remained alive in the original worktree and
  continued to use the original `shadowspill` environment.
- Baseline Ruff: passed.
- Baseline mypy: passed for 175 source files.
- Baseline Python suite: 674 passed, 4 skipped in 33.34 seconds.
- Baseline C/CUDA CTest suite: 27/27 passed in 21.99 seconds.
- Baseline `git diff --check`: passed.
- Production worktree status remained clean; the accepted internal plan and
  progress log are intentionally ignored implementation records.

## 2026-08-17 — Compiled-only planner and simulator milestone

- Removed the installed Python simulator implementation and made public
  `simulate()` require the compiled simulator unconditionally.
- Moved the readable simulator to `reference/python/simulator`; production
  configuration can no longer select it and `record_timeline` is no longer a
  production argument.
- Replaced the mixed Python/C PressureFit orchestration with one compiled
  program-context path. Candidate outcomes, physical admission, schedule
  selection, and makespan now always originate in C.
- Found that the compiled program-context validator incorrectly rejected
  zero-alias Programs. Updated the C context, residency, and portfolio scratch
  allocations to represent zero aliases safely while retaining non-null ABI
  storage. Workspace-only Programs now use the same compiled path.
- Added `shadowspill_validate_pressurefit_program_context()` and planner ABI
  version 12. The C preflight reports workspace overflow, required object
  capacity, and missing initial residency with device, alias, boundary, and byte
  evidence. Python only maps this structured result into public exceptions.
- During implementation, found and corrected a diagnostic-initialization bug:
  `prepare_context()` cleared caller-initialized error sentinels. Sentinel
  initialization now occurs immediately after the function's own reset.
- Moved the readable Python PressureFit facts, residency reducer, action emitter,
  dense residency adapter, and candidate selector under
  `reference/python/pressurefit`. The wheel continues to package only
  `src/shadowspill`.
- Added a compiled-versus-readable PressureFit differential test. The retained
  three-layer training chain produced identical schedule digest
  `911b75db998e38218a6db1775879434b9bb7d2705195c1ede212f7e13d071ed6`
  and 148,000-ns makespan.
- Updated architecture, C API, and Python API documentation to state the
  fail-closed compiled production contract and explicit reference location.
- Milestone validation:
  - focused planner/simulator suite: 283 passed;
  - full Python suite: 672 passed, 4 skipped in 27.94 seconds (three obsolete
    production-timeline/fallback tests removed and one differential test added);
  - C/CUDA suite: 27/27 passed;
  - Ruff, strict mypy (169 installed source files), and `git diff --check` passed.
- Passing milestone commit: `e31c587` (`Make planner and simulator production
  paths compiled-only`).

## 2026-08-17 — Generic topology ownership clarification

- The runtime-owned pool and route registries will be shared by any number of
  independently admitted callables.
- Each callable will own an immutable plan handle containing its pool/route
  role binding, task handles, physical-layout slice, and retained object
  references. There will be no runtime-global active-plan role binding.
- Releasing one callable will release only its plan records and pool leases; it
  will not clear another callable or close shared runtime pools and routes.
- Logical objects may be retained by multiple plans. Direct object references
  and residency generations, rather than duplicate IDs or global plan state,
  govern safe cross-callable binding.
- Calls will own plan-scoped invocation handles and terminal completion events.
  Dispatch returns without waiting for device completion, allowing one Python
  thread to enqueue multiple independent callables before synchronizing an
  individual result.
- Immutable task handles will contain no mutable per-invocation retirement or
  submission state. Such state belongs to the invocation, so overlapping plans
  cannot overwrite one another's worker batches or completion ownership.
- Shared-object plan bindings will default to causal consistency. An explicit
  per-binding unordered policy may capture and consume the currently visible
  generation without a cross-plan freshness wait. It still retains the exact
  lease through consumer completion and preserves generation-checked state
  commits, preventing use-after-free and stale-completion publication even
  though the observed value may intentionally be stale or concurrently raced.

## 2026-08-17 — Backend component split

- Replaced the monolithic backend vtable with independent generic
  `MemoryPool`, directed transfer-route, synchronization, and profiler
  contracts. Runtime construction now copies explicit pool and route
  registries and unwinds partially created components in reverse order.
- Converted the CUDA backend, mock backend, PyTorch adapter bootstrap, and all
  native canaries to the component topology. Pool storage semantics remain in
  concrete backends; neutral runtime code sees only pool IDs and routes.
- Bumped the neutral runtime ABI to 29 and aligned the Python adapter ABI
  expectation.
- Evidence: warnings-as-errors build completed; all 27 CTest canaries passed,
  including 12 PyTorch/CUDA integration canaries (62.71 seconds total).
- This commit intentionally establishes component boundaries before moving
  execution/spill role selection from runtime-global temporary defaults into
  immutable per-plan bindings.

## 2026-08-17 — ABI terminology audit

- Reserve “ABI” for true independently compiled binary boundaries: the C
  runtime shared library, PyTorch adapter, and backend vtables.
- Rename task allocator evidence and its diagnostics from
  `TaskAllocationABI` to `TaskAllocationContract`; it describes observable
  execution behavior, not a binary calling convention. User-facing Python
  surfaces use “API,” while binary compatibility versions remain explicitly
  labeled “ABI.”

## 2026-08-17 — Plan-owned topology and execution records

- Added an explicit runtime-owned plan registry. Each `ShadowSpillPlan` now
  retains direct execution/spill pool pointers, fetch/evict route pointers, an
  independent execution table, and its own fixed-layout certificate.
- Moved transfer-lane FIFOs into each directed route. Admitted actions retain
  their plan and resolved route directly; the worker no longer selects a
  runtime-global fetch or evict lane when dispatching an admitted action.
- Moved immutable execution records and physical-layout metadata out of the
  runtime and into their owning plans. Allocation callbacks resolve fixed
  placement from the active execution handle's plan.
- Added a focused native canary proving that two plans may share one runtime,
  admit the same dense task ID, resolve distinct handles, and execute without
  table collision. The same canary constructs three pools and four routes and
  validates a plan selecting the non-default spill pool and route pair.
- The legacy runtime-global entry points temporarily delegate to one internal
  default plan so existing PyTorch qualification remains runnable during the
  conversion. These wrappers and the default plan are explicitly scheduled
  for deletion after the adapter adopts explicit plan handles.
- Validation: warnings-as-errors build passed; all 28 native/CUDA/PyTorch
  canaries passed; `git diff --check` passed.
- Passing structural commit: `09a0394` (`Add plan-owned runtime topology
  bindings`).

## 2026-08-17 — Plan-local and shared-runtime object ownership

- Added an explicit per-plan object-binding table. A binding maps one
  Program-local object key to a retained runtime object and records either the
  default causal policy or the explicitly unordered policy.
- Execution admission now resolves inputs, mutations, and actions exclusively
  through that plan-local table. It retains direct object pointers in immutable
  execution records; repeated execution performs no runtime object-table lookup.
- Preserved both identities where their responsibilities differ: fixed-layout
  placements and action certificates retain `plan_object_id`, while object
  leases, generations, readiness, and failure diagnostics use the runtime
  object's identity.
- Updated fixed-layout validation and dependency resolution to compare
  plan-local identities rather than accidentally comparing runtime-global IDs.
- Added explicit PyTorch-adapter entry points for plan lifecycle, object
  binding, execution admission, fixed-layout admission/sealing, clearing, and
  handle resolution. The adapter ABI is now 37.
- Added a native canary proving that equal plan-local IDs in two plans can bind
  different runtime objects, and that rebinding one local ID inconsistently is
  rejected.
- Found a transitional lifecycle defect during the warm training canary: the
  default plan cleared execution records but retained its binding table, so a
  later planning call found a detached object under a reused local key. Plan
  clearing now releases execution records first and bindings second.
- Validation: warnings-as-errors build passed; all 28 C/CUDA/PyTorch canaries
  passed; the full Python suite passed after adding the new adapter symbols to
  its declarative signature fixture and API references; `git diff --check`
  passed.
- Passing structural commit: `a6af4d7` (`Bind plan objects to shared runtime
  ownership`).

## 2026-08-17 — Explicit PyTorch plan ownership and initial actions

- Converted Python planning and callable ownership from one implicit runtime
  plan to explicit native plan handles. Multiple completed callables may now
  remain attached to one runtime without clearing each other's execution
  records.
- Runtime bridges now preserve shared runtime-object identities separately
  from Program-local aliases. Output promotion allocates a runtime-global
  identity without incorrectly invoking the idle-only persistent-state import
  guard.
- Corrected action fixed-layout lookup to use the action's plan-local object
  key while native trace and failure records continue to expose the shared
  runtime-object identity.
- Updated transfer diagnostics to reconcile simulated plan-local aliases with
  native runtime-object IDs through the bridge binding, rather than assuming
  both identity domains were numerically equal.
- Root-caused a warm-replan failure: model materialization still submitted
  release actions through a legacy fake-task/default-plan path. Closing the
  first callable removed that accidental default plan, so the second planning
  call returned `INVALID_STATE`. Materialization now admits reusable,
  plan-owned action-only records before submitting each release; no fallback
  path is used.
- Validation: warnings-as-errors build passed; all 28 C/CUDA/PyTorch canaries
  passed serially; Ruff, mypy, and `git diff --check` passed; the full Python
  suite passed with four expected skips.

## 2026-08-17 — State terminology and pinned shared objects

- Accepted `import_*` for moving PyTorch state into runtime ownership and
  `export_*` for moving it back out. Relocate/externalize terminology and
  compatibility aliases will be removed exhaustively.
- Accepted reference-counted `SHARED` object residency. `SHARED` means
  guaranteed resident and read-only while shared. The runtime charges its
  lease once, while each PressureFit plan excludes that object from its
  movable set and reports the external shared footprint and remaining
  capacity. Mutating a `SHARED` object fails admission.

## 2026-08-17 — Shared-residency policy refinement

- Replaced the provisional single `SHARED` policy with two explicit policies;
  there is no `SHARED` or `PINNED_READ_ONLY` compatibility alias.
- `SHARED_READ_ONLY` retains one guaranteed-resident lease and rejects any
  declared mutation at admission.
- `SHARED_WRITABLE_UNORDERED` retains one guaranteed-resident lease, accepts
  only stable-address in-place mutation, and adds no inter-callable value
  dependency. This deliberately permits stale or concurrently changing reads
  without weakening lease lifetime, generation validation, or allocator
  safety.
- Replacement mutations are rejected for the unordered policy because
  changing the lease would invalidate another callable's stable binding and
  obscure ownership. Shared bytes remain charged once to the physical pool
  while excluded from each plan's movable-object decisions.

## 2026-08-17 — Persistent-state import/export rename

- Replaced every public, internal, native-operator, test, qualification, and
  documentation state-ownership name based on relocate/externalize with the
  canonical `import_*` and `export_*` APIs. No compatibility alias remains.
- Renamed the persistent-state canary and native PyTorch operators to
  `_import_cpu_storages` and `_export_cpu_storages` and clarified the internal
  flag that distinguishes a separate frontend storage copy from a direct
  runtime-backed storage view.
- Validation passed: warnings-as-errors build, 28/28 C/CUDA/PyTorch canaries,
  full Python suite, documentation tests, Ruff, strict mypy over 169 installed
  source files, `git diff --check`, and an old-symbol source audit.
- Passing commit: `4e293f7` (`Rename persistent state APIs to import and
  export`).

## 2026-08-17 — Task-allocation contract terminology

- Renamed `TaskAllocationABI` and every associated Python file, C type,
  ctypes field, runtime status, diagnostic key, test, canary, and document to
  `TaskAllocationContract`. The serialized contract begins at schema v1 and
  old allocation-ABI artifacts are intentionally unsupported.
- Retained “ABI” only for actual independently compiled boundaries such as the
  runtime shared library, adapter, and backend vtables.
- Validation passed: warnings-as-errors build, 28/28 native/CUDA/PyTorch
  canaries, full Python suite, Ruff, strict mypy over 169 installed source
  files, `git diff --check`, and an allocation-ABI old-symbol audit.
- Passing commit: `45cd5d8` (`Rename task allocation evidence to contracts`).

## 2026-08-17 — Indexed compiled representations

- Replaced ambiguous “dense” representation and identity terminology with
  explicit indexed types and operations: `IndexedProgram`,
  `IndexedMemorySchedule`, `IndexedExecutionPlan`, `index_program()`,
  `index_memory_schedule()`, and `index_execution_plan()`.
- Applied the same rename to C planner schedule types and helpers, compiled
  admission bindings, ctypes projections, reference implementations,
  diagnostics, tests, and documentation. Task and object numeric values are
  now described as plan-local indices rather than dense IDs.
- The only remaining production occurrence is the literal upstream PyTorch
  operator name `_local_scalar_dense`, which ShadowSpill must recognize
  verbatim.
- Validation passed: warnings-as-errors build, 28/28 native/CUDA/PyTorch
  canaries, full Python suite, planner/simulator/IR focused suites, Ruff,
  strict mypy over 169 installed source files, and `git diff --check`.
- Passing commit: `b7c5752` (`Rename compact planner representations to
  indexed`).

## 2026-08-17 — Structural-contract terminology

- Replaced non-binary uses of “ABI” with explicit contract or signature
  terminology across capture, graph-pair construction, compilation,
  profiling, execution entrypoints, diagnostics, cache manifests, tests, and
  documentation.
- Renamed public diagnostic fields such as `structural_abi_key` and
  `aot_unique_stage_abis` to `structural_contract_key` and
  `aot_unique_stage_contracts`. Compilation and profiling errors now expose
  `structural_contract`.
- Renamed execution entrypoint `abi_digest` to `contract_digest`, advanced the
  strict execution-plan wire schema to `shadowspill.execution_plan/v2`, and
  advanced graph-pair cache identity to `shadowspill.aot_graph_pair/v7`.
  Historical field names and cache formats are intentionally unsupported.
- Retained “ABI” only at real independently compiled boundaries: the runtime,
  planner, simulator, PyTorch adapter, profiler, and backend vtables.
- Validation passed: full Python suite with four expected skips, focused IR
  and PyTorch suites, Ruff, strict mypy over 169 installed source files, and
  `git diff --check`.
- Passing commit: `1c29fd0` (`Rename structural identities to contracts`).

## 2026-08-17 — Shared-residency IR contract

- Added the explicit alias-group policies `SHARED_READ_ONLY` and
  `SHARED_WRITABLE_UNORDERED`; no generic `SHARED` or legacy compatibility
  spelling exists.
- Distinguished residency sharing from object consistency. The read-only
  policy rejects mutations. The unordered writable policy accepts only the
  IR's stable-address mutation relation; replacement outputs are rejected for
  both policies.
- Made runtime ownership fail closed: shared aliases cannot appear in
  recomputation-retained sets, initial/final schedule residency, or memory
  actions. Schedule validation treats them as runtime-resident inputs.
- Advanced the strict Program schema to `shadowspill.program/v2`, added a
  lossless indexed policy code, refreshed the frozen canonical digest, and
  updated the IR, runtime, JSON, and public API documentation.
- Validation passed: full Python suite with four expected skips, focused IR,
  planner, simulator, and documentation suites, Ruff, strict IR mypy, and
  `git diff --check`.
- Passing commit: `5daa652` (`Define shared residency policies in program
  IR`).

## 2026-08-17 — Shared-residency physical accounting and callable composition

- Added `SHARED_WRITABLE_CAUSAL` as the only shared binding policy allowed to
  publish task outputs. `SHARED_READ_ONLY` accepts existing inputs only;
  `SHARED_WRITABLE_UNORDERED` permits stable-address in-place mutation only.
  The same runtime object may therefore be a causal producer binding in one
  Program and a read-only consumer binding in another.
- Kept the neutral ownership model tensor-free: a runtime object identity has
  generation, readiness, and per-pool leases. The planned PyTorch convenience
  layer will expose `TensorRef`, `StateRef`, `SharedOutput`, and `SharedInput`
  while C continues to expose only objects and object handles.
- Established the prefill/decode contract. Users select a declared output
  pytree path; lowering resolves its semantic storage roots without guessing
  model semantics. A follow-up plan binds the same object references through
  explicit `SharedInput` declarations.
- Projected shared aliases out of the compiled simulator's movable alias set,
  subtracted their physical execution/spill footprints before simulation and
  PressureFit, and restored those baselines in decoded peak diagnostics.
  Physical admission receives only callable-attributable capacity and cannot
  assign a duplicate placement or transfer action to a shared alias.
- Added `PlanReport` views for shared aliases, shared execution/spill bytes,
  and residual callable budgets. Advanced the strict Program and frozen IR
  schemas to v3; no old-schema reader was retained.
- Deliberately rejected a separate activation-snapshot subsystem. Users may
  expose diagnostic intermediates as ordinary callable outputs; this keeps
  the runtime and PressureFit contracts small.
- Validation: the complete Python suite passed with four expected skips;
  focused IR, planner, simulator, admission, fixed-layout, artifact, and
  documentation suites passed; Ruff, targeted strict mypy, and
  `git diff --check` passed.
- Passing commit: `776074c` (`Account for shared runtime residency`).

## 2026-08-17 — Imported-state identity is the sharing boundary

- Rejected an additional `exclusive`/`shared` scope option on
  `import_model_state()`. It would duplicate an ownership concept already
  represented exactly by the returned Python object's runtime-storage
  identity.
- One import call creates one independent runtime-state identity. Passing that
  same returned model to several plans explicitly requests shared storage;
  performing separate import calls creates independent runtime objects and
  bytes.
- No plan silently clones a model merely to obtain exclusive ownership, and no
  equal-valued model is treated as shared by inference. Concurrent frontend
  execution may create separate lightweight tensor views while retaining the
  same neutral runtime-object handles.

## 2026-08-17 — Runtime-global object ownership and retained forward outputs

- Added opaque, reference-counted runtime object handles. Registration, every
  plan binding, and every public reference are independent owners; removing a
  registration or closing a callable no longer destroys an object still used
  by another plan or public reference.
- Added framework-neutral `ObjectRef` and PyTorch `TensorRef`. `TensorRef`
  contains view geometry and one exact residency generation while the neutral
  handle remains tensor- and backend-independent.
- Added path-based `SharedOutput` declarations to `plan_forward()`. Lowering
  resolves public pytree paths to semantic storage roots, rejects partially
  shared alias bundles, and retains declared outputs without a caller copy.
- Established a bounded recurrent-slot contract. A live reference prevents a
  later invocation from overwriting that slot. After the reference closes,
  the executor releases only its exact prior generation and the next compiled
  output is published into the same logical runtime object.
- Root-caused an initial recurrence failure: invocation validation removed the
  closed-reference record before generation cleanup, so the old execution
  lease remained bound and the next promotion failed. Validation now only
  checks ownership; generation release owns the sole state-clearing step.
- Kept output-slot ownership conflicts outside fatal execution cleanup. A
  rejected invocation leaves the callable reusable after the outstanding
  reference closes.
- The public surface exports only the working output declaration and reference
  types. Symmetric shared-input declarations remain internal until plan-time
  binding and execution have end-to-end coverage.
- Focused native coverage proves generation-mismatch rejection, exact
  generation release, and final-owner reclamation. The CUDA forward canary
  proves a live-reference guard, stable logical identity across recurrence,
  generation replacement, callable-independent public lifetime, and final
  reclamation.

## 2026-08-17 — Zero-copy shared forward inputs

- Added the symmetric `SharedInput`/`shared_input()` frontend contract. Planning
  uses deterministic task-local representatives for compilation and profiling,
  while callable materialization binds a lightweight tensor shell directly to
  the referenced runtime object. Execution performs no caller copy and creates
  no duplicate logical object.
- Moved causal versus deliberately unordered binding policy into the neutral
  `ObjectConsistency` value. The runtime bridge now consumes only
  framework-neutral `ObjectRef` values; dtype, shape, stride, and storage-offset
  handling stays in the PyTorch sharing layer.
- Preserved alias geometry by constructing one profiling owner per runtime
  object and reconstructing all declared views. Integer and Boolean control
  inputs require an explicit authentic profiling value rather than a synthetic
  `{0, 1}` fallback.
- Root-caused the first producer-to-consumer planning failure: physical
  admission treated a runtime-resident shared input as a missing plan-owned
  initial allocation. Admission now validates shared roots as externally
  resident and assigns them no callable offset or replay operation.
- Added plan-owned consistency retention so a later task-admission pass cannot
  accidentally rebind an unordered object with the default causal policy.
- The CUDA canary keeps producer and consumer plans alive simultaneously,
  proves that an execution-resident shared input incurs no fetch, closes the
  first generation, overwrites the same logical object with a successor
  generation, and executes the already-admitted consumer against that new
  generation. Final ownership reclamation remains exact.
- Validation passed: 680 Python tests with four expected skips, all focused
  documentation/lint/type checks, the complete forward CUDA canary, and the
  native/CUDA lifecycle suite. Ruff, strict mypy over 176 installed source
  files, and `git diff --check` passed.
- Passing behavior commit: `26d69b1` (`Bind shared outputs as callable inputs`).

## 2026-08-17 — Task-local retirement ownership

- Root-caused a population-dependent task boundary: same-stream allocator
  frees and functional object-generation replacement both left unfenced leases
  in the runtime-global active-lease list. `after_task()` counted and attached
  those retirements by scanning that entire population twice.
- Added one intrusive task-local retirement chain owned by the dispatcher
  scope. Ordinary frees and replacement-style mutations enter the same
  tracking API. A lease reused within the task stays linked once; the boundary
  acts only on its final state.
- The first conversion hung the functional-mutation transition canary because
  replacement retires the previous generation directly rather than through
  `shadowspill_free()`. The old global scan had hidden that second producer.
  Explicit tracking at replacement publication fixed the missing completion
  source without restoring a scan.
- `after_task()`, allocator-pressure fence publication, and abort cleanup now
  traverse only retirements created by the active task. Population size no
  longer affects these operations.
- Validation passed: all 28 native/CUDA/PyTorch canaries and the complete
  Python suite (680 passed, four expected skips).

## 2026-08-17 — Handle-only frontend task boundaries

- Removed the Python raw-task fallback from both forward and training
  execution. Every repeated compiled task now owns one admitted native task
  handle and enters the runtime through the same fused storage boundary.
- Renamed frontend execution-handle state to task-handle state and renamed the
  private PyTorch storage operators to `_before_task_storages` and
  `_after_task_storages`. The old private operator names were deleted rather
  than retained as aliases.
- Folded output adoption, functional replacement, frontend-view rebinding,
  dematerialization, completion publication, and action submission into one
  native after-task call. The bridge's standalone output promotion,
  replacement, and raw task APIs are no longer used or exposed.
- The first fused mutation implementation exposed a real ordering race: it
  published the new object generation and its actions before Python rebound
  persistent parameter views. A fast worker could retire the new generation,
  closing the previous-generation validation window and producing
  `existing storage does not match the retired object generation`.
- Corrected the boundary transaction to perform `adopt replacement → validate
  and rebind every persistent frontend view → dematerialize → publish the task
  action batch`. The logical object and Python tensor identities remain stable;
  only the object's generation and backing lease change. The obsolete separate
  `_replace_storages` operator was deleted.
- Validation passed: all 28 native/CUDA/PyTorch canaries, including the
  functional-mutation canary; the complete Python suite (680 passed, four
  expected skips); focused execution tests; Ruff; strict mypy over the PyTorch
  source tree; and `git diff --check`.

## 2026-08-17 — Atomic worker submission acknowledgement

- Replaced the worker's timed condition-variable idle path with one
  continuously active, `cpu_relax`-based loop. The thread remains named
  `shadowspill_worker` and performs a short ordered pass over newly published
  batches, completion frontiers, retirement reclamation, and outstanding
  actions.
- Added a single release/acquire submission slot plus monotonic submission and
  acknowledgement sequences. An action-bearing `after_task()` publishes its
  immutable admitted record and spins until the worker has observed the batch.
  Fetch acknowledgement additionally requires the route wait, asynchronous
  copy, completion event, and object readiness event to have been published;
  it never waits for wire completion.
- Made fetch route submission stream-causal instead of host-polling the task
  trigger. The route lane waits on the already-published task event, so an
  immediate consumer can insert its readiness wait without a dispatcher sleep.
- Removed the timed readiness wait from admitted `before_task()`. Encountering
  an unpublished fetch after the preceding task was acknowledged is now an
  immediate invariant failure rather than a latent condition wait.
- Root-caused the first queued release-to-fetch failure to the compatibility
  action-only path's invocation value `0`: the initial
  `completed_generation == 0` was incorrectly accepted as proof of submission.
  A completed generation now counts only after that action has become inactive;
  an active queued action can never satisfy acknowledgement.
- Updated two transition/failure canaries that intentionally encoded the old
  behavior. They now require all fetches to be submitted before boundary return
  and require backend submission failure to propagate directly through
  `after_task()` without a readiness-waiter thread.
- Validation passed: warnings-as-errors build; all 28 native, CUDA, and PyTorch
  canaries; the complete Python suite with four expected skips; Ruff; strict
  mypy over 177 source files; and `git diff --check`.

## 2026-08-17 — Dedicated profiling allocation scopes

- Removed structural profiling's fake execution-task boundary. Profiling now
  opens a dedicated allocator-attribution scope, executes the isolated
  callable, and closes that scope against the actual compute stream.
- The neutral allocation-scope API cannot resolve execution records, publish
  mutations or outputs, decode actions, or enter a plan. It exists solely to
  apply the production allocator's causal retirement rules to isolated
  compilation and profiling work.
- Decoupled allocation-telemetry shutdown from task-scope shutdown. Telemetry
  now owns only capture state; the allocation scope owns its own lifetime and
  retirement fence.
- Scope completion records a backend event only when the scope actually
  retired allocations. Abort finalizes only the active non-execution scope and
  cannot accidentally close an admitted task.
- Updated the PyTorch adapter, ctypes boundary, profiler, native telemetry
  canary, tests, and C API references around the new contract. No compatibility
  alias was retained.
- Validation passed: warnings-as-errors rebuild; all 28 native, CUDA, and
  PyTorch canaries; the complete Python suite with four expected skips; 44
  focused allocator/compiler/profiling tests; documentation contract tests;
  Ruff; strict mypy over 177 source files; and `git diff --check`.

## 2026-08-17 — Dedicated initial-action and caller-acquisition handles

- Removed the two remaining fake frontend task boundaries. Initial placement
  now uses one admitted action-batch handle and caller return uses one admitted
  ordered object-acquisition handle.
- Action-batch submission opens no Python task pair. One call publishes its
  predecoded actions and returns after worker submission acknowledgement, not
  transfer completion. The handle is explicitly rejected by task resolution.
- Caller acquisition retains direct object pointers, deduplicates aliases on
  the cold path, snapshots each current generation once, inserts only
  published readiness-event waits, and expands bindings without opening an
  allocator scope or calling `after_task()`.
- Root-caused the first plan-adoption failure: public output objects had been
  registered lazily by their first producer, which is too late for cold handle
  admission. Plan adoption now registers those already-known outputs as empty
  placeholders. Execution publishes the produced lease into that same logical
  object; there is no copy or replacement object.
- Extracted the shared object-snapshot/wait loop so task input acquisition and
  public-object acquisition use one implementation. Acquisition handles are
  plan-owned and are cleared before plan object bindings.
- Added a native contract canary proving action handles are not task handles,
  immediate fetch-to-acquisition readiness, duplicate expansion, and exact
  handle lifetime.
- Validation passed: warnings-as-errors build; all 28 native, CUDA, and
  PyTorch canaries; the complete Python suite with four expected skips; focused
  execution tests; documentation tests; Ruff; strict mypy over 177 source
  files; and `git diff --check`.

## 2026-08-17 — Single-call task admission

- Added the canonical task API: one cold `plan_admit_task()` operation copies
  the immutable topology and returns its direct handle. Production no longer
  admits and then performs a second task-table lookup.
- Renamed the repeated neutral and adapter boundaries to
  `before_task_handle()` and `after_task_handle()`. The fused PyTorch storage
  operators call these names directly, matching the frontend orchestration.
- Kept the execution record internal: public code sees a task handle, while
  action batches and object acquisitions remain distinct handle types and are
  rejected by task operations.
- The old exported names remain only for canary migration and are not called
  by installed Python production code. Their deletion is the next compatibility
  cleanup milestone.
- Validation passed: warnings-as-errors build; all 28 native, CUDA, and
  PyTorch canaries; forward, mutation, and training handle-path gates; focused
  Python boundary tests; Ruff; strict mypy over 177 source files; and
  `git diff --check`.

## 2026-08-17 — Canonical runtime lifecycle

- Removed the lifecycle forwarding translation unit and its private
  `_legacy` declarations. Runtime creation, idle waiting, spill-pool resizing,
  close, and destroy now each have exactly one implementation in `runtime.c`.
- This is a structural deletion only: pool-role validation, default-plan
  compatibility, teardown ordering, and backend behavior are unchanged so they
  can be migrated and qualified independently.
- Validation passed: warnings-as-errors build; all 28 native, CUDA, and
  PyTorch canaries; the complete Python suite with four expected skips; Ruff;
  strict mypy over 177 installed source files; and `git diff --check`.

## 2026-08-17 — Strict current planning artifacts and diagnostics

- Removed annotated-plan schema-v1 acceptance and the old aggregate timing
  reconstruction. Deserialization now requires schema v2, complete per-attempt
  PressureFit/admission timings, and reconciled timing totals.
- Removed the flat PressureFit-diagnostics reader that reconstructed
  recomputation contexts from selection strings, including its synthesized
  `legacy_selection_*` identities. Candidate records likewise require the
  current nested policy/outcome/repair/work shape.
- Removed flattened diagnostics compatibility aliases. Callers now inspect the
  explicit recomputation-context and candidate-evaluation hierarchy; repair
  totals are read from their categorized repair diagnostics.
- Added fail-closed tests for old annotated-plan, timing, candidate, and
  PressureFit-diagnostics shapes.
- Validation passed: the complete Python suite with four expected skips; the
  focused planner, artifact, admission, and documentation suites; Ruff; strict
  mypy over 177 installed source files; and `git diff --check`.

## 2026-08-17 — Handle-only task boundary and explicit plans

- Removed every raw task-ID boundary and the duplicate “execution handle” API.
  The only production task operations now admit a plan-owned task handle and
  pass that same handle to `before_task`, `after_task`, or `abort_task`.
- Made abort handle-bound and validated against the calling runtime, owning
  plan, and active task scope. Runtime destruction retains one private
  current-scope cleanup helper; it is not a second public task API.
- Deleted the hidden runtime default plan and all default-plan fixed-layout,
  admission, resolution, and clear wrappers. Every canary now constructs its
  plan, bindings, action batches, and task handles explicitly.
- Deleted the old raw boundary implementation, including its runtime-global
  mutex, readiness condition wait, population scans, repeated action-array
  construction, and separate private boundary header.
- Migrated the PyTorch adapter, bridge, storage guard, forward/training
  executors, canaries, API references, and physical-admission documentation to
  the single handle path.
- Validation passed: warnings-as-errors native build; all 28 native, CUDA, and
  PyTorch canaries; the complete Python suite with four expected skips;
  documentation tests; Ruff; strict mypy over 177 installed source files; and
  `git diff --check`.
- Passing structural commit: `a2e01e6` (`Remove raw task and default-plan
  APIs`).

## 2026-08-17 — Current-only trace and benchmark evidence

- Removed the NSYS extractor's old `shadowspill.task.<phase>.task_*` parser.
  Task extraction now requires the semantic
  `shadowspill.pytorch.task.execution_*.<semantic-name>` label and correlates
  every boundary segment with that same execution identity.
- Removed frontier-summary reconstruction of incomplete point records from
  request and case sidecars. Aggregation now requires the current complete
  point record and rejects missing embedded request or case identity.
- Replaced compatibility fixtures with current semantic ranges and explicit
  fail-closed coverage for incomplete benchmark records.
- Validation passed: focused NSYS, benchmark, and documentation suites; Ruff;
  strict mypy over 177 installed source files; and `git diff --check`.
- Passing structural commit: `aae4156` (`Remove legacy trace and benchmark
  readers`).

## 2026-08-17 — Canonical task terminology

- Renamed the admitted topology from `ExecutionDescription`,
  `ExecutionRecord`, and `ExecutionTable` to `TaskDescription`, `TaskRecord`,
  and `TaskTable` across the neutral C API, PyTorch adapter, bridge, tests, and
  documentation.
- Renamed `execution_table.c` to `task_table.c`, the internal execution-scope
  entrypoint to task-scope entry, and plan cleanup to `clear_tasks()`.
- Renamed task-local update/action records and the plan's task table while
  retaining “execution” only for genuine concepts such as the execution pool,
  execution sequence, and measured execution timing.
- Validation passed: warnings-as-errors build; all 28 native, CUDA, and
  PyTorch canaries; the complete Python suite with four expected skips; Ruff;
  strict mypy over 177 installed source files; and `git diff --check`.
- Passing structural commit: `264cf44` (`Use task terminology for admitted
  records`).

## 2026-08-17 — Memory-pool-owned allocation state

- Moved lease ownership, active/reusable indexes, requested/live/peak
  accounting, waiter counts, and lock-free geometry snapshots out of the
  runtime-wide execution allocator and into each generic `MemoryPool`.
- Replaced the role-specific allocation/free/lookup/stream-recording and pool
  growth entry points with APIs that require an explicit pool ID. Allocation
  telemetry and structured failures now record that pool identity.
- Made allocation IDs and residency generations runtime-wide atomics so
  independent pools retain globally unambiguous diagnostic identities without
  sharing allocator metadata or locks.
- During the source audit, found that an allocator waiting in one pool could
  mistake a pending retirement or capacity action in another pool for usable
  future progress. Added per-pool progress counters and made the retirement
  worker reclaim through the lease's owning pool. This preserves immediate
  no-progress failure when only an unrelated pool can change.
- Added construction coverage for a one-pool runtime with no routes, a
  three-pool runtime with a sparse route graph, rejection of a plan missing its
  reverse route, and real allocate/free/retirement through the third pool.
- Runtime-global execution/spill statistics and object-location helpers remain
  explicitly transitional; the next milestone replaces them with pool- and
  plan-owned views rather than presenting this commit as complete topology
  generalization.
- Validation passed: warnings-as-errors build; all 28 native, CUDA, and
  PyTorch canaries; the complete Python suite with four expected skips; Ruff;
  strict mypy over 177 installed source files; and `git diff --check`.
- Passing structural commit: `e7124e1` (`Move allocation ownership into memory
  pools`).

## 2026-08-17 — Plan-owned object residency

- Changed object registration to name an explicit initial pool and changed the
  neutral object read/write APIs to require a pool ID. Object teardown now
  visits every owned location instead of assuming pools zero and one.
- Routed acquisition, projected action state, mutation publication,
  destination reservation, retirement, worker dispatch, and worker completion
  through the immutable execution/spill pools retained by the owning plan.
  These repeated paths no longer consult runtime-global pool roles.
- Strengthened the three-pool canary from a construction-only check into an
  end-to-end alternate-pair test: it imports a payload into pool 2, fetches it
  over route 2 into pool 0, consumes and evicts it over route 3, reads it back
  from pool 2, unregisters it, and proves the complete pool-2 range is reusable.
- Bumped the runtime ABI to 33 and PyTorch adapter ABI to 41 for the changed
  object description and neutral C object APIs.
- One first parallel validation run reported a non-reproducing native-canary
  segmentation fault. The affected canary then passed 20 isolated repetitions,
  and ten consecutive complete eight-way parallel suites passed. No failure
  state, core file, or deterministic path was found, so no speculative behavior
  change was made.
- Validation passed: warnings-as-errors native build; all 28 native, CUDA, and
  PyTorch canaries; ten repeated complete parallel canary suites; the complete
  Python suite with four expected skips; Ruff; strict mypy over 177 installed
  source files; and `git diff --check`.
- Passing structural commit: `49396cb` (`Route object residency through
  plan-owned pools`).
- Remaining transitional surface is now isolated to cold object publication,
  caller handoff, snapshots, and role-shaped aggregate statistics. The next
  milestone replaces publication with immutable task-owned records and then
  removes those runtime-global helpers.

## 2026-08-17 — Task-owned object publication

- Added immutable publication records to admitted tasks. Each record retains
  a direct runtime object, its plan-local identity, and whether the task binds
  an available logical object or replaces only its physical lease/generation.
- Replaced repeated output publication by runtime object ID and integer mode
  with `(task handle, publication ordinal, allocation address)`. Forward,
  backward-gradient, and optimizer outputs carry their ordinals directly from
  predecoded execution records through the C++ storage transaction.
- Registered and plan-bound every possible output during cold admission. The
  repeated path no longer creates objects, parses IDs, hashes object IDs, or
  binds new plan objects after compiled execution.
- Moved cold materialization onto
  `shadowspill_plan_publish_initial_allocation()`, which resolves the
  plan-local object once and publishes into that plan's selected execution
  pool without impersonating an execution task.
- Preserved the simple public invariant: publication overwrites the same
  logical object. Replacement publication changes an internal physical lease
  only when the prior generation must remain causally protected; it does not
  copy the payload or replace public object identity.
- Added a native two-plan canary proving that equal plan-local publication
  ordinals resolve to different directly retained runtime objects. Real
  device-backed public forward and training tests also passed through the new
  bind and replacement paths.
- Bumped the runtime ABI to 34 and PyTorch adapter ABI to 42 for the admitted
  publication schema and new handle-based adapter functions.
- Validation passed: warnings-as-errors native build; all 28 native, CUDA, and
  PyTorch canaries; the complete Python suite after documenting the new public
  C functions (four expected skips); focused real public forward and training
  execution; Ruff; strict mypy over 177 installed source files; documentation
  tests; and `git diff --check`.

## 2026-08-17 — Acquisition-owned caller handoff

- Replaced repeated caller handoff by runtime object ID with
  `(acquisition handle, object ordinal)`. The admitted acquisition owns a
  direct retained object reference and its plan-selected execution and spill
  pools, so handoff no longer performs object-table lookup or assumes the
  runtime's first two pools.
- Made pointer, generation, and allocation-ID validation part of the same
  locked object transaction that changes ownership. A stale caller is rejected
  before the lease or object is modified; there is no validate-then-commit race
  through a second raw-ID API.
- Kept public logical identity stable. Caller handoff detaches the current
  physical lease only because its lifetime becomes caller-owned; it neither
  copies the payload nor creates a replacement logical object.
- Added native coverage proving a stale generation fails without partial
  ownership transfer, followed by a successful handoff of the same acquired
  ordinal and complete caller-lease reclamation.
- Bumped the runtime ABI to 35 and PyTorch adapter ABI to 43 for the admitted
  acquisition handoff entry points.
- Validation passed: warnings-as-errors native build; all 28 native, CUDA, and
  PyTorch canaries; the complete Python suite with four expected skips; the
  real public forward path; Ruff; strict mypy over 177 installed source files;
  and `git diff --check`.

## 2026-08-17 — Single storage-install path

- Deleted the raw-ID `_rebind_storage` and `_rebind_storages` operators and
  their adapter-side object snapshot validator. Admitted task/acquisition
  handles now establish object identity and generation exactly once; the
  storage adapter only installs the validated addresses or dematerializes the
  selected storages transactionally.
- Converted cold/caller binding and native integration canaries to the same
  `_acquire_storages` and `_dematerialize_storages` primitives used by the
  canonical frontend. Generation mismatch rollback remains tested at the
  handle-owned publication and handoff transactions, where ownership can
  actually change.
- Removed 261 lines of redundant lookup, validation, and compatibility code.
  The PyTorch adapter ABI is now 44.
- Validation passed: warnings-as-errors native build; all 28 native, CUDA, and
  PyTorch canaries; the complete Python suite with four expected skips; Ruff;
  strict mypy over 177 installed source files; and `git diff --check`.

## 2026-08-17 — Plan-only PyTorch object publication

- Deleted the raw PyTorch adapter entry points for bind, replacement,
  promotion, and caller transfer by runtime object ID. Initial materialization
  now publishes only through an immutable plan/object binding; repeated
  publication uses an admitted task ordinal; caller transfer uses an admitted
  acquisition ordinal.
- Converted the low-level allocator and overlap canaries to construct the same
  plan-owned records as public forward/training execution. Test-only callers
  no longer keep a second production API alive.
- The remaining caller-allocation release function is intentionally physical:
  it is the DataPtr deleter for an allocation that has already left logical
  runtime object ownership.
- The PyTorch adapter ABI is now 45.
- Validation passed: warnings-as-errors native build; all 28 native, CUDA, and
  PyTorch canaries; the complete Python suite with four expected skips; Ruff;
  strict mypy over 177 installed source files; and `git diff --check`.

## 2026-08-17 — Remove raw neutral object publication

- Removed the public neutral-C bind, replacement, and caller-handoff functions
  that accepted runtime object IDs. The only public publication operations are
  now plan-local initial publication, task-handle publication ordinals, and
  object-acquisition ordinals.
- Kept one internal object-component implementation for binding and replacing
  a physical lease. Both initial publication and repeated task publication call
  those helpers after resolving a directly retained object and explicit pool;
  there is no duplicated transition implementation.
- Converted legacy transition canaries to plan-owned initial publication and
  explicit task replacement descriptors. Caller-handoff tests now admit and
  acquire object handles before transferring ownership.
- The first test-helper draft passed a null output binding to the canonical
  initial-publication API. This caused immediate `INVALID_ARGUMENT` failures
  and left one telemetry scope open during cleanup. The helper now supplies a
  local ignored binding when the test does not inspect it; no runtime behavior
  was changed to accommodate the test.
- One eight-way canary run subsequently left the planning-failure process
  stalled. The identical fresh-process canary completed in 6.4 seconds, and a
  complete repeated parallel run passed it in 14.52 seconds. No persistent
  failure or changed runtime state was found.
- The runtime ABI is now 36.
- Validation passed: warnings-as-errors native build; all 28 native, CUDA, and
  PyTorch canaries; the complete Python suite with four expected skips; Ruff;
  strict mypy over 177 installed source files; and `git diff --check`.

## 2026-08-17 — Direct zero-copy handoff records

- Removed the `after_task()` scan over every active execution-pool lease and
  the object-ID handoff chains that scan consumed. Its sole invariant was that
  a compiled output reusing another logical object's lease must release the
  prior logical owner in the same admitted task.
- Task admission now builds a sorted direct map from retained object pointer
  to its predecoded release action. Output publication performs one direct
  lookup, records the exact lease pointer and generation on that action, and
  fails before changing either object when the release is absent.
- Replaced each lease's numeric bound-object identity with a stable direct
  object reference. The worker validates the generation-matched handoff lease
  directly and never hashes a runtime object ID or walks a logical-owner
  chain. Chained zero-copy publication remains valid even while earlier
  release fences are pending.
- Added native coverage proving a missing admitted release produces
  `PLAN_VIOLATION` at publication with no partial binding, in addition to the
  existing one-hop and chained handoff cases.
- Validation passed: warnings-as-errors native build; all 28 native, CUDA, and
  PyTorch canaries; the complete Python suite with four expected skips; Ruff;
  strict mypy over 177 installed source files; and `git diff --check`.

## 2026-08-17 — Non-sleeping pool-capacity progress

- Removed the `MemoryPool` condition variable and every capacity-path
  `pthread_cond_wait`. Successful range release or causal range handoff now
  advances a monotonic atomic capacity epoch.
- Foreground allocator and task-action reservation paths retain their existing
  priority declaration, release the pool lock, and actively poll the epoch
  with `cpu_relax` while the always-active worker reclaims or coalesces ranges.
  They continue to inspect the latched failure and worker-stop state on every
  pass; no sleep, futex, or scheduler yield was introduced.
- Split task-retirement fence publication so event creation, backend event
  record, and completion submission occur outside the pool lock. Only the
  final lease-dependency attachment and retirement-queue publication reacquire
  pool ownership.
- The first focused run returned transient `OUT_OF_MEMORY` immediately because
  the new polling branch retained the failed probe status instead of entering
  its progress state. Resetting that local status before polling fixed the
  defect; delayed cross-stream reclamation and injected worker-query failure
  now both exercise and pass the epoch path.
- Validation passed: warnings-as-errors native build; all 28 native, CUDA, and
  PyTorch canaries; the complete Python suite with four expected skips; Ruff;
  strict mypy over 177 installed source files; and `git diff --check`.

## 2026-08-17 — Fine-grained output publication

- Removed the runtime-global mutex from task output publication. Binding now
  snapshots the lease's direct prior owner, locks the target and prior owner in
  deterministic pointer order, then validates and commits beneath only the
  plan-selected pool lock.
- Changed internal bind and replacement helpers to accept the compiled output
  pointer directly. This removes the prior pointer-to-allocation lookup followed
  by a second allocation-ID lookup; replacement performs one pointer-index
  lookup, while zero-copy bind performs one snapshot lookup and one protected
  revalidation because it must discover which admitted release record owns the
  handoff.
- The repeated path performs no runtime object-table lookup and never calls a
  backend while holding an object or pool lock. Public logical object identity
  remains unchanged; only the generation's physical lease is rebound.
- Validation passed: warnings-as-errors native build; all 28 native, CUDA, and
  PyTorch canaries; the complete Python suite with four expected skips; Ruff;
  strict mypy over 177 installed source files; and `git diff --check`.

## 2026-08-17 — Recycled-generation retirement race (`056b772`)

- A repeated native lifecycle run exposed an intermittent worker SIGSEGV in
  `shadowspill_memory_pool_try_lock_reclamation()`. The worker had detached an
  incomplete retirement for generation N; same-stream reuse recycled the same
  lease record for generation N+1, whose retirement could be processed first.
  Releasing N+1 cleared the mutable `lease->pool`, after which the stale N
  record dereferenced that field before it performed its generation check.
- Retirement records now snapshot their stable pool owner and allocation ID at
  enqueue time. The worker locks that runtime-owned pool first and then checks
  that the lease still names the same pool and generation. Stale records only
  release their retained event references.
- The best-fit canary now keeps its intended retirements pending explicitly,
  rather than asserting an asynchronous count against a zero-delay backend.
  Fifty repeated perturbed-heap runs and the complete 28-canary suite passed.

## 2026-08-17 — Runtime-owned neutral event records

- Found that the device backend already reused timing-disabled driver events,
  but the neutral runtime still allocated and freed one C event wrapper for
  every task, transfer, and retirement fence. This violated the steady-state
  host-allocation contract even though driver-event growth was already zero.
- Added a runtime-owned event-record pool. Cold physical sealing reserves its
  complete free inventory; hot acquisition pops one record, and final release
  returns it. Once sealed, exhaustion fails closed instead of falling back to
  `calloc`.
- Explicit cold reserve calls may grow the inventory for another callable
  sharing the runtime. The CUDA backend's matching explicit seal call now also
  grows its free driver-event inventory when necessary, while ordinary event
  creation remains forbidden from growing a sealed pool.
- Added runtime diagnostics for capacity, current/peak use, and rejected hot
  growth, plus native coverage for cold growth and reuse across 320 retirement
  fences. Runtime ABI 37 and PyTorch adapter ABI 46 describe the extended
  statistics and reserve function.

## 2026-08-17 — Intrusive completion-frontier records

- Removed the heap-allocated completion node created for every task, transfer,
  and retirement fence. A neutral event lease can belong to exactly one FIFO
  completion frontier during one lease generation, so it now owns its queue
  link and diagnostic object/allocation identities directly.
- Submission retains the pooled event lease, links it under the completion
  tracker lock, and fails closed if the same lease is submitted twice.
  Completion atomically unlinks the lease before releasing the tracker and
  polling references. No completion record is allocated or freed on the hot
  path.
- Validation passed: warnings-as-errors native build; all 28 native, CUDA, and
  PyTorch canaries; the complete Python suite with four expected skips; Ruff;
  strict mypy over 177 installed source files; and `git diff --check`.
