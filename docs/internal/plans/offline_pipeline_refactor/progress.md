# Implementation progress

## 2026-10-24 — preparation and offline pipeline

- User authorized commit/push, creating the branch, and autonomous implementation
  through the agreed order. No GPU reservations or experiment campaigns restarted.
- Chicago `master` is clean and pushed at `bec7950b`. Refactor work is on
  `codex/offline-pipeline-refactor` in the adjacent worktree, tmux
  `codex:offline-refactor`. MLOps is clean at `0274515`.
- Preparation validation: 100 CPU tests passed, two CUDA tests skipped. The frozen
  commit script was validated through complete/resumed runs in a disposable clone.
- Completed checkpoints: CPU-only artifacts/planning/reports, the three model
  builders, CPU selection of example/update orderings, standalone profiling,
  physical admission, and saved-plan execution for the qualified dense fixtures.
  Independent communication calibration and bounded checkpoint helpers also work.
  Current work is precision/provider qualification and normal caller migration;
  distributed support and final legacy removal remain pending.

### Findings

- Existing `ShadowSpillProgram` mixes structural IR with task runtime/workspace
  profiles. The new unprofiled Program needs structural task definitions without
  invented runtimes. Preserve the existing planner/simulator IR as the costed
  internal representation during migration.
- Existing search/simulation are CPU implementations, but the outer planning call
  requires ArtifactStore and always constructs an execution admission. The public
  CPU phase must work without the store, and estimates must never become execution
  evidence merely because a simulation succeeds.

### CPU milestone implementation and validation

- Added independent Program, Profiles, ProfiledProgram, Plan and SweepResult
  artifacts; one checksum-verified archive per result, atomic replacement.
- Added apply_profiles, plan_program, sweep_programs, CommunicationInfo and
  write_report. They use the existing native search/simulator, without a store.
- Shared structural IR validation now accepts task identities without costs;
  no zero-duration stand-ins are used for unknown computational work.
- Added semantic task/module names and alias source names to tables; HTML,
  SVG/PNG figures, CSV and JSON reports come from saved Plan data.
- Manual values are estimates. Changing timing preserves existing memory
  information; replacing memory without evidence resets it to estimated.
- Existing latency fields were metadata only. New host latency costs are
  explicit transfer phases, so search and simulation both charge them.
- Found eager imports through profiling -> planner -> step -> runtime bootstrap.
  Made the transitional exports lazy and the default search registration lazy.
  A fresh-process test blocks torch, mlops, runtime, PyTorch frontend, compilation,
  and ArtifactStore imports while loading, overriding, replanning and reporting.
- CPU checks: 513 tests passed across new offline tests and existing IR, planner
  and simulator tests. New modules pass focused strict mypy. Logs/scripts are in
  `validation/`. Subsequent minor typing/formatting cleanup is being rechecked.
- Local Python's editable finder overrides PYTHONPATH with the main checkout.
  CPU validation therefore uses `python -S` with explicit dependency paths,
  preserving the main environment and ensuring the new worktree is tested.
- Current limits: one logical rank/host transfers; logical workspace simulation
  only. No executable payloads or physical allocation certification yet. The
  report and Plan state that explicitly. GPU/distributed migration remains pending.
- Rechecked after consolidating MemoryBudgets: 513 passed. Fixed a missing
  explicit re-export exposed by the regression run; old callers now share the
  same budget class during migration.
- Visual inspection caught an empty memory-history panel: native simulation
  returns peaks, not snapshots. The report now shows honest peak/capacity bars;
  allocation-history reporting remains separate future work.
- Offline demo exercises five budgets with one diagnostic failure and four
  feasible plans, archive restoration, numeric CSVs and figures. It uses clearly
  labeled illustrative costs, not GLM measurements.

### Build audit (next milestone)

- Fake export, partition and Inductor compilation have reusable internal APIs.
  Current outer capture wrappers nevertheless require PlanMemory/Runtime just
  to obtain device/configuration/value sources. New builders will call the pure
  components directly and retain existing capture semantics.
- Current lowering reconciles compiler storage contracts with TaskMeasurement
  before emitting graph objects. This is the substantive build/profile coupling:
  separate compiler-resolved storage geometry from allocation/timing evidence,
  rather than supplying fabricated TaskMeasurements.
- Existing fake-model copying rejects meta tensors, despite needing only storage
  geometry. New build needs geometry-only model construction, preserving aliases.
- Compiler manifests contain output/root extents but not all anonymous scratch
  lifetimes. Unknown custom-op scratch must remain explicit until measurement.

### Build implementation (in progress)

- First offline milestone committed as `5b018465`; later implementation is on the
  same refactor branch. Main checkout remains untouched.
- Separated `TaskStorageGeometry` from measured `CompiledTaskLayout`. Forward
  lowering now needs compiler output geometry, not fabricated task measurements.
- Added meta-model capture, including tied storage, and complete positional and
  keyword call inputs. Existing capture/lowering checks: 159 passed with the GPU
  visible. Fake-CUDA autograd still checks for a live device; hiding CUDA caused
  device-index errors in existing autograd tests. Build may use the representative
  GPU as agreed; CPU profile application/planning/reporting remain independent.
- New `build_forward` exports, partitions, compiles and lowers without runtime
  pools or ArtifactStore. Graph identity and semantic module/parameter names are
  saved. New archives can contain opaque compiled blobs, deduplicated by checksum;
  CPU loading never deserializes or executes their code.
- Qualified PyTorch AOT task serialization with a new-process replay test, fresh
  Inductor/Triton cache directories, and compilation entrypoints replaced by
  failures. Restored stages matched direct PyTorch output. The test exercises
  nested kwargs, multiple stages, meta parameters, and structured outputs.
- Found AOT cache hits can bypass ShadowSpill compiler-metadata collection. Build
  now packages compiler facts with the code. Foreign entries lacking these facts
  are regenerated in a temporary build cache. Globally disabling caches is not
  suitable: PyTorch then drops the serialization metadata as well.
- Deriving scheduled buffer extents/lifetimes from Inductor wrapper records.
  Compiler-visible buffers are distinguished from opaque provider scratch and
  resource overhead, which remain explicit measurement requirements.
- Still pending: training builders, planning-time example/optimizer ordering,
  standalone measurement and physical admission, execution, caller migration,
  distributed support and final cleanup. A runnable compiled task is not yet an
  execution-admitted ShadowSpill Plan.
- Forward build checkpoint validation: 514 CPU artifact/planner/simulator tests,
  151 build/capture/lowering tests with the GPU visible, and focused strict mypy
  over 19 files passed. Fresh-cache numerical replay remains covered after adding
  compiler memory metadata. Nested archive tests verify blob deduplication and
  corruption detection; the CPU import-boundary test now carries opaque code.

### Training build and offline ordering

- Added `build_forward_backward` with a meta model, arbitrary args/kwargs,
  fixed `num_examples`, objective, partition, operator recomputation and gradient
  dtype controls. Gradients persist; there is no optimizer capture or state setup.
- Build retains compiled creator/accumulator forms for both save/recompute.
  CPU planning binds those forms to the requested example ordering using saved
  metadata. It does not import the capture/compiler stack or run another build.
- Numerical replay checks tied and untied two-stage models with a residual path,
  keyword targets and metrics. Save/recompute, three example traversals, and
  fresh/existing gradients all match PyTorch autograd. These are binding/code
  tests, not yet validation of the memory-managed Executor.
- The tied-weight test found a missing original-producer edge after intervening
  gradient mutations. Fixed by tracking creation and latest mutation separately.
  This matters when a single parameter contributes in more than one stage.
- A fresh CPU subprocess blocks framework/runtime/store imports, loads the saved
  training Program, plans three traversals and round-trips their reports/archives.
- Returned allocations used transiently by a particular binding contribute to
  its workspace. Compiler extents give a conservative simulation bound; measured
  allocation lifetimes can replace that bound without double-counting peaks.
- Effectful model mutations/RNG currently constrain example reordering. Broader
  ordering legality and distributed effects remain explicit later work.
- Next: optimizer capture/build_step and planning-time optimizer placement,
  followed by standalone measurement and physical allocation qualification.

### Full-step build checkpoint

- Forward/backward checkpoint committed and pushed as `0e17affc`; the earlier
  forward and CPU-only milestones are also on the remote refactor branch.
- Added `build_step` using the same capture/compile/lower pipeline, with explicit
  optimizer templates. State discovery and capture use storage-free copies;
  build does not populate the caller's optimizer state or allocate moments.
- Parameter, moment, counter and hyperparameter names/defaults travel with the
  Program. Nontraceable optimizers and state needing an unsupported first-use
  initializer fail explicitly instead of becoming process-local Python closures.
- Planning chooses tail or stage-interleaved updates from saved components.
  Read/write dependencies include aliased storage, all contributing examples,
  and tied parameters. An update cannot run before the last read of old weights.
- Compiled AdamW updates match eager PyTorch parameters, moments and counters
  over two invocations for both save/recompute and both update orderings, with
  and without tied weights. Original meta parameters/optimizer remain untouched.
- Full-step archives also pass CPU-only planning, restoration and report checks.
  General measurement, physical admission and Executor remain the next milestone;
  these numerical tests deliberately use an independent binding replay oracle.

### Ordinary-allocator measurement (in progress)

- Full-step checkpoint committed and pushed as `e3ac333d`.
- Added a bounded native observer using PyTorch's allocator trace callback. It
  records original allocation requests, frees, streams, device and pool identity
  without installing ShadowSpill's allocator or creating runtime pools. Overflow
  fails explicitly; raw pointers are normalized away before saving profiles.
- Public `measure_program` restores packaged tasks, prepares one stage pair at a
  time, uses actual forward-produced saved values for backward, times with CUDA
  events, and checks observed output roots/aliases against compiler contracts.
  Task definitions/digests do not change during measurement.
- Basic tests cover 37-byte requests versus native rounded segments, bounded
  telemetry, output views (including empty views), input mutations, anonymous
  retained memory, and private-stream work joined back to the task stream.
- Saved forward/backward and full-step Programs profile with the native
  allocator. A fresh-process test blocks build/runtime/store imports and compiler
  entrypoints, starts with empty compiler caches, and successfully measures the
  saved Program. Default stage coverage and explicit subsets are checked.
- Found a legacy convention: allocation contracts include output destruction,
  while the layout measurement's temporary trace excludes output frees. Keep
  both records, preserving complete lifetimes without weakening validation.
- Cross-allocator qualification now passes for every finalized task form of the
  two-stage AdamW fixture: native observations and runtime allocator observations
  agree on allocation contracts, workspace peaks and output/input aliases.
  This caught and fixed a charging error: the runtime aligns addresses to 256
  bytes but charges requested sizes; it does not round each extent to 256 bytes.
  Observed workspace totals alone do not qualify a Plan for execution; physical
  layout/admission and runtime/provider overhead checks are still required.
- Latest CPU regression: 516 artifact/IR/planner/simulator tests passed. New
  profiling modules pass focused strict mypy. Seven profiling tests pass,
  including fresh-worker restoration and cross-allocator qualification. Coverage
  currently establishes the small dense/optimizer cases, not all opaque providers
  or data-dependent allocation paths. Next: CPU physical layout and execution.

### CPU physical admission checkpoint

- Ordinary-allocator measurement is committed/pushed as `ae3e6ea7`.
- Complete saved allocation traces now drive physical placement entirely on CPU.
  Saved evidence is reconciled against the compiler storage contract before it
  reaches admission. Estimated memory remains simulation-only; inconsistent
  traces/layouts are rejected. Plan archives contain placement and admission
  facts and validate their program/schedule identities and total GPU capacity.
- Moved pure allocation/output projection out of PyTorch planning. A type-only
  runtime import also needed removal to keep this path framework-independent.
- Physical admission exposed a build metadata bug: in-place accumulated gradient
  outputs had been labeled new replacement storage. Input-alias updates now
  remain mutations. Numerical task behavior was already correct; the ownership
  declaration was not.
- Training tasks now use runtime-compatible internal IDs, stable across example
  traversal choices. Semantic task names remain separate and appear in reports.
  Caller-owned terminal outputs use dynamic leases outside the reusable arena.
- Validation: 516 CPU artifact/IR/planner/simulator tests and eight GPU build /
  measurement tests passed; fresh CPU workers block framework/runtime imports,
  round-trip physical Plans and reports, and check negative evidence cases.
  Focused strict typing and lint pass. Main checkout remains clean.
- Next: execute the saved Plan in a fresh worker using the existing runtime
  transfer/task boundaries, without build, profiling, calibration, or a model
  object. Then qualify repeated invocations, gradient persistence and bounded
  initialization before integrating Trainer and quickstart.

### Saved-plan Executor (in progress)

- Physical admission checkpoint is committed/pushed as `d1afcfb1`.
- Added a worker that restores finalized code, admits saved physical layouts,
  reserves spill storage directly and initializes/exports tensors in bounded
  chunks. Execution does no compilation, measurement or communication calibration.
- New native metadata-only CUDA storage lets tasks acquire existing runtime
  leases without allocating duplicate model storage. Runtime reservation now has
  an explicit uninitialized spill lease for the bounded loader to fill.
- Three-invocation comparisons cover forward, backward and AdamW state, retained
  caller outputs, and saved/recomputed stages. Fresh workers receive only Plan
  archives and input/state data. Compiler, capture, profiling and calibration
  entrypoints/imports are deliberately blocked.
- This stronger test found two old implicit dependencies: PyTorch package imports
  pulled capture/materialization into execution; the intrinsic gradient GEMM op
  was registered only through capture. Imports are now lazy, and the intrinsic
  registration lives independently of build. The profiling worker's import-block
  hook was also strengthened to run before importing measure_program.
- Tied parameters exposed an eager cross-stage gradient addition outside captured
  tasks. Build now captures the selective addition in the first-example backward,
  so its math/allocation/timing is part of the same compiled task everywhere.
  Updated tangent/prior-gradient binding handles those selective accumulators.
- Still in progress: runtime tensor hyperparameters, saved constants and initializer
  coverage, one-Plan fresh/existing gradients, retryable teardown and qualification
  beyond the small dense fixtures. These are not reported as complete.
- Validation checkpoint: 18 build/profiling/execution tests pass, including the
  strengthened cold-import worker tests and tied-parameter replay regressions.
  Focused typing and lint pass. Saved-plan execution remains a milestone under
  active development; the outstanding contracts above still block caller migration.

### Runtime controls and gradient invocation modes (in progress)

- Initial saved-plan execution checkpoint committed/pushed as `daeac9ca`.
- Executor is now exported from the small top-level API. Per-call tensor
  hyperparameters use their captured names; omitted values use captured defaults.
  Bounded CPU counters stay on the host, while device controls have planned
  storage. Unknown names/shapes are rejected before dispatch.
- Persistent tensor initialization validates coverage, supports scalar/array
  defaults and parameter-derived masters, and offers `restore=True` for a loader
  supplying complete saved state. Initialization/export use bounded spill views.
  Small lifted constant values travel in the Program; other constants are named
  loader inputs. Unresolved initialization must fail explicitly.
- Forward/backward Plans now contain fresh and accumulate paths. CPU planning
  derives both from the same Program/profiles, checks the shared arena capacity,
  and saves both paths in one archive with deduplicated code. Fresh calls retain
  creator kernels; they do not pay an unnecessary fetch of old gradients.
- Executor admits both paths into one GPU arena and shares persistent runtime
  object identities. Gradient storage is initialized to zero by default; a
  full-state restore can supply existing gradients before the first call.
- Numerical checks pass for fresh → accumulate → fresh calls, restored gradients,
  restored AdamW state, tied parameters, and changing tensor learning rates.
  CPU artifact/IR/planner/simulator regression remains 516 passing tests.
- Final lifecycle/cold-worker regression is running. Distributed execution,
  provider qualification, Trainer/quickstart migration and legacy cleanup remain.
- Additional output qualification exposed a pre-existing capture limitation:
  forwarding an input/parameter directly was rejected as not stage-produced.
  Build now inserts a profiled snapshot for returned input/state storage, retaining
  alias relationships among returned views. This implements caller ownership
  without exposing the reusable arena or allowing output edits to change weights.
  Literal CUDA tensors emitted by export are normalized to bounded host constants.
  The cold-worker alias/constant test passes, including output mutation and later
  calls. Mixed-dtype aliases of the same returned input storage remain an explicit
  unsupported case (caller can return explicit copies).
- Completed this checkpoint: 24 build/profiling/execution tests pass, including
  cold workers with disabled compilers/calibration, one shared arena for both
  gradient modes, restoring optimizer/gradient state, output aliasing, automatic
  invocation waiting and retrying close after releasing caller-held outputs.
  The independent CPU regression has 516 passing tests; focused typing/lint pass.
  Next is normal-caller migration, while keeping provider/distributed extension
  work explicit rather than claiming the dense qualification covers all backends.

### Independent communication calibration

- Executor/runtime-control checkpoint committed and pushed as `d18bbdd2`.
- Added public `measure_communication()` using ordinary pinned-host buffers and
  CUDA streams/events. No execution pools, model tasks, build, ArtifactStore or
  runtime initialization are imported. Solo rates/startup are fitted from a size
  sweep; concurrent directions use disjoint buffers. Raw observations, sample
  variability and fit errors remain inspectable in CommunicationInfo metadata.
- This measures one local GPU and current host placement. It does not claim a
  multi-GPU aggregate rate or complete distributed collective support. Those
  remain in the explicitly deferred distributed milestone.
- Validation: six numerical/failure fit checks and a cold-worker GPU calibration
  and archive round-trip test pass. Native allocation returns to baseline;
  focused lint and strict typing pass.
- Trainer migration is awaiting the user's answer on its public constructor:
  `Trainer(plan, ...)` with optional supplied Executor versus retaining the old
  model/prepare orchestration. Independent checkpoint work continues meanwhile.

### Bounded checkpoint helpers

- Added `save_checkpoint` / `load_checkpoint` above the Executor tensor API.
  A single atomic ZIP archive streams raw physical storage, retaining strided
  geometry, dtypes, CPU controls, checksums and caller JSON metadata. No full-model
  snapshot is assembled. Write/read chunk sizes can differ.
- Initial compatibility requires the same Program; planning budgets/order can
  change independently. This is full resume, not a Hugging Face/model-weight
  conversion layer. Application scheduler/RNG metadata remains caller-owned.
- Validation: eight cold-worker numerical replay cases now save after the first
  invocation, overwrite all persistent data, restore, and continue against the
  eager oracle. Four CPU tests check BF16/FP8/strided storage, bounded chunks,
  atomic failure, corrupted payloads and Program mismatch. Focused lint/typing pass.
- Continuing with precision qualification while the Trainer constructor question
  remains pending. Master-weight capture must migrate without changing compute
  or gradient dtype, and execution must initialize/export masters explicitly.

### Mixed-precision qualification

- `build_step(..., master_dtype=...)` now captures geometry-only master parameters,
  discovers optimizer state at that precision, and includes gradient casts and
  compute-weight refreshes in the compiled update. `grad_dtype` stays independent.
  Model compute weights and the supplied optimizer remain unchanged during build.
- Executor initializes masters from their named compute weights in bounded
  chunks and exports them explicitly as `master.<parameter-name>`. Full-resume
  checkpoints preserve both representations and CPU optimizer controls.
- Numerical tests cover BF16 compute weights with FP32 masters/moments and both
  BF16 and FP32 gradient storage. Three updates, including checkpoint overwrite /
  restore after the first update, match independent eager calculations.
- Current regression: **31** build/profiling/execution tests and **522** CPU
  artifact/IR/planner/simulator tests pass. Focused strict typing and lint pass.
  This still qualifies dense/local operators, not every MLOps/custom provider.
- Latest logs: `validation/precision-regression.log` and
  `validation/precision-cpu.log`. Design records are being tracked explicitly so
  the branch retains the agreed interfaces along with implementation progress.

### Operator registration portability

- Found that AOT custom-op call sites can depend on registrations performed only
  while importing the original model. Tasks now carry defining Python modules,
  source digests and schemas. The profile/execution worker restores and checks
  these before deserializing code. The dependency identity participates in task
  profile compatibility.
- Detection includes operators passed to higher-order/functionalized calls.
  Native/dynamic registrations and separately installed kernels require a future
  provider initializer; temporary dispatch overrides are rejected. This does not
  qualify transitive provider libraries, resource setup or internal autotuning.
- Consolidated forward task packaging onto the same CompiledTasks catalog used
  by training, removing duplicate identity/manifest/memory assembly.
- Cold-worker stateless custom-op execution passes, including checkpoint restore
  and rejection of a mismatched registration digest. Mixed-precision replay and
  forward-build regression also pass. The final combined build/profiling/execution
  regression is **33 passing tests**; the independent CPU suite remains **522
  passing tests**. Focused typing/lint pass. See `validation/portability-regression.log`
  and [validation commands](validation/README.md).
- Public Trainer construction remains the next review point. The pending dropdown
  recommends `Trainer(plan, ...)` with an optional supplied Executor; the other
  option retains model/prepare orchestration. No new constructor or distributed
  implementation was selected while that question remained unanswered.
