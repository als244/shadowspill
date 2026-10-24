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
  builders, and CPU selection of example/update orderings. Current work is
  standalone single-GPU profiling and allocation qualification; execution and
  caller/distributed migration remain pending.

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
