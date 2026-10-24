# Implementation progress

## 2026-10-24 — preparation and offline pipeline

- User authorized commit/push, creating the branch, and autonomous implementation
  through the agreed order. No GPU reservations or experiment campaigns restarted.
- Chicago `master` is clean and pushed at `bec7950b`. Refactor work is on
  `codex/offline-pipeline-refactor` in the adjacent worktree, tmux
  `codex:offline-refactor`. MLOps is clean at `0274515`.
- Preparation validation: 100 CPU tests passed, two CUDA tests skipped. The frozen
  commit script was validated through complete/resumed runs in a disposable clone.
- Current milestone: CPU-only artifacts, profile application, planning, persistence,
  simulation, and reports. Later milestones remain pending.

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
