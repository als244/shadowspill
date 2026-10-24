# Refactor implementation order

Status: approved by the user; autonomous implementation authorized. The user explicitly prioritizes
working offline planning, then a working single-GPU pipeline, with distributed
execution and sharding implemented last. Development runs on Chicago in the
`codex` tmux session, in the `offline-refactor` window.

## 0. Prepare Chicago master before creating the branch

- Update the existing checkouts while preserving local changes.
- Prepare and validate a commit script for remaining local changes; do not mix
  them into refactor commits.
- Require a clean ShadowSpill `master` before creating
  `codex/offline-pipeline-refactor` from its updated tip.
- Preserve the design notes currently in the detached refactor worktree. Its
  premature branch was removed after confirming it contained no new commits.

Completed preparation: ShadowSpill fast-forwarded to `5aa561e3`, MLOps to `0274515`.
The overlapping baseline README merged cleanly. All other local ShadowSpill
files match the protected pre-update snapshot. MLOps is clean. The ShadowSpill
commit script is in the main checkout at
`docs/internal/plans/offline_pipeline_refactor/chicago-preparation/commit-local.sh`.
It created commits `0bd79e66`, `44d080f0`, and `bec7950b`; all were pushed to
`origin/master` with explicit authorization. The new refactor branch starts at
`bec7950b` in the preserved worktree. No refactor changes are on master.

## 1. CPU-only artifacts, planning, simulation, and reports

Implement the smallest complete offline path through Program, Profiles,
apply_profiles, plan_program, the optional sweep helper, and write_report.
Establish the shared graph/storage facts and explicit save/load needed by that
path. Supplied timings, memory evidence/estimates, budgets, and communication
costs must suffice without loading model weights or creating runtime pools.

Use small known programs and compatible existing captured task data before
depending on the new model builders. Any transitional internal import adapter
must use the same Program representation, not become a second public pipeline.
Start with a single logical rank and host fetch/evict costs. Preserve the agreed
generic group/resource fields without requiring distributed implementation now.

Completion checks:

- A fresh CPU-only process loads Program/Profiles, applies supplied values,
  plans across memory budgets/rates, and writes simulated timelines/reports.
- Save/load round trips preserve schedules, memory facts, provenance, and
  diagnostics independently of the original artifact folders.
- Missing evidence is explicit; estimated memory yields simulation-only Plans.
- Known small schedules validate task order, live-memory peaks, and transfer
  accounting. No CUDA context, process group, allocator plugin, or real pool is
  needed to perform these operations.
- Use available GLM stage measurements as a realistic planning example once
  compatible Program structure is available. Supply omitted timings/memory
  explicitly and label estimates; do not manufacture execution evidence.

This is the first deliverable to demonstrate and validate before continuing.

## 2. Separate model construction into public build APIs

Implement build_forward, build_forward_backward, and build_step using the
already selected controls. Separate export/partition/compile/lower from cost
measurement, planning, and state initialization. Capture semantic names and
compiler-visible memory. Build save/recompute alternatives; leave example and
optimizer ordering to planning.

Start with single-rank models. Build may use a representative GPU when required
to finalize an implementation, but does not require ShadowSpill runtime pools.
Its Program must feed the unchanged offline path from item 1. All three
computation boundaries must work, including persistent gradients without an
optimizer.

## 3. Single-GPU profiling

Implement measure_program with bounded task/pair input preparation, warmup,
timing, and complete allocation/resource evidence. Profile all stages by default,
with the selected explicit stage-list override. Returned Profiles must save/load
and drive the existing CPU planner without further GPU work.

Qualify ordinary-allocator observation against admission requirements. Use the
permitted allocator-backed fallback only if necessary for complete information.
Measurement must not change finalized task definitions. Distinguish inputs,
saved values, outputs, temporary allocations, and shared fixed resources.

## 4. Single-GPU saved-plan execution

Implement Executor in a fresh worker process: setup, bounded initialization,
plan validation, invocation/result handles, caller-owned outputs, and bounded
export. Validate forward, forward/backward, and full training against direct
PyTorch results on small models. Check both fresh and explicitly accumulated
gradient invocations and multiple examples/orderings.

Validate actual allocation behavior against the planned memory description,
including runtime/provider overhead. Exercise archive execution without the
original compiler-cache paths. Reject simulation-only memory and incompatible
builds explicitly. No distributed group orchestration is required yet.

## 5. Trainer and quickstart on the single-GPU pipeline

Migrate Trainer and quickstart to compose the public APIs. Demonstrate the
ordinary end-to-end flow and separately invoked build/profile/plan/execute phases,
including reuse of saved profiles for CPU-only replanning. Preserve checkpoint
streaming, metrics, hyperparameters, precision, and selected advanced behavior.

Completion means the single-GPU pipeline is usable and tested through direct
public calls and the normal training entry points, not merely isolated mocks.

## 6. Distributed execution and sharding

Only after the single-GPU pipeline works:

1. Add simple replicated DP, logical-rank binding, and a shared saved Plan.
2. Add weight/gradient/optimizer sharding and their explicit transfers and
   reduction dependencies. Validate byte counts, buffers, and lifetimes.
3. Add distributed communication calibration/estimation and complete rank maps;
   preserve representative single-GPU measurement of local DP tasks.
4. Extend to compatible model-owned distributed operations, ordered group
   families, repeated replicas, and mixed placement/storage groups. EP and CP
   are validation examples of the generic model, not separate core designs.

Use the accepted Parallelism design throughout. Defer implementation complexity,
not the distinction between logical ownership, storage, inputs, and live groups.
Keep asymmetric per-rank computation explicitly unsupported initially.

## 7. Finish caller migration and remove redundant code

Migrate callers and delete obsolete paths as each replacement becomes usable;
complete removal after the distributed callers also work. Remove old public APIs,
duplicate orchestration/store implementations, temporary adapters, and stale
documentation. No deprecated public wrappers remain. Preserve advanced behavior
and run the final cross-phase acceptance checks.

## Implementation discipline

Each milestone ends with an independently usable result and focused checks.
Keep phase-owned files/functions small and cohesive, shared IR authoritative,
and the top-level API small. Finalize exact records/protocols needed by the next
milestone against the approved design; present substantive public design changes
before adopting them. Do not front-load distributed implementation or create
extra wrappers merely to prepare for it.
