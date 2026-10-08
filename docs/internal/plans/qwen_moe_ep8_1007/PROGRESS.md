# Progress

## 2026-10-07

- Read official configs and current Transformers implementation. Confirmed the
  head widths are independent of model width; Qwen3.5 adds interleaved per-head
  query gates, zero-centered norms, and a gated shared expert.
- Existing dense Qwen3.5 workload and qualification defaults will be retained.
- User set global batch to 2^22 and added quickstart before training, with six
  execution budgets and all resolution plans saved.
- Found the QuackMoE buffer currently requires an exact token count, not merely
  an upper bound. Candidate models must therefore own correctly sized resources;
  a single maximum-size buffer cannot silently serve all four geometries.
- No GPU allocation requested yet; preparing models and host-side checks first.

### Host preparation completed

- Added `mlops_qwen30b` and `mlops_qwen35b`, preserving dense `mlops_qwen35`.
  Public model code lives in `workloads/mlops/qwen_moe/` and contains no
  ShadowSpill imports. EP optional imports are delayed until model construction.
- Independent HF CPU comparison passes for logits and every parameter gradient.
  Corrected router auxiliary normalization to match HF: assignment counts are
  not divided by top-k, so uniform top-8 routing yields auxiliary ~= 8.
- Meta parameter counts match official HF models: 30,532,122,624 and
  34,660,610,688. Broader workload/quickstart checks: 52 passed, 1 skipped.
- Generic quickstart additions: optional model cleanup after state release,
  per-process search-worker limit, configurable external headroom, and Python
  `plan_store` argument for durable reuse. No model-specific lowering changes.
- Prepared geometry-specific quickstart orchestration, combined search reports,
  winner measurement, and subsequent real-data Trainer script. All resolution
  plans retained; all orderings use existing `default_orderings`.
- Prepared two ~50M-token FineWeb-Edu samples with exact official Qwen tokenizer
  revisions. Data remains under storage; no checkpoint weights were downloaded.
- Submitted job 15199147 (exclusive 8 GPUs / 96 CPUs / 960 GiB / 1 hour), pending
  with initial estimated start October 8 04:24 EDT. `--mem=0` was rejected by
  Slurm; node reports 1,024,000 MiB physical and 4,000 MiB system reserve.
- Watcher is a persistent head-node tmux window, `codex:qwen-watch`; heartbeat
  evidence verifies polling. A first shell-background launch did not survive
  the command session; replaced it with tmux before relying on it.

### Distributed search discussion

Current `_selection.choose` evaluates every point/resolution on every rank,
exchanges per-rank outcomes, and selects the minimum worst-rank predicted time.
This includes real rank-specific differences, not only redundant work. Proposed
future simplification: verify equivalent task graphs/objects/shards/budgets,
combine measured per-task times conservatively (max), search a symmetric problem
once with independent points/resolutions distributed across CPU workers, then
physically admit mapped plans on every rank. Keep asymmetric fallback. This is
**the initial design discussion; implementation is recorded below**. At that
point the experiment capped workers to avoid oversubscribing the node and retained correctness
and collective-order guarantees.

Timing symmetry and memory symmetry are separate conditions. For a homogeneous
group, all ranks still participate in profiling communicating tasks; task timings
can then share a conservative estimate instead of requiring identical measured
floats. Reusing the full planning problem additionally requires equivalent object
sizes, lifetimes, mutation/alias rules, initial placement, optimizer shard layout,
transfer assumptions, and effective budgets after external memory reserves.
Parameter values and GPU addresses need not match. Each rank still admits the
selected plan against its own runtime. Uneven batches/shards or heterogeneous
devices can retain today's independent local planning. Nothing about this fast
path would introduce global barriers between tasks during real execution.

### Final host-side orchestration checks

- Combined reports retain token-throughput metadata across all geometries.
- Measured tables and figures are consolidated across all budget winners for
  each rank; the subsequent short training run selects the fastest measured
  winner using the slowest rank's median step time.
- Resuming a sweep rejects changed experiment settings in the same output
  directory, while allowing selection of a different stage.
- CPU checks round-trip all timing fields, unknown terminal transfer timing,
  exact budget bytes, and per-step ordering. A two-rank/two-budget fixture checks
  consolidated figures and global token throughput without executing a GPU task.
  Evidence: `evidence/sweep_host_checks.json`.
- Allocation remains pending. Latest observed estimate: October 8, 06:44 EDT;
  the 10-second watcher heartbeat remains current. No GPU validation claimed.

## 2026-10-08 — symmetric planning implemented and DP2 checked

- User approved an opt-in verified symmetric mode. Added
  `Distributed(..., symmetric_planning=True)` and generic quickstart overrides.
  The Qwen EP8 harness enables it by default, with a disable flag for comparison.
- GPU capture/profiling remains per rank. CPU requirements are compared before
  sharing work; timing uses max task durations / min bandwidth / max latency.
  Each rank retains its own compiled bindings and physically admits its schedule.
- Standalone plans share resolution searches. Sweeps distribute full ascending
  budget chains across orderings and reuse identical lowered programs.
  Unequal requirements fall back to the existing independent-rank path.
- Updated public README, distributed API docs, quickstart docs, and artifact-store
  docs. Cache hits restore retained alternative plans. A completion manifest
  prevents interrupted writes from masquerading as complete retained artifacts.
- Broader CPU regression: 209 passed, 1 deselected. Targeted tests cover two real
  Gloo processes, local ABI/device remapping, physical admission, fallback,
  failures, duplicate orderings, warm-cache resume, and incomplete-cache repair.
- User requested fatnode validation and reminded us to isolate healthy devices.
  Used rootless Podman with only GPU device minors 0 and 2 exposed, selected by
  stable UUID. Bounded NCCL health check passed. No faulty GPU is exposed there.
- In `codex:0.0`, real DP2 checks passed for FP32 and FP16 recompute with sharded
  masters/optimizer, checkpoint replay, fresh restore, and forward evaluation.
  Checks assert symmetric mode actually activated.
- A two-geometry, five-ordering staged DP sweep searched 3 points on rank 0 and
  2 on rank 1. Both trained successfully; max parameter error 5.96e-8 versus a
  combined-data CPU oracle after three updates. Equivalent one-stage orderings
  also passed after fixing duplicate-result bookkeeping.
- Recorded a separate automatic-stage capture issue involving a mutable integer
  buffer before the linear layers. It reproduces with symmetric mode disabled
  and precedes CPU search; whole-stage mutable-buffer checks pass. See
  `SYMMETRIC_PLANNING.md` and `logs/fatnode/capture_control.log`.
- Repeating the standalone test in the same output directory correctly refused
  to overwrite its old checkpoint. The test launcher now picks a timestamped
  destination, and both full GPU checks passed again.
- No commits. Fatnode has the same uncommitted source changes on its existing
  checkout, with no additional worktree or branch. MLOps was not changed.
- Della job 15199147 remains pending; the 10-second watcher is still active.
  On allocation: tiny Qwen EP8 checks first, then resume the requested full-size
  quickstart matrix. These Qwen GPU runs and subsequent training remain open.

## 2026-10-08 — four-GPU real-model validation

- User requested four GPUs, then explicitly requested a real model. Four-device
  NCCL health passed in the container exposing physical indices 0, 2, 3, and 6
  (selected by UUID). The isolated DP4 toy test failed its first update on the
  nine-element bias. Repeating with symmetric planning disabled also failed;
  this is recorded as a separate unresolved numerical issue, not a passed test.
  Logs: `logs/fatnode/dp4/auto.log` and `auto-control.log`.
- Added `scripts/fatnode/llama.py`: public Trainer, full 1,179,699,200-parameter
  Llama workload, original verified weights and FineWeb-Edu sequence files from
  the prior DP1/DP2/DP8 experiment. Exact previous model dimensions and LR
  schedule are retained. No source data or old reference artifacts are changed.
- DP4: 32,768 global tokens/update, 2,048-token sequences; search 2K/4K tokens
  per microbatch/rank and their default orderings. FP16 compute, FP32 gradients,
  sharded FP32 masters/optimizer state, 10 GiB execution and 40 GiB spill/rank.
- Running visibly in fatnode `codex:0.0`; log `logs/fatnode/dp4/llama-symmetric.log`.
  Intended checks: verified symmetric decisions, distributed search ownership,
  physical admission, warmup/traced step, five actual updates, finite state,
  bitwise-equal final replicas, changed parameters, and reported differences
  from the original DP1 loss curve. The latter is historical evidence, not a
  claim of bitwise equivalence across accumulation orders.
- GPU/model validation is in progress; no new pass is claimed. No commits.

### Completed real-model check and fresh control

- Both Llama DP4 runs completed all five search points, physical admission,
  warmup/tracing, and five actual updates. All 111 parameter tensors changed;
  losses/weights were finite and all four replicas matched bitwise within each
  run. The symmetric path was explicitly verified with no fallback.
- Symmetric search owners: 0/1/2 for the three 2K orderings, 0/1 for the two 4K
  orderings. Each point searched once, then adopted/admitted by all ranks.
- Median actual step: 3.936 s symmetric vs 4.050 s independent. Preparation:
  193.44 s vs 223.51 s, including capture/profiling/search. Short validation,
  not a repeated performance experiment. Both selected 2K microbatches.
- The plans chose different save/recompute variants. Fresh-run loss differences
  were at most 0.002867; final parameters differ between runs. Within-run
  replicas are exactly equal. Historical DP1 differs by up to 0.062/0.064 in
  both modes, so strict parity against the older software run is not claimed.
- Full report: `REAL_MODEL_DP4.md`; JSON: `evidence/fatnode/dp4/llama_comparison.json`.
  Per-rank stores, startup simulated/traced timelines, logs, and hashes are
  retained on fatnode. Small evidence is copied to Della.
- The small-model bias failure remains open and is an explicit agenda item.
  No production fixes or commits were made during this real-model validation.

## 2026-10-08 — modularity review and commit preparation

- User asked why the batch is still uncommitted and requested reasonable module
  and function sizes. The earlier approval-before-commit preference was carried
  forward for this batch. Prepared four explicit groups in `COMMIT_PLAN.md` and
  a dry-run-by-default commit script; no commits, pushes, branches, or worktrees.
- Split candidate request/ownership/attempt/outcome handling into named private
  helpers. `_selection.choose` is now 75 lines, previously 182.
- Split geometry verification, owned search, and receiver admission; a private
  `_SharedSweep` holds one geometry's state. `prepare_geometry` is 39 lines,
  previously 168. Extracted shared plan exchange to `_plan_exchange.py` and
  removed the circular planner/sweep import. No math, API, collective order,
  search policy, or real-execution synchronization changes were intended.
- Five production modules are 139–385 lines; largest remaining routine is
  `DistributedPlanner.plan` at 103 lines including a 19-line callback. Recorded
  exact function metrics and checked the dependency graph in `modularity.json`.
- Ruff and `git diff --check` pass. Targeted Della-head tests had two subprocess
  SIGKILLs without Python failure evidence; no cgroup OOM kill was recorded.
  The same 33 tests pass on fatnode. A broader fatnode planner/step-search/
  quickstart/symmetry regression passes: 208 tests in 17.56 seconds.
- Started another real 1.18B Llama DP4 run against the refactored source, visibly
  in fatnode `codex:0.0`. Console: `logs/fatnode/dp4/llama-modularity.log`.
  Its result remains pending at this entry. The prior toy-model bias issue
  remains open and is not changed by this structural cleanup.

### Post-refactor validation completed

- All four ranks passed the real Llama rerun, with five searched points,
  symmetric ownership `[0, 1, 2, 0, 1]`, local admission, warmup/traced step,
  and five training updates. All 111 parameter tensors changed and final
  replicas matched bitwise. Container exited zero.
- New run: `evidence/fatnode/dp4/llama-symmetric-20261008T052534/`.
  Predicted step 3.3825 s; measured median 3.9048 s. The twelve per-rank
  timeline HTML files were checked on fatnode; compact results and the console
  log were copied to Della. Summary: `evidence/modularity_validation.json`.
- Confirmed the 44 source files in the recorded manifest match between Della
  and fatnode. Commit script preview passes and leaves the index untouched.
  No commits or pushes have been made. The earlier small-model bias issue and
  full Qwen EP8 validation remain open; the Della allocation watcher is active.

## 2026-10-08 — approved publication and DP4 bias investigation

- User approved committing/pushing the four prepared groups and updating Della,
  fatnode, Tubingen, and Chicago, then investigating the small-model bias issue.
- Publishing the validated snapshot first. The investigation will start from
  that shared revision; no extra branches or worktrees are being created.

- Published commits `cb739bab`, `40ab9403`, `9f6d65f2`, and `8dc458c9`.
  All four ShadowSpill checkouts reached `8dc458c9`; MLOps is `25972fb` on all
  machines. `evidence/rollout.json` records the paths and full revisions.
  Fatnode's GitHub DNS failed, so identical commits were fetched via a bundle
  from Della. Its local test copy was verified byte-for-byte before updating.
- Chicago's editable MLOps installation pointed at a directory the user had
  moved under `old_research`. Reinstalled that editable binding at its current
  location; imports now work. No source or dependency versions changed.
- Reproduced the DP4 failure with detailed snapshots and without metric logging.
  Found an intermediate alias overwritten by Inductor scratch reuse. Our direct
  compiler normalization had omitted functionalization. A standalone tensor
  reproducer confirms the issue without the runtime pool or communication.
- Added generic functional normalization, including opaque mutable operators
  and shared bases for aliased inputs. The original DP4 oracle now passes with
  max parameter error 2.98e-8. CPU regression probes pass; broader GPU compiler
  and gate validation is underway. Details and prototype limitations are in
  `DP4_BIAS.md`.

- All 34 compiler tests pass on the GPU container. DP4 recompute and the
  independent-planning control also pass on all ranks, including checkpoint
  replay/restoration; maximum parameter errors are 2.98e-8 and 3.73e-8.
  Per-rank summaries are saved in `evidence/bias_validation.json`.
- The full suite initially failed collection because the minimal container
  lacked Git. Mounted Git tools read-only and reran the unchanged suite; CPU
  canaries and the initial GPU canaries pass, with the rest still running.
- Chicago's MLOps directory is now back at `research/mlops`. Rebound its editable
  installation to that existing checkout and verified imports again. Its source
  remains at `25972fb`; no dependencies or repository content changed.

### Compiler validation complete

- `qualification.gates suite --run bias_functionalization_1008` passed on
  fatnode: 1,167 passed, one skipped, 38 fresh-process tests deselected from the
  outer pytest run; all 20 CPU and 51 GPU CTest canaries pass. Duration 22 minutes.
  The suite includes four distributed oracle configurations, represented
  parameters under save/recompute, and packed-parameter gradients.
- Production change is one 96-line compiler helper and its normalization call.
  Its longest routine is 40 lines including nested functions and documentation.
  Optimizer/distributed math and runtime code are unchanged. The new regression
  tests and lowering documentation are included with the compiler fix.
- Preparing the two approved follow-up commits (groups 5/6) for push and rollout
  to all four existing master checkouts. No branches or worktrees are added.
  The Della EP8 allocation remains pending, with its 10-second watcher active.

### Publication and final repository checks

- Pushed compiler fix `4b0778d7` and evidence `0ce11433`; all four machines reached
  that revision. Fatnode used a Git bundle because GitHub DNS is unavailable.
  Its temporary test edits were checked against the incoming committed bytes
  before fast-forwarding. MLOps remains unchanged at `25972fb`.
- Once the ignored diagnostic scripts became tracked, the repository CLI guard
  flagged options forwarded wholesale to the oracle. Made that forwarding
  explicit in `bias_probe.py`; all 114 repository checks now pass. No production
  code or GPU execution changed after the successful full suite.
- Publishing this diagnostic-only follow-up and finalizing the four-machine
  revision/import verification. Final machine revisions are recorded in the
  local evidence file `evidence/bias_rollout.json` after publication.
