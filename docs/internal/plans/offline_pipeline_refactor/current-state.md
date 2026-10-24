# Current pipeline: source audit

Audited 2026-10-23. Chicago HEAD was
`5dcde2100c25fc7ba8f71a98709b424d60c24c8f`; Della HEAD was
`5aa561e3b93add313fa683770426a24bd0390c89`. The worktrees also contain unrelated
baseline work. This document describes inspected behavior, not an approved
replacement architecture.

## Overall assessment

The framework-neutral IR, planner, simulator, allocation contracts, and serialized
planning results provide a useful foundation. Capture, graph construction,
compilation, profiling, lowering, planning, and admission already have separate
implementation files.

The difficulty is in the data dependencies between these files. Preparation needs
a runtime and real pools. Lowering consumes profiling results for physical memory
information as well as timing. Several intermediate objects retain live models,
profilers, or installed runtimes. The strongest portable CPU boundary currently
comes after compilation and profiling.

There is also an existing public prepare/search/admit workflow. The requested
refactor should account for this work rather than assume these boundaries do not
exist. The scheduler and runtime do not need to be replaced just to expose earlier
CPU-only stages.

## Existing public boundaries

| API/object | Current responsibility and limitation |
| --- | --- |
| `prepare_step` / `PreparedStep` | Captures exports and measured programs for reuse. Preparation requires a runtime. Save/load exists; the supported persistent path is currently single-rank. |
| `search_prepared_steps` | Searches already prepared programs without new export, state materialization, or GPU profiling. |
| `admit_step` | Binds a selected prepared plan to a model/runtime, loads selected kernels, and performs physical admission. |
| `build_step_programs` / `StepProgram` | Produces serializable measured planning problems. The build call still requires runtime pools and profiling. |
| `plan_program` / `AnnotatedProgramPlan` | Framework-neutral planning, search, simulation, and physical placement using an existing planning problem. |
| `plan_step` / `plan_step_search` | Convenience orchestration spanning substantially more than CPU plan selection. |

Sources: [public PyTorch API](../../../../src/shadowspill/pytorch/api.py),
[prepared operations](../../../../src/shadowspill/pytorch/prepared_api.py),
[prepared artifact](../../../../src/shadowspill/pytorch/prepared.py),
[step program](../../../../src/shadowspill/step/program.py),
[planner](../../../../src/shadowspill/planner/plan.py), and
[existing reusable-planning guide](../../../examples/reusable-planning.md).

## Where the dependencies cross phases

The training orchestrator currently performs:

```text
capture_training_graphs
  → materialize_training_state
  → profile_training_tasks
  → build_training_programs
  → plan_training_programs
  → compile_selected_training_tasks
  → admit_training_plan
```

This sequence is visible in
[training/build.py](../../../../src/shadowspill/pytorch/planning/training/build.py).
The existing helper names make the stages visible, but their arguments and results
do not yet provide the requested independent public artifacts:

- `TrainingCaptureArtifacts` contains `InstalledRuntime` and `device_ordinal`.
  The capture entry point takes `PlanMemory`, even though much of export uses fake
  tensors and can be structural work.
- Optimizer capture happens with materialized training state.
- `TrainingProfileArtifacts` retains a live `TaskProfiler`.
- `TrainingProgramArtifacts` includes a simulation configuration, admission facts,
  and workspace reserves derived using the runtime and profiling.
- The profile environment reads CUDA device properties.

Sources: [phase objects](../../../../src/shadowspill/pytorch/planning/artifacts.py),
[capture](../../../../src/shadowspill/pytorch/planning/training/capture.py),
[materialization](../../../../src/shadowspill/pytorch/planning/training/materialize.py),
[program construction](../../../../src/shadowspill/pytorch/planning/training/programs.py),
[profile environment](../../../../src/shadowspill/pytorch/profiling/environment.py).

## Structural information versus profiling information

### Microbatch order is currently fixed during final lowering

The current implementation already avoids repeating capture, compilation, and
GPU task measurements for each data ordering in the shared build path:

- `planning/training/steps.py` captures and profiles once, then calls lowering
  for each `StepDataOrdering` to publish a separate `StepProgram`.
- `planning/training/profile.py` measures both gradient-initializing and
  accumulating forms so the measurements can serve every ordering.
- `lowering/training/program.py` shares object/profile/boundary preparation,
  then `TrainingLowering.lower(ordering)` creates each ordered program.
- `lowering/training/bindings.py` selects a backward form using
  `ordering.creates(...)`. `lowering/training/tasks.py` emits the task sequence
  and adds dependencies between gradient contributors/mutations. Thus changing
  order after lowering is more involved than permuting an existing task list.

This supports the user's proposed boundary: reusable compiled tasks and semantic
dependencies in `Program`, with order-specific composition/binding and physical
planning performed on CPU inside planning/search. Current optimizer placement
(`stage_interleaved` versus `tail`) is also chosen during emission. The user has
now selected moving that policy into planning/search in the refactored API;
implementation has not started.

The current ordered-task model is explicit in `_TrainingTaskEmitter.build`:
dependencies describe data constraints, while task-list order is the chosen
compute schedule. `_emit_interleaved_optimizer` emits a stage's update components
after that stage has emitted every microbatch's backward; `_emit_tail_optimizer`
places remaining/all updates after backward, depending on policy. Optimizer
tasks declare input objects and mutations. Thus the proposed separation of DP
communication does not require runtime discovery of when old weights stop being
read. Legal ordering is established before execution. Standalone collective
actions producing resident shards still need representation and runtime support;
current fetch/evict recipes alone do not supply that general interface.

There is also an admission assumption to preserve or extend: explicit simulator
memory-reuse edges currently cover eviction completion, while other
task-completion reuse is treated as ordered by the compute stream. A new
asynchronous collective must expose completion to those memory/reuse checks.
Current eviction timing already retains device bytes until transfer completion;
the standalone collective path has not been implemented or validated. See the
[memory audit](memory-and-profiles.md#memory-while-standalone-dp-communication-overlaps-compute).

Sources: [shared preparation](../../../../src/shadowspill/pytorch/planning/training/steps.py),
[task profiling](../../../../src/shadowspill/pytorch/planning/training/profile.py),
[lowering](../../../../src/shadowspill/pytorch/lowering/training/program.py),
[variant binding](../../../../src/shadowspill/pytorch/lowering/training/bindings.py),
and [task emission](../../../../src/shadowspill/pytorch/lowering/training/tasks.py).

### Timing and memory dependencies

`ShadowSpillProgram` currently contains task profiles. A `TaskSpec` refers to a
profile ID, and `TaskProfile` requires both `runtime_ns` and `workspace_bytes`.
It is not currently the proposed unprofiled `Program`.

`TaskMeasurement` combines runtime measurements with allocation traces, physical
output extents, scratch/persistent memory observations, and resource/overlap
information. Lowering uses these measurements to reconcile executable layouts.
Simply making runtime optional would not remove those dependencies.

Compilation does produce static manifests with output-storage contracts. Opaque
provider workspaces and runtime allocation paths are not all statically described
by those manifests. A design for external timings must also state where required
memory information comes from.

Sources: [IR program](../../../../src/shadowspill/ir/program/__init__.py),
[task/profile specifications](../../../../src/shadowspill/ir/program/specs.py),
[measurements](../../../../src/shadowspill/task/profiles.py),
[executable manifests](../../../../src/shadowspill/task/manifest.py),
[layout/profile reconciliation](../../../../src/shadowspill/pytorch/lowering/profiles.py).

CPU export and fake-tensor lowering should be distinguished from final CUDA
kernel compilation or autotuning. The compiler invokes Inductor and provider
implementations; an explicit target alone does not prove that every supported
backend can finish specialization on a host without a GPU. The scope of the
build artifact and any deferred executable work remain design questions.

### GPU preparation and representative DP profiling

The source audit on 2026-10-23 found three separate dependencies:

- ShadowSpill prepares fake inputs, then invokes Inductor's compiler. In installed
  PyTorch 2.13.0, Inductor's `DeviceProperties.create` obtains device properties
  through the device interface; an empty property cache can trigger live CUDA
  queries. A target label alone does not replace every hardware property used.
- Inductor's Triton runtime can autotune among launchers on first execution.
  Likewise, Quack checks its configuration cache and benchmarks candidates on a
  miss. Compiling the outer graph does not necessarily exercise an opaque custom
  op's internal kernels or finish their tuning.
- MLOps provider selection can query device capability. Fake implementations
  describe output metadata, but do not establish that internal preparation has
  completed or describe every runtime allocation.

Therefore GPU need is backend-specific: distinguish supported offline metadata/
compilation, compatible cached or explicit configurations, and preparation that
actually benchmarks kernels. There is no general existing readiness flag to
inspect. Do not add an inferred graph-name test or treat every compiler failure
as missing GPU access. The user permits GPU-assisted build where needed; its exact
boundary and incomplete-build behavior remain under review.

For symmetric pure DP, local compute preparation can use one representative GPU
while retaining logical DP size for shard geometry. This is a refactor target,
not a capability demonstrated by the current distributed preparation path.
`distributed/_optimizer.py` currently inserts reduce-scatter/all-reduce in
`_gradient` and all-gather in `_gather` around the local update. Those operations
must be separated for single-GPU local optimizer profiling. Collective costs can
then be supplied independently; measuring their actual latency or interference
still requires the participating ranks. Distributed operators inside model tasks
remain a separate case.

Sources: [explicit compiler preparation](../../../../src/shadowspill/pytorch/compilation/inductor/explicit.py),
[Inductor invocation](../../../../src/shadowspill/pytorch/compilation/inductor/compiler.py),
and [distributed optimizer capture](../../../../src/shadowspill/pytorch/distributed/_optimizer.py).
Installed backend sources inspected were PyTorch 2.13.0+cu130
(`_inductor/runtime/hints.py`, `_dynamo/device_interface.py`, and
`_inductor/runtime/triton_heuristics.py`) and Quack 0.6.5 (`autotuner.py`).
This was source inspection, not GPU validation of an offline build path.

## Transfers and collectives today

There are several different measurements, not one collective profiling stage:

1. Runtime initialization calibrates host-to-device and device-to-host copies,
   including solo and concurrent behavior.
2. Parameter storage registration builds a sharded transfer recipe and calibrates
   all-gather at each unique group/message size.
3. Training materialization prepares gradient transfer recipes. It separately
   calibrates accumulator zeroing and reduce-scatter, including replica all-reduce
   where configured.
4. Task profiling measures tasks. For tasks containing collectives, additional
   observations identify their collective resource phases within the measured
   task duration.
5. Additional probes measure interference between ordinary compute tasks and
   background sharded transfers.

The storage-transfer calibrations run ten trials, discard the first three, use
the slowest rank for each trial, and retain the median of the remaining seven.
Results are reused for matching group/size/dtype configurations.

More precisely, these storage calibrations are cached in the active
`BoundDistributed.transfer_calibration` dictionary. All-gather uses the bound
group alias and full padded byte extent as its key; the operation moves `uint8`
storage bytes, with the local shard as a view into the full destination buffer.
Gradient calibration additionally keys by gradient dtype and replica-group alias.
It times reduce-scatter plus the optional replica all-reduce together, and times
full-accumulator zeroing separately. A Gloo barrier precedes each trial and a
Gloo MAX reduction combines rank durations afterward; neither control operation
is inside the CUDA-event timing interval. Host/device copies are not inside that
collective interval either.

This is reuse within an already bound distributed setup, not a portable table
with explicit hardware/topology compatibility matching. The group alias encodes
the logical name and ordered live rank membership. Storage setup/lowering calls
`storage_layout`, which may trigger all-gather calibration; optimizer
materialization calls `prepare_gradient_transfers`. The resulting durations are
embedded in the transfer IR. There is currently no bandwidth/topology fallback
for an unmeasured collective in that path.

Task-internal collective observations use a different path. When transfer
sharding is active, the profiler invokes the task ten times, retains the last
five instrumented samples, and uses NCCL Work start/end events for observed c10d
collectives. It normalizes median phase proportions to the ordinary task timing;
the intervals describe resource occupancy within that duration, not extra time
to add. This observer does not inspect arbitrary provider-private communication
that bypasses c10d. Separate overlap probes compare task-only, transfer-only,
and concurrent execution to characterize interference where supported.

| Transfer | Current sequence |
| --- | --- |
| Fetch sharded parameters | H2D local shard, then all-gather into the full execution object. |
| Evict sharded parameters | D2H owned shard. The optimizer already publishes updated weights. |
| Fetch sharded gradients | Zero full accumulator, then H2D the local shard. |
| Evict sharded gradients | Reduce-scatter, optional replica all-reduce, then D2H local shard. |

The IR separates local spill size from full execution size. A transfer contains
copy phases priced by bandwidth and timed phases carrying already measured
durations. The current `TransferBandwidths` override reprices host copies; it does
not replace those embedded collective durations with a topology model.

Sources: [runtime calibration](../../../../src/shadowspill/runtime/calibration.py),
[distributed storage and calibration](../../../../src/shadowspill/pytorch/distributed/_backing_storage.py),
[transfer IR](../../../../src/shadowspill/ir/program/transfers.py),
[task collective phases](../../../../src/shadowspill/pytorch/profiling/profiler/collectives.py),
[overlap probes](../../../../src/shadowspill/pytorch/profiling/profiler/overlap.py),
[planning inputs](../../../../src/shadowspill/planner/program_inputs.py).

The planner currently models one execution device, with symmetric distributed
behavior represented through the rank-local transfer recipes and collective
resource. It is not a general multi-rank topology simulator. A scale-up/scale-out
cost model and its relationship to that representation need an explicit design.
Collectives already included in task durations must not be charged a second time.

### Small-collective granularity

The user reports many inefficient small collectives in simulated traces. The
source confirms two ways these can arise; this audit does not identify every
event in the unspecified trace:

- `storage_layout` creates a transfer description per storage object. The C++
  `storage/sharded.cpp` adapter issues an all-gather for each fetched sharded
  weight object, or a reduce-scatter and optional replica all-reduce for each
  evicted gradient object. These paths have no explicit cross-object bucketing.
- `distributed/_optimizer.py` composes gradient reduction per parameter when
  needed, and calls `_gather` for each updated parameter with sharded optimizer
  ownership. `optimizer/tasks.py` can group update components by backward-ready
  stage, but grouping them into one task does not itself combine their collective
  calls.

The update publication is also an optimization candidate: when updated weights
remain authoritatively sharded and only local shards will be spilled before the
next full-weight use, eager all-gather can be avoidable. Current update outputs
materialize the full compute weight, so changing this requires an explicit
shard-output representation and dependencies, not simply deleting the call.
Replicated outputs or intervening full-weight consumers still require publication.

Communication aggregation remains deferred until after the refactor. The user
subsequently proposed separating DP communication from optimizer tasks as part
of the architecture: local updates consume reduced shards and publish updated
shards, with communication represented separately. This also enables deferring
weight gathering until a consumer needs it, where declared layouts permit.
That separation is now a design direction under discussion; neither it nor
collective aggregation has been implemented.

## Persistence and caches today

`ArtifactStore` manages build and planning trees and modes such as reuse/refresh.
Specialized stores hold exports, graph pairs, optimizer captures, compiled
manifests, measurements, step archives, and planning results. The step archive has
an early lookup identity in addition to intrinsic program identities.

`PreparedStep` already supports explicit save/load using a manifest, hashes,
export payloads, metadata, and program JSON. This is a PyTorch-backed preparation
artifact. The framework-neutral `StepProgram` JSON and plan JSON have a smaller
dependency surface for CPU planning.

The design review should cover the overlapping responsibilities and distinguish
what users need to retain from what can be regenerated. Content identity,
compatibility validation, and complete artifact publication have value even if
the public store interface changes.

Sources: [artifact store](../../../../src/shadowspill/store/artifacts.py),
[planning stores](../../../../src/shadowspill/pytorch/planning/stores.py),
[step archive](../../../../src/shadowspill/step/archive.py),
[prepared persistence](../../../../src/shadowspill/pytorch/prepared.py).

Della already has changes keeping Inductor/Triton caches outside the artifact
store by default. Some Chicago documentation still describes the older layout;
the refactor must account for these checkout differences.

## Evidence and limits of this audit

With CUDA hidden and imports of `torch` explicitly blocked, importing
`shadowspill.planner.plan_program`, `shadowspill.step.StepProgram`, and
`shadowspill.ir.ShadowSpillProgram` succeeded. This verifies the existing neutral
import boundary; it is not a completed planning experiment or a CPU-build test.

The latest GLM trace harness on Della demonstrates fake-tensor stage capture and
reusable graph-pair artifacts. Its 40 cases cover 45 stage occurrences, omitting
the first stage (embedding plus layer 0) and optimizer work. It records diagnostic
timings under Nsight and pair-level memory peaks, not the planner's physical
allocation contracts. These are inputs to a future example after the public
interfaces are designed, not complete planning evidence by themselves.

No new simulation, GPU profiling, library refactor, or removal of existing APIs
was performed during this audit.

## Source names: existing information and where it is lost

The automatic partition policy already reads FX `nn_module_stack` metadata to
find repeated model blocks. Export input signatures preserve parameter/buffer
targets, and `TaskInputProvenance.source` carries those names into stage inputs.
`RegistrationBinding` relates registered parameter/buffer names and quantized
component paths to lowered object IDs. Registration preserves duplicate names
for tied parameters.

Stages still receive generated `stage_0000` identifiers. `Stage.module_target`
names the split graph submodule, which need not be the original model module.
The neutral `ObjectSpec`/`AliasGroupSpec` contain generated IDs, sizes, roles,
and storage relationships, without those module/value source annotations.
The object catalog creates names such as `object_000000` and `alias_000000`.
Runtime transfer labels currently add roles, byte counts, and triggering tasks,
but do not recover the original tensor names from the model.

The missing work is chiefly preserving and publishing this attribution across
partitioning, differentiation, lowering, persistence, and reporting. An alias
group represents shared storage, so it may legitimately have several source
tensor names. Compiler-generated activations may have no original Python local
name. A source map needs to represent these cases instead of assuming one name
per storage or one module per stage.

Sources: [partition metadata](../../../../src/shadowspill/pytorch/partition/policy.py),
[stage provenance](../../../../src/shadowspill/pytorch/partition/provenance.py),
[stage records](../../../../src/shadowspill/pytorch/partition/artifacts.py),
[object registration](../../../../src/shadowspill/pytorch/lowering/catalog.py),
[neutral specifications](../../../../src/shadowspill/ir/program/specs.py), and
[transfer labels](../../../../src/shadowspill/runtime/transfer_labels.py).

## Allocator and profiling process boundary

`TaskProfiler` currently requires the installed allocator library, runtime handle,
and plan ID. Its `AllocatorBoundary` opens runtime allocation scopes, drains
retirements, reads slab statistics, and uses runtime event wrappers even for
timing. Workspace observation records runtime telemetry and identifies returned
storage using the slab's pointer lookup. This is substantial coupling, not merely
an optional pool argument.

The frontend's `refuse_unusable_build` rejects installation once
`torch.cuda.is_initialized()` is true. `Runtime.close()` leaves the process-global
allocator shim installed; it does not restore the ordinary PyTorch allocator.
The installed PyTorch source and
[PyTorch allocator documentation](https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.change_current_allocator.html)
state that switching the current allocator fails once that allocator has been
used/initialized. Importing `torch` alone is not the same event; ShadowSpill's
current guard is explicitly tied to CUDA initialization.

The user requires the selected plan to execute in a fresh process after ordinary
GPU profiling, with an explicit worker API and an optional convenience launcher.
A fresh interpreter or appropriate spawn-based worker avoids inheriting
initialized CUDA state; a plain fork after GPU use is not a suitable shortcut.
See [PyTorch multiprocessing guidance](https://docs.pytorch.org/docs/stable/notes/multiprocessing.html).

Sources: [profiler](../../../../src/shadowspill/pytorch/profiling/profiler/__init__.py),
[allocation boundary](../../../../src/shadowspill/pytorch/profiling/profiler/boundary.py),
[workspace observation](../../../../src/shadowspill/pytorch/profiling/profiler/workspace.py),
[frontend allocator installation](../../../../src/shadowspill/pytorch/frontend.py),
and [runtime lifecycle](../../../../src/shadowspill/runtime/core.py).

## Shared distributed execution artifacts

Existing distributed planning already checks symmetric planning requirements and
shares CPU schedules. Its symmetry layer normalizes device identifiers for those
checks, uses shared timing/transfer assumptions, and rebinds a received schedule
to local admission facts. Compiled ABI identities and actual entrypoints remain
rank-local. Fixed runtime layout records contain offsets and dependency indices,
not process pointers. These are useful foundations for a portable execution plan.

Remaining process-specific dependencies include:

- Lowering publishes a `DeviceSpec` containing the current CUDA ordinal.
- `UpdateLayout.group_rank` determines `start`, and distributed optimizer capture
  uses that concrete value in shard slicing.
- Logical group aliases exist, but their creation currently requires live groups
  and actual ordered membership.
- `admit_step` explicitly rejects distributed persistent preparation, and the
  existing saved plan does not contain all executable entrypoints/state bindings
  needed by a fresh worker.

Thus saving the current plan JSON is not yet the requested distributed executable
bundle. The refactor needs a logical binding contract plus packaged compatible
executable payloads, reusing the existing schedule/layout machinery.

Sources: [symmetric planning](../../../../src/shadowspill/pytorch/distributed/_symmetry.py),
[rank-local allocation reconciliation](../../../../src/shadowspill/pytorch/distributed/_allocation_symmetry.py),
[lowered devices](../../../../src/shadowspill/pytorch/lowering/program.py),
[optimizer shard offsets](../../../../src/shadowspill/pytorch/distributed/_optimizer.py),
[logical group aliases](../../../../src/shadowspill/pytorch/distributed/_groups.py),
[persistent binding API](../../../../src/shadowspill/pytorch/prepared_api.py), and
[relative runtime layouts](../../../../src/shadowspill/runtime/fixed_layout.py).

The [EP path and distributed contract audit](distributed-contracts.md) follows
the current workload through MLOps custom operators, replica/gradient ownership,
coordinated task selection, stream completion, and process-local resource handles.
It distinguishes existing working contracts from the additional information
needed for offline construction and logical-rank rebinding.
