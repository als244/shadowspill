# Independent build, profile and planning phases

The refactor is being implemented in phases. The CPU path, builders and
ordinary-allocator GPU measurement and initial saved-plan execution work; caller
migration is still in progress on `codex/offline-pipeline-refactor`.

```python
from shadowspill import (
    Program, Profiles, apply_profiles, plan_program,
    MemoryBudgets, CommunicationInfo, write_report,
)

program = Program.load("program.ssp")
profiles = Profiles.load("profiles.ssp")
profiled = apply_profiles(program, profiles)
plan = plan_program(
    profiled,
    memory_budgets=MemoryBudgets(
        execution_bytes=40 * 2**30, spill_bytes=128 * 2**30,
    ),
    communication_info=CommunicationInfo(
        fetch_bytes_per_second=24e9, evict_bytes_per_second=20e9,
    ),
)
plan.save("plan.ssp")
write_report(plan, "report")
```

All the calls in this example run on CPU. They do not import PyTorch, initialize
CUDA, install an allocator, open pools, or require ArtifactStore. Planning uses
the existing native search/simulator library. Loading and inspecting an artifact
does not load that native library.

`Program` stores task dependencies, storage extents, alternative task groups,
source names and implementation identities without durations. `Profiles` stores
plain numeric task durations and unresolved workspace sizes for one setup.
`apply_profiles` checks complete coverage and snapshots the supplied values.
An explicit zero is valid; a missing workspace or runtime/provider overhead is
an error, not zero. Known compiler memory is retained as a lower bound.

The forward builder already separates export, partition, compilation and lowering
from timing and runtime pools:

```python
from shadowspill import build_forward

program = build_forward(
    model, example_args=(features,), example_kwargs={"mask": mask},
    partition="auto", target="cuda:0",
)
program.save("program.ssp")
```

The model may have CPU or meta parameters. This initial backend uses one
representative GPU to finalize CUDA task implementations. It accepts normal
positional and keyword arguments, including nested structures and specialized
non-tensor values. It records semantic module/parameter names, output storage
geometry, scheduled compiler-buffer lifetimes and explicit unresolved memory
requirements for provider calls. No persistent-state initialization or task
duration profiling occurs. Generated kernels are finalized by executing complete
tasks with valid example controls and outputs from their producer tasks. This
avoids benchmarking indexing kernels on unrelated synthetic integer buffers
during standalone compilation. Selected binaries are saved in the Program;
measurement and execution do not select new kernels or depend on an earlier
process's compiler cache.
Finalization also resolves backend outputs whose device was provisional in
compiler metadata, such as attention RNG controls returned on CPU. Those values
have host lifetimes and no GPU allocation; conflicting observed storage locations
fail build before profiling or planning.
An optional `forward_fn(model, *args, **kwargs)` selects another model method or
computation without changing registered parameter names. For example, an
evaluation loss can use `forward_fn=lambda m, *a, **kw: m.loss(*a, **kw)` with
targets in `example_kwargs`. The default calls `model.forward`. The callback is
only needed during build; the saved Program contains the captured computation.

`build_forward_backward` adds an objective and persistent gradients without an
optimizer. It compiles gradient-creation and accumulation forms for each stage:

```python
from shadowspill import build_forward_backward
from shadowspill.step import StepDataOrdering

program = build_forward_backward(
    model, objective=objective, example_args=(features,),
    example_kwargs={"targets": targets}, num_examples=8,
)
# After measurement or explicit profiles, choose traversal entirely on CPU:
plan = plan_program(
    apply_profiles(program, profiles),
    ordering=StepDataOrdering(depth=2, breadth=4),
    memory_budgets=memory_budgets, communication_info=communication_info,
)
```

The objective receives `objective(model, *args, **kwargs)` and returns a scalar
loss, optionally with metrics. `num_examples` fixes how many complete calls the
program includes; ordering is a planning choice. Changing it reuses compiled
forms and task profiles. Model mutations or RNG effects currently restrict
reordering. Tied parameter contributions preserve both creation and mutation
dependencies. `operator_recompute` controls operator retention during build;
the planner separately chooses among saved stage variants.

For differing example shapes or static metadata, both training builders accept
`example_calls=[(args0, kwargs0), (args1, kwargs1)]` instead of
`example_args`/`example_kwargs`/`num_examples`. Matching signatures reuse captures;
each example retains its own signature and constants through save/load. The
default one-example-plus-count form remains sufficient for uniform inputs.

`build_step` takes the same training arguments plus `optimizer=optimizer`, whose
parameters must belong to the supplied model. It discovers optimizer tensor
shapes and initialization defaults without allocating persistent state or
changing that optimizer. Traceable local updates are packaged with the stages.
`plan_program(..., optimizer_ordering="stage_interleaved")` places updates after
their required gradient writes and all remaining reads of old weights;
`optimizer_ordering="tail"` puts updates after backward. Both use the same
Program and profiles. Opaque updates and unsupported first-use state creation
are currently rejected explicitly.

`master_dtype=torch.float32` captures separate optimizer master weights while
preserving the model's compute dtype. `grad_dtype` independently selects stored
gradient precision. The optimizer determines moment/state dtypes from its own
rules. Execution initializes masters from compute weights using bounded chunks;
exports/checkpoints name them `master.<parameter-name>`. The update contains the
gradient conversion and write-back to compute weights, so those costs and
allocations are profiled and planned. Build does not modify the supplied model
or optimizer.

```python
from shadowspill import TaskProfile

overrides = Profiles(
    setup=profiles.setup,
    tasks={profile_key: TaskProfile(duration_seconds=0.0072)},
)
profiles = profiles.merge(overrides)
```

Later supplied values overwrite earlier values. Omitted fields are preserved,
so this timing override preserves existing memory information. Supplied values
without classification metadata are estimates. Task compatibility and setup
mismatches are errors. Changes to editable Profiles never alter an already
created ProfiledProgram or Plan.

Explicit latency is charged as a transfer phase. Concurrent fetch/evict rates
default to independent full-duplex lanes; callers can provide different rates.
The communication description is saved alongside the plan, independently of
compute measurements. Symmetric distributed simulations account for local host
shards, weight gathers and explicitly lowered gradient reductions.

`measure_communication()` optionally measures pinned-host copies on one GPU,
using ordinary PyTorch allocation and CUDA events. It returns CommunicationInfo;
save it independently and pass it to later planning runs. A size sweep fits
startup latency and bandwidth, and simultaneous copies measure overlapping lane
rates. Raw medians, sample counts, variability and fit errors are retained.
Neither planning nor execution triggers this probe. Copy buffer use is bounded
by twice the largest requested size on each of host and GPU.

`measure_communication(program=program, rank_map=..., device=...)` derives
collective probes from saved transfer geometry. With multiple workers, initialize
distributed rendezvous first and call on every worker with the same complete
worker-to-logical-rank map. No compiled task is executed. Host copies are sampled
concurrently across workers; the shared result uses the minimum observed rates
and maximum fitted latency, preserving every worker's observations. Each collective
group is then probed separately, using coordinated repetition counts and completion
timing. Logical member order is retained after worker remapping. Topology is
derived from physical hostname membership. Every worker receives the same result.
Calibration records NCCL version, group options, relevant environment settings
and configuration-file digests separately from compute setup. Workers must agree
on this backend setup. Changing it invalidates matching communication timings
and memory records without invalidating single-GPU compute profiles.

```python
# All workers have initialized rendezvous and loaded the same Program.
communication = measure_communication(
    program=program,
    rank_map={worker: worker for worker in range(8)},
    device=device,
)
if worker_rank == 0:
    communication.save("communication.ssp")
```

An explicit `collectives=[...]` replaces derived probes; `collectives=[]` selects
host copies only. A probe is a plain dictionary, for example
`dict(operation="reduce_scatter", ranks=[0, 1], tensor_bytes=64 << 20,
dtype="torch.bfloat16")`. Supported operations are all-gather, reduce-scatter and
all-reduce. Bytes describe the full padded gather output/scatter input, or the
full tensor on each all-reduce participant. Sizes must contain whole elements in
each shard. Duplicate probes are removed. Calls without Program, overrides or
rank_map retain the original local host-only behavior.

Gather probes accept `inplace=True` when the input shard occupies its ordered
slot inside the full output buffer. Program-derived weight gathers select this
form. Calibration preserves that layout, and cost lookup distinguishes it from
a gather with separate buffers. All-reduce is in-place. Native gradient evictions
use in-place reduce-scatter; optimizer tasks use a separate output so the input
accumulator remains available to an overlapping spill.

Derived requests cover lowered weight fetches, gradient completion and update
publication tasks. Model-owned communication requires its own lowering before
automatic derivation is supported. Exact observed collective timings do not infer
a general scale-up/out rate or overlapping-group contention.
CPU/Gloo coordination, single-GPU copies and two-H100 NCCL calibration have
hardware checks. Calibration and saved-plan execution have separate tests.

CommunicationInfo also owns communication memory evidence. Each collective row
can contain a `memory` mapping with `workspace_bytes`, `memory_kind` and per-rank
allocation traces/contracts. Calibration observes the same allocating-output
callable used for execution, separately from event timing. Returned collective
outputs are accounted as Program objects, not counted again as workspace.
`fixed_device_bytes` covers retained communicator resources; its metadata records
the setup, groups, topology and per-rank native/NVML observations. This remains
separate from compute/provider overhead in Profiles.

NCCL-owned allocations use ordinary external headroom. Executor warms the saved
collective forms before admitting its arena, adds measured communicator memory
to the reservation, and checks physical usage at admission and completed calls.
These external resources are distinct from task-local PyTorch allocations, which
still require allocation contracts. Estimated task memory remains simulation-only.

The IR also supports explicit framework collective tasks with declared input,
output and completion dependencies. Their durations and workspace are resolved from
CommunicationInfo during planning, so applying compute profiles does not require
a process group or dummy collective profiles. The saved ProfiledProgram can leave
both fields unspecified for those tasks. Planning requires their missing memory
information, including an explicit zero when appropriate; unknown is never zero.
Manually supplied numeric memory values default to estimates and permit simulated
timelines only. A memory-only observation can omit `duration_seconds`, allowing
timing to come from another compatible observation or the supplied rates.
Model/provider operations that communicate internally retain their ordinary
measured task duration and memory in Profiles.

Generated training plans place optimizer reductions, local updates and weight
publication on the same ordered compute resource as forward/backward. Separate
collective cost records preserve single-GPU compute profiling; they do not put
communication on an independent task stream. Each reduction/update/publication
sequence is placed together, either interleaved after the corresponding backward
work or at the tail. Reports retain phase labels and ring-equivalent traffic.

Parameter gathers remain inside compound fetches. Gradient reduce-scatter and
optional cross-replica all-reduce remain inside compound evictions. These transfers
can overlap unrelated compute using the existing readiness/completion boundaries.
The generic IR can describe independent task lanes, but concurrent shared-reader
retirement is not qualified; generated DP plans deliberately retain ordered task
execution.

The runtime's existing task-boundary handshake acknowledges submission of queued
collective transfers before the next explicit task is dispatched. Within a batch,
collective evictions precede collective fetches, independent of event polling.
This is a host submission-order guarantee, not a GPU completion wait. A native
mock-backend test verifies interleaving under delayed callbacks while GPU work
remains gated. Native two-H100 transfer tests cover FP32/BF16 gradient spill and
restore; the saved Executor has a separate end-to-end test.

Distributed workers initialize CPU rendezvous before constructing Executor. Each
worker loads the same Plan and supplies the same complete launcher-to-logical-rank
map. Executor owns the groups it creates, checks backend/topology compatibility,
and leaves caller-owned rendezvous intact. Each worker needs a distinct GPU.

```python
import os
import torch.distributed as dist
from shadowspill import Executor, Plan

dist.init_process_group("gloo")
try:
    with Executor(
        Plan.load("training-plan.ssp"),
        device=int(os.environ["LOCAL_RANK"]),
        rank_map={0: 1, 1: 0},
    ) as executor:
        executor.initialize(fill_owned_chunk)
        result = executor.run(args=local_args, kwargs=local_kwargs).result()
finally:
    dist.destroy_process_group()
```

The initial training integration uses symmetric DP with weights, gradients and
optimizer data sharded over the same groups. Other ownership combinations and
model-owned communication still require their lowering/binding implementation.
Tensor loaders/exporters see original tensor geometry, global element offsets,
and the worker's owned region. Checkpoint helpers save one archive per logical
rank; callers supply distinct paths. A checkpoint rejects the wrong logical rank
but can be restored by a different launcher rank mapped to that same logical rank.

The GPU allowance includes fixed compute/provider and communication overhead.
Planning subtracts both before scheduling objects/workspace, and reporting adds
them back to the total peak. Background weight-fetch and gradient-eviction callbacks have no task
allocation scope, so their observed scratch is conservatively reserved per
sharded object and collective operation rather than assumed to retire at the next asynchronous submission.
Explicit communication tasks account for scratch over their task lifetimes.
Spill capacity is separate. No physical memory is allocated for these calculations.

Task resource lanes bind to stable CUDA streams in the worker. Native task and
physical-reuse dependencies insert stream waits; independent lanes can overlap.
Each invocation joins its lanes before output acquisition and result completion,
including when successive calls originate on different caller streams. Retained
BLAS workspaces are initialized before the fixed execution arena is admitted.
Profiling includes per-resource library capacity in fixed memory, and the saved
setup records the BLAS backend and workspace settings. The cuBLAS/cuBLASLt bound
is conservative when the libraries share a workspace.

`sweep_programs({name: profiled_program, ...}, memory_budgets=[...],
communication_info=...)` is an optional loop. It keeps every successful Plan,
diagnostic failures, and a `best_per_budget` lookup. It does not construct or
profile candidates. Invalid arguments and implementation errors propagate.

Each phase result saves as one atomic archive with a versioned JSON description
and checksums. Compiled payloads are deduplicated inside the archive and remain
opaque bytes during CPU inspection. ProfiledProgram contains its Program and profiles; Plan contains
the selected schedule, simulation, assumptions and diagnostics. No compiler cache
directories are included. Reporting is explicit and separate from persistence.
The native simulator currently records memory peaks rather than a complete
allocation history. Reports therefore show peak-versus-capacity bars alongside
the compute/transfer timeline; they do not invent a memory curve.

`measure_program(program)` restores the finalized tasks on a representative GPU
and returns a reusable Profiles collection. It uses PyTorch's ordinary native
allocator with bounded allocation observation, not ShadowSpill pools. By default
it measures all unique task forms; `stages=[...]` selects stage indices and omits
standalone optimizer work. Backward inputs include saved values from the exact
forward form. An `inputs` mapping can supply flat task arguments by profile key
when automatic values are unsuitable or control semantics are unknown.
Build and measurement preserve the caller's default CPU and target-device random
generator states. Execution consumes random values normally; preparation does
not advance the training stream.

Structural capture does not read model payloads to compute inter-stage controls.
When an integer/boolean input is produced by another stage, build-time provider
finalization and profiling obtain it by replaying the compiled producer with the
ordinary profiling inputs. This preparation runs outside task timing and also
works when profiling a stage subset. Caller overrides apply to producer inputs
as well. Only the needed dependency slice is materialized, with CPU snapshots
between producers; generated index arrays are not embedded in Program archives.
Unknown root controls still require explicit values. CPU task outputs retain
their dependencies but have zero GPU extent, and Executor holds their host values
until the last ordered consumer.

Measurement records durations, allocation lifetimes, output aliases and observed
fixed memory. Completed native frees permit a later equal-sized allocation on
the same allocation stream to reuse that slot, independent of native addresses.
This lets fixed placement share a task-local slot across sequential temporaries;
otherwise it would reserve every temporary until task completion. Different
extents or streams remain separate leases. The saved allocation contract still
describes requests and ownership, independently of native pointer choices.

`external_headroom_bytes` (512 MiB by default) additionally reserves
worker headroom in that fixed-memory total; Executor inherits this saved setting.
This is conservative: profiling observes loaded library/context memory, while
the worker reserves headroom above its startup baseline. Planning therefore
includes the reservation before fitting the task arena into the total GPU budget.
The build/target setup must match. It does not compile or change
task definitions. Provider resources that cannot yet be observed completely
remain an execution-qualification concern; they are not silently treated as zero.

Workspace totals permit logical simulation. They do **not** establish allocation
order, storage handoffs, charged alignment, persistent provider allocations or
the task code required for execution. Built Programs now contain packaged
task code that has been tested in a fresh process with empty compiler caches.
With complete observed allocation evidence, planning also constructs a physical
`memory_layout` entirely on CPU. It validates the saved traces against compiler
storage contracts, allocates ranges, and includes dependencies needed for safe
range reuse in the simulated timeline. Layout and admission facts travel in the
Plan archive, with identities checked against its program and schedule.

Estimated memory, missing code, or incomplete allocation evidence keep a Plan
simulation-only, as reported by `simulation_only_reasons`. Inconsistent evidence
is an error. Physical placement is necessary for execution but does not replace
the worker's dependency, provider-resource and available-memory checks. The
new saved-plan Executor now exercises these checks in a fresh worker. Its initial
qualification covers small dense forward/backward/AdamW tasks, saved/recomputed
stages, tied weights, bounded initialization/export and repeated calls. Runtime
tensor hyperparameters and per-call gradient accumulation also work. General
provider qualification and normal caller migration remain in progress.

Packaged tasks retain importable Python custom-op registration modules, their
source digests and operator schemas. Measurement and execution restore/check
these registrations before loading task code; the original model need not be
imported. Native or dynamically created registrations, cross-module kernel
overrides and provider-owned resources still require a provider initialization
contract. They are not certified by this stateless-Python-op qualification.
Likewise, the registration digest is not a fingerprint of all transitive native
libraries or proof that a provider's internal autotuning/resources are portable.

For Python providers that use Triton's autotuner, build now runs each affected
task with bounded representative inputs and records its selected configurations.
Backward uses the corresponding forward's tensors. Kernel source, tuning keys
and candidate configurations identify each selection, and selections contribute
to profile compatibility. Measurement and execution bind these saved choices;
they reject unseen specializations or changed candidates before benchmarking.
No GPU timing records or runtime pools are produced by this build pass.

Only compact configuration records belong to the Program archive. Compiler
caches remain disposable, and provider code remains an external dependency.
Fixed kernels may compile lazily without selecting a different implementation.
Other autotuner systems and shared provider resources still need their own
initialization support; this does not certify arbitrary provider state.

MLOps AdamW also passes cold-worker execution with BF16 compute weights, FP32
gradients/masters/moments, CPU step counters, changing learning rates and
checkpoint restore against independent PyTorch updates. Its fixed-block Triton
kernel compiles lazily in an empty cache; this does not select a new algorithm,
change the observed allocation requests, or recompile on subsequent updates.
Such fixed provider compilation is allowed by the build contract. Separately,
the tiny Qwen3.5 MoE LoRA recipe passes cold checkpoint resume with the FLA
Triton choices finalized at build, including exact tensor and loss comparison.
Neither check qualifies providers that recreate shared resources.

Implementation ownership:

- `artifacts/`: immutable phase results, editable profile collections, archive codec.
- `build/`: model capture, finalized task packaging, static memory and lowering.
- `profiling/apply.py`: CPU validation/application of supplied profiles.
- `profiling/measure.py` and `profiling/observation/`: ordinary GPU measurement
  and normalized allocator observations; separate from execution setup.
- `planning/`: public CPU planning/sweep and capacity/cost conversion.
- `communication/`: saved communication assumptions and standalone GPU calibration.
- `reporting/`: semantic task/transfer tables and timeline/memory figures.
- `ir/` and existing native planner/simulator: shared structural validation and engines.

### Execution in a fresh worker

Load the saved Plan before constructing an Executor. The worker must not have
initialized CUDA with another allocator. Build/measurement belong in a different
process; this worker needs neither the original model object nor compiler caches.

```python
from shadowspill import Executor, Plan

executor = Executor(Plan.load("training.ssp"))
executor.initialize(fill_tensor)  # fills provided, bounded CPU spill views
pending = executor.run(args=[(features,)], kwargs=[{"targets": targets}])
result = pending.result()
```

`run` returns a result handle; the next `run` waits for an outstanding invocation.
Forward results preserve the model's output structure. Training returns losses
and metrics per example. Outputs are caller-owned across invocations; release
held GPU outputs before closing the Executor. A refused close can be retried
once those outputs have been released.

An Executor can admit named forward Plans alongside its default Plan:

```python
executor = Executor(training_plan, plans={"evaluation": validation_plan})
executor.initialize(fill_tensor)
update = executor.run(args=training_examples).result()
predictions = executor.run(plan="evaluation", args=validation_examples).result()
```

All Plans must use the same target/build and memory budgets. Model tensors are
matched by qualified registration name, exact dtype/shape/stride/storage offset,
and aliasing. Internal object IDs may differ. Evaluation reads the current compute
weights and buffers; master weights and optimizer data remain training-owned.
Additional forward Plans must not mutate shared model tensors. Different input
shapes are supported by separately built Plans; execution never rebuilds one.

Admission checks capacity for all Plans before opening pools. Sequential calls
reuse one fixed GPU arena sized for the largest saved layout. Persistent spill
storage is the union of shared model tensors and each Plan's private inputs and
constants, plus active invocation requirements. Measured provider overhead is
conservatively summed across distinct Programs until provider resource sharing is
explicitly described. Plans that fit separately can therefore fail joint admission.
Initialization/export visit shared model tensors once; private constants use
`plans.<name>.<tensor-name>`. Checkpoint restore requires the same named Programs.

A forward/backward Plan contains both fresh and accumulating invocation paths.
The default starts fresh gradients. `accumulate_gradients=True` uses stored
gradients. Both paths share a physical arena and persistent tensor identities;
the archive includes both schedules and memory admission records. There is no
runtime planning or hidden eager gradient addition.

`hyperparams={"optimizer_group.0.lr": value}` overrides a captured tensor
hyperparameter for that call. Omitted entries use the captured defaults;
Python numeric optimizer settings remain specialized constants. CPU controls
remain on the host, and device control tensors occupy planned memory.

Initialization uses `TensorChunk` metadata and fills an Executor-owned buffer.
Normal initialization loads model tensors and applies optimizer/gradient defaults.
`initialize(fill_tensor, restore=True)` instead asks the loader for all persistent
tensors, including optimizer data and gradients. `export_tensors(write_chunk)`
streams the same named storage in bounded views. Callbacks must consume/fill each
view before returning and must not retain it. Checkpoint file formats belong to
helpers above this interface, not the allocator or task dispatcher.

Worker shard descriptions preserve offsets into the original tensor storage.
Load/export callbacks see only valid owned elements; padded tail elements are
zeroed internally. Master initialization converts directly from the corresponding
local compute-weight shard into each bounded destination chunk. Optimizer bindings
record their associated parameter explicitly, including names containing dots.
This shard-loading path is qualified independently; distributed Executor admission
and distributed checkpoint publication remain pending.

`save_checkpoint(executor, path, metadata={...})` streams persistent tensor storage
into one atomic archive. `load_checkpoint(executor, path)` fills the same bounded
views and returns the caller metadata. The initial helper requires the same
Program identity; memory budgets and scheduling order may differ. It preserves
storage geometry, exact dtypes and CPU optimizer controls. Chunk sizes can differ
between writing and reading. Each tensor is a packed archive member, so there
are no per-tensor filesystem writes or complete-model host snapshots. Format,
geometry and checksum checks precede successful initialization. Application
scheduler/RNG state belongs in the caller's JSON metadata; it is not discovered
implicitly.

Returned views of input or parameter storage become owned snapshots during build.
Those copies are compiled and profiled as part of the task. Views sharing that
storage still share their returned snapshot; changing it cannot change stored
parameters. Mixed-dtype views of one returned input storage currently require
explicit model-side copies. TensorChunk includes stride and storage offset for
loaders handling noncontiguous registrations; chunk offsets address physical
storage rather than a flattened logical tensor.

### Trainer over a saved Plan

`Trainer(plan, ...)` runs in the same fresh execution worker as Executor. It does
not accept a model, compile, profile, calibrate communication, or plan. The usual
caller can let it create the Executor; advanced callers pass `executor=existing`.
The latter remains caller-owned. A separate launcher (for example, quickstart)
starts the worker; Trainer does not hide a child process behind its methods.

```python
from shadowspill import Plan, Trainer

with Trainer(
    Plan.load("training.ssp"),
    evaluation_plan=Plan.load("validation.ssp"),  # optional build_forward Plan
    schedules={"optimizer_group.0.lr": lambda completed: 1e-3 / (completed + 1)},
) as trainer:
    trainer.initialize(fill_tensor)  # same bounded-buffer callback as Executor
    result = trainer.step(
        args=[(features0,), (features1,)],
        kwargs=[{"targets": targets0}, {"targets": targets1}],
    )
    validation = trainer.evaluate(args=[(validation_features,)])
    trainer.save("checkpoint.ssp")
```

The Plan fixes the number and structure of rank-local examples. `step` accepts
Executor's sequences or indexed `input_loader`, waits for completion and returns
host-only losses/metrics. It sums the objective's contributions without adding
normalization. Schedules receive the zero-based completed-update count and may
only set declared runtime hyperparameters; a tensor-valued optimizer learning
rate during build makes it such an input. Per-call values override schedules.

`evaluate` calls the admitted `evaluation` Plan through the same Executor and
returns its normal output structure on CPU. It does not update the optimizer,
advance schedules, or increment the training step count. Include any validation
loss/metrics in that forward computation so they are also built, profiled and
planned. Evaluation cadence belongs in the caller's loop or `fit` callbacks.
For a supplied Executor, Trainer uses its named `evaluation` Plan; if an explicit
`evaluation_plan` is also supplied, it must match the admitted one.

`fit(source, steps=N)` takes an iterable whose items are keyword argument maps
for `step`; `N` is the total completed-update target, including resumed updates.
For example, each item can be `{"args": [(features,)], "kwargs": [{"targets":
targets}]}`. No new batch wrapper or automatic input sharding is required.
`run_dir`, logging/table cadences, callbacks and periodic checkpoints are optional.
Callbacks run before the checkpoint for that update.

Optional `parameter_norms_every=N` collects pre-update compute-weight and final
gradient norms through Executor, then supplies CPU scalars to the existing
`metric_reducer`. It adds no compiled tasks or full-tensor copies. Fixed reduction
buffers must fit inside the existing execution pool; per-task allocation contracts
remain unchanged. Direct Executor callers enable `parameter_norms=True` and can
select collection per invocation. See [parameter observations and logging](../python/api/training.md#parameter-norms-and-custom-logging)
for timing, ownership and sharded aggregation behavior. Arbitrary GPU observation
callbacks are not supported by this bounded path.

`load("checkpoint.ssp")` restores all persistent tensors and loop progress.
A source with `state_dict`/`load_state_dict` resumes before the next `fit` reads
it. Stateful schedules use the same methods. Default Python, NumPy and PyTorch
RNG states are preserved; custom generators remain application-owned. Loop state
supports small plain Python containers and CPU tensors/arrays, without pickle.
Checkpoints are single bounded-stream archives; partial weight-only export and
model checkpoint conversion are separate from this full-resume path.

Trainer runs saved `build_step` Plans and optional read-only evaluation sharing
trained parameter storage. The [distributed regression recipe](../../workloads/recipes/regression/README.md)
uses the same phase APIs, with separate collective calibration. Remaining
model/prepare callers explicitly use the private preparation implementation until
migrated; the public Trainer has only the saved-Plan API.

### Logical parallelism and distributed plans

All builders accept `parallelism=Parallelism.dp(...)`. Omitting it describes one
rank. The declaration contains no process groups or devices and survives saving
the Program. `ranks_per_replica` describes a group collectively owning the whole
model; `num_replicas` repeats that group. Simple DP defaults to sharded weights,
gradients and optimizer tensors, with masters following optimizer ownership.

`Parallelism`, `ParameterStorage`, `Replicated`, `Sharded` and `GroupRef` are
framework-neutral. Advanced declarations use named ordered groups plus separate
`parameter_placements` and `parameter_storage` maps. Exact names take precedence
over wildcard patterns. Tied names resolve to one entry; conflicts at the winning
priority and unmatched selectors are errors. Storage entries inherit omitted
fields only from `default_parameter_storage`. `parallelism.group(name)` supplies
a typed reference exposing static membership and size, without creating a live
communicator.

The initial distributed build paths support forward, forward/backward and full-step
computation with replicated local parameters and storage sharding. They build and
measure on one representative GPU without a process group. Planning then
models local host-shard copies followed by the full weight all-gather:

```python
program = build_forward(
    model, example_args=(features,),
    parallelism=Parallelism.dp(ranks_per_replica=8),
)
profiles = measure_program(program)
profiled = apply_profiles(program, profiles)

# This portion can run in a CPU-only process with no Torch installation.
plan = plan_program(
    profiled,
    memory_budgets=MemoryBudgets(execution_bytes=60 << 30, spill_bytes=96 << 30),
    communication_info=CommunicationInfo.load("communication.ssp"),
)
```

The communication archive may contain calibration or explicit simulation estimates.
The shared schedule describes one symmetric rank
and uses the slowest corresponding group instance's transfer cost. Memory and
traffic totals are **per rank**, not summed over the job. Complete compute and
communication memory evidence allows distributed admission and worker binding;
estimated memory remains simulation-only. Model-sharded execution, annotated
distributed input layouts and already-reduced-gradient overrides remain pending.

`build_forward_backward` additionally supports sharded gradient storage. Local
backward uses full logical accumulators, but an eviction can reduce and spill the
owned shard between examples. A fetch zeros the full accumulator and restores
only that shard. With repeated model replicas, only the first corresponding
replica injects the old value; new local contributions then sum without repeating
the previous total. Final gradients use the same reduce-and-spill path.
For uneven sizes, build includes equal-shard padding in the compiled backward's
allocation contract before reconciling it with the canonical storage layout;
the physical output is larger than its logical tensor view where necessary.

One saved Plan supports fresh gradients and cross-call accumulation. The latter
starts from existing spilled shards and uses the same compiled accumulation
forms. No separate old-plus-new shard addition is needed. Measurement on one GPU
covers local compiled work without executing collectives. Calibration distinguishes
in-place transfer reductions from allocating optimizer reductions and records the
buffer-zero time used on gradient fetch. Missing zero timing is explicitly
estimated using the host-fetch rate. Collective scratch must still be supplied;
manual memory estimates permit simulation only. CPU archive/planning tests forbid
Torch, build and runtime imports. Replicated gradient storage remains later work.

`build_step` supports the common case with weights, gradients and optimizer data
sharded over the same groups. Gradient completion feeds shard-shaped local
optimizer inputs. Masters and moments have independent owned allocations.
The compute-weight shard is a view of the full compute buffer, with its logical
group-relative offset recorded for worker binding. Build also accounts for the
alignment of every owner's interior view: one rank's shard may be aligned while
another's starts between vector boundaries. Code generation must support those
views without inserting an unprofiled alignment copy. Masters and optimizer
moments retain their separate, aligned allocations.

An in-place all-gather on the same ordered compute resource
publishes the updated weights immediately after the local update. Update placement waits for
all contributions and reads of old weights, including aliased views.

This initial path keeps the full compute-weight buffer available during the
update; the plan accounts for that allocation and any required fetch. It does not
claim shard-only device residency during optimization. Differing component groups
need explicit redistribution and are currently rejected. The saved local AdamW
updates are checked over four logical owners on one GPU, including padded tails,
with and without FP32 masters. Local compute measurement does not stand in for
NCCL calibration: communication memory/timing evidence comes from
`measure_communication`, and the saved worker path needs multi-GPU qualification.

CommunicationInfo accepts flat scale-up/scale-out rates and latencies plus
`measurements`, a sequence of plain mappings. An observation contains
`operation` (`all_gather`, `reduce_scatter`, or `all_reduce`), ordered `ranks`,
`tensor_bytes`, `dtype`, optional `duration_seconds`/`memory`, hardware/runtime
`setup`, and `topology`.
For gather/scatter, tensor bytes are the **padded full output/input**, not the
local shard. Exact operation, membership/order, size, dtype, setup and topology
must match. The latest matching observation wins. Uncovered cases use a flat-ring
estimate, with missing topology/rates/latencies reported as errors. Scale-out
costs use the slowest link type in that ordered ring; extra shared-NIC contention
and alternative collective algorithms are not yet modeled.

Reports distinguish host bytes from ring-equivalent collective traffic. Combined
copy/collective intervals are purple, and do not claim an isolated host-copy
rate. The simulator currently retains parent transfer timestamps, not individual
phase intervals. Summary `fetch_bytes`/`evict_bytes` include both legs, alongside
explicit host and collective byte totals. These totals count each direction's
per-rank traffic, not a sum of both send and receive bandwidth.

### Saved code and worker devices

The current Inductor artifact format embeds a CUDA ordinal. Build records it;
measurement and execution require the same process-local ordinal. In particular,
setting `device=1` does not relocate code compiled for `cuda:0`. Device visibility
can be arranged before CUDA initialization so separate workers each see their
assigned physical GPU at ordinal zero. Older artifacts without the recorded
binding can still be inspected/planned on CPU, but need rebuilding for execution.
This limitation is separate from logical rank assignment.

Worker-group binding preserves explicit logical membership order
after applying the complete worker-to-logical-rank map and rejects inconsistent
maps/Plan identities before backend group creation. CPU multi-process checks
cover group ordering and SUM/reduce-scatter geometry. Separate two-GPU tests
exercise complete build/profile/calibration/CPU-plan/fresh-execution paths,
including odd-sized shards, repeated calls, checkpoint restore and shared
evaluation. Primitive group checks alone do not establish runtime correctness.
