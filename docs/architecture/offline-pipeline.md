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
requirements for provider calls. No state initialization or task timing occurs.
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

```python
# All workers have initialized rendezvous and loaded the same Program.
communication = measure_communication(
    program=program,
    rank_map={worker: worker for worker in range(8)},
    device=device,
)
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
a gather with separate buffers. All-reduce is in-place; reduce-scatter currently
uses separate buffers.

Derived requests cover lowered weight fetches, gradient completion and update
publication tasks. Model-owned communication requires its own lowering before
automatic derivation is supported. Exact observed collective
timings do not infer a general scale-up/out rate, overlapping-group contention,
or communicator memory admission.
The CPU/Gloo coordination and single-GPU copy paths are tested; multi-GPU NCCL
calibration has an acceptance test but is not yet hardware-qualified.

The IR also supports explicit framework collective tasks with declared input,
output and completion dependencies. Their durations are resolved from
CommunicationInfo during planning, so applying compute profiles does not require
a process group or a dummy zero-time collective measurement. Workspace information
is still required; unknown workspace does not become zero. The saved
ProfiledProgram retains an unspecified duration for these collective forms until
planning. Model/provider operations that communicate internally retain their
ordinary measured task duration.

Explicit framework collectives share the ordered communication resource used by
weight fetch gathers. Their buffers and scratch remain live while the task is in
flight, and consumers depend on completion. Reports show a separate collective
lane and label ring-equivalent traffic. The analytic CPU test exercises overlap
with independent compute and replans with different collective timings from the
same saved ProfiledProgram. Distributed Executor resources still need qualification.

The GPU allowance includes fixed runtime/provider overhead. Planning subtracts
that overhead before scheduling objects/workspace, and reporting adds it back to
the total peak. Spill capacity is separate. No physical memory is allocated to
perform this calculation.

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

Measurement records durations, allocation lifetimes, output aliases and observed
fixed memory. The build/target setup must match. It does not compile or change
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

MLOps AdamW also passes cold-worker execution with BF16 compute weights, FP32
gradients/masters/moments, CPU step counters, changing learning rates and
checkpoint restore against independent PyTorch updates. Its fixed-block Triton
kernel compiles lazily in an empty cache; this does not select a new algorithm,
change the observed allocation requests, or recompile on subsequent updates.
Such fixed provider compilation is allowed by the build contract. This check
does not qualify providers that perform autotuning or recreate shared resources.

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

The internal `_legacy.split_costed_program` imports old captured IR for validation
while callers migrate. It is not a second public build path and does not claim
that imported workspace totals carry allocator evidence.


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

`load("checkpoint.ssp")` restores all persistent tensors and loop progress.
A source with `state_dict`/`load_state_dict` resumes before the next `fit` reads
it. Stateful schedules use the same methods. Default Python, NumPy and PyTorch
RNG states are preserved; custom generators remain application-owned. Loop state
supports small plain Python containers and CPU tensors/arrays, without pickle.
Checkpoints are single bounded-stream archives; partial weight-only export and
legacy checkpoint conversion are separate from this full-resume path.

This milestone qualifies local `build_step` plans and read-only evaluation sharing
trained parameter storage. Distributed Trainer callers and old recipe migration
are still pending. Remaining model/prepare callers explicitly use the private
legacy implementation until migrated; the public Trainer has only the new API.

### Logical parallelism and the first distributed simulation path

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
    communication_info=CommunicationInfo(
        fetch_bytes_per_second=25e9, evict_bytes_per_second=25e9,
        topology={"node0": tuple(range(8))},
        scale_up_bytes_per_second=300e9, scale_up_latency_seconds=2e-6,
    ),
)
```

These rates are illustrative. The shared schedule describes one symmetric rank
and uses the slowest corresponding group instance's transfer cost. Memory and
traffic totals are **per rank**, not summed over the job. These distributed Plans
are currently simulation-only: communicator resources, distributed allocation
admission and execution binding still need qualification. Model-sharded execution,
distributed inputs, and
already-reduced-gradient overrides also remain pending.

`build_forward_backward` additionally supports sharded gradient storage. Local
backward tasks accumulate full contributions across the call's examples. Optional
padding and explicit reduce-scatter produce the owned gradient shard; repeated
model replicas then all-reduce matching shards. Planning inserts each completion
after its last local contributor, independently of parameter declaration order.
All in-flight inputs, outputs and scratch remain part of memory accounting.

One saved Plan supports both fresh gradients and cross-call accumulation. The
latter reduces only the new contribution, then adds it locally to the stored
already-reduced shard. Measurement on one GPU covers the local padding/addition
tasks, without executing collectives. Collective scratch must be supplied if
unresolved; a manual estimate permits simulation only. The archive/planning path
is tested in a process that forbids Torch, build and runtime imports. Replicated
gradient storage and distributed execution remain later work.

`build_step` supports the common case with weights, gradients and optimizer data
sharded over the same groups. Gradient completion feeds shard-shaped local
optimizer inputs. Masters and moments have independent owned allocations.
The compute-weight shard is a view of the full compute buffer, with its logical
group-relative offset recorded for worker binding. An explicit in-place all-gather
publishes the updated weights after the local update. Update placement waits for
all contributions and reads of old weights, including aliased views.

This initial path keeps the full compute-weight buffer available during the
update; the plan accounts for that allocation and any required fetch. It does not
claim shard-only device residency during optimization. Differing component groups
need explicit redistribution and are currently rejected. The saved local AdamW
updates are checked over four logical owners on one GPU, including padded tails,
with and without FP32 masters. This validates local math and bindings, not NCCL
execution. Collectives and communicator memory still require separate qualification.

CommunicationInfo accepts flat scale-up/scale-out rates and latencies plus
`measurements`, a sequence of plain mappings. An observation contains
`operation` (`all_gather`, `reduce_scatter`, or `all_reduce`), ordered `ranks`,
`tensor_bytes`, `dtype`, `duration_seconds`, hardware/runtime `setup`, and `topology`.
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

Worker-group binding is being qualified independently before distributed
Executor admission is enabled. It preserves explicit logical membership order
after applying the complete worker-to-logical-rank map and rejects inconsistent
maps/Plan identities before backend group creation. CPU multi-process checks
cover both group ordering and SUM/reduce-scatter geometry. They do not establish
multi-GPU runtime correctness or communicator memory requirements.
