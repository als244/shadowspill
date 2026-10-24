# Independent build, profile and planning phases

The refactor is being implemented in phases. The CPU path, builders and
ordinary-allocator GPU measurement and initial saved-plan execution work; caller
migration are still in progress on `codex/offline-pipeline-refactor`.

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

The current offline milestone supports one logical rank and host transfers.
Explicit latency is charged as a transfer phase. Concurrent fetch/evict rates
default to independent full-duplex lanes; callers can provide different rates.
The communication description is saved alongside the plan, independently of
compute measurements. Collective calibration, topology-aware collective estimates,
and distributed transfer layouts are deferred to the distributed milestone.

`measure_communication()` optionally measures pinned-host copies on one GPU,
using ordinary PyTorch allocation and CUDA events. It returns CommunicationInfo;
save it independently and pass it to any later planning run. A size sweep fits
startup latency and bandwidth, and simultaneous copies measure overlapping lane
rates. Raw medians, sample counts, variability and fit errors are retained in
metadata. These are local measurements under the caller's host placement, not
aggregate multi-GPU bandwidth. Neither planning nor execution triggers this probe.
Buffer use is bounded by twice the largest requested size on each of host and GPU.

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
