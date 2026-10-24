# Consolidated public API review

Status: design review record. See [progress](progress.md) for implementation
status; distributed examples remain planned. This document brings the selected
phase boundaries, defaults, execution behavior, and migration policy into one
common workflow. [README.md](README.md) records the decisions. Exact supporting
record/callback protocols still requiring review are listed at the end; a
working-looking example does not silently finalize those details.

## Main calls and objects

| Public call | Result | Responsibility |
| --- | --- | --- |
| `build_forward` | `Program` | Model forward computation. |
| `build_forward_backward` | `Program` | Objective, backward, and persistent gradients over the declared examples. |
| `build_step` | `Program` | Objective, backward, and optimizer computation. |
| `measure_program` | `Profiles` | GPU task timing and required memory observations. |
| `apply_profiles` | `ProfiledProgram` | CPU validation/application of complete supplied profiles. |
| `measure_communication` | `CommunicationInfo` | Optional, independent communication calibration. |
| `plan_program` | `Plan` | CPU scheduling, placement, and simulation for supplied budgets/costs. |
| `sweep_programs` | Per-case results and plans | Thin convenience over planning supplied candidates/budgets. |
| `write_report` | Written report | CPU reporting from an existing Plan or sweep result. |
| `Executor.run` | Invocation result handle | Dispatch a saved plan inside a prepared execution worker. |

Trainer and quickstart compose these calls. Internal compiler passes, runtime
pools, and caches are not extra prerequisites in the public phase pipeline.
Build finalizes implementations/layouts and may require a representative GPU;
measurement records costs without changing the task definitions. CPU profile
application/planning/reporting never performs hidden compilation or measurement.

`Program`, `ProfiledProgram`, and `Plan` have read-only public structure;
`Profiles` is editable. Each supports explicit save/load as one archive.
ProfiledProgram includes its Program and applied profiles. Plan contains the
selected executable payloads and logical-rank declarations, independently of the
original compiler-cache paths. Checkpoint values and provider libraries remain
external. No mandatory ArtifactStore or additional Pipeline class is needed.

## One replica using eight DP ranks

The caller supplies an ordinary `nn.Module`, a scalar training objective with
signature `objective(model, *args, **kwargs)`, a compatible optimizer definition,
and representative rank-local arguments. The model can be constructed on meta;
build does not initialize full training state. The precise optimizer/target
descriptors and precision defaults belong to the build/provider review.

`example_args` supplies positional arguments and `example_kwargs` supplies named
arguments. Training builders call `objective(model, *example_args,
**example_kwargs)`; `build_forward` calls `model(*example_args, **example_kwargs)`
through the module's normal call/forward boundary. For example:

```python
def objective(model, features, targets, mask=None):
    predictions = model(features, mask=mask)
    return loss_fn(predictions, targets)
```

Here the model's own forward accepts `features` and `mask`; only the objective
needs `targets`. These argument names are illustrative and impose no model- or
dataset-specific schema.

```python
parallelism = Parallelism.dp(ranks_per_replica=8)

program = build_step(
    model,
    objective=objective,
    optimizer=optimizer,
    example_args=(features, targets),
    example_kwargs={"mask": mask},
    num_examples=8,
    target=target,
    parallelism=parallelism,
)
program.save(program_path)
```

The shorthand supplies the selected fully sharded storage default for weights,
gradients, and optimizer data; master weights follow optimizer ownership. It
describes one complete-model replica with eight DP workers. Their rank-local
inputs can differ. `num_examples=8` means eight complete objective calls per rank
within this Program invocation; each call may itself contain a batch.

The common defaults already selected are automatic stage partitioning and the
existing operator-recomputation heuristic, with built alternatives available to
planning. An advanced caller supplies `partition=...` or
`operator_recompute=...` explicitly. Neither data ordering nor optimizer ordering
is a build argument. Loss normalization remains the objective's responsibility.

The two other computation boundaries use the same control style:

```python
forward = build_forward(
    model,
    example_args=(features,),
    example_kwargs={"mask": mask},
    target=target,
    parallelism=parallelism,
)

forward_backward = build_forward_backward(
    model,
    objective=objective,
    example_args=(features, targets),
    example_kwargs={"mask": mask},
    num_examples=8,
    target=target,
    parallelism=parallelism,
)
```

The objective and forward call have separate argument examples because labels or
other objective inputs need not be accepted by the model's forward. The
forward/backward Program retains gradients and does not update parameters. It
supports both fresh and accumulating invocations, with fresh gradients the
selected default. Final gradients are reduced into the declared storage layout
unless synchronization was explicitly deferred. Both behaviors are represented
in the same saved artifact; no runtime replan is needed.

## Measurement and CPU planning

```python
# In a GPU measurement process. Pure DP local tasks need one representative GPU;
# this example does not measure real distributed collectives.
profiles = measure_program(program, device="cuda:0")
profiles.save(profiles_path)

# In a separate CPU-only process if desired.
program = Program.load(program_path)
profiles = Profiles.load(profiles_path)
profiled = apply_profiles(program, profiles)

plan = plan_program(
    profiled,
    memory_budgets=MemoryBudgets(
        execution_bytes=60 * GiB,
        spill_bytes=96 * GiB,
    ),
    communication_info=communication_info,
)
plan.save(plan_path)
write_report(plan, report_path)
```

`communication_info` is supplied independently, manually or through the optional
`measure_communication` API. It contains topology, flat rates/latencies with
explicit units, and optional measurements. Use compatible measurements first;
estimate gaps from supplied rates and topology. Required missing costs are
reported rather than silently assumed. See [the communication review](communication-review.md).

The example budget pair applies per rank. Execution bytes include accounted
runtime/provider overhead, rather than representing only usable pool space.
Whether the plan is executable depends on the selected memory evidence, not
merely on successful simulation. Estimated memory permits simulation only;
execution rejects it. Replace that evidence and replan before execution.

`plan_program` returns a Plan or raises an exception with diagnostics.
`sweep_programs` accepts already-profiled candidates and budget choices, retaining
each successful case's Plan and unsuccessful-case diagnostics. Ordinary caller
loops can use `plan_program` directly. Neither entry point creates new model
candidates or profiles them. Data ordering and optimizer ordering are planning
choices over the Program's legal dependencies.

Saved paths are individual archive files, except `report_path`, which is the
report destination. No particular filename extension or archive codec is selected
here. In-memory objects also pass directly between phases without saving.

External measurements or estimates use the same profile collection. With an
existing compatible collection, illustrative task/profile access is:

```python
task = program.find_task(
    module="layers.3", phase="backward", variant="recompute",
)
overrides = Profiles(
    context=profiles.context,
    tasks={task.profile_key: TaskProfile(duration_seconds=0.0072)},
)
profiles = profiles.merge(overrides)
profiled = apply_profiles(program, profiles)
```

This is an illustrative duration, not a measurement. Lookup must report an
ambiguous match rather than choose one silently. Program inspection exposes the
task/stage inventory and source attribution for disambiguation. The compatibility
key identifies reusable task measurements, while the module name is for discovery.
Manual values without measurement metadata default to estimates. Later supplied
fields overwrite matching earlier values; omitted fields are retained. This
duration-only override retains the existing memory evidence. Applying profiles
captures an independent snapshot and still requires complete compatible coverage.
The displayed lookup/record/merge spellings remain the concrete API proposal.

The minimum profile record has plain numeric timing and, when needed, workspace
fields. Source/classification data belongs in ordinary metadata, not wrappers
around each number. Build supplies known storage, aliases, and allocation facts;
profiles fill unresolved requirements. A whole-device or pair-level memory peak
is not a task workspace description or proof of execution-safe allocation
behavior. Shared provider allocations are counted once by resource identity.
The full [memory evidence review](memory-and-profiles.md) explains that distinction.

## Saved-plan execution in a fresh worker

Each worker loads the same Plan archive and receives the same complete
launcher-rank-to-logical-rank map. For an eight-rank single-node plan, an example
nonidentity map is shown below. Its logical ordering must be preserved by actual
communication bindings; changing the map cannot silently change the planning
topology or capacity assumptions.

```python
# Fresh worker process, before CUDA initialization.
plan = Plan.load(plan_path)
rank_map = {worker: 7 - worker for worker in range(8)}

with Executor(plan, rank_map=rank_map, device=device) as executor:
    executor.initialize(fill_tensor)
    pending = executor.run(
        args=rank_local_example_args,
        kwargs=rank_local_example_kwargs,
    )
    # CPU work may proceed while this invocation completes.
    result = pending.result()
    executor.export_tensors(write_chunk)
```

The sequences contain one args tuple / kwargs mapping per example slot. An
indexed `input_loader` is an alternative, not an additional mandatory abstraction.
The saved plan determines task order; every replay of a slot sees the same data.
Per-call `hyperparams` can supply values declared runtime-changeable during build.

`fill_tensor(request, destination)` fills an Executor-provided bounded tensor
region. `write_chunk` consumes bounded exported regions. The descriptors carry
semantic names, component/dtype, and owned logical regions so helpers do not need
to understand generated alias IDs. Checkpoint formats and random initialization
helpers use these interfaces rather than requiring a full-model host snapshot.
Exact descriptor fields and callback lifetime rules remain in the final protocol
review below.

Executor performs setup, including the allocator and required live groups and
provider resources. It validates the full map, plan identity, relevant exact
software builds, target/capacity requirements, and executable memory evidence.
CPU Plan loading itself does not initialize GPU providers. Execution never
silently builds, profiles, or replans to repair incompatible artifacts.

`run` returns after dispatch with a result handle. `.result()` waits for that
invocation, including required writebacks, and returns its outputs or failure.
The next `run` automatically waits for the previous invocation before reusing
storage. There is one active invocation initially. Returned outputs remain valid
while the caller retains them, with their memory accounted for across calls.
Forward returns the model's native output structure; training returns losses and
metrics in stable example order. Gradients remain Executor-managed.

For a forward/backward Plan, the selected default is explicit in this example:

```python
first = executor.run(args=first_inputs).result()  # Fresh gradients.
second = executor.run(
    args=next_inputs,
    accumulate_gradients=True,
).result()  # Add to the stored gradients from the first invocation.
third = executor.run(args=third_inputs).result()  # Fresh again.
```

All examples within each invocation accumulate regardless of that flag. Already
reduced incoming gradients must not participate in a second reduction. The plan
must cover both modes' memory, dependencies, and communication before execution;
the flag does not authorize unplanned allocation or schedule mutation. This
control does not change the optimizer semantics of `build_step`.

## One sixteen-rank replica with two expert-exchange groups

This example uses explicit model parameter names. The local model already uses
one eighth of the model's experts, with the same local shapes on every rank.
The group-aware distributed MoE implementation must exist; the declaration does
not transform an ordinary local MoE into a distributed one.

```python
parallelism = Parallelism(
    ranks_per_replica=16,
    num_replicas=1,
    groups={
        "all_ranks": [tuple(range(16))],
        "exchange": [tuple(range(8)), tuple(range(8, 16))],
        "same_experts": [(i, i + 8) for i in range(8)],
    },
    default_parameter_placement=Replicated(group="all_ranks"),
    parameter_placements={
        "layers.*.experts.w13": Sharded(group="exchange", dim=0),
        "layers.*.experts.w2": Sharded(group="exchange", dim=0),
    },
    default_parameter_storage=ParameterStorage(
        group="all_ranks",
        weights="sharded",
        gradients="sharded",
        optimizer="sharded",
    ),
    parameter_storage={
        "layers.*.experts.w13": ParameterStorage(group="same_experts"),
        "layers.*.experts.w2": ParameterStorage(group="same_experts"),
    },
)

exchange_group = parallelism.group("exchange")
model = MyDistributedModel(config, exchange_group=exchange_group)
# Use the same build_step(model, ..., parallelism=parallelism) interface above.
```

`MyDistributedModel` is a stand-in for a compatible user model, not a new
ShadowSpill class. Construct it on meta when appropriate. `exchange_group` is a
typed logical reference with static group size; it is not a live process group.
The symmetric local module uses eight-way expert partitioning. Rank binding and
initialization select the corresponding expert values later. Model-specific
routing and its forward/backward communication belong to the compatible layer.

| Parameters | Local computation uses | Persistent tensor storage |
| --- | --- | --- |
| Attention and other default parameters | The same complete logical parameter on each rank | Split across all 16 ranks |
| Expert `w13`/`w2` | One eighth of the experts, with corresponding ranks in the two exchange groups using the same subset | Each subset split across its matching pair, such as ranks 0 and 8 |

The expert overrides inherit the sharded component modes from the explicit
storage default. Master weights follow optimizer ownership. Matching parameter
elements determine gradient reduction obligations; persistent storage sharding
does not eliminate those reductions. Model-specific already-reduced overrides
remain available as previously selected. Optimizer scalar/control values are not
blindly treated as parameter-shaped tensor slices.

Setting `num_replicas=2` repeats this complete sixteen-rank description for a
32-rank job and adds the required reductions between corresponding parameter
copies. The declared groups remain local to one replica. Input sharing is
independent of parameter sharing; distinct group instances/replicas use
independent inputs unless an input placement explicitly declares otherwise.

## Generic distributed inputs

The same representation can describe parameter-free distributed operations.
For example, a model may receive a typed sequence group and declare its local
input as a sequence slice:

```python
# Within Parallelism, using a separately declared group named "sequence":
input_placements={
    ("kwargs", "x"): Sharded(group="sequence", dim=1),
}
```

Input paths refer to the builder's `example_args`/`example_kwargs` tree, not to
parameter names. A composed input layout accepts a list of placements, as
selected. Runtime callers still supply local tensors and any positions/masks
needed by the actual distributed algorithm. The previously discussed optional
non-contiguous mapping remains in the full API draft for the concrete mapping
schema review; this example does not drop that requirement.

## Source organization and migration

The implementation uses phase directories and a small top-level public API.
Keep files/functions cohesive, use hierarchy for real components, and define
shared graph/storage facts once in IR/artifacts. Avoid forwarding layers and
duplicate wrapper types. Trainer and quickstart use the same public phases as
direct callers. The [source layout review](source-layout-review.md) records the
accepted ownership and old-to-new entry-point map.

Work occurs on `codex/offline-pipeline-refactor` in the separate checkout. Migrate
repository callers, then remove superseded public APIs and duplicate code.
Deprecated public compatibility wrappers were not selected. Preserve advanced
capabilities such as tied storage, precision controls, parameter observations,
and bounded checkpoint operations under the new owners before removal.

## Final protocol review, without more configuration layers

The two last identified behavioral choices are settled: remove old APIs after
migration, and default to fresh gradients. The common phase/API/execution shape
above incorporates those choices. It does not imply that every public record
field or extension callback has been approved. Finish these details together:

| Boundary | Concrete detail still to specify under the selected behavior |
| --- | --- |
| Task and profile inspection | Exact task lookup/key and numeric record fields; allocation/evidence metadata sufficient to distinguish simulation from execution. Reuse authoritative IR facts. |
| Build/provider interface | Explicit target/preparation parameters and portable resource initialization/binding descriptions, retaining existing precision and optimizer capabilities. No blanket new registry or model base class. |
| Execution helpers | Exact init/export descriptor fields and borrowed-buffer lifetime; indexed-input replay/lifetime rules; runtime hyperparameter keys and gradient presence/loading validation. |
| Communication calibration | Operation/size/dtype and complete rank-binding arguments; measured-record matching and cost-model byte/latency conventions. |
| Archives | Concrete container/schema/version/payload encoding under the selected one-archive and exact-relevant-build policy. No new public manifest object. |

Normal-allocator observation, portable compiler payloads, and preservation of
logical collective ordering require engineering validation. They are not new
preference polls. If a concrete feasibility result requires changing selected
behavior or adding a public abstraction, present that specific issue before
adopting it. No GPU allocation or source implementation starts as part of this
design-note update.
