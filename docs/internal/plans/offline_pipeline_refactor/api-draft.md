# Public API and object draft

Status: proposal for review, not implemented Python. The phase boundaries and
behaviors recorded in the [decision log](README.md) are agreed. Provisional
signatures, fields, policies, and serialization details below are marked as such;
selected defaults and the migration/removal policy are recorded explicitly.

## Public functions

The shared pipeline would have these entry points:

```python
build_forward(...) -> Program
build_forward_backward(...) -> Program
build_step(...) -> Program

measure_program(program, ...) -> Profiles
apply_profiles(program, profiles) -> ProfiledProgram

measure_communication(...) -> CommunicationInfo

plan_program(profiled_program, ...) -> Plan

# Optional convenience over public planning calls:
sweep_programs(candidates, ...) -> SearchResult
```

The builders run the common export/partition/differentiation/compilation/lowering
passes as appropriate to their computation. Build finalizes task definitions,
using a GPU when compiler/provider preparation requires one. Exact target and
preparation arguments remain to review. `measure_program` obtains task measurements
without changing those definitions. Measurement is optional
when supplied profiles cover the requirements. Profile application, planning, and
search are CPU-only and never trigger a build or profiling run.

The target is for every phase above to work without a ShadowSpill runtime,
runtime pools, or pluggable allocator installation. The user subsequently allowed
allocator-backed measurement if it proves necessary for admission/correctness;
information completeness takes priority over that profiling simplification.
Build, profile application, planning, and search remain independent of the runtime
and pools. The profiling backend and any fallback have not been selected.

| Phase | Device work | Allocation/process boundary |
| --- | --- | --- |
| `build_*` | Finalize task definitions/layouts; CPU structural work and GPU-assisted compilation/preparation where required | No ShadowSpill runtime or pools. |
| `measure_program` | GPU task measurement when requested | Prefer ordinary PyTorch/provider allocations with observation hooks; allocator-backed profiling is allowed if required for complete correctness evidence. |
| `measure_communication` | Optional host/device or collective calibration; collective probes require their participants | Separate from compute measurement; no ShadowSpill runtime/pools required. Returns reusable communication information. |
| `apply_profiles` | CPU coverage and compatibility checks | No GPU allocation or live runtime. |
| `plan_program`, `sweep_programs` | CPU scheduling, simulation, and placement calculations | Numeric capacities and allocation descriptions only; no real pools or runtime. |
| Execution | GPU execution of the selected plan | Fresh execution process installs the allocator and creates runtime pools. |

### Builder arguments to review

The user chose explicit keyword arguments for important controls, with defaults
and focused policy objects for advanced choices. The signatures below illustrate
that agreed style; exact argument names, policy types, and defaults remain open.

```python
def build_forward(
    model,
    *,
    example_args=(),
    example_kwargs=None,
    target,
    forward_fn=None,
    partition="auto",
    parallelism=None,
    compiler_options=None,
) -> Program: ...

def build_forward_backward(
    model,
    *,
    objective,
    example_args=(),
    example_kwargs=None,
    target,
    num_examples=1,
    partition="auto",
    operator_recompute=None,
    parallelism=None,
    grad_dtype=None,
    defer_gradient_sync=False,
    compiler_options=None,
) -> Program: ...

def build_step(
    model,
    *,
    objective,
    optimizer,
    example_args=(),
    example_kwargs=None,
    target,
    num_examples=1,
    hyperparams=(),
    partition="auto",
    operator_recompute=None,
    parallelism=None,
    grad_dtype=None,
    master_dtype=None,
    compiler_options=None,
) -> Program: ...
```

The [model authoring contract](model-authoring-contract.md) defines proposed
requirements on the `model` input. The user chose a symmetric initial scope:
one ordinary module describes the common rank-local computation. Factories and
collections of different role modules are outside the initial API. Asymmetric
computation/planning is explicitly unsupported for now.

These are a proposed core surface, not a decision to remove existing advanced
features such as parameter metrics or accumulation-fusion controls. Their exact
placement must be covered by the migration inventory before anything is removed.

`example_args` and `example_kwargs` describe one forward call for
`build_forward`, or one objective call for a differentiation builder. The user chose
one rank-local input example plus a count for the two differentiation builders,
then requested the order-neutral name `num_examples` instead of
`accumulation_steps`. Retain an explicit per-example form for differing shapes.
One example is the proposed default. Exact argument packaging and the
mutual-exclusion rules for the advanced form remain to settle. Either form
defines one concrete candidate with fixed input geometry for each example;
build does not invent a sweep over example counts.

An example here is one complete rank-local model/objective call, including all
its input arguments and metadata. Its tensors may themselves contain a batch.
For instance, a text input shape `[2, 8192]` with `num_examples=8` describes eight
calls containing two sequences each: sixteen sequences per rank in that program
invocation. It does not describe eight optimizer updates.

Under the agreed shorthand, the example supplies input structure/geometry and
required static controls. Each of the eight runtime calls receives independent
data of that form; the example payload and storage are not implicitly reused.
The count does not select depth-first or breadth-first traversal and does not
silently change the objective's normalization. Gradient contributions still
accumulate for either legal traversal in the differentiation builders.
Build should not initialize or load model weights into a runtime pool.

The user selected rank-local inputs as the core boundary. Builders describe the
actual rank-local model calls, and execution workers receive rank-local tensors.
Optional loaders/splitting helpers can accept global batches without making
global-data distribution a prerequisite of build or execution. This preserves
the agreed common rank-local model and does not require data payloads for CPU
build.

For this local boundary, ordinary DP could use `[1, 8192]` tokens per
rank, while four-way context processing of an 8192-token sequence could use
`[1, 2048]` per rank. Those local shapes are the compiled task shapes. Where
needed, separate logical layout/position information describes how local chunks
relate to the sequence and distinguishes them from unrelated local examples.
The distributed operator describes communication. This does not yet select
layout classes, require annotations for every intermediate, or imply that
ShadowSpill automatically rewrites attention from a tensor-layout declaration.

### Model-independent input description

The user requires inputs to work for arbitrary model families, including their
custom arguments and metadata. The core should describe a Python call, not
require fields such as tokens, labels, or sequence length. For training, this is
the objective call: labels and other loss inputs need not be arguments accepted
by the model itself. An optional forward wrapper serves the same purpose for
forward-only programs.

Proposed representation principles, with exact API packaging still open:

- Positional and keyword arguments can contain nested tuples, lists, dictionaries,
  and supported custom containers. A dictionary passed as one positional batch
  must remain distinguishable from keyword arguments.
- Tensor examples supply shape, dtype, strides, and relevant alias information.
  CPU/meta examples should describe geometry without GPU allocation; offline
  capture support must be qualified before claiming all paths support this.
  The compiler target determines the intended execution device.
- Runtime tensor contents can change between compatible executions. Lengths,
  masks, positions, offsets, or other metadata that changes with the data belongs
  in runtime inputs. Shape examples alone do not supply representative values
  for value-sensitive profiling.
- Static Python options such as a mode string or a causal flag specialize the
  captured computation and are retained in its compatibility description. Changing
  them can require another build. Do not silently freeze metadata that the caller
  intends to vary at runtime. Exact scalar/symbolic input support remains open.
- A padded sequence's capacity can be inferred from tensor shape. Actual lengths
  or packed-sequence boundaries cannot in general be inferred from that shape;
  the model/objective must receive them when its computation needs them. Global
  positions and distributed input layouts remain explicit where needed.
- Custom Python objects need a supported flatten/reconstruction and persistence
  representation. Supporting arbitrary model domains does not promise export of
  arbitrary Python control flow, opaque objects, or live external resources.

For example, the following proposed call structure needs no special sequence
fields in ShadowSpill itself:

```python
example_args = ({
    "tokens": torch.empty((2, 8192), dtype=torch.int64, device="meta"),
    "labels": torch.empty((2, 8192), dtype=torch.int64, device="meta"),
    "lengths": torch.empty((2,), dtype=torch.int32, device="meta"),
},)
example_kwargs = {"causal": True}
# Describes objective(model, *example_args, **example_kwargs).
```

An image model, graph network, or scientific model supplies its own argument
structure instead. The user selected separate `example_args` and
`example_kwargs`, without a required call-description object. For example,
the following describes
`objective(model, features, targets, mask=mask, mode="train")`:

```python
program = build_forward_backward(
    model, objective=objective,
    example_args=(features, targets),
    example_kwargs={"mask": mask, "mode": "train"},
    num_examples=8, target=target,
)

```

This supports nested argument structures and distinguishes a positional batch
dictionary from keyword arguments. It introduces no model-specific metadata
schema. Empty positional arguments and omitted keyword arguments are the
proposed defaults, allowing positional-only or keyword-only examples. The
representation of the advanced heterogeneous-example form remains open.

Logical parallelism, target, and build policies stay separate from model inputs.
Measurement annotations such as routing or length distributions are also
distinct: they characterize profile applicability and are not automatically
passed to the model. The current `profiling_metadata` argument already serves
that latter role; its replacement API remains to design.

### Microbatch ordering belongs to planning

The user requests moving data/microbatch ordering out of build. This is feasible
without moving graph capture or compilation into planning. The fixed candidate
still declares its input shapes and `num_examples`. Build prepares
reusable stage tasks, including the backward forms that initialize and accumulate
gradients, and records the dependencies and effects that every order must obey.
It does not choose depth/breadth traversal, reversed backward traversal, or loss
pairing.

Planning selects an allowed traversal and binds the appropriate backward form to
each occurrence, then derives lifetimes, transfers, and physical memory placement
for that order. Search can compare several orders of the same `ProfiledProgram`
without another build or GPU measurement. The selected order is saved in `Plan`.
Task/profile identities should not depend on traversal or task-list numbering.

Illustrative planning syntax, not approved names or defaults:

```python
program = build_step(
    model, objective=objective, optimizer=optimizer,
    example_args=example_args, example_kwargs=example_kwargs,
    num_examples=8, target=target,
)
profiled = apply_profiles(program, profiles)
plan = plan_program(
    profiled, memory_budgets=memory_budgets,
    communication_info=communication_info, ordering=ordering,
)
results = sweep_programs(
    candidates={"batch2_examples8": profiled},
    memory_budgets=budget_sweep, communication_info=communication_info,
    orderings=orderings,
)
```

This requires a structural change, not sorting the current task list after it
has been built. Today lowering selects which backward initializes each gradient,
binds its arguments, and adds dependencies between mutations. The order-independent
program must retain enough information for planning to make those selections.
Different shapes may need both forms measured before any order can select them.
Incoming gradients and shared/tied parameters must be accounted for per logical
gradient destination, rather than assuming one independent gradient per stage.

Data dependencies, reads of old parameter values during recomputation, model
mutations, RNG semantics, and distributed collective/effect ordering restrict
legal schedules. The planner cannot treat every permutation as interchangeable.
Reordering gradient additions can change floating-point roundoff; any required
exact accumulation order must be represented as a constraint. Measurements remain
associated with task variants and their workload assumptions; schedule-level
contention/cache effects are a separate simulation-accuracy issue.

The user also selected moving optimizer placement (`stage_interleaved` versus
`tail`) into planning/search. Build captures update computations and their
dependencies; planning chooses when those tasks run. Stage-interleaved updates
can execute once their gradient contributions/reductions are complete and all
uses of the old weights, including recomputation, have finished. Tail placement
waits for all backward work. Shared parameters and global operations such as
gradient clipping can introduce dependencies that prevent earlier updates.
Changing placement reuses captured tasks and profiles; changing the optimizer's
math or compiled task granularity is a separate build change. Exact scheduling
fields and defaults remain open.

### DP communication outside optimizer tasks: user-proposed direction

The user proposes that an optimizer task consume already-reduced gradients in
the required layout, owned master/weight tensors, and optimizer tensors, with
DP communication outside that task. This is the recommended boundary for the
refactor. The current wrapper's combination of reduction, local update, and
weight publication is an implementation choice, not an optimizer requirement.
Exact IR representation and public fields remain to review.

#### Simpler initial scope: local profiling without collective overlap

The user prioritizes DP build/profiling without live process groups and is
concerned about expanding this refactor into asynchronous collective scheduling.
These are separable requirements. The recommended initial scope is:

- Build uses logical group sizes/ownership to derive local tensor and shard
  geometry. Local forward/backward/update executables contain no ShadowSpill DP
  collectives and need no live process group to construct or measure.
- Measurement prepares ordinary local inputs, including gradient shards in the
  optimizer's declared input layout. Producing representative input values does
  not require actually running a distributed reduction. Communication timings
  remain separate, supplied through rates/topology or previous measurements.
- Execution can initially place separate DP communication operations at explicit
  ordered boundaries, completing/joining their GPU work before advancing past
  that boundary. NCCL may use private streams internally; the operation joins
  those streams into the caller's ordered stream. This does not require a CPU
  wait or device-wide synchronization after every operation.

For example, an ordered sequence can perform gradient completion, local update,
and any required weight publication. The optimizer stays independently compilable
and measurable even if execution chooses to place these operations next to each
other. Ordinary operation input/output/workspace accounting is still required,
but a new collective need not remain in flight across later compute tasks.
Existing supported fetch/evict overlap is a separate mechanism and need not be
removed to take this approach.

This is a recommendation for scope, not a selected new scheduling API. Independent
collective/compute overlap can be added later using the same external pipeline
APIs, with the completion and memory accounting described below. It is not a
prerequisite for representative single-GPU DP profiling, nor for CPU-only
planning with supplied profiles and transfer costs. Actual distributed execution
still binds real groups; directly measuring communication requires participants.

For a shard-local optimizer, the conceptual computation is:

```text
update(reduced_gradient_shard, master_or_weight_shard,
       optimizer_tensor_shards, hyperparameters/control_values)
    -> updated master/weight shard, updated optimizer shards,
       updated compute-weight shard when a separate compute copy is required
```

These are ordinary task inputs/mutations; this sketch does not introduce another
public optimizer API or require callers to manually arrange communication. Dtype
conversion into the local compute-weight shard may stay fused with the update.
With unsharded ownership the same boundary accepts complete local tensors.

Build derives required reductions and redistribution from the central parallelism
declaration and records their data dependencies separately from local update
code. Examples include reduce-scatter into an owned gradient shard, any further
reduction across copies of that shard, all-reduce for replicated gradients, and
redistribution when gradient and optimizer layouts differ. Already-completed
reductions must not be repeated. The same gradient-completion operations serve
`build_forward_backward` without requiring an optimizer task.

With the optional asynchronous communication extension, planning can
reduce ready gradient contributions and fetch master/moment shards while other
backward work executes, then update at its selected position once its required
inputs and gradient contributions are complete. Overwrites must follow readers
of the old contents of the storage actually mutated. A separate master can be
updated before readers of the old compute copy finish; an update fused with
overwriting that copy must follow those readers too. Tied parameters,
recomputation, and global statistics such as clipping constrain legal ordering.
No mandatory host round trip is introduced between reduction and update: a
reduced shard may remain on the GPU when the plan permits.

Updated owned shards are the local update results. Required all-gathers or other
publication remain separate operations with explicit consumers. For sharded
weights, this permits a gather before their next full-weight use instead of
requiring it inside every update; the actual schedule follows declared storage
and output requirements. Replicated consumers still need complete weights.
Masters and moments need no gather merely to perform a shard-local update.

Any execution plan must preserve collective order across participants, completion events,
buffer lifetimes, memory reservations, and communication resource contention.
Separating operations enables later overlap; it does not require implementing
that overlap in the initial refactor or guarantee free concurrency.
Local optimizer profiles exclude these separately costed collectives. This is
what permits single-GPU measurement of DP optimizer arithmetic.

Do not assume every optimizer is independently shardable. An optimizer that
needs cross-shard statistics must expose that work and its dependencies, or use
a supported unsharded layout. Global clipping/overflow decisions, when enabled,
cannot be silently moved after the updates they control. Small-collective
aggregation remains deferred; this boundary does not require implementing it.

#### Fit with explicit task ordering

The user asked how this fits ShadowSpill's explicit order. Keep that model:
planning constructs a concrete order and execution follows it. The earlier
wording about running an update "once ready" describes dependencies that the
selected schedule must satisfy, not a proposal for a runtime ready-task queue
or a new algorithm that discovers weight readers during execution.

- Build captures reusable forward/backward/local-update forms and records their
  tensor reads, outputs, mutations, gradient contributions, and required logical
  reductions/layout changes. It does not choose the microbatch/optimizer order.
- Planning expands those forms into the requested concrete order, binds gradient
  initialization/accumulation, and schedules communication and memory actions
  around it. The saved Plan fixes the compute order, collective submission order,
  buffers, and cross-stream completion dependencies.
- Executor submits that plan. Actual completion events enforce dependencies;
  simulated timestamps are predictions, not wall-clock launch requirements.
  Sequential compute work already obeys the compute stream's order.

For example, with all earlier work omitted, an allowed tail-update order is
`B2 -> B1 -> B0 -> U2 -> U1 -> U0`. Here B denotes the last backward contribution
for that stage and U its local update, assuming ordinary untied parameters.
Reduce-scatter of stage 2's gradient starts after B2 and can overlap B1/B0;
master/moment fetching may overlap too. U2 remains at its fixed position and
waits for those transfers if they are not complete. With the order
`B2 -> U2 -> B1`, that reduction has no intervening backward work to hide behind.
Separating communication alone does not make an immediately following dependent
update overlap the reduction.

The implementation change is to expose reduction and publication separately
from optimizer code, including a reduced shard usable on the GPU without a forced
spill. Existing transfer recipes are tied to fetch/evict operations, so the exact
representation of standalone GPU communication must still be reviewed across
IR, simulation, and runtime. This does not require selecting a general dynamic
DAG scheduler or splitting master updates from compute-copy conversion solely
for extra overlap. That latter split is optional future granularity tuning.

If independent collective overlap is implemented, its buffers must remain accounted for while later compute
tasks run. The full reduction input, its output shard, required scratch, and
prefetched optimizer tensors can coexist with backward workspace. Compute order
alone cannot authorize reusing an in-flight communication buffer. See the
[overlap memory requirements](memory-and-profiles.md#memory-while-standalone-dp-communication-overlaps-compute)
for the current admission assumption that needs extending and validation cases.

### Remaining builder controls

`build_forward_backward` must support optional incoming gradients as agreed.
The user selected one saved Program/Plan with fresh and accumulating behavior
selected per call, as described below. Fresh gradients are the selected default,
with explicit `accumulate_gradients=True` for existing gradients. The gradient
loading/presence protocol and internal representation remain to design. No
incoming gradient means the first contribution creates it; otherwise accumulation
adds new contributions to it.
Existing reduced gradients must not be multiplied by reducing them again on
every rank.

`target` is explicit here so CPU build does not infer a CUDA architecture from
whatever device happens to be present. A convenience orchestrator could resolve
the target for an actual allocation. The target's exact hardware/compiler fields
and any backend specialization that must be deferred remain open.

### Build preparation and GPU requirements: selected boundary

The user permits build to use a GPU when this is a necessary/common compiler or
provider requirement. The earlier proposal to keep build always CPU-only and
return unresolved kernel choices was not selected. Prefer offline work where
supported without complicating the public API merely to avoid GPU preparation.
Build remains independent of ShadowSpill pools, and profile application,
planning/search, and reporting remain CPU-only.

The user selected finalizing task implementations and layouts during build,
using a representative GPU when required. Measurement warms up and records
timing/memory without changing those definitions. Missing dynamic memory remains
explicit for profiling. See the [build-completion review](build-completion-review.md).

Distinguish hardware queries, code generation, and empirical autotuning. A target
description can replace a hardware query only when the backend supports those
supplied facts. Code generation may run on CPU. Selecting a configuration by
timing candidates requires a GPU unless an applicable configuration is already
selected. Some custom-op tuning occurs on first execution, after the enclosing
graph has compiled. A successful compile alone does not establish readiness.

There is no universal unresolved-kernel flag in the current implementation.
Preparation should follow the supported compiler/provider path and its exact
configuration/cache requirements; do not infer readiness from graph names or
classify arbitrary compilation errors as requests for a GPU. The required
invariant is compatible executable behavior, tensor layouts, and allocation
requirements, with profiles describing that implementation. It need not expose
every internal CUDA kernel name chosen by a library. Build owns preparation that
determines the task definitions. Exact provider hooks, configuration persistence,
and compatible load-time compilation policy remain to review.
Execution cannot silently change a specialization in a way that invalidates its
planned memory requirements or profile compatibility.

For symmetric pure DP with no collectives inside local compute tasks, one
representative target GPU can compile/prepare and profile all unique local
forward, backward, and optimizer forms, one task or pair at a time. Logical DP
size still determines owned shard geometry; preparation must not substitute
world_size=1 and thereby change the task. Distinct local shapes, if any, need
their own coverage. The complete model need not be resident simultaneously.

DP transfers remain explicit planning work. Supplied rates/topology or compatible
previous measurements provide their costs. Direct collective and communication
contention measurements require the actual participants; isolated local task
timings do not measure that interference. Tasks containing distributed operators
do not qualify for this single-GPU path without an explicitly supported local
preparation/measurement method. Today optimizer captures contain DP collectives,
so the refactor must separate these from local updates to meet this target.

Unknown graph semantics are still errors: required dependencies, layouts,
mutations, and resource descriptions cannot be invented. Explicit memory
estimates support simulation only. When a backend cannot finalize its task
definitions, build reports the missing requirement or failure rather than return
an incomplete Program as if successful. This does not introduce another artifact
kind or mandatory registry.

### Defaults and advanced controls to review

| Control | Proposed common default | Advanced control |
| --- | --- | --- |
| Partition | User selected `"auto"` | `"whole"` or custom `PartitionPolicy`. |
| Operator recomputation | Build both save and recompute graph pairs with current retention behavior | The user selected `operator_recompute` for the build argument, supporting a custom operator-level retention policy in addition to the heuristic. Exact policy interface remains to design. Planning constraints select among built alternatives. |
| Parallelism | Single participant when omitted; the DP convenience defaults to sharded weights, gradients, and optimizer data, with masters following optimizer ownership | One declarative specification with presets, named groups, and parameter-specific overrides. |
| Gradients/masters | Preserve existing API dtype defaults | Explicit `grad_dtype` and `master_dtype` where applicable. Model/provider precision still defines operator math and frozen weight representation. |
| Microbatch ordering | Chosen during planning; default still to review | Planning/search can compare legal traversals of one built candidate. No build-time traversal argument. |
| Optimizer ordering | Chosen during planning; default still to review | Stage-interleaved or tail updates subject to semantic dependencies. No build-time optimizer-ordering argument. |
| Gradient completion | Reduced final gradients | Explicit deferred synchronization for forward/backward-only calls. |
| Compiler controls | A documented default configuration | Explicit compiler/provider options recorded in compatibility identity. |

The policy classes should group related controls, not reproduce every keyword
inside another catch-all configuration tree. In particular, no separate
`shard_optimizer` argument should compete with `parallelism`.

### Retention controls: agreed scope, protocol to review

Automatic stage partitioning is now selected. The existing save/recompute pair
remains the starting point, including the user's earlier requirement to retain
the save variant for profiling/reporting even when planning selects recompute.
The build/planning distinction is:

- Build constructs the retained-value and recomputation variants and their
  backward forms. Changing which intermediates a form retains changes its code
  and memory requirements.
- Planning selects among those built variants; it does not trigger graph
  differentiation or compilation to invent another retention scheme.

Today `RetentionPolicy(memory_bound_flops_per_byte=16.0)` governs the save
variant. Save does not mean retaining every forward intermediate: the min-cut
may regenerate inexpensive operations, while compute-heavy and unknown custom
operations are retained. The recompute variant retains stage inputs and reruns
the stage work required for backward. This behavior is implementation evidence,
not a claim that the threshold is optimal for every model or kernel.

The user selected exposing the existing heuristic plus an advanced custom
policy to influence operator-level save/recompute choices. This is not limited
to the current threshold. The exact policy protocol, override fields, and
interaction with each graph variant still need concrete review. Overrides must
respect operator replay and effect constraints and cannot authorize unsafe
recomputation. No custom-policy implementation has been added yet.

The user selected `operator_recompute` as the build-argument name, distinguishing
operator-level retention during task construction from planning's selection of
stage variants. The current heuristic makes cheap operations eligible for
regeneration; min-cut may still retain a value when that is the better cut.
The argument therefore does not promise that every eligible operation is always
recomputed. Strict overrides in a custom policy, their interaction with each
variant, and their replay restrictions remain to specify.

The user's latest direction is to describe one complete replica and repeat it
with `num_replicas`. Total participants are ranks per replica times the replica
count. Groups, parameter placements, and sharding settings are described within
one replica and repeated identically. Sharding stays within a replica; reductions
across corresponding copies between replicas are generated automatically.
The previous proposed world-size-only DP convenience needs to be reviewed in
terms of this boundary. Exact constructor and storage-sharding fields remain open.

The user defines a replica by complete model-state ownership: its ranks
collectively own the entire logical model and the training state applicable to
the program. It is not a single data-parallel worker or a requirement that the
ranks process the same data. For example, eight DP ranks processing different
microbatches and sharding weights, gradients, and optimizer state form one
eight-rank replica. `num_replicas=1` does not remove their gradient reduction:
local data contributes gradients to the same logical weights, which are reduced
and stored in their declared shards. Two such replicas use sixteen ranks and
add synchronization between corresponding copies across the two groups.

Input/data placement therefore remains distinct from the replica count.
`num_replicas` alone does not determine the number of independent data streams
or the global batch. Tensor ownership inside a replica may be sharded or may
include local copies; the requirement is complete collective ownership.

This description supports generic model operations inside a replica, including
computation/input sharding and distinct parameter replica groups. EP and CP are
examples, not a closed set of modes.
The user chose explicit named, ordered groups of logical ranks, with an optional
mesh helper that generates them. Exact group/reference fields and model/operator
extension contracts remain under discussion; the DP example does not settle
those schemas. See the
[generic interface example](build-and-parallelism.md#worked-generic-interface-proposal)
for the earlier group sketch and its clarification; its broad `rules` argument
is not a selected interface.

The user chose explicit model-specific state ownership declarations inside
`parallelism`, optionally generated by reusable model helpers. Build validates
them against the model and resolves them with requested storage sharding. It
does not discover ownership through a required model method. Parameter/buffer
selection uses module-qualified names and patterns, with defaults and named
overrides resolved during build. The user selected exact names over wildcard
matches, conflicting wildcard matches as build errors, and an explicit default
when no entry matches. Resolve the placement and storage maps independently of
entry order. Callers may use any registered name of a tied Parameter; build
resolves declarations across those names as one identity, applies the selected
layout to every use, and rejects conflicts at the winning priority. Preserve all
names and shared parameter/gradient/optimizer ownership. This does not merge
independent equal-valued parameters or arbitrary storage-sharing views. The user
selected filling omitted storage fields from the explicit storage default after
selecting one entry by name precedence. Do not merge wildcard entries or inherit
one into an exact-name entry. The resolved configuration must be complete. Exact
public fields, pattern syntax, and unmatched-selector behavior remain to design.

The user chose the term `Sharded` for tensor placement to avoid overloading
computation partitioning, and rejected the public names `StateContract` and
`StateRule`. The following spelling illustrates one replica repeated twice;
constructor/field names and complete placement fields are still proposed:

```python
parallelism = Parallelism(
    ranks_per_replica=4,
    num_replicas=2,
    groups={"workers": [(0, 1, 2, 3)]},
    default_parameter_placement=Replicated(group="workers"),
    parameter_placements={
        "layers.*.experts.w13": Sharded(group="workers", dim=0),
        "layers.*.experts.w2": Sharded(group="workers", dim=0),
    },
)
```

Group rank numbers here are local to one replica. Build expands this description
to eight logical participants. The model already computes with different expert
shards on different ranks within a replica. These declarations describe that
ownership; they do not rewrite
ordinary modules into distributed modules. ShadowSpill storage-sharding fields
are not shown in this partial example and remain separate information to
represent in the central declaration.

The user selected equal contiguous slices in logical group order as the default
meaning of `Sharded(group=..., dim=...)`. A local extent L and group size G
describe a global extent GL, with position p representing indices [pL, (p+1)L).
Corresponding positions in repeated groups identify copies of the same logical
parameter slice. Advanced explicit mappings are allowed within the initial
common-local-shape/graph scope; their exact fields and multidimensional
composition remain to design. Parameter storage is configured independently.

The user's [sixteen-rank example](build-and-parallelism.md#required-example-two-ep8-groups-sharing-storage-inside-one-16-rank-replica)
makes that distinction a concrete requirement: two EP8 compute groups live inside
one replica; corresponding expert subsets are storage-sharded over pairs
`(0,8)`, `(1,9)`, and so on, while attention weights are storage-sharded over all
sixteen ranks. The final parameter API must express these separate compute and
storage groups. The partial sketch above does not yet include those fields;
their spelling and defaults remain to review.

Storage configuration must also account for master weights, gradients, and
optimizer tensors. For the fully sharded example, these can use the same groups
and matching logical parameter-index ranges as the model-weight shards, with
dtypes independent. The user selected a common storage group per parameter by
default, with optional group overrides for weights, gradients, and optimizer
data. Master weights follow optimizer ownership; independent component sharding
controls remain available. Differing layouts require explicit redistribution in
the program/plan, including its memory and communication costs. Exact constructor
fields remain to design. See the same example for the update behavior and
scalar/non-parameter-shaped optimizer caveats.

The user selected a flat storage object: one common group, per-component sharding
modes, and optional weight/gradient/optimizer group overrides. Proposed spelling
uses `ParameterStorage`, `group`, modes (`weights`, `gradients`, `optimizer`), and
optional `weight_group`, `gradient_group`, and `optimizer_group` overrides.
Modes below use the provisional values `"sharded"` and `"replicated"`. Example:

```python
ParameterStorage(
    group="all_ranks",
    weights="sharded",
    gradients="sharded",
    optimizer="sharded",
    weight_group="weight_groups",
)
```

For eight ranks, `all_ranks` could be one eight-rank group and `weight_groups`
two four-rank groups. Weights are then sharded four ways within each group,
while gradients and optimizer/master data are sharded over all eight. The plan
must include redistribution needed to update those weight shards. Omitting all
component-group overrides uses the common group for all three components.

No nested component-layout object is required. Omitted fields inherit from the
explicit storage default as already selected. Type names, exact field spelling,
and omitted/reset representation remain provisional.

The user chose to infer gradient reductions from placements by default, with
optional per-parameter overrides for gradients already combined by model
backward. Build identifies all copies of the same logical parameter or shard
and requires a SUM of their trainable gradient contributions. It does not
classify parameters as needing synchronization based on a technique name or
on whether their placement is `Replicated` versus `Sharded` alone. Frozen
parameters have no parameter-gradient reduction. Objective normalization remains
separate from the inferred reduction.

This inference uses the logical weights involved in computation and each rank's
gradient contributions, not just which persistent storage shards coexist. In
sharded DP, ranks reconstruct the weights needed for their tasks and contribute
gradients from different data even though persistent parameter storage is
sharded. Required reductions inside the complete-state replica must be preserved.

For example, two copies of a four-way expert-sharded model can use groups
`(0,1,2,3)` and `(4,5,6,7)`. If corresponding positions own the same logical
expert shards, reductions are required across `(0,4)`, `(1,5)`, `(2,6)`, and
`(3,7)`. The complete parameter placement must identify those copies and their
logical shard mappings; matching local tensor shapes alone is insufficient.
With the replica-based description, callers do not list these outer copy groups
or attach an extra outer `Replicated(...)` entry to each parameter. Build derives
them from the repeated layout. The earlier proposal to require a sequence of
sharding/replication entries for this case was not accepted and is superseded by
the replica-based direction. Richer combinations inside one replica still need
precise placement semantics; the outer replica count alone cannot describe every
internal model layout.

The executable plan retains the replica layout and count. Worker binding maps
logical replica/index coordinates to physical processes and devices; a replica
does not have to coincide with a physical node. This remains within the agreed
symmetric scope and does not introduce asymmetric role modules.

The user clarified that the replica size directly chooses the sharding domain:
one eight-rank replica permits global sharding across an eight-rank job; two
four-rank replicas permit sharding within each four-rank unit, with corresponding
parameter, gradient, master, and optimizer copies across units. Eight one-rank
replicas provide ordinary outer replication with no within-replica sharding.
These are configurations of the same interface, not separate global/hybrid
mechanisms. Actual per-parameter sharding still follows the declared groups and
storage settings. Precise storage-sharding controls remain to review.

This organization resembles FSDP2's HSDP mode: the 2D mesh replicates along its
first dimension and shards along its second, with reduce-scatter within the
shard groups followed by all-reduce across matching replica shards. FSDP2 is a
reference for this grouping pattern, not an automatic implementation of arbitrary
model communication inside a replica. Sources:
[FSDP2 API](https://docs.pytorch.org/docs/2.14/distributed.fsdp.fully_shard.html)
and [PyTorch 2.13 collective implementation](https://github.com/pytorch/pytorch/blob/v2.13.0/torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py).

Gradient storage sharding can implement the required sum with reduce-scatter;
it does not remove the reduction obligation. Existing incoming gradients that
are already reduced retain their completion status. The representation of the
optional named reduction overrides remains open.

The same object is accepted by all builders. Optimizer ownership matters only
when an optimizer is part of the computation; parameter and gradient ownership
still apply to forward/backward-only programs. The concrete policy on irrelevant
fields, and representation of TP/EP/CP model transformations, need review.

### Simple DP convenience and computation independent of parameters

The user requires a simple common DP configuration, while retaining the general
named-group and parameter-specific declarations for advanced cases. Proposed
spelling, not a selected constructor:

```python
parallelism = Parallelism.dp(ranks_per_replica=8)
```

This would describe one complete-state replica by default, eight ordinary DP
workers using the same local model on distinct microbatches, and automatic
group/parameter/reduction declarations. `num_replicas` could repeat that unit as
already agreed. Ordinary model authors would not annotate parameters or supply
distributed operators just to use DP. Omitting `parallelism` from a builder can
still mean the simple single-participant case.

The user selected the DP storage default: shard model weights, gradients, and
optimizer data, with masters following optimizer ownership. Advanced settings
retain independent component controls, including optimizer-only sharding or
fully replicated storage. This default trades persistent copies for weight
reconstruction communication; it is not a claim of universally better
performance. The shorthand spelling and detailed control fields remain proposed.

The user also pointed out that parameter placement is not a complete description
of distributed computation. Distinguish these pieces without requiring separate
top-level configuration objects for all of them:

| Information | Meaning | Example |
| --- | --- | --- |
| Parameter placement for computation | Which logical parameter or shard each rank's local model uses | Same full projection matrix, or different expert subsets |
| Parameter-related storage | Where model weights, gradients, masters, and optimizer data are persistently owned | Shard projection storage over all ranks, or expert storage over matching pairs |
| Input/activation layouts and distributed operations | Which logical values each rank receives/produces and how operations cooperate over groups | Distinct sequence chunks, with attention communicating across their owners |

The central declaration provides logical participants, named groups, parameter
defaults/overrides, and the needed input-layout description. The model and
registered operators refer to those groups and expose their own computation,
tensor-layout behavior, and effects. Exact input-layout and operator-registration
fields remain to design. These references must not create live process groups
during model construction or CPU build.

The user accepted separate parameter-placement and parameter-storage maps inside
`parallelism`. The earlier proposed `compute` field described only the logical
parameter slice used by a local module, not arbitrary distributed computation.
Separate maps make this distinction clearer and allow changing storage without
reorganizing the model's parameter placements. Input/activation layouts and
distributed operations remain independent of both parameter maps. Exact
constructors and field schemas remain under review.

For the sixteen-rank example, a proposed fragment inside `Parallelism` is:

```python
parameter_placements={
    "layers.*.experts.*": Sharded(group="expert_exchange", dim=0),
}
parameter_storage={
    "layers.*.experts.*": ParameterStorage(
        group="same_experts",
        weights="sharded",
        gradients="sharded",
        optimizer="sharded",
    ),
}
```

Here `expert_exchange` contains the two ordered eight-rank groups and
`same_experts` contains the matching pairs. The placement identifies each local
module's one-eighth expert subset (with the matching logical shard mapping in
each group); storage further splits that subset's persistent training tensors
over the pair. This declaration does not create the expert-routing computation.
The compatible model/operator still performs that computation using its typed
logical group reference.

Context-parallel attention illustrates a different concern: its Q/K/V inputs
cover different token positions, and the attention operation exchanges data over
a declared group. That operation needs no parameter entry. Projection weights
can still have replicated parameter placement while their persistent storage is
sharded. Input/activation layouts and operator communication must remain expressible
independently of both parameter maps. The same requirement applies to arbitrary
compatible distributed operations, including parameter-free ones. Group names
identify ordered participants, not a closed set of parallelism techniques.

For example, one eight-rank replica could contain two context groups:

```python
groups = {"context": [(0, 1, 2, 3), (4, 5, 6, 7)]}
```

Each group processes a different sequence/batch, split into sequence chunks
among its four members. Shared projection parameters can still have storage
sharded over all eight ranks. The core attention operation works on Q/K/V,
not on a parameter selected by `parameter_placements`. Schematic model code:

```python
q_local, k_local, v_local = self.project(x_local)
y_local = distributed_attention(
    q_local, k_local, v_local,
    group=self.context_group,  # typed reference supplied during construction
    positions=local_positions,
)
```

Here `self.context_group` comes from the central specification, for example via
the proposed accessor `parallelism.group("context")`. The provider implements a
real supported distributed attention operation; declaring the group alone does
not transform ordinary attention into one.
Its definition describes local/global input and output relationships, relevant
sequence positions/masks, communication participation and ordering, backward
contributions, memory/resources, and replay/completion behavior. The chosen
algorithm may use ring exchanges, collective redistribution, or another
supported protocol without requiring a new core parallelism mode. Its layout
description concerns logical values, not an inventory of tensor contents.

Authors should describe external input layouts and opaque/distributed operator
boundaries. Build should propagate ordinary inferable layout information rather
than requiring annotations for every intermediate. Same-shaped inputs cannot
by themselves distinguish independent data, copies, and disjoint pieces of one
sequence. The common symmetric-program limitations still apply.

The user selected a separate external input-placement map inside `Parallelism`,
indexed by paths into the selected call's args/kwargs tree. Keep `example_args`
and `example_kwargs` as ordinary example values; per-tensor layout descriptors
are not required. Proposed field names and path syntax below are illustrative:

```python
# Addition to Parallelism; "sequence" names one declared four-rank group.
input_placements={
    ("kwargs", "x"): Sharded(group="sequence", dim=1),
}

# Passed to the builder as before.
example_kwargs={"x": x_local}
```

For x_local with shape (2, 1024, 4096), this example declares a dimension-1
slice of a (2, 4096, 4096) logical input distributed over that group. It does not
load a global tensor, split input data, or transform ordinary attention into
distributed attention. Runtime workers still receive rank-local input payloads;
global positions or other metadata used by the computation remain actual inputs.

The separate map keeps model-call examples reusable when layout annotations
change. The common independent rank-local input case should remain simple. This
does not settle all input-layout semantics: layout propagation and validation,
supported tree paths/custom containers, and output-layout exposure still require
design. The selected map
packages external input-layout metadata, not annotations for every intermediate
activation.

The user selected independent logical inputs by default for distinct instances
of an input's declared group and for repeated complete-state replicas. Intentional
input sharing is explicit. Independence means there is no required sharing
relationship; input values can happen to coincide. For example:

```python
groups={"sequence": [(0, 1, 2, 3), (4, 5, 6, 7)]}
input_placements={
    ("kwargs", "x"): Sharded(group="sequence", dim=1),
}
```

Ranks 0–3 receive different chunks of input A; ranks 4–7 receive chunks of
independent input B. With local shape (2, 1024, H), each group describes its own
logical (2, 4096, H) input. Corresponding parameter placements can still represent
the same weights across these groups. Parameter identity must not be reused as an
implicit input-sharing declaration. The composed form below expresses intentional
sharing; exact advanced mappings and validation remain open. The declaration does
not make the runtime duplicate, split, or load input payloads.

The user selected one placement or a list of placements as the value of an
input-map entry. The list expresses a composed layout. For example:

```python
groups={
    "sequence": [(0, 1, 2, 3), (4, 5, 6, 7)],
    "same_chunk": [(0, 4), (1, 5), (2, 6), (3, 7)],
}
input_placements={
    ("kwargs", "x"): [
        Sharded(group="sequence", dim=1),
        Replicated(group="same_chunk"),
    ],
}
```

The explicit replication identifies the chunks held by ranks 0/4, 1/5, 2/6, and
3/7 respectively. Both sequence groups then work on copies of the same logical
input. Omitting replication retains the agreed independent-input default. Group
names identify participants, not special core EP/CP/TP modes. This is input-layout
composition; it does not require callers to enumerate repeated parameter layouts
that `num_replicas` already generates.

The single-placement/list form avoids a mandatory wrapper for the common case.
Build must validate compatible groups and slice mappings.
Exact composition semantics for multiple shard dimensions, overlapping groups,
and custom mappings remain to design. The list describes a layout; it is not a
request to insert runtime input redistributions.

The user raised non-contiguous sequence chunks, such as paired first/last chunks
for balancing causal attention, as an additional expressiveness example. An
optional mapping can describe global ownership and local packing order; it does
not require a strided physical tensor. For eight equal chunks A through H and
four logical group members:

| Position in group | Local sequence contents | Global chunk indices |
| --- | --- | --- |
| 0 | A, H | 0, 7 |
| 1 | B, G | 1, 6 |
| 2 | C, F | 2, 5 |
| 3 | D, E | 3, 4 |

One possible syntax follows. `ChunkMapping` and its fields are proposals, not
approved API names:

```python
Sharded(
    group="sequence",
    dim=1,
    mapping=ChunkMapping(
        num_chunks=8,
        chunks_by_rank=[(0, 7), (1, 6), (2, 5), (3, 4)],
    ),
)
```

Rows refer to positions in the declared logical group, not physical process/GPU
IDs. Each local tensor contains its chunks in the listed order. The initial
common-local-shape restriction still applies. Build can validate coverage,
overlap, and compatible local geometry once the mapping semantics are agreed.
The paired ordering must remain available in the artifact for layout validation
and group binding. It can compose with an explicit replication placement.

The model/operator remains responsible for the corresponding distributed
attention algorithm, communication, causal mask, and position handling. Actual
position IDs or equivalent metadata needed by the operation are still model
inputs or operator configuration; merely annotating a local tensor must not make
ordinary local attention incorrectly treat A and H as adjacent global tokens.
The example establishes the use case without settling general irregular/ragged
mappings or requesting an automatic runtime redistribution.

Context participants can contribute distinct token gradients to the same
projection parameters, so parameter-gradient completion still follows the
logical parameter contributions unless an operator has already combined them.
Parameter gathers/scatters and attention activation communication are distinct
operations for planning; embedded communication already measured in a task must
not be charged twice. As a concrete reference, [Megatron's parallel-state
documentation](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.parallel_state.html)
describes sequence partitioning independently of weight ownership and the need
to combine the associated weight-gradient contributions. That implementation's
public configuration is not a proposed requirement for ShadowSpill.

This example establishes what the generic model/operator interface must express;
its registration and binding schema remain for a separate design decision. The
simple DP storage default above is now agreed.

For both EP layers and parameter-free operations, the user selected typed logical
group references during construction. The proposed accessor
`parallelism.group("expert_exchange")` supplies CPU-readable logical metadata;
the measurement/execution worker later resolves it to a real process group and
binds provider resources. Exact type/accessor spelling and fields remain open.
See the [logical group and resource binding proposal](distributed-contracts.md#logical-group-references-and-runtime-binding-proposal)
for the lifecycle and changes needed in the current MLOps EP constructor.

For backend resources beyond a process group, the user selected portable resource
descriptions saved with the program/plan and initialized by compatible providers
in measurement/execution workers. Advanced callers may supply compatible existing
resources. Multiple operators/layers can refer to the same resource identity.
The description includes configuration and memory requirements, while real
buffers/handles belong to the worker. Exact fields, versioning, initialization,
cleanup, and binding methods remain to design. This choice does not require a
general per-operator registration API.

### Measurement, profile application, and planning arguments

```python
def measure_program(
    program,
    *,
    device,
    stages=None,
    inputs=None,
    bindings=None,
    options=None,
) -> Profiles: ...

def apply_profiles(program, profiles) -> ProfiledProgram: ...

def plan_program(
    profiled_program,
    *,
    memory_budgets,
    communication_info,
    ordering=None,
    optimizer_ordering=None,
    options=None,
) -> Plan: ...

def sweep_programs(
    candidates,
    *,
    memory_budgets,
    communication_info,
    orderings=None,
    optimizer_orderings=None,
    options=None,
) -> SearchResult: ...
```

- `stages=None` profiles all stages by default; an explicit stage list restricts
  the run. The user selected stage-level selection rather than a missing-task
  helper or automatic skipping based on existing Profiles. Compatible internal
  task forms may share measurements; every selected stage remains covered.
  Full coverage is checked by `apply_profiles`, not by requiring every partial
  measurement run to cover the whole program.
- `inputs` would override the selected automatic preparation of supported task
  inputs, supplying values or generation logic for real integer/index inputs,
  value-sensitive routing, and other custom requirements. Its concrete interface
  is not specified yet.
- `bindings` would supply actual distributed groups/resources for measurement
  where needed, matching the program's logical declarations.
- `apply_profiles` would resolve static memory plus supplied dynamic information,
  check compatibility and completeness, and preserve measurement/estimate
  provenance. It does not execute kernels or silently fill gaps.
- `memory_budgets` names the execution and spill memory limits. For one plan,
  reusing the existing `MemoryBudgets` execution/spill byte-pair avoids another
  equivalent type. The search example supplies a collection of such pairs;
  its exact container and any per-rank/per-resource overrides remain to design.
  The plural does not require a sweep in `plan_program`: there are multiple
  memory limits even for one plan. `options` would reuse existing
  measurement/search options where their responsibilities match.
- `candidates` would be a named collection of `ProfiledProgram` objects. A sweep
  compares the caller's concrete candidates and budgets without adding new ones.
- Microbatch and optimizer ordering are planning/search controls, as selected by
  the user. The sketch uses `ordering`/`optimizer_ordering` for one plan and plural
  forms for search choices; exact packaging and defaults remain proposals.
- `communication_info` contains topology, bandwidth/latency assumptions, and
  optional measured transfer/collective timings, separately from compute profiles. Rates
  cover spill fetch/evict and scale-up/scale-out communication. The user selected
  this combined input instead of a separate measured-transfer argument. Exact
  fields, latency, and measurement matching remain to design. Compatible measured
  timings take precedence; supplied rates/topology fill gaps. The existing
  `TransferBandwidths` class covers host fetch/evict rates and optional latency;
  it does not yet provide the full proposed distributed cost interface.
  Changing links does not manufacture new GPU compute measurements. Group/layout
  changes may require another program and new profile forms.
- Topology within `communication_info` describes the assumed placement/connectivity of logical ranks and
  devices: node membership, host/device connections, and shared scale-up or
  scale-out links. It supplies the connectivity needed to interpret rates and
  model shared-link contention. It does not redefine the program's logical
  groups or tensor ownership. Exact representation and defaults, including
  simple single-device/single-node helpers, remain to design.

The user selected `memory_budgets` and `communication_info` as the planning
resource arguments, with topology included inside `communication_info`. There
is no separate topology argument to planning/search. Their object schemas and
omission/default rules remain open. A bandwidth number plus connectivity still does not choose a
collective algorithm or supply its latency; those cost details and the measured
collective override interface remain to design.

For example, a sixteen-participant logical group could be placed with ranks 0-7
on one node and 8-15 on another. The logical group remains defined by
`parallelism` in `Program`; topology in `communication_info` describes this placement; transfer rates
price the relevant host, within-node, and between-node traffic. Execution still
binds real workers/devices later. Arbitrary physical worker numbering is allowed
when the logical mapping preserves the plan's topology/capacity assumptions;
different placement costs cannot silently inherit the same performance claim.

### Memory budget scope: selected direction

The user selected the existing simple byte-pair shape for a common per-rank
budget, with `execution_bytes` describing the rank's total GPU allowance rather
than only its suballocatable execution pool:

```python
memory_budgets = MemoryBudgets(
    execution_bytes=60 * GiB,
    spill_bytes=96 * GiB,
)
```

The common pair applies to each logical rank. Accounted
runtime/device baseline, external provider resources, and explicit headroom
consume the execution allowance; planning derives the remaining pool capacity.
For example, a 60 GiB allowance with 4 GiB of fixed overhead leaves 56 GiB before
other pool reservations. These numbers are illustrative, not a measured overhead
for a particular model. Shared resources must be counted once rather than also
charged as task workspace. Retained caller outputs and other known live memory
must be accounted for when admitting another invocation.

`spill_bytes` is that rank's spill-storage capacity, not a cap on all CPU-process
memory. Eight ranks assigned 96 GiB each require up to 768 GiB of aggregate spill
capacity when those allocations occupy the same host memory resource. Node-wide
memory limits, per-rank overrides, backing other than host RAM, and accounting for
non-spill host usage remain to review; topology's nodes are not model replicas.

Budgets remain CPU-only numeric inputs, independent of allocated runtime pools.
Fixed overhead must come from supplied/derived evidence, not a hidden GPU query
during planning. Estimated overhead is estimated memory and preserves the
simulation-only restriction. Exact overhead fields, measurement/refinement, and
execution validation remain to design. Current prepared planning inputs already
subtract fixed/shared overhead from physical budgets; the legacy APIs also have
pool-domain budgets, so the refactor must make one meaning explicit.

Having `execution_bytes` specify only usable pool capacity, with overhead
requiring additional memory, was not selected. Exact overhead inputs and advanced
per-resource capacities remain to review.

### Fallback transfer estimation: selected direction

For cases without compatible measured timings, the user selected a built-in
default collective cost model selected from the declared topology, with an
advanced explicit model override. The common caller supplies rates/topology without
having to specify a collective algorithm for each transfer. Exact model/override
types, field names, and default algorithm recipes remain to review.

The proposed basis for each communication/copy phase is startup latency plus
phase bytes divided by effective bandwidth. Collective recipes must describe
their own phases, participant count, traffic, and scale-up/scale-out resources;
the model must not price all-gather or reduce-scatter using just the local host
shard's byte count. A sharded fetch has a local host-copy leg and a separate
collective leg. Reductions and evictions likewise retain their actual payloads,
layouts/dtypes, padding, ordering, and replica-group operations. This is a
performance estimate, not a claim about which algorithm a backend will run.

Report the selected recipe, rate/latency assumptions, byte conventions, and
measured-versus-estimated costs. Missing required rates/latencies must be exposed;
the exact latency defaults are still open, and zero latency cannot silently
masquerade as a measurement. Shared-resource contention and overlap must be
modeled separately from isolated transfer duration. Do not charge communication
already included in a task's measured duration a second time.

Compatible measured transfer timings retain precedence. This choice does not
change the selected memory-evidence rules or introduce GPU work into planning.
Requiring callers to choose a collective model explicitly whenever estimates are
needed was not selected. The concrete formulas/algorithms still need review.

### Measured transfer costs: selected organization

The user selected combining supplied rates and optional measured transfer/
collective timings in one input, now named `communication_info`, and subsequently
requested that it also contain topology. It covers host/device transfers as well
as inter-GPU communication. Planning/search take `memory_budgets` and
`communication_info`; do not add separate topology or `transfer_measurements`
arguments. Exact fields and the reusable calibration-data representation remain
to design.

```python
plan = plan_program(
    profiled_program,
    memory_budgets=budgets,
    communication_info=communication_info,  # Topology, rates, latencies, measurements.
)
```

The distinction is between supplied rates for estimating transfer work and
observations such as an all-gather duration for a particular group, message size,
dtype, and hardware/runtime setup. Optional GPU calibration produces reusable
data; CPU planning does not run calibration. The public calibration function is
selected; its exact arguments, probe selection, and save/load interface remain to design. Search must use the
same communication-input organization as planning.

#### Communication calibration API: selected direction

The user selected public `measure_communication(...) -> CommunicationInfo` separately
from `measure_program(...) -> Profiles`. Callers can instead construct/import
`CommunicationInfo` with supplied topology, rates, latencies, and timings, so
calibration is optional. Host-copy calibration can use one representative GPU;
collective calibration needs the participating devices/groups. Pure-DP local
task profiling does not call this function or require those live groups.
CPU planning never triggers calibration. Keeping calibration only in an
internal/recipe helper was not selected. Exact signature, probe selection,
returned fields, and persistence details remain to review.

Rates and measured durations remain distinguishable within that combined input.
Compatible measurements take precedence, with uncovered cases estimated from
supplied rates and topology/cost assumptions. Missing costs are reported rather
than silently filled. Exact matching, interpolation, latency, and collective
estimation algorithms remain to review. Bandwidth and topology alone do not
determine a complete collective cost model.

Preserve the existing distinction between storage transfers and communication
already included in a task's duration. Do not charge the latter again through
these transfer inputs or silently claim that changed link rates retime an opaque
distributed task measurement.

### Communication field shape: selected direction

The user selected flat numeric rate and latency fields with explicit units,
alongside topology and optional measured timings, rather than a nested record
for each transfer category. For example:

```python
communication_info = CommunicationInfo(
    topology=Topology(nodes={"node0": tuple(range(8))}),
    fetch_bytes_per_second=25e9,
    evict_bytes_per_second=25e9,
    scale_up_bytes_per_second=300e9,
    fetch_latency_seconds=5e-6,
    evict_latency_seconds=5e-6,
    scale_up_latency_seconds=2e-6,
    measurements=[],
)
```

These numbers illustrate units; they are not measurements or hardware defaults.
The corresponding `scale_out_bytes_per_second` and
`scale_out_latency_seconds` fields support between-node estimation. Optional
detailed topology overrides remain available. The exact measurement/link record
schemas, startup/traffic conventions, and estimation algorithms are not decided
by selecting flat fields. See [the communication review](communication-review.md)
for the proposed conventions and remaining details.

### Transfer cost precedence: selected default

The user selected using a compatible measured duration when one is available for
the requested transfer/collective. Otherwise estimate using supplied rates and
the declared topology/cost model. If neither source provides a cost, report the
missing requirement rather than invent bandwidths or perform GPU calibration.

Here, "compatible measurement" means an observation of the relevant operation
under a matching setup, not a new public wrapper class or a requirement to use
the same model. For example, a single-node eight-GPU all-gather observation is
not automatically a measurement of a multi-node group or a different message
size. A future portable matcher should describe hardware/group placement and
message semantics rather than require identical physical rank numbers. Exact
matching fields and curve reuse remain proposals. Today's storage calibration
cache instead assumes its active bound setup and keys by group alias, padded
extent, and (for reductions) dtype and replica group; see the current-state audit.

For example, if a matching all-gather measurement is 0.8 ms and the rate-based
estimate is 1.1 ms, this default uses 0.8 ms. These are illustrative numbers,
not experiment results. A transfer without a compatible measured entry uses the
explicitly supplied estimate inputs. Compatibility must include the relevant
operation, message geometry/dtype, participants, placement, backend, and setup;
a measured row cannot be reused just because its byte count matches.

Callers do not have to select a measured-versus-estimated mode for each planning
run. The exact override mechanism for deliberately ignoring measurements, if
provided, remains to design. Interpolation/extrapolation and estimation
algorithms remain separate decisions; selecting this precedence does not settle
curve handling. Reports retain whether a chosen cost was measured or estimated
so changing rate assumptions has an inspectable effect.

### Topology organization: selected direction

The user selected a simple node-to-logical-rank map for ordinary homogeneous
systems, with optional explicit links/shared-resource details for nonuniform
host/device or network connections. Do not require every caller to supply a
complete resource graph. Topology belongs inside `communication_info`, following
the user's later simplification. For example, with provisional type/field spelling:

```python
communication_info = CommunicationInfo(
    topology=Topology(nodes={
        "node0": tuple(range(8)),
        "node1": tuple(range(8, 16)),
    }),
    # Rates, latencies, and optional measured timings are supplied here too.
)
```

This describes the placement assumed by planning; the node labels need not be
actual hostnames. The program's parallelism specification still determines
logical groups and parameter/input ownership. A replica can span nodes, and a
node can contain ranks belonging to different replicas. Execution binds workers
to logical ranks while respecting the saved placement/capacity assumptions.

Node membership alone does not describe every PCIe, NUMA, NVLink, or NIC sharing
arrangement. The common case needs documented homogeneous resource assumptions;
advanced overrides would describe different connectivity or shared bottlenecks.
Transfer rates supply costs for that topology. Exact resource/link fields,
defaults, collective algorithms, and contention rules are still open. Requiring
a full explicit device/host/link graph for every configuration was not selected.

### Deferred optimization: small collective aggregation

The user explicitly deferred this optimization work until after the refactor.
The following investigation is retained for later; batching policies and buffer
packing are not prerequisites for the current design or its initial
implementation. The user's later proposal to move DP communication outside
optimizer tasks is a separate architectural boundary described above.

The user observed many inefficient small collectives in simulated traces and
asked whether they could be handled better. Current storage transfers issue
collectives per object, and distributed optimizer capture emits per-parameter
reductions/publication. Propose bounded grouping of nearby small communication
operations while preserving individual parameter names and declared ownership.
Scope, default policy, size limits, and public controls are not selected yet.

Two implementations should be distinguished:

- Group independent collective calls into one backend submission, retaining
  separate buffers. NCCL grouped operations can reduce launch overhead without
  requiring a flat parameter representation. Current per-call completion joins
  would need a batch-aware implementation, with completion attached after the
  batch is actually enqueued.
- Pack compatible small payloads into a bounded communication buffer and perform
  one larger collective, then expose/copy the results to their program objects.
  This adds possible packing/unpacking work and temporary memory; both belong in
  the costs and memory layout. Reduction buckets require matching dtype/reduction
  semantics. Packing must preserve each parameter's per-rank shard mapping;
  simply concatenating full parameters and re-sharding the concatenation would
  generally change ownership.

A conservative initial candidate is small operations needed/ready at the same
stage boundary with matching participant groups and compatible semantics. Avoid
holding unrelated large tensors or waiting for distant stages merely to fill a
bucket. All ranks use the same planned membership and collective order; runtime
local readiness must not independently choose incompatible batches.

Separating weight publication from local updates, as discussed in the optimizer
boundary above, also permits deferring a gather until an actual full-weight
consumer. This does not depend on collective aggregation and does not permit
omitting publication for consumers requiring complete copies.

Proposed responsibility split: build records tensor ownership, communication
requirements, and legal dependencies; planning selects supported groupings under
the memory budget; measurement/transfer cost data describe the resulting batch
and any packing work; the execution bundle fixes buffers, order, and completion.
Model builders should not have to flatten their parameters to enable this.
This applies to ShadowSpill-owned communication; opaque model/provider-internal
collectives cannot be regrouped without an appropriate backend interface. A
planning rewrite must not change a previously measured opaque task and keep its
old profile as though the executable were unchanged.

Simulation/reporting must charge grouped work using a measured or explicitly
modeled batch cost, account for extra residency/scratch and readiness delays, and
retain component tensor names/bytes. Summing individual launch latencies would
erase the benefit in simulation; assuming cost-free packing would overstate it.
Validation should compare counts, size distribution, exposed communication time,
temporary memory, and end-to-end runtime. No GPU experiment or speedup is claimed.

Primary references: [NCCL grouped operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/groups.html#aggregated-operations-2-2-and-later)
and [PyTorch FSDP2 parameter grouping](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html#model-initialization).

## Proposed object contents

These are field groups for review, not final dataclass definitions. Reuse existing
IR records and storage/task identifiers wherever they satisfy the boundary.

| Object | Proposed contents | Kept outside it |
| --- | --- | --- |
| `Program` | Computation kind; input/output and persistent-state contracts; stages/tasks/dependencies/alternatives; tensor views and shared-storage identities/layouts; static memory requirements and unresolved entries; resolved parallelism; build target/compatibility identity; source attribution; references to executable graph payloads. | Task durations, transfer bandwidths, chosen memory budgets, live tensors, pools, process-group handles. |
| `Profiles` | One hardware/runtime setup; task records keyed by compatible executable identity; runtime; missing/dynamic memory information; optional allocation observations; per-field provenance distinguishing measurements and estimates. | Planning budgets, selected schedules, live profiler objects, multiple hardware/runtime setups requiring a selector. |
| `ProfiledProgram` | A `Program` with compatible profiles applied and required task timing/memory information resolved; source/evidence references preserved. | A live runtime, chosen memory budgets, or silently chosen transfer/topology assumptions. |
| `CommunicationInfo` | Topology, supplied host/device and inter-GPU rates/latencies, optional compatible measurements, and their setup/source metadata; constructed manually, imported, or returned by `measure_communication`. | Live process groups, runtime buffers, compute task profiles, or automatically running calibration during planning. |
| `Plan` | The chosen program/profile identity; transfer/topology assumptions; memory budgets; selected alternatives and executable dependencies; task/transfer schedule; simulated timeline and metrics; memory placement information with its evidence/validation status; logical rank/group binding requirements. | Model weights, process-local device/group handles or addresses, an initialized runtime, or a claim that estimated allocation inputs were observed. |
| `SearchResult` | Candidate/budget outcomes, a Plan for every successful case, unsuccessful-case diagnostics, and a best-per-budget lookup. Object name remains provisional after renaming the helper to `sweep_programs`. | Another independently defined program or plan representation. |

### Public artifact mutability: selected direction

The user selected inspectable, read-only public structure for `Program`,
`ProfiledProgram`, and `Plan`, while keeping `Profiles` editable for manual
measurements/estimates and overrides. Build establishes code/shape/layout
relationships; planning establishes schedule/memory relationships. Arbitrarily
editing those results can invalidate their identities and dependent evidence.
Use build/apply/plan again when changing the corresponding inputs.

For example, with an existing complete profile collection, using provisional
accessor/field spelling and omitting source-metadata edits:

```python
task = program.find_task(
    module="layers.3", phase="backward", variant="recompute",
)
profiles.tasks[task.profile_key].duration_seconds = 0.0072
profiled = apply_profiles(program, profiles)
```

The selected `apply_profiles` behavior captures the applied values so later
edits to `profiles` do not change an existing ProfiledProgram or Plan. Updating
task costs does not require recompilation when executable compatibility is
unchanged; apply and plan again. Read-only here describes the public interface,
not a selected frozen-dataclass implementation or a new wrapper hierarchy.

Editable built artifacts were not selected. Existing choices about numeric
fields, profile merging, and explicit save/load remain unchanged. Exact
accessors and the implementation of read-only inspection remain to design.

### Task profile organization: selected direction

The user selected one combined record per compatible task containing timing and
any required missing memory information. GPU measurement and manual/external
inputs use the same organization and preserve the source of each value. Measured
timing can coexist with estimated memory. The user rejected value-wrapper classes:
durations and memory sizes are plain numeric fields. Exact accessors, type/field
names, and separate metadata representation below remain illustrative.

```python
# Illustrative numbers, not measurements from the GLM traces; metadata omitted.
task = program.find_task(
    module="layers.3",
    phase="backward",
    variant="recompute",
)
profiles = Profiles(
    context=profile_context,
    tasks={
        task.profile_key: TaskProfile(
            duration_seconds=0.0072,
            workspace_bytes=768 * MiB,
        ),
    },
)
# Supply all other required entries before applying the collection.
profiled = apply_profiles(program, profiles)
```

Sources and measured/estimated labels belong in ordinary metadata, not classes
wrapping each number. Profiling/import helpers can populate that metadata. Its
exact fields and defaults remain to review; a plain numeric value alone does not
establish that it came from GPU measurement. Keep manual construction as a small
data record and avoid introducing equivalent wrapper types under different names.

#### Manual-value classification: selected direction

The user selected treating manually supplied values without measurement metadata
as estimates. This keeps a minimal numeric `TaskProfile` useful for CPU simulation
without implicitly asserting execution-safe memory evidence. Requiring an
explicit classification on every manually supplied field was not selected. This
default classifies supplied values; it does not fill in missing numbers.

`measure_program` supplies its own measurement metadata. External imports may
declare measured timings and supply the corresponding source/setup information;
no wrapper around individual numeric values is required. A record may contain a
measured duration and estimated memory. The default applies to supplied values,
not to memory already determined from the Program's compiler/provider information.

Estimated timing alone does not make a plan unsafe to execute; selected estimated
memory makes it simulation-only, as already agreed. Marking memory as measured or
naming a trace is not sufficient by itself: its information must meet the
allocation/lifetime requirements of execution. A manually changed memory field
must not silently retain evidence for its previous value; changing only a timing
need not invalidate unchanged memory evidence. Exact metadata fields and how
edit/evidence consistency is checked remain to design, without introducing
Measured/Estimated value wrappers.

This example concerns a task whose full temporary workspace remains unresolved
after build. The workspace bound excludes inputs, retained outputs, and
persistent/provider storage already represented separately; it must be reconciled
with known compiled allocations rather than blindly added to them. A supplied
peak alone is not an allocation trace or proof of an executable physical layout.
Memory estimates are simulation-only, as the user clarified. Exact memory
fields, bounds, overlap/lifetime interpretation, and execution-status reporting
remain separate design work.

Semantic names locate a task; reusable compatibility identities associate
profiles with supported executable forms. Stage occurrences can share a record
when their computation, input/storage signatures, configuration, and relevant
conditions are compatible. Identical shape or display name alone is insufficient.
An accessor must report ambiguous/missing matches; it must not guess a stage from
a module path that spans several tasks. The exact lookup interface and key/context
schema remain to design. `profile_context` represents GPU/provider and relevant
input/communication conditions, not planning budgets or a live device/group.

Measured timing can coexist with estimated memory, and fields known from build
need no duplicate manual entry. A partial collection is allowed during data
gathering; `apply_profiles` still requires complete compatible coverage and no
implicit estimates, as already agreed. Manual imports assert their association
with a task and retain provenance; simply naming an external trace does not
establish graph or environment compatibility.

Separate timing and memory tables were not selected. The combined record keeps
known/missing facts for a task together while allowing independently collected
fields. This does not require shared resource memory to be repeated in every task
record. Resource evidence is associated with the resource identity and counted
once. The selected merge behavior follows below. Metadata fields and
collection/context persistence remain to design.

### Combining profiles: selected direction

Measurement and imports may each supply partial records. The user selected
later-values-win merging by default for compatible task/context entries. Missing
fields are combined; later supplied fields replace earlier conflicting values.
For example, a measured duration of 7.2 ms and a manual workspace estimate of
768 MiB fill different fields and combine naturally. A later duration of 8.1 ms
replaces 7.2 ms by default. These numbers are illustrative.

Illustrative syntax; the default behavior is selected, exact spelling is not:

```python
profiles = measured.merge(manual)
profiles = profiles.merge(new_measurements)  # Later supplied fields win.
```

Replacement affects fields explicitly supplied by the later source, preserves
other fields, and retains the provenance of the selected values. It does not
bypass compatibility checks for changed executable signatures, input assumptions,
or environments. It does not average samples, choose the fastest result, or invent
missing requirements. An explicit overwrite flag is not required for the default
operation. Each collection describes one hardware/runtime setup, as selected
below; merging cannot silently combine incompatible setups.

### Profile collection context: selected direction

The user selected one `Profiles` collection per declared hardware/runtime
measurement setup. A collection may contain many tasks, shapes, and save/recompute variants,
combine multiple measurement/import runs, and be reused across compatible
Programs. It is not limited to one GPU rank or one measurement run. Task-specific
kernel identities, input assumptions, and provider requirements remain associated
with their records. Exact setup metadata and compatibility fields remain to
design.

For example, keep H100 and H200 measurements in separate collections and apply
the desired collection to a compatible Program:

```python
h100_profiles = Profiles.load("profiles-h100")
profiled = apply_profiles(program, h100_profiles)
```

The example assumes the Program is compatible with that collection's target;
changing the collection does not retarget incompatible compiled code. Measurement
helpers populate hardware/runtime metadata. Manual collections declare their
assumptions using simple metadata; exact required fields/defaults remain open.
Relevant distributed conditions must accompany task timings that include
communication, independently of the planning transfer/topology inputs.

Select the setup through the collection passed to `apply_profiles`. A collection
holding multiple setups with an additional context selector was not selected.
Keep compatibility checks: one collection must not silently combine measurements
from incompatible setups. Per-task memory evidence and ordinary planning resource
inputs retain the already agreed phase boundaries.

### Measurement inputs: selected direction

The user selected automatic preparation where valid task inputs can be
constructed, with caller-supplied values or generators for unsupported cases and
realistic input distributions. Do not require manual inputs for every supported
measurement run. Exact override scope and callback/data signatures remain to
design.

```python
# Illustrative syntax; automatic input preparation must be supported for this task.
profiles = measure_program(program, device="cuda:0")
profiles = measure_program(program, device="cuda:0", inputs=my_inputs)
```

Generating dense floating-point activations can be reasonable, but geometry alone
does not establish valid indices, masks, lengths, quantization metadata, or custom
operator inputs. Parameters and buffers also need suitable initialization or
supplied values; a portable Program does not contain checkpoint weights. Unknown
requirements must be reported rather than silently filled with arbitrary values.
Input assumptions affecting runtime or allocation behavior accompany the record.

Backward measurement must retain the current producer relationship: run the
corresponding forward and use its saved tensors, including statistics and RNG
state, rather than inventing saved tensors independently. The existing profiler
already implements this in `pytorch/profiling/profiler/saved_values.py`; input
materialization in `pytorch/profiling/inputs.py` also distinguishes supplied values,
synthetic floating-point values, and values requiring genuine initialization.
The refactor must preserve these semantics while removing pool coupling where
possible. Prepare only the tasks/pairs and required dependencies being measured;
do not require the full model's weights and intermediates to fit on one GPU.

### Selecting stages to measure: selected direction

The user chose profiling all stages by default, unless given an explicit stage
list. Replace the proposed task-selection interface with `stages=None`. Do not
require a helper to select missing profiles or add implicit skipping based on an
existing collection.

```python
profiles = measure_program(program, device="cuda:0")
# Illustrative stage IDs from the Program inventory; selector syntax is provisional.
measured = measure_program(
    program, device="cuda:0", stages=["stage_0003", "stage_0004"],
)
profiles = profiles.merge(measured)
```

Selected stages include their built forward/backward variants and optimizer work
where present. The selection is not an instruction to omit save/recompute forms
that planning may need. Compatible executable forms may still reuse one
measurement internally; stage-level coverage and source attribution are retained.
Profiles remain organized by compatible task form so forward/backward timings
and memory requirements are distinct and reusable.

The stage list controls requested measurement coverage, not which producer
kernels may run to prepare valid inputs. Preserve the corresponding forward's
saved values for backward. Measurement returns records that callers can merge;
`apply_profiles` validates complete compatible timing and memory coverage. Shared
resource requirements remain separately accounted for. No hidden build, planning,
or execution-worker launch is introduced by stage selection.

### Source attribution: agreed boundary and proposed representation

The user chose internal IDs with attached semantic source metadata, displayed in
reports and traces. Propose a portable source map in `Program` associating
stage/task/object/storage IDs with display names, original module paths,
registered tensor names, graph node/output names, and available source locations.
Concrete fields and the public inspection interface are not yet approved.

The mapping is not necessarily one-to-one:

- One stage can cover a layer, part of a layer, or several modules. Preserve all
  relevant source paths and the operation range, rather than claim it is exactly
  one `nn.Module`. A reused module also needs a call occurrence distinction.
- A parameter, its gradient, and its optimizer moments should retain the same
  parameter attribution while identifying their different roles. Quantized
  storage components should retain the parent parameter and component path.
- Several tensor views or tied parameter names can refer to one storage object.
  Preserve all known names and the sharing relationship; display naming must not
  duplicate its byte count or change residency semantics.
- Forward, backward, recompute, and accumulation tasks should inherit source
  attribution. Generated intermediates should point back to their source
  operations when that information exists. Compiler fusion may have several
  source operations; it must not invent a single original Python variable.
- Python local variable names are not reliably retained by export. Module paths,
  graph signature names, FX values, and source locations provide progressively
  weaker fallbacks. Optional user labels could fill gaps without being required
  for ordinary models; that interface remains to design.

Illustrative displays, not claims about the current GLM stage mapping:

```text
layers.3.self_attn + layers.3.mlp  [stage_0003]
layers.3.self_attn / backward / recompute / microbatch 1
layers.3.self_attn.q_proj.weight  [parameter, BF16]
layers.3.self_attn.q_proj.weight.grad  [gradient, FP32]
```

Reports, simulated timelines, graph-pair inventories, memory diagnostics, and
NVTX ranges should consume the same source map. Generated IDs remain available
for exact lookup. Source attribution belongs to the persistent program rather
than being reconstructed from a live model only when generating a report.
Occurrence names should remain separate from structural profile compatibility,
so repeated equivalent layers can still reuse measurements. Renaming a display
label should not by itself require another GPU measurement.

`Parallelism`, the compiler target, memory budgets, and communication information are
configuration values. Their exact schemas still need agreement. Distributed
transfer estimation must connect named logical groups to physical link placement
and costs, and distinguish measured collective curves from a topology/bandwidth
model. Topology is included in `communication_info`; its schema and the
measured-cost interface remain open. The planning call no longer proposes a broad
`machine` argument.

Prefer data objects with inspection and persistence methods, with the main work
performed by the public functions above. Whether the concrete classes are frozen
dataclasses is an implementation/design choice still to approve.

## Persistence: agreed interface and proposed packaging

The user chose explicit `.save(path)` / `.load(path)` on phase artifacts and an
optional cache. Exact signatures, file formats, and bundle layouts remain open.

The optional cache accelerates phases without being the mechanism by which one
phase passes its result to another. Portable results, disposable
compiler caches, and run reports have different lifetimes. Inductor/Triton caches
remain separate from portable artifacts.

This preserves useful content identity, integrity checks, and complete publication
without requiring users to manage multiple stores simply to call the pipeline.
The user selected one archive per saved phase artifact. The exact container,
metadata schema, and payload representations remain to design.

Program packaging remains a proposal; the executable-plan bundle is agreed:

1. Keep one public `Program`, with an inspectable planning structure and backend
   executable payloads that are loaded only when requested. CPU planning/loading
   would not import CUDA providers or deserialize live executable objects.
   Measurement/execution would load or specialize the executable payloads under
   their recorded compatibility requirements. The alternative is separate public
   planning-only and executable artifact types. Deferring payload loading does
   not, by itself, solve GPU-dependent compilation or autotuning.
2. The user chose a portable executable `Plan` bundle containing selected task
   code/graphs, schedule, memory layout, logical rank/group declarations, and
   input/state contracts. Every rank loads the same bundle. Model checkpoints,
   compatible runtime/provider libraries, and disposable compiler caches remain
   external. Its exact format and internal deduplication scheme remain open.
   The user also selected including Program and applied profiles when saving
   ProfiledProgram, as detailed below.

### ProfiledProgram packaging: selected direction

The user selected that `ProfiledProgram.save(path)` includes its Program and the
applied profile records/setup metadata needed to use it. Copying this artifact should
not require also copying the original Program/Profiles folders or keeping the
original cache/store available. This extends the already selected portable
Plan packaging to the preceding CPU planning input; it does not change the
separately reusable Profiles collection.

```python
profiled = apply_profiles(program, profiles)
profiled.save("profiled-glm8k")

# In a different process/machine, using the copied artifact.
profiled = ProfiledProgram.load("profiled-glm8k")
plan = plan_program(profiled, **planning_inputs)
```

Include the program metadata and required graph/code payloads, relevant profile
data, and compatibility information. Do not copy an entire multi-program
Profiles collection when only its applied records are needed. Model checkpoints,
provider/runtime libraries, and disposable Inductor/Triton caches remain external,
as for the agreed Plan bundle. Required program payloads must not rely on paths
inside a disposable cache. Artifact loading for CPU planning must not initialize
CUDA or runtime pools.

This can duplicate program payloads when callers save Program and
ProfiledProgram separately. Optional internal deduplication may reduce that cost,
but a copied saved artifact must still contain what it needs. Exact archive
format, payload loading, and the deduplication mechanism remain separate design
questions. Saving only references to separately persisted Program and Profiles
artifacts was not selected as the default.

### Artifact file format: selected container direction

The user selected one archive file per saved phase artifact. It contains readable
metadata and required graph/code/data payloads. A manifest is simply the small
metadata file describing the artifact type/version, its contents, and relevant
compatibility requirements; it is handled by save/load rather than a separate
public class or user step. JSON metadata and ZIP are illustrative choices; exact
container, extension, compression policy, and schema remain to design. This
avoids many small writes/opens on shared or cold storage and makes copying one
saved artifact straightforward.

```python
# One archive is selected; the ZIP format/extension remains illustrative.
profiled.save("profiled-glm8k.zip")
profiled = ProfiledProgram.load("profiled-glm8k.zip")
plan.save("glm8k-plan.zip")
```

The manifest and planning data must be readable on CPU without importing GPU
providers or extracting a full compiler cache. Required executable payloads are
included independently of disposable caches. If a backend needs real files,
required payloads may be extracted to local working/cache storage when used;
those copies do not become external dependencies of the portable artifact.
Reports/checkpoints remain separate from phase artifacts.

A directory of separate metadata/payload files was not selected. Saving an
updated artifact generally rewrites the archive; selective reads and temporary
local extraction can still be supported. Keep Inductor/Triton caches separate.

### Representative public calls

Illustrative calls using the agreed phase names and control style. Argument
shapes and constructor details remain proposals; these are not runnable examples
of an implemented API.

```python
# Build: fixed input geometry/count; GPU preparation may be required.
program = build_step(
    model,
    objective=objective,
    optimizer=optimizer,
    example_args=example_args,
    example_kwargs=example_kwargs,
    num_examples=8,
    target=target,
    parallelism=parallelism,
    partition="auto",
    operator_recompute=operator_recompute_policy,
)
program.save(program_path)

# Optional GPU measurement, potentially in a different process or machine.
program = Program.load(program_path)
profiles = measure_program(program, device="cuda:0")
profiles.save(profiles_path)

# CPU profile application and planning. Profiles can instead come from external
# measurements and explicitly supplied estimates, with complete coverage.
program = Program.load(program_path)
profiles = Profiles.load(profiles_path)
profiled = apply_profiles(program, profiles)
profiled.save(profiled_path)

plan = plan_program(
    profiled,
    memory_budgets=memory_budgets,
    communication_info=communication_info,
)
plan.save(plan_path)

# CPU search over concrete candidates; this builds/profiles nothing.
results = sweep_programs(
    candidates={"batch1": profiled_batch1, "batch2": profiled_batch2},
    memory_budgets=budget_sweep,
    communication_info=communication_info,
)
```

Each boundary also accepts its predecessor's in-memory result; saving is an
explicit user operation, not a mandatory disk round trip after every function.
The convenience orchestrator can choose persistence locations for a run. The
manual `Profiles` construction/merge interface and worker execution calls remain
to design explicitly.

## Planning results and unsuccessful searches: selected direction

The user selected `plan_program(...) -> Plan`, raising a diagnostic planning
exception when this single request cannot produce a plan. Do not add a mandatory
result wrapper around successful plans.
The existing `PlanInfeasibleError` and `PlanSearchExhaustedError` already
distinguish infeasibility from a bounded search that stopped without an answer;
preserve that distinction. A search limit does not prove that no valid plan
exists. Missing inputs and incompatible artifacts are separate validation
errors, not evidence that a memory budget cannot work.

For `sweep_programs`, record those expected unsuccessful candidate/budget cases
inside `SearchResult` and continue evaluating the other requested cases. Retain
the cause and diagnostics, and the best discovered feasible plan for each budget.
Do not suppress unexpected implementation errors or turn them into infeasibility.
If no case succeeds, the search result should still contain its outcomes. Exact
fields and accessors below are illustrative, not selected:

```python
plan = plan_program(profiled, **planning_inputs)
plan.save("selected-plan")

results = sweep_programs(candidates, **search_inputs)
for outcome in results.outcomes:
    if outcome.plan is None:
        print(outcome.status, outcome.reason)
```

Each successful Plan retains its simulated timeline, predictions, selected
alternatives, schedule, and memory placement/evidence. Search outcome summaries
must not require saving every intermediate explored schedule. One resulting Plan
per successful candidate/budget case is retained, as selected below. The exact
inspection/report API remains to design.

A mandatory result record containing an optional Plan and status was not
selected for single-plan calls. The user separately clarified that plans relying
on estimated memory are simulation-only and cannot execute.

### Role of the multi-program sweep: selected direction

After asking whether a multi-program helper is needed when callers can loop over
`plan_program`, the user accepted keeping it thin and optional, using the clearer
name `sweep_programs` instead of the earlier `search_programs`. It is a convenience,
not another fundamental pipeline phase. `plan_program` performs the actual
scheduling/recomputation/memory-plan search for one candidate and budget; the
wrapper repeats that work over caller-supplied cases and organizes the outcomes.

A caller-written loop must be supported without losing core planning capability.
The wrapper's useful responsibilities are continuing after expected per-case
planning failures, collecting comparable summaries and full case Plans, selecting
budget winners, and potentially forwarding compatible earlier Plans as search
starting points. The current sweep carries feasible plans to larger budgets;
preserving that optimization should use core planner inputs rather than require
calling a special orchestrator. Exact starting-plan syntax is not selected.

Keep this wrapper small, composing public calls without owning a separate
planning algorithm or requiring build/profile/runtime orchestration. The selected
retention behavior below describes its results; users need not use the wrapper
to plan or write reports for individual cases. `SearchResult` remains a provisional
return-type name; its organization is unchanged by the helper rename.

### SearchResult plan retention: selected direction

The user selected retaining one resulting Plan for each successful
candidate/budget case, plus a convenient lookup of the best discovered Plan per
budget. Retain status
and diagnostics for unsuccessful cases. This lets users inspect or save a
nonwinning candidate's timeline without repeating planning. It does not retain
every internal schedule explored while solving a case.

Illustrative accessors and caller-chosen labels, not selected field signatures:

```python
result = sweep_programs(candidates, **search_inputs)
best = result.best["40GiB"]
other = result.outcomes[("batch1", "40GiB")].plan
write_report(other, "results/batch1-40GiB-report")
best.save("results/best-40GiB-plan")
```

The example assumes those cases succeeded. A failed case has no Plan and retains
its failure explanation. A budget with no successful cases has no winner.
Shared immutable program/code/profile payloads can be referenced by multiple
plans rather than requiring independent copies for each outcome; exact in-memory
and archive sharing mechanisms remain implementation work. Per-case schedules,
memory layouts, and timelines still increase result size.

Keeping only best-per-budget Plans and summaries for other candidates was not
selected. Exact outcome keys, ordering-policy detail, best-plan ranking/ties,
and save/load fields remain to design separately. Retaining a plan does not change
whether its memory evidence permits execution.

### Comparing different amounts of work: proposal for review

Recommend minimizing predicted call time by default, assuming the candidates
represent equivalent useful work per call. Allow explicit per-candidate work
counts to select by predicted throughput when their work amounts differ. This
is a proposed default/extension, not yet selected. The current sweep compares
makespan directly (`src/shadowspill/pytorch/step_search/sweep.py`,
`src/shadowspill/search/report.py`).

Illustrative additional argument; exact spelling and reporting-unit metadata
remain to review:

```python
results = sweep_programs(
    candidates={"small": small, "large": large},
    work_per_call={"small": 16_384, "large": 32_768},
    **planning_inputs,
)
```

For example, a 16,384-token call taking 0.10 seconds is 163,840 tokens/s, while
a 32,768-token call taking 0.16 seconds is 204,800 tokens/s. These are invented
illustrations, not experimental results. Time-only ranking selects the first;
throughput ranking selects the second. Counts can instead represent images,
samples, or another explicitly chosen common unit. They do not change the
program, input distribution, objective normalization, or training semantics.

If this option is selected, counts must be positive and cover the compared
candidates in the same unit/scope. They describe useful work across the full
invocation, with replicated work counted according to the caller's chosen metric;
ShadowSpill must not blindly multiply by the number of ranks or infer tokens from
arbitrary tensors. `num_examples` counts complete rank-local calls and is not a
universal throughput quantity. The caller remains responsible for a meaningful
comparison; equal counts alone cannot establish equivalent workloads.

The alternative is requiring explicit work counts for every search, including
equal-work comparisons. Either approach keeps runtime and supplied throughput
visible in outcomes, and preserves every successful case's Plan. Tie-breaking,
custom scoring, and exact placement of optional unit metadata remain open.

### Inspection and report generation: selected direction

The user selected the following inspection fields on Plan, backed by canonical
stored data and derived views rather than independent copies of the same values:

| Field | Contents |
| --- | --- |
| `summary` | Predicted call runtime and execution/spill memory peaks; simulation-only classification when selected memory requirements are estimated. Work-specific throughput is reported only when its unit/count is explicitly available, not by assuming every model processes tokens. |
| `schedule` | Selected tasks/variants, example and optimizer ordering, dependencies, and storage transfers. |
| `timeline` | Simulated events with start/end times, logical ranks/resources, task/source identities, transfer kinds and byte counts. Reuse the semantic source metadata instead of exposing only generated IDs. |
| `memory_layout` | Planned object/workspace placement and relevant memory information. A simulation using estimated memory must not claim to have a complete executable allocation layout. |
| `diagnostics` | Decisions, constraints, search statistics, and their attribution to the selected result. |

The user selected one CPU-only `write_report(result, path)` function accepting
either Plan or SearchResult. A Plan report renders its timelines, memory information, and
summary; a search report adds candidate/budget comparisons and unsuccessful-case
diagnostics. It consumes existing result data without repeating build,
measurement, search, or simulation, and can render after artifact loading.
Exact output formats, selective rendering, and lazy loading remain to design.

```python
plan = plan_program(profiled_program, **planning_inputs)
print(plan.summary)
write_report(plan, "results/report")
plan.save("results/plan")

results = sweep_programs(candidates, **search_inputs)
write_report(results, "results/sweep-report")
```

Saving the phase artifact and rendering a human-readable report remain separate
explicit operations. Report rendering does not initialize CUDA or runtime pools.
Timing and memory quantities retain their units and measurement/estimate
provenance. Host copies and collectives retain distinct events/accounting;
communication already included inside task timing must not be charged twice.
Keep per-rank/shared-resource memory meanings explicit rather than presenting an
unlabeled cross-rank total as a per-GPU peak.

Report methods on Plan/SearchResult were not selected; use the standalone
function. The exact nested summary/diagnostic schemas remain to design. Avoid
requiring callers to navigate a parallel hierarchy of planning-report wrappers
to obtain the timeline of the Plan they already have.

## Plan as a distributed executable

The user wants the saved plan to serve as an executable artifact: plan once,
save it, then let every participating worker load that same artifact and run its
part. `plan_program` returns one such `Plan`; `sweep_programs` returns selected
plans for its feasible candidate/budget outcomes. Search diagnostics need not be
loaded to execute a selected plan.

The artifact is independent of the particular process that loads it. It remains
specialized for the declared input geometry, target compatibility, memory
requirements, and logical parallelism configuration. An eight-participant plan
does not become a four-participant plan merely by loading it with another rank
environment.

| Stored execution description | Bound in each execution worker |
| --- | --- |
| Named logical devices, rank roles, and ordered groups | A consistent mapping from logical participants to physical workers/devices and actual process groups. |
| Parameter/gradient/optimizer ownership rules and shard geometry | This worker's shard and actual state storage. |
| Logical input/activation partitions and communication dependencies | This worker's data/context partition, peer bindings, and communication resources. |
| Selected task entrypoints and compatibility requirements | Loaded code/provider entrypoints in the compatible environment. |
| Relative memory placements, capacities, and dependencies | Actual pool bases, addresses, streams, and event handles. |
| Input/output and persistent-state contracts | Current data, checkpoint/initialization source, and persistent state. |

The user clarified that the declaration and shared bundle must express other
models/parallel techniques and their combinations, with workers able to bind to
arbitrary logical ranks. EP and CP are examples. The user subsequently scoped
the initial refactor to symmetric programs: the artifact describes a common
rank-local computation and schedule plus logical ownership/communication bindings
for all participants. It does not require a general asymmetric multi-rank planner.
Concrete schedule/template structures remain proposals:

- Supported symmetric distributed models use one common per-rank schedule with
  rank-dependent state, input, and group bindings. The memory/allocation contract
  must be valid for every participant;
  a measurement on one rank alone does not establish that for every
  value-dependent allocation path.
- Different per-rank graphs, incompatible tensor geometries, or asymmetric task
  schedules are outside the initial supported contract and must be reported as
  such. A later extension could package role-specific entrypoints/subplans with
  explicit cross-rank dependencies, but this refactor does not introduce their
  construction API or planning algorithm.
- Rank-specific constants currently baked into captured graphs must become
  explicit binding inputs where possible, or declared specializations within
  the bundle. Merely replacing `cuda_0` strings in a serialized plan is
  insufficient.
- Loading/binding should verify plan identity, logical group membership/order,
  code compatibility, memory capacity, and the required allocation contracts.
  It must not silently re-export the model, measure tasks, or search again.
- Logical rank binding selects matching state/expert ownership, input/context
  partitions, peer order, and code. Process IDs and CUDA ordinals are launch
  details. The mapping must cover every logical participant consistently;
  changing a worker's logical role without rebinding its state is not valid.
- A physical placement can satisfy execution requirements yet have different
  communication costs from the planning scenario. Preserve and check these
  assumptions separately; the exact mismatch policy is still open. Arbitrary
  startup binding does not by itself provide live migration during execution.

See [composed parallelism and binding](build-and-parallelism.md#composed-parallelism-and-logical-rank-binding)
for the agreed group representation, per-parameter replica groups, value-dependent EP
routing contracts, and the separation of logical roles from physical placement.

Like an executable using shared libraries, this bundle still depends on a
compatible runtime/provider environment and separately supplied application data.
Portable artifact payloads must not depend on a disposable compiler-cache path
existing on the original machine. Packaging source/graphs is not itself a claim
that every CUDA kernel is already compiled: offline backend specialization and
the explicit policy for any load-time compilation remain to settle.

The [artifact review](artifact-review.md) assembles the proposed archive contents
and CPU-load/execution-check boundaries. The user selected exact matching of
relevant recorded framework/compiler/runtime/provider builds initially, including
implementation identities for locally modified dependencies. CPU inspection and
reporting remain independent of execution validation. This does not match every
installed package or physical machine identifier. Supported version ranges were
not selected initially; compatible lazy binary loading/compilation must still
preserve finalized task and allocation requirements.

### Estimated memory: selected simulation-only boundary

The user clarified that estimated memory is for planning and simulated timelines
only. A Plan whose selected work relies on estimated memory cannot execute. Keep
that distinction in its inspection and saved metadata, and have execution reject
it with the affected tasks/resources identified. Exact fields remain to design;
this does not introduce a separate public Plan type.

To produce an executable plan, replace estimated memory with complete information
from compiler/provider descriptions and profiling as needed, then apply profiles
and plan again. Do not introduce worker-side measurement or replanning as a way
to execute a simulation-only plan. Orchestrators may explicitly compose the
existing phases before starting execution.

Compiler-derived sizes/lifetimes and adequate provider declarations can establish
memory requirements without GPU measurement. The requirement is complete,
non-estimated allocation information, not a mandatory measurement for every
statically known value. Measured peaks alone still do not supply unknown output
extents, aliases, or all allocation behavior required by the runtime. Supported
data-dependent paths need valid bounds/guards; one observation does not establish
all possible paths.

This restriction concerns memory estimates. Duration and transfer-cost estimates
remain performance-model inputs and do not by themselves make an otherwise valid
executable memory layout incomplete. Only selected executable work needs its
execution requirements resolved; estimates for unused alternatives need not
block execution. Environment/capacity checks and enforcement of the declared
allocation behavior remain required for executable plans. Exact admission
fields, guard coverage, and mismatch handling remain to design.

## Measurement and execution process boundary

The user requires ordinary profiling and ShadowSpill execution to use separate
processes. This does not require a separate process for each CPU phase or for
every training step. A fresh execution worker can own a long-lived training loop.

Proposed execution startup sequence:

1. Load portable program/profile/plan metadata and check compatibility on CPU.
2. Install the ShadowSpill allocator before CUDA initialization, then create the
   execution/spill pools and bind live distributed resources. Exact bootstrap
   ordering within this step must respect the allocator implementation.
3. Materialize/bind model state and input sources, validate executable allocation
   requirements against the plan, then run repeated steps.

The user chose a worker-side execution API usable from a caller's fresh process
or `torchrun`, plus an optional launcher used by convenience orchestrators.
Exact API names, state loading, and results/checkpoint handoff remain open.
Artifacts describe structure and plans; they do not silently become checkpoint
containers or transfer live CUDA tensors/process-group handles across this
boundary.

### Trainer and public Executor: selected relationship

The user clarified that Trainer will normally interact with the Executor API,
while Executor remains public for fine-grained control. Trainer and direct users
must use the same execution implementation. Use "worker" for the fresh process
hosting an executor, rather than introducing an additional required public
ExecutionWorker abstraction. Concrete signatures remain to design.

Trainer manages the surrounding training workflow: input iteration, repeated
calls, runtime hyperparameter updates, reporting, and checkpoint orchestration.
It can compose the public build/measurement/planning phases or use their saved
outputs. The Executor runs a prepared Plan with caller-provided inputs and
persistent tensors, and returns its outputs. An optimizer already captured in a
build_step plan runs as part of that plan, not as an additional Trainer optimizer
update.

The fresh execution-process boundary still applies. Trainer's orchestration must
arrange that the executor starts in a suitable worker; constructing a Trainer in
a process that has already profiled on CUDA cannot make allocator replacement
safe. The explicit caller/torchrun worker path and optional launcher remain
available. This relationship does not decide an RPC/data-loading protocol or
require the execution worker to own the outer training loop.

### Executor startup: selected direction

The user selected that creating an Executor handles normal allocator/pool
bootstrap and logical-group/provider-resource binding from the saved Plan, given
the caller's device, complete rank mapping, and distributed launch context.
Advanced callers can supply compatible bindings through an explicit setup path.
Trainer would use this same entry point in its execution worker. Do not require
every launch script to reproduce the internal bootstrap sequence.
Exact names and arguments are provisional:

```python
# In a fresh execution process, before CUDA initialization.
plan = Plan.load("plan.zip")
with Executor(plan, rank_map=rank_map, device=device) as executor:
    executor.initialize(load_values)
    for call_inputs in data:
        pending = executor.run(...)  # Supply runtime args/kwargs or an input loader.
        outputs = pending.result()
```

Construction first checks CPU metadata, rejects a simulation-only plan, then
performs allocator/runtime setup before any CUDA-dependent group/provider
initialization. Group creation follows a consistent global ordering and the
launch-wide logical-to-physical mapping. It does not spawn other processes,
build/profile tasks, or plan again. An optional launcher remains separate.

The user selected the same complete launcher-worker-to-logical-rank mapping as
an input on every worker. Each Executor derives its logical rank from its launch
identity and this map. Coordinated startup validates map agreement, one-to-one
coverage, shared Plan identity, and compatible physical placement. Each worker
loads the same archive; the logical rank assignment does not change its content.
This supersedes the earlier per-worker logical-rank choice. `rank_map` is the
proposed argument spelling; exact rendezvous/context inputs remain to specify.
Logical group member order must be preserved when physical worker numbering
differs. Missing-worker timeouts and mapping failures must be reported before
dependent GPU collectives. See the concrete [Executor example](executor-review.md).

The executor retains live parameter/buffer/optimizer storage and resources across
calls. Initialization/loading is explicit and must support bounded, shard-aware
writes into the declared storage without requiring a full model payload on every
rank. `load_values`, checkpoint/random helpers, incoming gradients, and runtime
argument forms need separate review; this example does not select those APIs.

Current `runtime/bootstrap.py` installs the framework allocator once per process
and cannot uninstall it. An executor close operation can release its owned plan
and backend resources, but must not promise to restore the ordinary PyTorch
allocator or make the process suitable for ordinary profiling afterward. Reusing
an installed runtime, caller-owned resources, and exact cleanup semantics remain
to design.

Requiring Trainer or every direct caller to reproduce runtime/group setup was
not selected for the common path. The fresh-process boundary and optional
advanced compatible bindings remain part of the design.

### Tensor initialization interface: selected direction

The user selected a fill-into-destination interface for loading or initializing
persistent tensors. Executor determines the owned regions and presents bounded chunks with
their semantic tensor/parameter names, role/component, global geometry, dtype,
and logical slice information. A caller-provided function or helper fills the
provided destination. Exact request fields and signatures are provisional:

```python
def fill_tensor(request, destination):
    # Read or generate only the requested tensor region into destination.
    ...

executor.initialize(fill_tensor)
```

Destinations are views into the declared storage where directly accessible, or
bounded staging buffers which Executor publishes to that storage. Avoid requiring
an in-memory copy of the full model or a full parameter on every rank. The same
bounded interface can cover parameters, buffers, masters, optimizer tensors, and
explicit incoming gradients; which values are required/defaulted and how loaders
are composed remain separate decisions. The request must distinguish their roles
without requiring the caller to interpret generated alias IDs.

Trainer and checkpoint/random-initialization helpers use this same public path.
Checkpoint readers which return tensor chunks can be adapted using a bounded
copy into destination. A helper must honor the recorded dtypes, tied ownership,
quantized components, and declared sharding. Random initialization needs
model-appropriate rules and stable logical indexing so copies of the same
parameter agree across ranks/replicas; do not infer valid initial values solely
from dtype/shape. Source formats and initializer helper APIs remain to design.

Returning a tensor chunk for Executor to copy was not selected as the canonical
interface. The fill interface permits direct writes and can adapt readers that
return chunks. Callback collectives, invocation ordering, ownership/lifetime,
and failure propagation need an explicit protocol before implementation.

### Runtime input delivery: selected direction

The user selected sequences of positional-argument tuples and keyword-argument
mappings for the common Executor invocation, with an optional indexed input
loader for streaming. This is separate from build's one representative example plus
`num_examples`: execution supplies the actual values for all of those slots.
Each slot is one complete rank-local model/objective call, possibly containing a
batch of dataset records. Names and exact signatures below are provisional:

```python
# A Plan whose Program has num_examples=2.
pending = executor.run(
    args=[(features0, targets0), (features1, targets1)],
    kwargs=[{"mask": mask0}, {"mask": mask1}],
)

# Alternative delivery for the same fixed example slots:
pending = executor.run(input_loader=get_example)
# get_example(i) returns (args, kwargs) for slot i of this invocation.
```

There is one outer entry per example slot, including for `num_examples=1`;
`kwargs` may be omitted when unused. Nested argument values retain their built
structure without introducing an LLM-specific schema or a mandatory call-wrapper
class. Static arguments/guards and tensor geometry must match the Program. A
sequence of values does not require all examples to reside on GPU at once.
The sequence form and loader form are mutually exclusive for one invocation.

Planning chooses execution order; neither list order nor input loading order
replaces the saved task schedule. An example index names a stable slot within
the invocation, not an instruction to advance a dataset iterator. Replay must
use the same values. The callback invocation count, retention/release protocol,
staging scopes, and any prefetching need separate definition. Trainer assigns
dataset items to the slots and can adapt its data loader to this public path.

Requiring an indexed loader for every call was not selected. Invocation completion,
output ownership, and runtime hyperparameter delivery are specified below.

### Invocation completion: selected direction

The user selected a per-invocation result handle, reusing the existing
`InvocationResult` concept. `Executor.run` returns after dispatch without an
unconditional final GPU-completion wait. `pending.result()` waits for that
invocation, including its required writebacks, and returns its caller-owned
outputs or propagates execution errors. Keep `executor.synchronize()` for waiting
on outstanding executor work. See [Executor review](executor-review.md).
The user selected the model's native output as the forward payload and a small
training result record with per-example losses/metrics for the two training
builders. Preserve example-slot order and auxiliary metric structure; Trainer
or callers perform deliberate aggregation. Gradients remain Executor-owned.
Exact record naming and optional observations/diagnostics remain to finalize:

```python
pending = executor.run(args=[...])
# CPU work may overlap completion here.
outputs = pending.result()
```

Callers may instead wait with `executor.synchronize()` before retrieving the
payload. Repeated `result()` calls return the same completed result; neither
re-dispatches the invocation.

Returning after dispatch permits CPU work to overlap outstanding GPU work; it
does not promise immediate return, no task-boundary waits, concurrent invocations,
or an unbounded queue of steps. Executor must preserve caller-stream ordering,
input/output storage lifetime, and mutable parameter/optimizer dependencies.
CPU-visible results must be ready when exposed for ordinary CPU consumption;
unfinished CPU buffers cannot silently be returned as ready values. Checkpoint
access and teardown must respect completion. An explicit wait should cover the
executor's outstanding work rather than require a device-wide synchronization.

Waiting for completion before every `run` returns and returning outputs directly
were not selected. The user selected one active invocation per Executor, with a
subsequent `run` waiting automatically for the previous invocation and its
required writebacks before reuse/dispatch. Explicitly retrieving the prior
outputs is not required. The earlier completed handle and outputs remain valid;
retained output storage remains accounted. This does not introduce simultaneous
optimizer updates or an unbounded invocation queue. If the prior invocation
failed, propagate the error before dispatching the next. Output ownership and
runtime hyperparameter delivery are specified below.

### Returned output lifetime: selected direction

The user selected caller-owned returned results which are not overwritten or
invalidated by subsequent invocations. A caller can retain an earlier forward result, loss,
or metric without explicitly cloning it solely to protect it from executor
buffer reuse. The selected native-forward/training-record payloads share this
lifetime; the example applies to either plan kind:

```python
first = executor.run(args=first_inputs).result()
second = executor.run(args=second_inputs).result()
# first still represents the first invocation's returned results.
```

This concerns returned computation results, not automatic snapshots of persistent
parameters, gradients, or optimizer tensors. Those remain executor-managed;
access/export of persistent tensors needs its own interface. Public results that
would otherwise alias mutable executor storage need a safe handoff or a copy to
satisfy this lifetime. Do not require copying every output when ownership
can be transferred safely.

Retained outputs consume memory. Their handoff must be modeled, and later
invocation admission must account for storage the caller still owns. A successful
single-invocation plan cannot promise arbitrary output retention at no memory
cost. Exact storage location, budgeting, admission, release, and external-stream
rules remain to design. Dropping a Python reference must not release storage
before outstanding GPU uses complete.

Borrowed outputs valid only until the next invocation were not selected as the
common API. The returned container/schema and access to persistent tensors remain
separate decisions.

### Runtime hyperparameters: selected direction

The user selected passing runtime-changeable settings through a plain `hyperparams`
mapping on each `Executor.run`, rather than requiring a separate mutation of
executor settings before calling it. Trainer's scheduler supplies this mapping through
the same public interface. A learning-rate example, with provisional spelling for
the build-time declaration, is:

```python
program = build_step(..., hyperparams=("lr",))
# Measure/apply profiles and plan as usual, then initialize an Executor.
pending = executor.run(args=[...], hyperparams={"lr": 2e-4})
```

The saved Program/Plan must identify which values may vary at runtime and validate
their expected structure. A supplied value must reach the captured computation;
changing a captured constant cannot silently do nothing. Runtime values belong to
their invocation and must not be overwritten while an earlier invocation still
uses them. Per-call mappings do not require a new public wrapper class or embedding
the scheduler in the Plan.

A persistent setter followed by `run` was not selected as the common interface.
Exact declaration/key schemas, optimizer-group overrides, model-buffer settings,
omission/default behavior, validation, and checkpoint handling remain to review.

### Optional incoming gradients: selected direction

For `build_forward_backward`, the user selected one saved Program/Plan supporting
both fresh-gradient execution and accumulation into persistent incoming gradients.
The user selected fresh gradients by default (`accumulate_gradients=False`),
with explicit `accumulate_gradients=True` to add to existing gradients:

```python
# Executor holds a forward/backward-only plan; both calls use its fixed shapes
# and num_examples. Each call may accumulate across several examples internally.
executor.run(args=first_inputs).result()  # Fresh gradients by default.
executor.run(args=next_inputs, accumulate_gradients=True).result()
```

Fresh execution replaces prior gradients; accumulation adds new contributions to
existing gradients. Build must represent the required backward forms and
dependencies. Profiling and planning must cover both supported behaviors,
including initialization, persistent-gradient reads/writes, reductions, and any
additional temporary storage. Runtime selects prepared behavior; it does not
silently mutate the schedule or perform unplanned allocations/communication.
The representation may share code and schedule structure; it need not duplicate
the entire artifact. Exact representation and planning strategy remain open.

For already-reduced incoming gradients, the mathematical result is the incoming
gradient plus the reduced new contributions. Reducing the old replicated value
again would overcount it. Incoming layout and reduction status must therefore be
validated and preserved. Deferred synchronization remains explicit. External
gradient loading uses the agreed bounded initialization path; exact loading,
missing/per-parameter presence, reset, and validation APIs remain to review.

Requiring separate Programs/Plans for fresh execution and incoming-gradient
accumulation was not selected. This choice does not change `build_step` optimizer
behavior.

### Persistent tensor export and checkpointing: selected direction

The user selected a bounded, format-independent tensor export interface on
Executor, with checkpoint-format helpers using it. This complements initialization
into Executor-provided buffers. Trainer and direct callers use the same export
path. Illustrative names and callback fields, not final signatures:

```python
def write_chunk(tensor_info, values):
    writer.write(tensor_info, values)

executor.export_tensors(write_chunk)

# A common helper can supply the writer and checkpoint metadata:
save_checkpoint(executor, path)
```

The exporter describes semantic names, components/dtypes, global geometry, and
the actual logical tensor regions exported. It should support bounded access to
owned shards without requiring full-model materialization or a full gather on
every rank. A writer can consume bounded views/staging buffers; the exact
callback/iterator and buffer-lifetime protocol remains to review. This is
different from the selected stable lifetime of ordinary `run` results.

Executor owns safe access to live tensors and their consistency with completed
work. Checkpoint helpers own encoding/file organization, reconstruction metadata,
and distributed publication. Trainer adds its own progress/data/scheduler
information as appropriate. Full resume also needs the applicable optimizer,
master, gradient, RNG, and control values; weights-only export is a different
selection. Exact contents, shard deduplication/resharding, frozen-base handling,
failure/atomic-publication behavior, and any asynchronous snapshot support remain
to review. This choice does not imply background checkpointing concurrent with
updates to the exported storage.

`Plan.save` continues to save the executable artifact, not live model/checkpoint
values. The selected archive format direction for phase artifacts does not by
itself choose a model checkpoint format. Current `PlannedStep.save` already
streams pool objects into a checkpoint without creating a full CPU snapshot;
the refactor should preserve bounded export while exposing a reusable public
boundary instead of requiring every format adapter to access runtime internals.

Built-in checkpoint encoding and file writing inside the core Executor was not
selected as the responsibility boundary. Exact helper and export signatures
remain to review.

### Profiling implementation considerations

Pool-independent measurement requires replacing the existing profiler's runtime
allocation scopes, event wrappers, and spill-backed saved-value handling. A
candidate implementation uses CUDA events for timing and ordinary allocator
allocation history/provider declarations for missing memory evidence. This is a
proposal requiring validation against the installed PyTorch/providers, not a
settled measurement backend. Timing and detailed allocation recording should be
separate passes so instrumentation overhead is not mistaken for task runtime.

The user permits a ShadowSpill allocator profiling backend if observation hooks
cannot preserve the information required by admission/execution. Such a backend
would be contained within measurement and would still return portable `Profiles`;
it must not reintroduce runtime/pool arguments into build or planning/search.
The [memory evidence audit](memory-and-profiles.md) records the existing native
PyTorch hooks and the qualification needed before selecting a backend.

Compiler-visible lifetimes, observed requested allocations, allocator-reserved
memory, external provider allocations, and runtime-specific placement evidence
must remain distinguishable. Planning can perform static placement checks without
real pools. Validation that actually uses the execution allocator belongs to the
execution process; an estimated memory bound is not proof of that validation.

## Questions this draft deliberately leaves open

These details are grouped into the [six remaining design reviews](README.md#six-remaining-design-reviews).
They are an inventory, not nine additional preference questions. Review concrete
examples together and ask only about unresolved choices that affect public
behavior; investigate implementation feasibility separately.

1. Optional incoming gradients: loading/presence protocol and internal
   representation for the selected per-call modes in one artifact. Fresh is the
   selected default; explicit `accumulate_gradients=True` enables accumulation.
2. Communication inside compiled tasks: how to expose costs without double
   counting, while retaining useful fused providers.
3. Backend compilation that currently queries hardware or autotunes: exact
   preparation hooks and compatibility/persistence requirements. Build owns
   task finalization and may require a GPU; measurement does not replace task
   definitions. An unfinished build does not return a completed Program.
4. Exact reporting of simulation-only plans and the non-estimated allocation
   information required by execution; the estimated-memory boundary is selected.
5. Runtime binding/execution APIs and ownership of mutable state.
6. Exact class fields, defaults, cache/persistence format, and source layout.
7. Source-map representation, display/override policies, and provenance through
   fused and differentiated graphs.
8. Pool-independent memory observation and how execution handles a mismatch
   between supplied memory evidence and actual allocation requirements.
9. Composed logical parallelism, symmetric shared-plan communication contracts,
   and physical worker binding. Asymmetric programs are deferred.

## Migration and cleanup

The [source layout and migration review](source-layout-review.md) maps current
entry points and callers to the selected phase boundaries. The user accepted the
small top-level public API and phase-organized directories, with small focused
files/functions and useful internal hierarchy. Work uses the separate
`codex/offline-pipeline-refactor` branch. The user selected migration of repository
callers followed by removal of superseded APIs and duplicate implementations,
without deprecated public compatibility wrappers. No source implementation or
deletion has started for this refactor.

After agreement, map current APIs and their callers to these responsibilities.
The inventory includes `plan_step`, `plan_step_search`, `build_step_programs`,
`prepare_step`, `search_prepared_steps`, `admit_step`, intermediate planning
artifact classes, and the several store/archive wrappers. Preserve capabilities
under their new owners before removing superseded entry points or redundant code;
the migration policy is not authorization to discard advanced behavior.

Trainer, quickstart, recipes, benchmarks, and tests should use the agreed public
boundaries. Remove redundant implementations and obsolete documentation after
their uses have migrated. Temporary internal adapters do not remain as public
compatibility APIs; repository cleanup is part of completing the refactor.
