# Build controls and parallelism: proposal for discussion

Status: source findings, agreed builder/parallelism boundaries, and proposed
controls. The user chose named build arguments with focused policy objects and
one central parallelism argument. Specific policy classes, fields, and defaults
below remain proposals.

## Where these choices enter today

| Choice | Current location | Effect |
| --- | --- | --- |
| Stage partitioning | `plan_step(partition=...)`, applied during capture/partitioning | Defines stage boundaries before graph-pair construction. Supports `"auto"`, `"whole"`, or a `PartitionPolicy`. |
| Intra-stage retention | `memory_bound_flops_per_byte` and the graph-pair retention policy | Changes what the save variant retains versus regenerates. The save variant is not necessarily free of recomputation. |
| Available graph pairs | `graph_pair_variants` | Determines which save/recompute graphs are built and profiled. |
| Save/recompute selection | Planner `SearchOptions` | Chooses among built alternatives at a given memory budget. |
| DP replication and remaining gradient contributions | `Distributed`, bound at public import/preparation boundaries | Describes replicas, gradient reductions, and parameter-specific overrides. It does not transform ordinary model layers into TP/EP layers. |
| Weight and gradient storage sharding | `TransferSharding`, `Distributed.shard_gradients`; state materialization | Changes spill ownership and fetch/evict recipes while computation uses full rank-local tensors. Calibration currently happens while constructing these recipes. |
| Optimizer/master sharding | Separate `shard_optimizer` training argument, optimizer materialization/capture | Sizes owned optimizer state and captures update/reduction/publication operations. Defaults to enabled over parameter replicas in `plan_step`. |
| Expert parallel model layers | Workload model construction and MLOps EP layers | Changes expert ownership and the computation/communication graph before capture. Existing construction needs live groups and CUDA communication buffers. |

`Trainer.DataParallel` is an existing convenience configuration that creates the
execution/control groups during preparation. Advanced callers supply
`Distributed` with their own groups and per-parameter overrides. These objects
currently mix logical ownership with live distributed resources.

Sources:

- [Training API](../../../../src/shadowspill/pytorch/api.py)
- [Partition policy](../../../../src/shadowspill/pytorch/partition/policy.py)
- [Retention policy](../../../../src/shadowspill/pytorch/capture/retention.py)
- [Distributed declarations and binding](../../../../src/shadowspill/pytorch/distributed/__init__.py)
- [Trainer DP convenience](../../../../src/shadowspill/training/parallel.py)
- [Transfer sharding](../../../../src/shadowspill/pytorch/distributed/_backing.py)
- [State/optimizer materialization](../../../../src/shadowspill/pytorch/planning/training/materialize.py)
- [Distributed optimizer capture](../../../../src/shadowspill/pytorch/distributed/_optimizer.py)
- [Workload EP construction](../../../../workloads/models/shared/expert_parallel.py)

## Agreed build entry points; proposed controls

The user requested three public entry points that share their internal passes:

| Entry point | Computation/state boundary |
| --- | --- |
| `build_forward` | Forward computation. |
| `build_forward_backward` | Forward and backward over multiple microbatches, excluding optimizer work; optionally takes existing gradients, creates absent gradients at their first contribution, accumulates later contributions, and retains final gradients persistently. |
| `build_step` | Forward, backward, and optimizer update. |

All produce the agreed `Program` abstraction and feed the same measurement,
profile application, and planning phases. This supersedes the earlier provisional
name `build_program`.

The existing graph-pair implementation already has initialization and accumulating
backward forms. The forward/backward-only builder should share that machinery;
it needs its own persistent output-state boundary and must not require an
optimizer capture. Sharing internals with `build_step` should not force an extra
spill of all gradients before an optimizer that could otherwise run earlier.
The user selected one Program/Plan supporting fresh gradients and accumulation
into existing gradients, selected per call. Both behaviors must be represented,
profiled, and planned with their memory and reduction requirements. Exact runtime
arguments, defaults, presence validation, and internal representation remain to
design; see the [API draft](api-draft.md#optional-incoming-gradients-selected-direction).

For distributed forward/backward calls, the user chose reduced gradients in the
declared sharded/replicated layout on completion, with an explicit deferred-sync
mode. The no-optimizer path therefore still needs gradient-completion operations;
it cannot simply delete every task that currently surrounds an optimizer update.
Incoming gradients' layout and reduction status must be respected so already
reduced contributions are not reduced again as though they were rank-local.

The user chose important controls as named keyword arguments. A caller can omit
them to use defaults, choose a built-in policy, or supply an advanced policy
object. An all-in-one `BuildOptions` object is not mandatory.

Illustrative argument groups, not an approved signature:

```text
build_step(
    model,
    example_args=...,
    example_kwargs=...,
    num_examples=...,
    objective=...,
    optimizer=...,
    target=...,
    partition=...,
    operator_recompute=...,
    parallelism=...,
    precision=...,
)
```

| Proposed control | Responsibility |
| --- | --- |
| `partition` | Stage boundaries. The user selected `"auto"` by default, retaining `"whole"` and custom `PartitionPolicy` support. |
| `operator_recompute` | Operator-level retention during graph construction: which intermediates a built variant retains or regenerates. The user selected this argument name and the existing default heuristic plus custom policies; exact protocol remains open. Stage-variant selection for a memory budget remains planning work. |
| `parallelism` | Logical participants/groups, computation and activation partitioning, and state ownership needed to construct a distributed program, including EP/CP combinations. Real process-group binding is separate. |
| `precision` | Build-time gradient/master policies; model parameter representations and operator math still come from the model/provider configuration. |
| `num_examples` | Number of independent rank-local model/objective calls with the common input structure. Each example may contain a batch. The user chose this order-neutral name; it does not select traversal or count optimizer updates. An advanced explicit form handles differing shapes. |
| `target` | Compiler/provider target information needed for graph specialization, distinct from memory budgets and transfer bandwidths. |

The user requests choosing microbatch ordering during planning/search, not build.
Build records semantic dependencies and reusable initialization/accumulation
backward forms. Planning binds those forms to a legal traversal and determines
its lifetimes and placement. See the
[ordering boundary](api-draft.md#microbatch-ordering-belongs-to-planning).
The user also selected optimizer scheduling as a planning/search control. Build
retains the update computation and semantic dependencies; planning selects legal
stage-interleaved or tail placement. This includes completion of reductions and
all reads of the old parameters, plus any shared/global optimizer dependencies.

The user further proposes moving DP communication outside optimizer tasks.
Local updates would consume reduced gradients in the required shard layout,
owned masters/weights, and optimizer tensors, then produce updated owned shards.
Build describes required reductions/redistribution separately. They may initially
execute at ordered boundaries; independent overlap with later compute is an
optional extension, not a prerequisite for local profiling. Required weight
gathers also remain separate consumers of updated shards. No mandatory spill
between reduction and update is implied. This shares gradient-completion logic
with forward/backward-only calls and supports representative single-GPU update
profiling. See the [optimizer boundary](api-draft.md#dp-communication-outside-optimizer-tasks-user-proposed-direction)
for dependencies and the limits on shard-local optimizers.

The user selected automatic stage partitioning as the default. The remaining
controls should be reviewed explicitly. Current behavior provides a starting
point: both save and recompute variants, with retention controlled by arithmetic
intensity. Building both alternatives also preserves
the save timing used to explain recomputation overhead when a planner selects
recompute everywhere.

## Agreed central declaration; proposed representation

Distinguish three kinds of information:

1. **Build declaration:** logical group membership/size; parameter replication;
   which gradient contributions need reduction; sharding of compute weights,
   gradients, masters, and optimizer state; any model/operator-level TP/EP/CP
   transformation and input/activation partitioning.
   These choices affect tensor geometry, ownership, or required communication.
2. **Planning resource inputs:** explicit `memory_budgets` and
   `communication_info` for memory limits and communication costs. The latter
   contains rates, latencies, measurements, and topology describing the assumed
   placement/connectivity of logical participants. Exact rank placement,
   shared-link, latency, and measured collective representations still need
   review. A broad `machine` argument is no longer proposed. Changing link costs
   alone should not rebuild compute graphs.
3. **Live bindings:** actual NCCL/Gloo groups, CUDA devices, communication buffers,
   and model state. Measurement may bind these using ordinary allocations.
   ShadowSpill runtime pools and allocator installation belong to execution in
   its fresh process, except that the user allows an allocator-backed measurement
   fallback if required for correctness. Neither kind of live binding is required
   to build a structural program, apply profiles, plan, or search offline.

The user approved one central `parallelism` argument to all three builders, resolved
against the model once into a description retained by the `Program`. Storage
construction, optimizer capture, and runtime setup would consume that resolved
description; they would not independently choose sharding policies. The separate
`shard_optimizer` training flag would move into this specification if this design
is implemented. The resolved description belongs to the persistent program and
does not contain live process-group handles.

The user also chose to supply model-specific state ownership explicitly inside
this central argument, rather than discover it through a required model method.
Reusable model helpers may produce the declarations. Ownership describes facts
about the local computation, such as replicas, logical shards, and remaining
gradient reductions; requested storage sharding describes how ShadowSpill stores
and updates that state. Build validates and resolves both together. The exact
contract schema remains open. These declarations do not themselves transform
ordinary layers into a distributed algorithm.

Parameter/buffer selection now uses module-qualified names and patterns, with
defaults and overrides resolved during build. Parameter-object references are
not the selected public selector style. The user prefers `Sharded` for logical
tensor placement, reserving partitioning terminology for computation stages.
The earlier public names `StateContract` and `StateRule` were rejected as unclear.
The user selected separate parameter-placement and parameter-storage maps inside
the central declaration. Parameter placement describes the logical tensor used
by a local model; storage describes its persistent training tensors. Neither map
owns the model's input/activation layouts or distributed operations. Exact
container/field names and detailed matching semantics remain open.

The user selected order-independent name resolution in each map: an exact
parameter name takes priority over matching wildcard patterns; otherwise use
matching patterns if they agree, or report conflicting patterns as a build error;
use the explicit default when no entry matches. Do not infer which wildcard
pattern is more specific or use last-match precedence. Exact pattern syntax and
unmatched-selector behavior remain open.

The user selected accepting any registered name of a tied parameter.
Build collects all names of the same Parameter, resolves selections over that
shared identity using the agreed precedence, and rejects conflicting declarations
at the winning priority. An explicit selection through one name applies to every
use of that parameter; an unmentioned alias does not introduce a competing
default. Preserve all source names and one shared parameter/gradient/optimizer
identity. This concerns aliases of the same Parameter, not deduplicating distinct
parameters with equal values or collapsing arbitrary views sharing storage.

The user selected inheritance of omitted storage fields from the explicit storage
default. For example, an expert entry can change only the common storage group
while inheriting sharded weights, gradients, and optimizer data. First select an
entry by the agreed name precedence, then fill its omitted fields from the
default. This does not layer multiple wildcard entries or inherit a wildcard entry
into an exact-name entry. The resolved configuration must be complete. The user
selected a flat storage object with a common group, per-component sharding modes,
and optional weight/gradient/optimizer group overrides. Exact public field
spelling and omitted/reset representation remain under review.

Model sharding describes the part of a logical tensor used by each rank's
computation. ShadowSpill storage sharding describes how that compute tensor is
stored and reconstructed across ranks computing with the same logical tensor or
shard, within the complete-state replica. Both belong in the central declaration,
but their fields must distinguish these different meanings.

The user chose inferred gradient reductions with optional named overrides.
Build identifies ranks holding copies of the same logical parameter or shard
from the complete placement description and sums their trainable gradient
contributions by default. This applies to replicated expert shards as well as
ordinary dense parameters. For two copies of a four-way sharded model, with
groups `(0,1,2,3)` and `(4,5,6,7)`, corresponding shards require reductions across
`(0,4)`, `(1,5)`, `(2,6)`, and `(3,7)`. Being sharded within each four-member group
does not remove the copies across those groups. Technique labels and equal
local tensor shapes alone cannot establish parameter/shard identity.

Overrides cover backward implementations that already combine some or all of
the contributions. Frozen parameters require no parameter-gradient reduction.
Loss normalization is separate. Storage sharding can change the realization of
the sum to reduce-scatter, without removing the mathematical reduction.

### Replica-based public description

The user's definition of a replica is a group whose ranks collectively own the
entire logical model and its associated training state. One replica may itself
contain multiple data-parallel workers. In particular, eight DP ranks processing
different data with weights, gradients, and optimizer state sharded across them
are one eight-rank replica. `num_replicas` counts complete-state groups, not
data-parallel workers or distinct input streams. With `num_replicas=1`, gradients
from those eight workers still need reduction within the group.

The user now requests describing one complete replica and providing
`num_replicas`, rather than manually enumerating outer replication groups. All
groups, parameter placements, and sharding settings describe ranks within that
replica. The system repeats the description, multiplies ranks per replica by
replica count, and generates gradient reductions across corresponding copies.
All sharding stays within a replica under the proposed boundary. Exact API
spelling and storage-sharding controls remain to review.

The user clarified that globally sharding a job simply uses a larger replica:
for eight total participants, one eight-rank replica permits global sharding,
while two four-rank replicas permit sharding within each repeated unit. This
applies to the storage-sharding domain as well as the model's declared groups;
the actual choices remain per-parameter and configurable. No separate mechanism
for sharding across outer replicas is needed for these configurations.

For a four-rank replica repeated twice, group indices `0..3` are replica-local.
An expanded internal representation may number the units `0..3` and `4..7`,
deriving copy groups `(0,4)`, `(1,5)`, `(2,6)`, `(3,7)` for uniquely owned local
shards. Parameters replicated inside the unit also need their intra-replica
gradient contributions combined; both levels must be counted once. Physical
worker bindings remain independent of the logical numbering.

Persistent storage sharding does not eliminate local DP contributions to the
same logical weights. Inference must account for the weights reconstructed for
computation and their resulting gradients, rather than infer no reduction merely
because stored shards have unique owners. Input/data placement is described
separately and does not follow solely from the number of complete-state replicas.
The user selected independent logical inputs across group instances and repeated
replicas by default, with intentional sharing declared explicitly. This does not
require numerical values to differ or make the runtime distribute input data.

The earlier global group examples below illustrate resolved memberships. Under
this newer public description, callers supply the within-replica groups and the
replica count; they do not have to write those expanded copy groups themselves.
Explicit named ordered groups and optional mesh helpers remain useful inside a
replica. The system is still generic with respect to the computation performed
there, subject to the agreed symmetric-program scope.

FSDP2's HSDP organization provides a reference: sharding within one mesh dimension
and replication along another. Its gradient path reduces and scatters within a
shard group and then all-reduces corresponding shards between replicas. Outer
replication does not extend the storage-sharding factor; when the boundary covers
all storage, optimizer moments and masters are repeated as well. See the
[API draft](api-draft.md) for the proposed replica-local example and sources.

### Required example: two EP8 groups sharing storage inside one 16-rank replica

The user supplied this practical expressiveness requirement. It is one replica
of sixteen ranks (`num_replicas=1`), not two eight-rank replicas: the complete
persistent model is collectively owned by all sixteen ranks. This example fixes
the intended behavior, not new constructor names or a selected field schema.

The logical groups within the replica are:

```python
# Illustrative group names; their ordered memberships are the relevant facts.
groups = {
    "expert_exchange": [tuple(range(8)), tuple(range(8, 16))],
    "same_experts": [(r, r + 8) for r in range(8)],
    "all_ranks": [tuple(range(16))],
}
```

Within each `expert_exchange` group, rank position `r` computes with the same
global expert subset. Thus ranks 0 and 8 compute with the same experts, ranks 1
and 9 with the next subset, and so on. The global expert-index mapping must make
this identity explicit; matching shapes or group positions without that mapping
are insufficient. The model's expert dispatch/combine uses its eight-rank group.

The user selected a default mapping for `Sharded(group=..., dim=...)`:
equal contiguous chunks along the declared dimension, assigned
in the group's explicit logical-rank order. For a local extent `L` and group size
`G`, group position `p` owns global indices `[p*L, (p+1)*L)` and the global extent
is `G*L`. Other dimensions are unchanged. Corresponding positions in the group's
repeated memberships represent copies of the same logical slice. This is an
explicit meaning of the placement declaration, not inference from tensor shape
alone; it does not transform the model's local computation.

For illustration, a local tensor with 32 expert rows in each eight-rank group
would describe 256 global expert rows: ranks 0 and 8 use rows 0–31, ranks 1 and 9
use rows 32–63, and ranks 7 and 15 use rows 224–255. These are illustrative counts,
not a claim about a particular model checkpoint. The two-rank storage sharding
of each subset remains a separate declaration.

The user also selected allowing advanced explicit logical-index mappings when the
model uses a different assignment, subject to the initial common local shape and
graph restriction. Unequal local geometries are still outside that scope. Exact
advanced mapping fields, multidimensional composition, and physical storage
mapping for compressed tensors remain to design. Build must validate declared
global geometry and mappings against the available model/checkpoint metadata.

| Parameters | Tensor needed for local computation | Persistent weight storage | Required gradient sum, unless already performed by model backward |
| --- | --- | --- | --- |
| Attention | The same complete attention parameter on all sixteen ranks | Shard across `all_ranks` | Across all sixteen ranks |
| Routed experts | One eighth of the global expert set, shared by corresponding ranks `r` and `r+8` | Shard that expert subset across the two ranks in `same_experts` | Across the matching pair, after each rank accumulates contributions from its own EP8 group's routed tokens |

Before a task needs the full compute weight, attention reconstructs it over the
sixteen-rank group, while an expert weight reconstructs its local expert subset
over a pair. These are separate from the EP8 dispatch/combine communications.
When gradient storage is also sharded over those groups, reduce-scatter can
implement the corresponding sums. Optimizer/master storage may use the same
groups if requested; the user's example specifies weight storage, so this does
not select gradient/optimizer defaults or require identical policies.

For a layer with `A` bytes of attention weights and `E` bytes of expert weights,
balanced storage assigns `A/16 + E/16` weight bytes per rank (ignoring padding).
Local computation needs `A` attention bytes and `E/8` expert bytes at their
respective tasks. Whether these coexist or remain resident is a planning choice.
An EP8 subgroup's persistent shards alone do not contain the entire model.

The user also asked how master weights, gradients, and optimizer tensors relate
to this storage description. For the fully sharded form of this example, the
corresponding layout is:

| Stored component | Attention | Expert subset computed by ranks `r` and `r+8` |
| --- | --- | --- |
| Model weights used by forward/backward | Shard over all sixteen ranks | Shard over the matching pair |
| Master weights, when present | Matching parameter-index shards over all sixteen ranks | Matching parameter-index shards over the pair |
| Completed accumulated gradients | Matching parameter-index shards over all sixteen ranks | Matching parameter-index shards over the pair |
| Parameter-shaped optimizer tensors, such as Adam moments | Matching parameter-index shards over all sixteen ranks | Matching parameter-index shards over the pair |

Matching means the same logical parameter element range, not the same byte
range: model weights, masters, gradients, and moments may have different dtypes.
For an elementwise optimizer such as Adam, each rank can update its owned range
using its reduced gradient, master weight, and moments. It then produces the
updated model-weight shard. Only the compute weight needs reconstruction for
forward/backward; this layout does not require gathering the full master or
moment tensors for that update. Scalar counters and non-parameter-shaped
optimizer tensors require their optimizer-defined placement rather than blindly
applying a parameter slice. Frozen weights have no corresponding trainable
gradient/master/optimizer tensors unless explicitly required by the algorithm.

The user selected a common storage group per parameter by default, with optional
component-specific group overrides for model weights, gradients, and optimizer
data. Master weights follow optimizer ownership. Independent sharded/replicated
controls accommodate replicated compute-weight storage with sharded optimizer
data as well as the fully sharded example. Exact constructor fields remain to
design. Gradient reduction is still determined by contributions to the logical
parameter; its mathematical requirement does not disappear when gradient storage
is replicated.

These overrides apply to components of the *same* parameter. Different parameters
also need different groups, as the attention versus expert example demonstrates.
For example, one eight-rank replica could store each full parameter across two
four-rank weight groups, while storing its reduced gradients and optimizer/master
data across all eight ranks. Updating the four-rank weight shards from eight-rank
optimizer ownership then requires explicit redistribution, represented and
costed by the program/plan. This capability is agreed but not implemented; build
must validate compatible logical parameter coverage across the chosen groups.

Consequences for the public API and resolved program:

- Parameter declarations must distinguish model compute ownership from storage
  sharding and allow different named storage groups for different parameters.
  One global sharding boolean or one group per replica is insufficient.
- A storage group can cross the model's communication-group boundaries while
  remaining inside the declared complete-state replica.
- Build must verify that a storage group reconstructs the intended logical
  parameter/shard, and infer reductions from computation contributions rather
  than from uniquely owned persistent fragments.
- Planning needs the pairwise expert gathers/scatters, sixteen-rank attention
  gathers/scatters, and EP8 model communication with their distinct topology and
  memory effects. Model communication already included in task measurements must
  not be charged again.
- The shared plan can retain the same symmetric local graph and schedule with
  different group and expert-index bindings. This example does not require
  asymmetric role-specific programs.

### Convenience APIs and model operations

The user requires a simple default DP configuration, with no manual group or
per-parameter inventory for ordinary models. The [API draft](api-draft.md#simple-dp-convenience-and-computation-independent-of-parameters)
contains a proposed shorthand and a parameter-free context-attention example.
The user selected sharded weights, gradients, and optimizer data as the simple
DP default, with masters following optimizer ownership. Exact shorthand spelling
remains proposed. Parameter placement alone cannot express all distributed
computation: input/activation layouts and operator communication need a generic
interface independent of weight storage.

For DP, a convenience specification could construct the replica layout and count
with independent weight/gradient/optimizer sharding choices. Advanced specifications would support
named groups and parameter-specific ownership overrides. Exact names, fields,
and defaults remain open. The user chose explicit named, ordered logical groups
as the core representation, with an optional mesh helper that generates them.

TP/EP/CP are more than storage-sharding flags: they change local model operations
and communication. Existing EP workloads construct those modules before capture.
Supporting offline capture will require decoupling their structural description
from creation of live communication buffers. A generic parallelism option alone
does not make arbitrary models TP/EP/CP-capable. The central specification must
be consumed by compatible model/operator transformations or declarations;
unsupported combinations must be explicit rather than silently treated as DP.

Pure DP layer computation can use compatible measurements collected on one GPU.
Tasks that execute distributed collectives need corresponding distributed
measurements or a separately defined communication-cost representation. Their
already measured total runtime must not also be charged for the same collective
in the planner. The exact decomposition remains a design question.

## Composed parallelism and logical rank binding

The user requires an extensible description for other models, techniques, and
their combinations. EP and CP were examples; the core must not be a fixed list
of parallelism modes. Physical workers should be able to bind to arbitrary
logical ranks. The user subsequently chose symmetry as the initial scope:
participants share the rank-local computation/schedule while their data, state
shards, and group bindings can differ. Generic contracts must accommodate other
symmetric techniques, but asymmetric local graphs/schedules are unsupported for
now. This refactor does not introduce factories or collections of role modules.

The following information needs a representation; concrete public classes and
fields remain proposals:

| Information | Example responsibility |
| --- | --- |
| Logical participants and ordered groups | Which ranks exchange tokens, context fragments, or gradient contributions; group-local order and peer relationships. |
| Parameter and state placement | Expert ownership, shared-parameter replication, and weight/gradient/optimizer storage sharding within the appropriate groups. |
| Input and activation placement | Which data replica and sequence/context partition a rank processes, including CP's position/layout semantics. |
| Computation and communication | The rank's operations and dependencies, including EP dispatch/combine and CP attention exchanges. |
| Execution binding | The mapping from logical participants to actual processes/devices and live communication resources. |

The user chose explicit named, ordered groups of logical ranks as the core
representation, with an optional mesh helper that generates those groups. A mesh
is not required. For example, an eight-participant layout can have exchange
groups `(0,1,2,3)` and `(4,5,6,7)`, and same-shard groups `(0,4)`, `(1,5)`, `(2,6)`,
and `(3,7)`. A two-by-four mesh helper could generate the same memberships.
The exact constructors, group-reference fields, and helper spelling remain open.

Group names describe participation, not a parallel technique or an instruction
to run a collective. Names alone cannot determine every parameter's replica or
reduction group. Expert parameters and shared dense parameters can have different
replication groups. EP groups may subdivide the participants used for dense DP;
multiplying every scheme's degree together is not a valid universal world-size
rule. Explicit groups also permit arrangements outside rectangular grids, subject
to the agreed symmetric-computation scope and supported operator contracts.

The shared plan covers the logical participants through a common rank-local
schedule/code description and explicit communication/ownership bindings.
Structural symmetry and compatible participation/order are required; tensor
values and measured runtime need not be identical. Memory requirements must
admit a common plan valid for all participants. Exact profile aggregation and
memory reconciliation remain to design.

Different per-rank programs, such as splitting different layer ranges between
workers, need additional cross-rank planning and are deferred. Merely packaging
unrelated local schedules would not establish collective ordering or memory
lifetimes. If asymmetric support is added later, the user tentatively prefers
explicit modules over factories; no initial interface is added for either.

EP routes and message sizes can depend on input values. The plan should describe
the supported routing/communication contract and capacity requirements rather
than pretend to know every future token destination. Profiles and memory
evidence must cover those contracts. Communication embedded in a fused provider
still needs declared group/dependency/resource effects; cost accounting must
avoid charging it both inside the task and as a separate transfer.

At worker startup, an arbitrary consistent mapping may assign physical worker 3
to logical rank 6. It must then receive logical rank 6's state, input partition,
expert ownership, peer relationships, and applicable task code. Group-local rank
order must be preserved or explicitly translated by the binding mechanism;
passing reordered physical rank IDs to a backend is not sufficient by itself.
The complete mapping must cover the declared participants without duplicate
ownership and satisfy device, memory, and provider requirements.

Physical placement also affects communication costs. The plan records its
topology/cost assumptions separately from logical ownership. A different mapping
may leave execution valid while invalidating the predicted runtime; hard provider
requirements and performance assumptions need distinct validation. Binding must
not silently change the logical program or replan. The policy for reporting or
rejecting a different placement remains to agree on. Initial binding does not
imply live rank migration during a step; that would require an additional state
and communication protocol.

## Worked generic interface proposal

This earlier sketch prompted the user's request to clarify `groups` and `rules`.
The `rules` argument combines responsibilities and is not a selected public
interface. The [existing EP audit and explicit contract proposal](distributed-contracts.md)
now recommend defining ownership, communication, resources, and binding directly,
with model transformations treated as an optional convenience to review.

The purpose of the declaration is to give model construction, build, planning,
and execution one description of participants, ownership, and communication.
The following spelling is illustrative, not an approved API or existing code:

```python
parallelism = Parallelism(
    mesh={"replica": 2, "partition": 4},
    groups={
        "exchange": along("partition"),
        "replicas": along("replica"),
    },
    rules=my_model_parallel_rules,
)

program = build_step(model, ..., parallelism=parallelism)
```

This example names eight logical coordinates `(replica, partition)`. Varying the
partition coordinate at fixed replica creates two ordered four-member exchange
groups; varying replica at fixed partition creates four two-member replica
groups. The names are labels, not instructions to run any particular collective.
An explicit ordered group description would serve arrangements that do not fit
a rectangular mesh. The user has now chosen explicit groups as the underlying
representation, with an optional mesh helper. This earlier constructor spelling
and its `rules` argument remain unapproved.

`my_model_parallel_rules` was a placeholder for a model/operator extension point:

- It can transform supported ordinary modules, or describe modules that already
  implement distributed operations using logical group references.
- It specifies parameter/buffer/input/activation ownership, state sharding, and
  remaining gradient-reduction obligations. Different tensors can use different
  groups and placements.
- It describes or emits local computation and communication operations against
  those groups. A group declaration alone neither partitions a model nor
  inserts a correct distributed algorithm.
- It supplies source/ownership mappings so a local state shard can be related to
  the original model/checkpoint and semantic source names.

That broad rule interface has not been selected. State ownership is now explicitly
supplied inside the central `parallelism` argument, optionally generated by
reusable model helpers, rather than discovered through a required model method.
Operator registration and optional model-transformation helpers remain separate
design questions; neither should introduce a competing sharding configuration.

During CPU build, the builder consumes the model's structural contracts and
exports the common local graph with logical group and rank references. Supported
distributed operators need export/fake and memory/effect contracts; build must
not create NCCL communicators or CUDA communication buffers. The current EP
workload construction needs changes to support this boundary.

The resulting `Program` records tensor placement and communication dependencies
with local work. `Profiles` supplies compatible costs and missing memory
evidence. Planning uses the declared participants and machine scenario to choose
schedules and placements. At execution, a worker binds its selected logical
rank, materializes the matching state, and resolves logical group references to
live resources. The model definition should not need physical rank IDs or
device ordinals baked into the portable artifact.

A future technique should normally add a model/operator rule or provider with
the required contracts rather than require a new parallelism mode or plan
format. This is an extensibility goal, not a promise that every dynamic algorithm
already fits the initial planner. Unsupported effects, dynamic membership, or
unbounded allocation behavior must be explicit; exact extension contracts and
initial implementation coverage remain to agree on.

## Decisions needed

- Concrete configuration classes, defaults, and custom-policy contracts.
- Exact named-group/reference fields and optional mesh-helper API.
- How distributed operator structure is represented before real groups exist.
- How the model/operator rule interface is exposed and consumed, including
  ordinary-model transformation versus already-distributed model support.
- How changes in parallel layout affect required measured profile coverage.
- Shared symmetric plan representation, communication contracts, and execution
  binding, including topology-assumption validation. Asymmetric role programs
  are a future extension.

No new API or model behavior is implemented by this proposal.
