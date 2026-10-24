# Existing distributed models and explicit contracts

Status: source audit and proposal for review. No implementation or exact public
contract schema has been agreed. EP is the concrete existing example; the target
contracts describe distributed operations generally.

The [model authoring contract](model-authoring-contract.md) is the forward-looking
proposal for compatibility guidelines. This document audits the existing EP path
and migration considerations; existing recipe/provider interfaces are not
requirements that the refactor must preserve.

The user subsequently chose a symmetric initial scope: one model, equivalent
rank-local computation, and a common schedule with rank-dependent bindings.
Contracts remain generic across techniques, but asymmetric graphs/schedules and
their construction API are deferred. The audit's discussion of more general
cross-rank dependencies identifies a future limitation, not an initial feature
commitment.

## How the existing EP path works

The workload supplies an already distributed PyTorch model. ShadowSpill does not
transform an ordinary MoE into EP, infer expert ownership from shapes, or implement
token dispatch itself.

1. The recipe creates a live NCCL group and constructs the model with `ep_group`,
   token capacity, device, and provider selection. The shared model constructor
   installs MLOps `QuackMoE`/`TEMoE` modules and shares communication resources
   across layers. In the Quack path inspected here, routing, MoonEP transport,
   expert computation, and their backward operations are implemented in MLOps.
2. The recipe separately supplies `Distributed(group,
   replica_overrides=[(model.expert_parameters(), None)],
   groups={"experts": group})`. Thus the expert parameters are unique local
   parameters, while remaining parameters are replicas across the group. The
   optimizer policy is sharded over each parameter's replicas: unique experts
   retain their own optimizer state; replicated parameters have sharded state.
   This recipe does not transfer-shard computation weights or gradients.
3. The Quack module calls custom forward/backward operators with explicit tensor
   inputs, a serialized configuration `spec`, and a process-local integer runtime
   `handle`. Fake implementations derive shapes, dtypes, and padded saved-state
   backing extents from the spec without reading the live handle. Autograd saves
   explicit state tensors and invokes the registered backward operator.
4. Those operators are registered as `ORDERED` effects. ShadowSpill preserves
   their local compiler ordering and joins replayed forward and derivative
   effects into one backward chain. Ordered effects do not by themselves declare
   replay safety or the complete distributed communication protocol.
5. Preparation coordinates profiling participation and matching task roles
   across ranks. Planning agrees on an ordered task sequence and feasible
   save/recompute choices. Memory schedules can remain local; the verified
   symmetric path can share schedules when its compatibility checks pass.
6. During execution the live MLOps runtime performs its internal communication.
   Its stream wrapper joins communication/transfer streams back to the caller
   stream before the task completion event can finish. This is GPU stream
   ordering, not a host/device synchronization after every call. The current
   contract does not support untracked communication outliving the task.

The current ownership contract is already more general than DP. Parameters can
have different replica groups, and `gradient_group`/`gradient_overrides` describe
which reductions remain after model backward. Remaining contributions are SUM
reduced; the caller supplies objective normalization and rank-local input data.
For replicated copies of an expert shard, callers can declare that shard's
replica group instead of `None`.

The group binding layer gives explicit PyTorch functional collectives stable
aliases based on a caller-provided name plus actual ordered membership. It
rewrites `_c10d_functional` group arguments. It does not inspect the MoonEP
protocol hidden behind a Quack runtime handle or turn that handle into a portable
resource reference.

Sources:

- [EP recipe](../../../../workloads/recipes/text/expert_parallel.py)
- [Model resource construction](../../../../workloads/models/shared/expert_parallel.py)
- [Qwen model](../../../../workloads/models/shared/qwen_moe/model.py)
- [Ownership API](../../../../src/shadowspill/pytorch/distributed/__init__.py)
- [Ownership records](../../../../src/shadowspill/pytorch/distributed/_ownership.py)
- [Optimizer reduction/sharding](../../../../src/shadowspill/pytorch/distributed/_optimizer.py)
- [Coordinated profiling](../../../../src/shadowspill/pytorch/distributed/_profiling.py)
- [Plan selection](../../../../src/shadowspill/pytorch/distributed/_selection.py)
- [Effect replay ordering](../../../../src/shadowspill/pytorch/capture/effects.py)
- [Functional collective group binding](../../../../src/shadowspill/pytorch/distributed/_groups.py)
- [Current documented execution contract](../../../../docs/python/api/distributed.md#preparation-and-execution-boundaries)
- [MLOps Quack module](../../../../../mlops/src/mlops/expert_parallel/quack/layer.py)
- [MLOps custom operators](../../../../../mlops/src/mlops/expert_parallel/quack/operators.py)
- [MLOps resources](../../../../../mlops/src/mlops/expert_parallel/quack/runtime_resources.py)
- [MLOps stream completion](../../../../../mlops/src/mlops/expert_parallel/quack/streams.py)

This audit inspected source, including the EP training comparison harness. It
did not run new GPU qualification tests.

## What is missing for the new boundaries

- Public EP construction requires live groups, CUDA devices, buffers, runtime
  registration, and synchronization. The fake operator already has much of the
  static shape description, but reaching it through model construction currently
  depends on those live resources.
- The exported custom-op arguments contain a process-local handle. It is valid
  for the current process and capture copies that share its registry entry; it
  is not a binding contract for another process loading the same plan.
- Ownership records describe local shapes, replicas, contributions, and physical
  tensor components. They do not express the complete mapping from a global
  expert tensor to a logical rank's expert indices for arbitrary worker rebinding.
- Ordered effects and matching local task sequences support the current EP
  schedule. They do not provide a general cross-rank dependency description for
  different roles or arbitrary new distributed algorithms.
- Communication buffer capacities, external storage, shared-bank reuse, stream
  constraints, and handle lifetime are implemented inside the provider. A
  portable plan needs explicit descriptions of the relevant requirements,
  including persistent memory outside per-task allocator observations.
- Timing a complete EP task includes its internal communication. Estimating a
  different network from local compute measurements requires additional provider
  cost/effect information; the mesh cannot infer that decomposition. Do not add
  communication costs again to an already measured complete task duration.

## Logical group references and runtime binding: proposal

The user asks for EP to use the same mechanism as parameter-free distributed
operations: a compatible layer receives a logical group description that later
resolves to a real process group. This fits CPU-only construction and the shared
executable plan. The user selected a typed reference obtained from the central
parallelism declaration. Constructors can read static metadata such as group
size without a live communicator or a duplicated `ep_size` setting. Exact public
type names, fields, accessors, and binding methods remain under review.

The tradeoffs reviewed before that selection were:

| Concern | Typed reference from the central specification | Plain group-name string |
| --- | --- | --- |
| CPU model construction | Can expose validated static metadata such as group size directly | Needs the specification/resolver as additional context, or separately supplied and validated metadata |
| Validation | Reference creation can check declaration existence and retain its scope; build still checks usage and compatibility | Names and their metadata can be validated at build, or earlier if the constructor receives a resolver |
| Model/provider integration | Requires support for a small metadata interface or adapter | Easy argument type, but still requires metadata lookup and deferred live-resource binding |
| Export and persistence | Needs explicit lowering to stable group identity/metadata; do not pickle a live Python object as the executable interface | Naturally representable as a graph string or serialized key, but the key alone does not describe membership/order |
| Runtime behavior | Resolves to the same real communicator/resources during worker binding | Resolves to the same real communicator/resources during worker binding |

The selected direction uses typed references at the model/layer boundary and
portable identities/metadata in exported IR/artifacts. Keep the reference
small and immutable, derive its metadata from the central declaration, and keep
communicators, buffers, automatic initialization, and physical process rank out
of it. This need not make providers depend on the full ShadowSpill package; the
small provider-facing interface or adapter location remains to decide. Neither
representation inherently improves GPU performance or eliminates the need to
separate structural construction from runtime setup.

Illustrative spelling, not an implemented interface:

```python
parallelism = Parallelism(
    ranks_per_replica=16,
    num_replicas=1,
    groups={"expert_exchange": [tuple(range(8)), tuple(range(8, 16))]},
    # Parameter placement and storage declarations are supplied separately.
)
experts = parallelism.group("expert_exchange")
model = MyModel(..., ep_group=experts)
program = build_step(model, ..., parallelism=parallelism)
```

The reference identifies the declared group family, not one process's current
group. In this example its static group size is eight. Worker binding for logical
rank 3 selects the first group; binding for logical rank 11 selects the second.
Both have group-local position 3. The physical worker IDs may differ under the
chosen logical-to-physical mapping. Membership and logical order must be
preserved or translated correctly by the binding layer; a raw physical rank
number must not be baked into model construction or expert ownership.

The proposed lifecycle is:

1. **Construct/build on CPU.** The layer consumes static logical group metadata,
   configuration, and tensor geometry. Fake/export implementations need no
   communicator, CUDA buffer, or live runtime lookup. Rank-dependent ownership
   is represented by logical mappings/binding inputs within the common symmetric
   computation, not a query of the physical process rank.
2. **Persist the program/plan.** Save logical group references and portable
   descriptions of provider resources, their capacities/sharing, and task uses.
   Do not save communicator objects or a process-local runtime registry ID as
   the identity of a required resource.
3. **Bind for measurement or execution.** Resolve references for each worker's
   logical rank to real groups, validate membership/order/backend requirements,
   and initialize the provider's declared communication buffers and resources.
   Group creation, if needed, must be coordinated in a compatible order across
   workers. Caller-supplied versus helper-owned groups and teardown details
   remain an execution-API decision. Execution binding follows the agreed
   allocator/bootstrap ordering in its fresh worker.
4. **Run repeatedly.** Bound tasks use their resolved resources. Keep group and
   resource construction outside the measured task and training loop. A
   process-local fast handle is fine in the bound execution environment; it
   must be regenerated from the saved descriptions when another worker loads
   the artifact. Binding is scoped to the loaded program/worker, not a single
   globally mutable mapping from a short group name to a communicator.

This is not a transparent replacement that can be passed unchanged to arbitrary
`torch.distributed` calls. Compatible layers/operators need an explicit
unbound-construction path and a binding implementation. A provider can retain an
ordinary eager path accepting a real `ProcessGroup`; the portable construction
path uses logical references. The provider-facing metadata/binding interface
should remain small; where its types or adapters live is still to design and
must not force every provider to depend on the full ShadowSpill planner.

For the current MLOps EP implementation, this requires separating more than
the `ep_group` argument:

- `QuackMoE.__init__` currently queries CUDA, binds a live buffer, creates a
  runtime, registers its process-local handle, and synchronizes parameters.
  Construction must instead be able to establish parameter geometry and
  portable operator/resource descriptions without those actions.
- Communication banks, buffers, and external storage are runtime resources with
  memory and sharing requirements; replacing only the process-group object
  would not make them portable or account for their storage.
- The custom-op fake functions already derive shapes and saved-state extents
  from serialized configuration without resolving the handle. This is useful
  existing structure to preserve.
- The current live-handle arguments in forward/backward must become portable
  resource references or explicit binding inputs in the artifact. Their exact
  operator schema and binding representation are not selected yet.
- Existing ShadowSpill functional-collective aliases illustrate late lookup,
  but are derived from already live groups and actual rank membership. They
  need adaptation for offline logical references and arbitrary worker mapping;
  they do not already solve the opaque EP runtime-handle case.

The same group-reference mechanism serves EP, CP, or other compatible distributed
operations. It identifies participants; parameter placement, storage ownership,
tensor layouts, and operator effects remain explicit, distinct information.

## Proposed contracts, independent of technique names

The user requires explicit contracts for non-DP parallelisms. The following are
proposed responsibilities, not selected classes or field names:

| Contract | What must be explicit | Main producer |
| --- | --- | --- |
| Logical participation | Ordered groups, rank roles, and the relationship between logical participants and operator invocations. | Central parallelism declaration, resolved with the model. |
| State and data placement | Global tensor/source identity, local shard mapping, replicas, ties, and input/output/activation layouts where needed across ranks. | State ownership is explicitly supplied in central `parallelism`, optionally generated by model helpers. Exact data/activation contract interfaces remain open. |
| Gradient obligations | Which copies represent the same logical parameter/shard; which contributions the model already combines; the remaining reductions and objective normalization. | ShadowSpill infers SUM reductions from placements by default. Optional named overrides describe gradients already combined by model backward; the caller defines objective normalization. |
| Communication effects | Participating group/peers, operation identity/order, matching protocol, dependencies, and capacities or bounds for value-dependent messages. Opaque fused operations can expose these effects without splitting their implementation. | Distributed operator/provider, connected to model group references. |
| Memory and resources | Outputs/views/mutations, saved state, temporary allocation requirements, persistent/external buffer requirements, shared-resource reuse and concurrency restrictions, and acquisition/release ownership. | Operator/provider and compiler-visible graph structure. |
| Completion and replay | When communication/storage becomes safe on the caller stream; whether forward may be replayed; required RNG/state handling and restrictions on ordering or overlap. | Operator/provider. |
| Portable binding | Stable resource descriptions and compatibility requirements; creation of process-local handles/groups/buffers after logical-to-physical rank binding. | Provider binding implementation and execution worker. |

The central declaration remains the source of group/ownership intent. The model
and provider expose how their computation uses it; they must not independently
invent a conflicting sharding configuration. Contracts become portable program
metadata, allowing build, measurement, planning, and execution to validate the
same resolved facts.

Some properties are verified structurally; others are provider assertions that
need tests/measurement evidence. The schema must distinguish those. Ordered
effects alone do not prove replay safety, bounded dynamic memory, or freedom from
distributed deadlock.

## Operator integration: requirements and options under review

The user selected saved resource descriptions with provider initialization,
including an advanced caller-supplied binding option. A second general-purpose
operator registry is not selected. Distinguish the information/binding
requirements from the mechanism used to provide them, and reuse existing
metadata and group binding wherever possible.

The current source already demonstrates several reusable pieces:

- MLOps custom-op fake implementations describe output and saved-tensor geometry
  from the serialized configuration without resolving the live handle.
- PyTorch operator schemas and autograd registrations describe existing tensor
  behavior; ShadowSpill preserves ordered effect chains during capture/replay.
- ShadowSpill's group bindings already give explicit functional collectives
  named process-group aliases, although these currently require live groups and
  use physical membership when forming identities.

The concrete portability gap in the opaque EP path is the process-local integer
handle and the live communication buffers/resources it refers to. A fresh worker
needs a portable description and an initialization/binding path. Their capacity,
lifetime, sharing, ordering, and replay restrictions must also be represented
where relevant to planning and execution. Ordinary task timings and observable
scratch can come from profiling; hidden persistent/external memory needs separate
accounting. None of this alone selects a per-operator `register_operator` API.
The selected resource direction extends logical group/resource binding using
portable descriptions. Exact public fields and any supplemental operator
metadata representation remain undecided.

### Resource initialization options

Resolving typed logical groups is already selected. The following choice concerns
additional provider resources, such as persistent/shared dispatch buffers hidden
behind the current EP runtime handle. The user selected portable descriptions
with provider initialization, retaining compatible caller-supplied bindings as an
advanced option. The alternatives reviewed are recorded below. All require
portable resource identities, configuration, memory accounting, and relevant
sharing/lifetime constraints.

| Option | How resources are created | Tradeoff |
| --- | --- | --- |
| Caller-supplied resources | After worker bootstrap, the caller creates provider resources using resolved groups and binds them to the plan's resource names. The worker validates the bindings against saved requirements. | Small core interface; execution entry points must supply the provider setup code. |
| Portable resource descriptions with provider initialization | A layer/provider produces a CPU-only description with a stable resource identity and configuration. A compatible provider implementation supplies memory requirements and initialization/cleanup. The plan persists the description; each worker creates the live resource during startup. | Supports loading and initializing a plan without reproducing backend setup in every launch script; needs a small resource interface and provider compatibility checks. |
| Per-operator adapters | Register an adapter keyed by operator to supply extra metadata and binding behavior. Shared resources still need their own identities. | Can cover broader missing operator information but introduces a more general registry than resource initialization alone requires. |

Portable resource descriptions are the selected normal path, with caller-supplied
compatible resources as an advanced binding option. This does not require a
second registration for ordinary PyTorch operators or select the exact
supplemental effect/layout metadata interface.

Illustrative resource description, created inside the provider/layer integration:

```python
# CPU-only: no communicator or device allocation is created here.
dispatch = MoEDispatchSpec(
    group=parallelism.group("experts"),
    token_capacity=16384,
)
```

The plan would save a stable provider/type identity and configuration, with
sharing expressed by resource identity; live buffers and process-local handles
are produced at worker initialization. Multiple operators and layers may refer
to one resource. How wrappers expose those references in captured calls, type
resolution without a general operator registry, versioning, and exact
memory/create/close signatures remain to review. Compatible provider code remains
an external dependency. Unresolved dynamic memory is still an explicit profile
requirement under the already agreed phase boundaries.

The caller-supplied variant would run after worker bootstrap and before tasks:

```python
dispatch = backend.create_dispatch(worker.group("experts"), ...)
worker.bind_resource("moe_dispatch", dispatch)
```

This is also illustrative syntax. Both paths must respect allocator/bootstrap
ordering, collective initialization order, resource ownership/cleanup, memory
admission, and sharing; supplying a live object does not waive those requirements.

### Earlier per-operator registry illustration

The earlier illustrative registry proposal is retained below for comparison;
it is not an agreed requirement for compatible models or all custom operators.

This option registers the additional compatibility information once per custom
operator/backend, with external adapters supported. Model authors using a
supported operation supply its configuration and typed logical group references;
they do not repeat the backend's communication/resource implementation in each
model or recipe. This is a proposal, not an approved registration API.

Illustrative spelling:

```python
register_operator(
    torch.ops.my_backend.distributed_attention.default,
    describe=describe_attention,       # CPU-only description for this invocation
    bind=bind_attention_resources,    # Live resource setup in a worker
)

attention = DistributedAttention(
    group=parallelism.group("sequence"),
    # Model/operator configuration follows.
)
```

`describe` would supply only information that existing graph/schema/fake metadata
cannot establish: distributed layouts and participation, communication effects,
scratch/shared-resource requirements, and replay/completion restrictions. It may
depend on input geometry, configuration, and logical group metadata. Known static
memory requirements become Program metadata; unresolved dynamic memory remains an
explicit profiling requirement. Reuse PyTorch's existing shape, alias, mutation,
and autograd information instead of introducing duplicate declarations.

`bind` would resolve logical groups and create or acquire the declared provider
resources during worker initialization for measurement or execution. It is not
called by CPU build or planning and is not part of each timed task invocation.
The provider's registered implementation remains a code dependency; saved
artifacts contain portable descriptions rather than live callback closures,
communicators, or registry handles. Exact callback signatures, resource-sharing
identity, registration lifetime/versioning, and teardown are still open.

An integration adapter may live outside the provider repository. The design must
not require every upstream kernel library to import the ShadowSpill planner.
Pure tensor operations with sufficient existing metadata should need no special
registration. Fused computation/communication may remain opaque when its
participation, memory, and completion obligations are adequately described.

A model-level method supplying integration descriptions was also considered;
it was not selected as the resource setup mechanism. The selected resource
descriptions keep backend setup reusable across models. Parameter ownership
remains explicitly declared in `parallelism`, as already agreed.

## Current private-stream boundary and optional extension

The user asked who is responsible for private communication streams. ShadowSpill
does not enumerate, schedule, or join a provider's private streams. The provider
owns their ordering and exposes a callable that is safe to use from its caller's
stream. The task-completion boundary relies on that existing operator behavior;
it is not a proposal for ShadowSpill to take ownership of those streams.

The current MLOps Quack EP `Streams.finish` records events on its private lanes
and enqueues waits on the caller stream. This makes a subsequent task-completion
event on the caller stream cover those lanes. It does not wait for the GPU on the
CPU or synchronize every device stream. The implementation still overlaps its
internal communication, transfer, and computation.

Both forward and backward invoke `Streams.finish`. At the ShadowSpill task
boundary, `shadowspill_after_task_record` records a completion event on the
supplied compute stream when memory actions or retirements require one. The
runtime attaches that event to triggered memory actions and eligible allocation
retirements; it does not discover the provider's private streams. Allocator
`record_stream` notifications can track tensor lifetime on additional streams,
but they do not replace the provider's responsibility for output readiness and
communication ordering.

Sources: [provider stream dependencies](../../../../../mlops/src/mlops/expert_parallel/quack/streams.py),
[forward/backward calls](../../../../../mlops/src/mlops/expert_parallel/quack/runtime.py),
[runtime task boundaries](../../../../csrc/src/runtime/tasks/task_boundaries.c),
and the [existing distributed execution contract](../../../../docs/python/api/distributed.md#preparation-and-execution-boundaries).

Propose retaining this completion boundary in the initial refactor: a task's
completion event covers all GPU work launched for that task, including use of
its inputs, outputs, temporary memory, and shared resources on private streams.
The provider must establish correct ordering for dependent operations inside
the task as well; a final completion event alone does not fix internal races.
This lets profiling measure complete task work and lets the scheduler release
or reuse memory after its last declared use completes.

For example, an MoE task can dispatch tokens on a communication stream while
computing expert chunks on a compute stream. Before recording completion, it
joins the required stream events. Separately scheduled work can overlap subject
to declared dependencies/resources, but backend work must not silently continue
accessing task memory after the completion event.

The alternative is an initial API for explicitly split asynchronous work across
tasks: launch a communication operation in one task, carry a completion token
and storage/resource obligations, and await it at a later task. This can expose
additional overlap, but requires explicit dependencies, extended lifetimes,
profiling/cost treatment, and matching distributed ordering. It is distinct from
untracked work outliving a task, which is not safe under either option.

The recommendation is to retain this existing completion boundary, deferring
explicit cross-task asynchronous operations. The earlier dropdown was about
adding that extra capability; it did not imply that ShadowSpill currently owns
provider streams. No cross-task asynchronous extension has been selected. This
does not alter ShadowSpill-managed transfer and collective lanes, whose
dependencies and completion are explicitly tracked.

## Model-side migration proposal

The existing model-authored distributed computation is a useful implementation
example. It does not require choosing a generic transformation framework or
preserving today's recipe API. Model implementations and automatic transformation
helpers should produce the same agreed explicit contracts.

For the existing workload, the main changes would be:

1. Supply logical group/layout/resource descriptors during CPU/meta model
   construction. Keep the forward's use of distributed MLOps modules.
2. Replace the coarse `expert_parameters()` ownership convention with, or augment
   it by, declarations of global-to-local state mapping, replica groups, and
   remaining gradient reductions. Reuse normal module/parameter source names.
3. Split MLOps module/resource construction into a static description and a live
   binding step. Existing spec-based fake/autograd implementations provide a
   starting point; the current live integer handle becomes an internal binding
   result rather than an unrebindable artifact constant.
4. Publish the provider's communication, memory, completion, and replay contracts.
   Preserve task-contained communication as the existing supported behavior;
   asynchronous cross-task effects need an additional explicit protocol before
   they can be supported safely.

Most GPU-resource changes belong in MLOps and the ShadowSpill binding layer.
The model's mathematical architecture need not be rewritten. The user chose
explicit state ownership declarations inside `parallelism`, optionally generated
by reusable model helpers, rather than discovery through a required model method.
The exact public descriptor types and operator effect/resource interfaces remain
to agree on.

## Clarification of the earlier `groups` / `rules` sketch

The user has selected explicit named, ordered logical groups as the core
representation, with an optional mesh helper that generates them. Concrete
group/reference fields are still open; live process groups are not build input.

`groups` named participation sets (or mesh-derived families of sets); it did not
select a collective algorithm or mean that every tensor is replicated over that
group. In the 2-by-4 example, the two four-member exchange groups and four
two-member replica groups follow from varying different coordinates.

`rules` was a broad placeholder combining ownership declarations and optional
model transformations. It is not an existing API and has not been selected.
The recommendation after auditing the existing EP path is to define the explicit
contracts above first, retaining already-distributed models as a direct path.
Whether an optional policy/helper exposes transformations is a separate decision.
