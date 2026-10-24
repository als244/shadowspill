# Model authoring contract: proposal for review

Status: proposed requirements for the refactored system. The user requested a
concrete, provider-independent definition of compatible model construction.
These requirements and their exact public representation need review before
implementation. Existing recipes, MLOps interfaces, and construction conventions
may change to satisfy the new contract; they do not define it.

## What a compatible model provides

A compatible model provides exportable computation and an inspectable description
of its state, distributed ownership, and effects. Ordinary PyTorch tensor
operations supply much of this information through export and differentiation.
Custom or distributed operations supply the parts the compiler cannot infer.

Propose keeping an ordinary `nn.Module` as the basic model interface, with no
mandatory ShadowSpill base class. The user chose explicit model-specific state
ownership declarations inside the central `parallelism` argument, rather than
discovery through a required model method. Reusable model helpers may construct
these declarations; build validates them against the model and resolves them
with the caller's requested storage sharding. The user chose module-qualified
parameter/buffer names and patterns for selection, with defaults and overrides.
Exact names take precedence over wildcard matches; conflicting wildcard matches
are build errors, and unmatched parameters use the explicit default. Exact
fields and pattern syntax remain to design. Callers can configure any registered
name of a tied Parameter; build resolves all its names together and rejects
conflicting declarations at the winning priority. Operator
effect/resource registration and entry-contract interfaces
are separate pending decisions. Single-participant and ordinary replicated cases
should use documented defaults rather than require extensive annotations.

The user chose a symmetric initial scope: one model describes the common
rank-local computation. The initial API does not accept a model factory or a
collection of distinct rank-role modules. Programs requiring different local
graphs or coordinated asymmetric schedules are unsupported for now. If that
support is designed later, the user tentatively prefers explicitly supplied
modules over a factory; no such API is selected for this refactor.

Symmetry means equivalent computation/task structure, tensor geometry, and
communication/effect order after substituting logical rank/group bindings. It
does not mean identical tensor values or identical ownership. Different data,
weight shards, input partition offsets, and corresponding group members remain
valid. Value-dependent work inside an operator must stay within its declared
capacity, memory, and communication contract.

Timing and scratch use may still vary across ranks. Shared execution must satisfy
every participant's memory/compatibility requirements; symmetry alone does not
prove that one sampled profile is valid for all runtime inputs. The exact cost
aggregation and supported memory reconciliation remain to design.

## Required model behavior

### 1. Construction must work without live execution resources

- The computation and state geometry must be describable on CPU/meta without
  initializing CUDA, process groups, communication buffers, or ShadowSpill pools.
  Existing CPU parameter values are allowed; build must not need their payloads
  merely to determine the graph and storage structure.
- Constructors may depend on configuration, the declared input signature, and
  logical parallel metadata. They must not derive logical ownership from the
  physical process rank, current GPU, environment-specific group handles, or
  machine addresses.
- Any backend resource needed by an operation is described structurally and
  bound separately during preparation/measurement/execution. Target architecture
  and dtype are explicit build facts; constructors must not infer model ownership
  from the current GPU. Compiler/provider preparation may inspect a permitted
  target GPU when needed. That does not require live execution resources for
  structural model construction.
  The user selected saved resource descriptions with provider initialization in
  workers; advanced callers may supply compatible existing resources. Shared
  resources retain one identity across their uses. Exact resource schemas and
  lifecycle methods remain to design; a general per-operator registry is not
  required by this choice.
- The user selected typed logical group references from the central parallelism
  specification for compatible layers/operators. Constructors can consume static
  metadata without a live process group; workers later bind real communicators
  and provider resources. These references must not smuggle in a physical rank,
  a CUDA buffer, or automatic distributed initialization. Exact fields/accessors
  remain to design.
- The built computation must not depend on checkpoint I/O or random weight
  initialization as a prerequisite for structural construction. Those belong to
  state initialization/loading. An algorithm that actually depends on constant
  tensor values must expose those required constants as such.

### 2. Tensor state and entry contracts must be visible

- Register parameters and model buffers under stable module paths. Declare which
  tensors are trainable. Mutable state must be registered or passed explicitly;
  provider resources use the resource contract below.
- Preserve tied parameters and views. Report real aliasing, strides, storage
  offsets, physical extents, and dtypes; do not represent shared storage as
  independent memory or hide overlapping mutable state.
  Keep all registered names of a tied Parameter available for selection and source
  reporting. Its configuration applies to the shared identity and all uses; equal
  values alone do not identify two parameters as tied.
- Compressed/wrapped tensors must expose their physical storage components and
  reconstruction metadata, including scales and auxiliary tensors. Logical
  arithmetic dtype is not a substitute for the physical storage description.
- Inputs/outputs have an exportable structure, fixed geometry for this Program,
  and explicit static controls. Different shapes or static program choices
  produce another candidate. Document which state a call reads, mutates, and
  returns, including persistent gradients for forward/backward-only calls.
- The core receives rank-local inputs, as selected by the user. Models do not
  require a global batch to be present in each worker. Optional helpers may
  prepare/distribute local inputs; their relation to distributed layouts must
  remain explicit where needed. Example geometry for build describes the actual
  local call, independently of runtime input payloads.
- External input placements are supplied through a separate map in the central
  parallelism declaration, keyed by paths into the example args/kwargs tree.
  Models continue to receive ordinary input values. Exact path syntax and layout
  propagation/validation remain to design; the map does not replace the real
  masks, positions, lengths, or other runtime metadata used by the computation.
  Distinct group instances and repeated complete-state replicas use independent
  logical inputs by default; intentional sharing is declared explicitly. Equal
  input values alone do not establish shared logical identity, and parameter
  replication does not imply input replication.

The compiler should derive ordinary views, dependency edges, and temporary
lifetimes. Authors should not hand-write an inventory of every intermediate.

### 3. Computation and differentiation must be capturable

- Forward uses operations with supported export/fake behavior. Required tensor
  computation must not disappear into untracked Python side effects or hidden
  reads/writes of process-global tensor state.
- Training additionally requires supported derivative definitions. An opaque
  operation must expose its backward behavior and saved-state contract, or have
  a supported differentiable decomposition. Forward-only programs do not need
  backward support.
- Python decisions based on configuration/static arguments may specialize a
  Program. Decisions based on runtime tensor values must be represented by
  supported graph control flow or an operation with declared behavior and memory
  requirements. Observing one example branch is not a valid general contract.
- Value-dependent routing is allowed within a declared shape/capacity contract.
  It does not permit silently changing group participation, output geometry,
  or allocation requirements beyond the admitted behavior.

### 4. Distributed state and data semantics must be declared

A replica means a rank group collectively owning the complete logical model and
its associated training state. It can contain several DP ranks processing
different data while persistent weights, gradients, and optimizer state are
sharded within the group. Authors must therefore describe input/data semantics
separately from `num_replicas`; that count alone is not the data-parallel degree.

State ownership is an explicit contract supplied in the central `parallelism`
specification, as agreed with the user. Helpers may generate it from a supported model, but build
does not rely on discovering a mandatory ownership method on `nn.Module`.

Resolve the following against the single central parallelism specification:

- Each state tensor's logical identity and local-to-global shard mapping, where
  it is sharded; which logical participants hold replicas of the same value.
- The input/output partitioning needed to interpret the local computation,
  including which participants receive distinct data versus pieces/copies of
  the same data. Relevant intermediate layouts follow from the model/operator
  contracts and graph propagation.
- The complete placement must identify copies of each logical parameter or
  shard. Build infers SUM reductions of trainable gradient contributions across
  those copies, including copies across different model-sharding groups. A
  `Sharded` placement alone does not mean no reduction. Authors provide optional
  named overrides when model backward already combines some or all contributions.
  A communication group is not automatically a replica group, and matching
  shapes alone do not establish logical parameter/shard identity.
- Storage sharding must not hide DP contributions: ranks that reconstruct the
  same logical weights for computation on different data still contribute to
  the same gradients. Their reduction is required even inside one complete-state
  replica with no outer replication.
- The objective's normalization convention. ShadowSpill must not guess a world-
  size divisor or reduce an already completed contribution again.

For example, a sharded matrix can declare its global shape, the row range
owned by each logical coordinate, copies of that same range, and which copies
still hold partial gradients. The description does not need a name for the
parallel technique that produced it.

Storage sharding selected by the caller and logical partitioning inherent in
the model must resolve into one consistent state description. They must not be
independently inferred in several pipeline phases. A model constraint can reject
an incompatible requested placement; it must not silently override it.

### 5. Distributed operations must expose communication effects

- This interface is independent of parameter ownership. A parameter-free
  operation such as attention on Q/K/V must be able to reference a logical group
  and describe distributed input/output layouts, communication, and backward
  contributions. A module's parameter list cannot stand in for its distributed
  computation. Context parallelism is an example, not a required core enum.
- Declare the relevant logical input/output relationship at distributed or
  opaque operator boundaries: independent values, copies, or shards with
  coordinates/segments and needed position/mask semantics. Ordinary inferable
  layout propagation should be handled by build, not manually repeated for
  every intermediate. Exact registration fields remain to review.
- Identify logical groups/peers and the participation/order contract for each
  invocation, including backward and recomputation. Declare matching messages,
  dependencies, or an opaque matched protocol that the planner can treat as a
  single operation.
- Declare shared resource use and ordering/concurrency restrictions. Tensor
  dependencies alone may not order communication or shared scratch correctly.
- A fused compute/communication operation may stay fused. It still needs a
  public effect description; an arbitrary integer runtime handle or the label
  "ordered" alone does not identify its participants or dependencies.
- If message sizes are value-dependent, describe the capacity/bounds and how
  runtime counts participate in the protocol. No claim is made that CPU build
  knows every future routing result.

Model authors using a compatible distributed operator bind its logical group
references; they need not repeat its complete protocol at every call site.
Neither MLOps nor a particular transport library is required by this contract.

### 6. Memory, completion, and recomputation must be truthful

- Operations expose accurate output/view/alias/mutation metadata, including
  allocation padding behind returned views and saved backward state.
- Compiler-visible temporary storage is derived by build. For opaque scratch,
  providers describe known requirements or explicitly unresolved requirements
  to be supplied by Profiles. Unknown storage must not silently count as zero.
- Persistent/external buffers and caches need capacity, lifetime, sharing, and
  ownership descriptions. Memory outside the PyTorch allocator still consumes
  the device budget. Reuse between calls/modules must be declared.
- Providers own their private streams and establish correct dependencies on the
  caller stream. ShadowSpill does not enumerate or manage those streams. The
  existing supported path relies on provider event dependencies so task
  completion on the caller stream covers the provider's work, without requiring
  a host synchronization. Propose retaining that boundary. Explicit cross-task
  asynchronous work would require an additional dependency/lifetime protocol;
  no such extension has been selected.
- Declare whether forward is safe to replay, including mutations and RNG/state
  handling. An operation with irreversible effects cannot be replayed merely
  because its calls are ordered. The builder must preserve required saved state,
  select a valid alternative, or reject an incompatible recomputation request.
- Numerical state, caches, and resource reuse must remain correct when ordinary
  tensor storage is relocated between invocations. Retaining a tensor pointer
  beyond its declared lifetime is incompatible with planned memory reuse.

Profiles may contain explicit estimates for simulation, as already agreed.
Those estimates are not proof of allocation-contract correctness for execution;
the required evidence and validation must remain explicit.

### 7. Saved artifacts must be rebindable

- Persist logical identities, tensor/state contracts, code dependencies, and
  resource descriptions. Do not rely on the original process's Python object
  identity, registry index, communicator, CUDA ordinal, or pointer.
- A logical worker binding selects the corresponding state shards, inputs,
  operations, and group-local order. Physical workers may be permuted when the
  binding satisfies the plan's requirements.
- Resource/provider compatibility and initialization/teardown requirements are
  explicit. Loading should report an unsupported binding or missing dependency;
  it must not silently recapture or change the distributed program.

## Responsibility split

| Author/component | Supplies |
| --- | --- |
| Model author | CPU/meta-constructible computation, registered state, input/output semantics, logical distributed ownership, and use of compatible operations. |
| Custom/distributed operator author | Export/fake/autograd behavior, communication and mutation effects, memory/resource descriptions, replay/completion guarantees, and live binding implementation. |
| Caller/recipe | One parallelism declaration, concrete inputs and objective, build policies, profiles or measurement request, machine/budgets, and state/data sources for execution. Recipes may be rewritten around this contract. |
| ShadowSpill | Infer ordinary graph/storage facts; resolve declarations; validate coverage and compatible effects; construct variants, profiles, schedules, placements, and portable artifacts; bind and execute under their contracts. |

One author may fill several roles, but a normal model using supported operators
should not need to implement allocator or communication instrumentation.

## Compatibility checks to provide

These are proposed validation obligations, not an implemented checker API:

1. Build the common rank-local model without CUDA/process-group initialization;
   verify its declared symmetric applicability and report unresolved metadata
   using module, parameter, and operator names. Report asymmetric structures as
   unsupported rather than silently duplicating one rank's graph.
2. Verify state mappings, ties, replica/reduction groups, and cross-rank
   participation/order compatibility for the selected graph variants.
3. Check actual custom-op outputs/mutations/storage against fake contracts and
   compare gradients to an appropriate reference when qualifying an operator.
4. Exercise replay, saved-state reuse, and stream completion under the supported
   behavior, including declared routing/capacity boundaries.
5. Save/load and bind in fresh processes with a different valid physical-worker
   mapping, verifying logical ownership and program identity.

Static validation cannot prove arbitrary provider code correct. Distinguish
compiler-derived facts, provider assertions, measured evidence, and estimates.
Never treat a successful sample as proof of all possible unbounded behavior.

## Decisions still required

- Exact ownership fields and name/pattern matching semantics in `parallelism`;
  entry-contract interfaces and operator effect/resource registration. Ownership
  discovery through a required model method is not the selected entry point.
- Initial supported control-flow, concurrency, and replay capabilities, with
  explicit errors for unsupported cases.
- The minimum evidence needed for execution after simulation with estimates.

Asymmetric module construction and planning are deferred. They are not a pending
choice that blocks the initial symmetric interface.

The [existing EP audit](distributed-contracts.md) identifies reusable mechanisms
and migration work. It is an example against which to check this proposed
contract, not the definition of compatibility.
