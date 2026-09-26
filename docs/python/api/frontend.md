# Frontend and lifecycle API

The symbols on this page are exported by `shadowspill.memory` or
`shadowspill.pytorch`: the pool and route configuration a runtime is built
from, the runtime itself, the state-import operations, the planning entry
points, and the callables planning returns.

## Memory pool configuration

`shadowspill.memory` holds the values that describe a machine to ShadowSpill.
They are frozen dataclasses; `device()`, `pinned_host()` and
`transfer_route()` are keyword-only constructors for them, and
`MemoryPoolConfig` is the union `DevicePool | SpillPool`.

`DevicePool` is an execution-device pool, built by `device()`:

| argument | type | default | meaning |
|---|---|---|---|
| `physical_capacity` | `int` | required | The whole process-attributable device memory cap, provider headroom included. |
| `device` | `int` | `0` | Accelerator ordinal the pool allocates on. |
| `provider_headroom` | `int` | `1280 << 20` | Bytes inside `physical_capacity` left to the provider's own allocations. Non-negative and smaller than `physical_capacity`. |

`SpillPool` is what every pool that is not the execution pool answers:

| member | type | meaning |
|---|---|---|
| `capacity` | `int` | Bytes the pool takes at construction and holds for its life. |
| `kind` | `int` | The pool-kind value the runtime looks its memory up by. |
| `kind_name` | `str` | What the pool registry reports for it. |
| `library` | `Path \| None` | The shared object supplying this kind, or `None` for one the runtime implements. Loaded once per distinct path, before the runtime is created, and kept open for its life. |
| `configuration()` | `ctypes.Structure \| None` | What this pool's kind is told about *this* pool, forwarded untouched and read by nothing in between. |

**Nothing enumerates which kinds exist**, here or in the runtime: a pool's kind
selects an acquire/release pair from a list the runtime seeds and a loaded
library appends to. A kind implemented elsewhere subclasses `SpillPool`, names
its library, and is admitted by the same validation as a built-in one.

`PinnedHostPool` is a registered pinned-host spill pool, built by
`pinned_host()`:

| argument | type | default | meaning |
|---|---|---|---|
| `capacity` | `int` | required | Pinned host bytes the pool registers. |

`RemotePool`, in `shadowspill.network`, is a spill pool held by a daemon on
another machine, built by `remote()`:

| argument | type | default | meaning |
|---|---|---|---|
| `capacity` | `int` | required | Bytes the daemon is asked for. Declared, not discovered: construction fails if the daemon cannot serve them, so a plan stays reproducible from its configuration. |
| `host` | `str` | required | Where the daemon is. |
| `port` | `int` | required | Its TCP port. |
| `selector` | `str` | `"host"` | Which memory the daemon should serve, in the daemon's own vocabulary. |

It needs `libshadowspill_network.so`, which a build without it does not
produce; asking for a remote pool without one raises where the pool is
configured.

`TransferRoute` is one directed relationship between two named pools, built
by `transfer_route()`:

| argument | type | default | meaning |
|---|---|---|---|
| `source` | `str` | required | Name of the pool copied from. Must be a Python identifier. |
| `destination` | `str` | required | Name of the pool copied into. Must differ from `source`. |

Direction is immutable: a route is never handed a copy direction later. The
backend behind it is resolved from the endpoint pools when the runtime is
constructed.

Provider headroom is inside `DevicePool.physical_capacity`. The runtime
reports the derived suballocatable capacity after initialization.

## Runtime

`Runtime` installs the allocator, registers the configured pools and routes,
and calibrates the real directed transfers between their addresses. It is
`shadowspill.runtime.Runtime` with PyTorch supplied as its frontend -- [the
neutral page](neutral.md#runtime) documents the runtime itself -- and takes the
same arguments minus that one. Construct it once, before any workload state
exists: model and optimizer state are then
created and imported into an initialized runtime, and planning reads the
published `transfer_capabilities` snapshot rather than recalibrating.

<!-- source-signature: src/shadowspill/pytorch/runtime.py:Runtime.__init__ -->
```text
Runtime(
    *,
    pools: Mapping[str, MemoryPoolConfig],
    routes: Mapping[str, TransferRoute],
    library_path: str | Path | None = None,
    calibrate: bool = True,
    worker_poll_nanoseconds: int = 1_000,
    background_transfer_window_bytes: int = DEFAULT_BACKGROUND_WINDOW_BYTES,
    backend: str | None = None,
)
```

| argument | type | default | meaning |
|---|---|---|---|
| `pools` | `Mapping[str, MemoryPoolConfig]` | required | Pool name to configuration. Exactly one `DevicePool`, and at least one `SpillPool` beside it. |
| `routes` | `Mapping[str, TransferRoute]` | required | Route name to directed pool pair. A plan can only move bytes along a route registered here. |
| `library_path` | `str` \| `Path` \| `None` | `None` | The PyTorch adapter library to load; `None` resolves the one installed beside the package. |
| `calibrate` | `bool` | `True` | Measure every registered route at construction. With `False`, planning refuses a route that was never calibrated until `calibrate_transfer_capabilities()` has run. |
| `worker_poll_nanoseconds` | `int` | `1_000` | How long the C transfer worker waits between polls. |
| `background_transfer_window_bytes` | `int` | `64 << 20` | How far a lane may run ahead with transfers the plan did not schedule. |
| `backend` | `str` \| `None` | `None` | Which backend shared object the adapter loads: `None` the one accelerator backend installed beside the libraries, a name resolves to `libshadowspill_backend_<name>.so` there, and a path is used as given. |

`background_transfer_window_bytes` bounds unscheduled work such as the opening
restore of a step's initial device set: a transfer the plan did schedule never
waits behind more than this many background bytes. Zero removes the bound. See
[transfers](../../architecture/transfers.md#dispatch).

Pool and route names are user-defined identities. A planning call binds its
own `execution` and `spill` pool names and resolves the matching directed
fetch/evict routes; those roles are not global properties of `Runtime`.
Unsupported pool or route combinations fail during runtime construction or
plan resolution.

`Runtime` owns the installed allocator, the initialized `MemoryPool` and
route registries, transfer calibration, the active callable count, the
persistent state registry, and the latest failure. Its public properties are
`pools` (name to `MemoryPool`), `routes` (name to `RuntimeRoute`),
`transfer_capabilities` (a `TransferCapabilities` matrix of `TransferProfile`
entries) and `last_failure` (a `RuntimeFailureDiagnostics` or `None`).

```text
Runtime.calibrate_transfer_capabilities(
    *,
    routes=None,
    small_copy_bytes=4096,
    large_copy_bytes=256 << 20,
    warmup_copies=4,
    measured_copies=16,
) -> TransferCapabilities
```

| argument | type | default | meaning |
|---|---|---|---|
| `routes` | `Sequence[tuple[str, str]]` \| `None` | `None` | The `(source, destination)` pairs to measure; every registered route when `None`. |
| `small_copy_bytes` | `int` | `4096` | The copy size that measures latency. |
| `large_copy_bytes` | `int` | `256 << 20` | The copy size that measures bandwidth. |
| `warmup_copies` | `int` | `4` | Copies performed and discarded before measuring. |
| `measured_copies` | `int` | `16` | Copies kept per size. |

It atomically publishes the new matrix, which it also returns. This runtime must
be locally idle, but ShadowSpill performs no cross-process barrier: callers may
coordinate several processes and calibrate concurrently to measure contended
links.

`Runtime.close()` verifies that no planning, callable, persistent imported
state, public object reference, or caller-owned device output remains, then
tears the runtime down as [memory
runtime](../../architecture/memory-runtime.md#failure-and-teardown) describes.
Release or copy ordinary device outputs before this call. PyTorch's
process-global allocator shim cannot be uninstalled, so it remains in a
permanently closed state and rejects later device allocations.

### What a pool holds

`Runtime.pool_statistics(pool="execution")` reports one pool's own numbers:
`capacity_bytes`, `allocated_bytes` and `peak_allocated_bytes`,
`requested_allocated_bytes` and its peak, `free_bytes`, `free_prefix_bytes`,
`largest_free_range_bytes`, `external_fragmentation_bytes`, `live_allocations`,
`blocked_allocators`, and the lease-record reserves. They are asked of a pool by
name rather than flattened into a runtime-wide record, because a runtime may own
any number of pools and which of them a plan uses for execution and spill is the
plan's choice. The allocator's own pool also arrives with the adapter's
statistics as `allocator_pool`, which is the one a caller on the allocation path
usually wants.

Those numbers say how many ranges a pool holds. The occupancy queries say
*which*, and whose. They are functions over a runtime rather than methods on it,
in `shadowspill.runtime.occupancy`: reading occupancy needs the runtime's handle
and pool registry and nothing of its state machine.
`live_allocations(runtime, pool="execution")` returns one
`PoolAllocation` per live range in pool order. That is what a refusal for want of
a contiguous range actually turns on: one small allocation in the wrong place
costs the largest free range and leaves the free total almost untouched.

| field | meaning |
|---|---|
| `allocation_id` | The lease's identity. |
| `offset`, `charged_bytes`, `requested_bytes` | Where it sits in the pool, and its size charged and asked for. Position is what explains a refusal. |
| `origin_plan_id` | The plan whose scope made it, or `None` when no plan did. A pool outlives any one plan, so this is what separates a range an earlier plan left behind from one the current plan made; a task id cannot, being plan-local. |
| `origin_task_id` | The scope that made it, or `None` when it was made outside any task or allocation scope -- a provider's retained state, say. |
| `origin_task_invocation`, `origin_task_allocation_ordinal` | Which invocation of that scope, and which allocation within it. |
| `object_id` | The object the range is bound to, or `None`. An unbound range is workspace: it holds no value the program named. |
| `references`, `scratch`, `plan_owned`, `ever_plan_owned`, `logical_freed`, `framework_free_seen` | The reference count; whether it was asked for as task workspace; whether the plan owns it now; whether the plan ever owned it, which set without `plan_owned` means the range was promoted out to a named owner and is no longer the plan's to release; whether the frontend has already given it up and only retirement is outstanding; and whether the framework's free has arrived. |

Three properties read those fields rather than adding to them.

`role` is what the range is for as far as the runtime alone can tell:
`planned` where a plan placed it or an object is bound to it, `runtime-object`
where it backs an object the program named that no plan owns, `workspace` where
a task made it and nothing named it, `op-internal` where a profiling probe left
it behind as provider or custom-operation state, and `unscoped` where no scope
made it at all. It stops short of whether a planned object is a parameter or an
activation: that is the program's to say, and a caller holding the program
resolves `object_id` against it.

`origin` names the scope rather than numbering it, and names the plan first,
because a task id is plan-local -- task 1112 exists in every plan, so the pair is
the identity and neither half is one on its own. It reads `plan 7 task 1112` for
a task of the program; `plan 7 profiling scope #40`, `runtime object`,
`materialization task` or `initial actions` for the synthetic scopes, whose ids
sit far above any task and would read as noise as bare numbers; and `no scope`
for a range no scope made.

`unclaimed_scope_workspace` is workspace a scope made that nobody took ownership
of. Not every range a scope made is the scope's to reclaim: one promoted out to a
named owner -- an output the caller holds, an object registered for sharing --
belongs to that owner now, one the plan placed is released by the plan's own
teardown, and one already logically freed is awaiting retirement rather than
surviving. What is left is the thing that outlives a scope by accident, and the
only thing a closing plan may take back.

`describe_live_allocations(runtime, pool="execution")` renders the whole
enumeration, one line per range: allocation id, offset, sizes, role, plan and
scope, what it is bound to, and the flags that say who owns it now, `unclaimed`
among them. A plan that has finished is named as such -- `plan 5 task 1112
(closed)` -- since a range still held by a closed plan is the signature of
something that should have gone. This reads a pool's occupancy; deciding what to
do about it is a separate question.

`plan_slices(runtime, pool="execution")` returns the other half of a pool's map:
one `PlanSlice` per admitted plan's fixed layout, in pool order. A layout is a
reserved range rather than an allocation -- what a plan's tasks place inside it
are the allocations above. `plan_id`, `offset` and `bytes` say where the plan's
own layout lies; `slab_plan_id` and `slab_bytes` name the plan that reserved the
range it lies in, and that range's size: the plan itself, unless its layout
shares another's slab.

`occupants(runtime, allocations)` maps each range to the framework objects whose
storage lies inside it -- what the range is in PyTorch's terms. A range with no
occupant is held by something the framework does not own, and that is itself the
answer: there is no reference for a caller to drop.
`retainers(held)` then names where each of those objects is
*referenced from*, since a reference can only be dropped where it is held.
Pass `ignore` the containers the question itself built, so the answer names
holders rather than the query. Both
compare addresses and return descriptions; neither keeps a reference, which
would otherwise extend the lifetime of what is being investigated.

### What a closing plan leaves behind

Nothing a plan's own scopes allocated outlives the plan.
`plan_scoped_residue(runtime, plan_handle)` returns one description per range of
that plan's unclaimed scope workspace still standing in the execution pool: the
range, the scope that made it, the object occupying it, and where that object is
referenced from.
Ranges carrying no plan -- a provider taking its own workspace between tasks --
belong to no plan and are not counted. It reports and does not release, because a
reference can only be dropped by whoever holds it, in that component's own
teardown.

`force_release_plan_scope(runtime, plan_handle)` is the forcing path, for a plan
that is closing: the ranges its own scopes made go whether or not the framework
has released them, and it returns `(storages detached, leases reclaimed)`. The
frontend half detaches the storages, so a tensor over one of those ranges stops
referencing bytes about to be reclaimed and a later read raises on an empty
storage rather than reading whatever now lives there. The runtime half reclaims
the leases, in bytes, which is what it owns; a lease the framework has not freed
keeps its pointer indexed, so the free that eventually arrives still resolves.

`reclaim_plan_scoped_residue(runtime, plan_handle)` runs both, and a closing
planned callable calls it as one of its cleanup steps: it names the residue,
reclaims it, and reports what it took in a `RuntimeWarning` rather than
raising, since a kernel is entitled to keep state between tasks and what is
wanted during teardown is visibility. Setting
`SHADOWSPILL_REPORT_LIVE_ALLOCATIONS` to a non-empty value adds a second
`RuntimeWarning` carrying `describe_live_allocations()` for the whole execution
pool, emitted before anything is released, so the residue can be read against the
rest of the pool rather than on its own.

### Plan identity

`Runtime.next_plan_id()` takes an id no other plan on this runtime will be given,
and `plan_state(plan_id)` says what became of one: `PlanState.UNKNOWN`, `LIVE`,
`CLOSED`, or `DESTROYED`. The id is taken before the plan is created, so the same
id can name the allocation scopes opened for it -- profiling runs under the plan
but outside any of its tasks, where the runtime cannot infer it. An id stays
answerable after its plan is gone, because a closing plan releases the ranges its
scopes made and so a live allocation naming a closed or destroyed plan is a
defect worth seeing. Plan resolution takes an id for each plan it creates and
records it on `PlanMemory.plan_id`; see
[plan identity](../../architecture/plan-identity.md).

The immutable runtime values are `MemoryPool`, `TransferProfile`,
`TransferCapabilities`, `ExecutionTaskIdentity`, `RuntimeFailureDiagnostics`,
and the `RuntimeRoute` records reached through `Runtime.routes`.
Configuration and execution failures use `RuntimeConfigurationError` and
`RuntimeExecutionError`.

## Persistent state

These operations move model and optimizer state between ordinary host memory,
a checkpoint file, and a runtime pool. Every one of them takes `runtime` (the
open `Runtime` that owns or will own the state) and either the `nn.Module` or
the `torch.optim.Optimizer` whose tensors they act on.

### Importing

<!-- source-signature: src/shadowspill/pytorch/state/model.py:import_model_state -->
```text
import_model_state(
    model,
    *,
    runtime,
    pool,
    release_source=True,
) -> ModelT
```

<!-- source-signature: src/shadowspill/pytorch/state/optimizer.py:import_optimizer_state -->
```text
import_optimizer_state(
    optimizer,
    *,
    runtime,
    pool,
    release_source=True,
) -> torch.optim.Optimizer
```

| argument | type | default | meaning |
|---|---|---|---|
| `model` / `optimizer` | `nn.Module` / `torch.optim.Optimizer` | required | Whose registered tensors are imported. |
| `runtime` | `Runtime` | required | The runtime that will own the resulting objects. |
| `pool` | `str` | required | Name of the pool in `runtime.pools` the state lives in, normally the spill pool. |
| `release_source` | `bool` | `True` | ShadowSpill retains no reference to the input; Python frees it when the caller drops theirs. Pass `False` only when the original must stay usable on its own. |

`import_model_state()` returns a model whose registered tensors point at
runtime-owned pool leases; `import_optimizer_state()` returns the same
optimizer object, rebound.

`import_model_state()` takes a model in either of two states, and what it
returns follows from which.

A model **on `meta`** has structure but no storage, so there is nothing to
copy: every tensor is built in ordinary memory, `reset_parameters()` writes the
values, the whole of it is imported once every module has initialized, and the
same module is returned, rebound. The transient is the model, until the import
releases it; where that is what decides whether the model fits, fill from a
checkpoint instead, which maps the file rather than reading it. The model must
satisfy the
contract in [importing state](../../architecture/state-import.md) -- constructible
on meta, dtype fixed at construction, and `reset_parameters()` on every module
that owns state -- and is refused, naming the offenders, if it does not.

A model **already materialized** is copied into the pool as it stands, and a
copied module hierarchy is returned with distinct Python identities but the
same topology, ties, views, values, and metadata. That copy is what costs a
host transient, which is why the meta path exists. Assign the result back over
the input so no other reference keeps the source alive.

### Importing from a checkpoint

<!-- source-signature: src/shadowspill/pytorch/state/model.py:import_model_state_from_file -->
```text
import_model_state_from_file(
    model,
    path,
    *,
    runtime,
    pool,
) -> None
```

<!-- source-signature: src/shadowspill/pytorch/state/optimizer.py:import_optimizer_state_from_file -->
```text
import_optimizer_state_from_file(
    optimizer,
    path,
    *,
    runtime,
    pool,
) -> None
```

`path` is the checkpoint to fill from; `runtime` and `pool` mean what they
mean above. `import_model_state_from_file()` and
`import_optimizer_state_from_file()` rebind the object they were passed rather
than returning a copy, so the caller keeps the object it has, and both return
`None`.

They fill pool state without building the checkpoint in ordinary host memory
first. The file is mapped rather than read, so its pages are reclaimable cache,
and the import happens before the copy, so the values land in pool memory
directly. The checkpoint must name every tensor the target enumerates and agree
with each on dtype and shape; raw bytes cannot be converted, so a disagreement
is refused rather than reinterpreted, and extra names in the file are ignored.
One file per call: a checkpoint sharded across several files is refused. The
optimizer form is keyed by the paths `import_optimizer_state()` enumerates,
which is what `read_optimizer_state()` writes, so a checkpoint saved from one
reads back through the other.

A model need never occupy ordinary host memory on its way into a pool.
Construct it under `torch.device("meta")`, so its parameters have no storage;
assign a mapped checkpoint onto it with
`model.load_state_dict(torch.load(path, mmap=True, weights_only=True),
assign=True)`, so its parameters become file-backed pages rather than
anonymous allocations; then `import_model_state()` copies those into the pool
and releases the source. Only the pool copy is anonymous host memory at any
point. This needs no ShadowSpill-specific call:
`import_model_state_from_file()` is the shorter form when the model is already
built.

### Exporting and releasing

<!-- source-signature: src/shadowspill/pytorch/state/model.py:export_model_state -->
```text
export_model_state(
    model,
    *,
    runtime,
    release_runtime=False,
) -> ModelT
```

<!-- source-signature: src/shadowspill/pytorch/state/optimizer.py:export_optimizer_state -->
```text
export_optimizer_state(
    optimizer,
    *,
    runtime,
    release_runtime=False,
) -> torch.optim.Optimizer
```

<!-- source-signature: src/shadowspill/pytorch/state/model.py:release_model_state -->
```text
release_model_state(
    model,
    *,
    runtime,
) -> None
```

`export_model_state()` and `export_optimizer_state()` copy the authoritative
bytes into ordinary CPU allocations and rebind the same registered tensor
identities to them, returning the object they were given. `runtime` is the
runtime that owns the state; `release_runtime=False`, the default, keeps the
runtime objects for later reuse, and `release_runtime=True` releases them after
the copy.

`release_model_state()` releases those runtime objects without materializing
any CPU copy and returns `None`: the module's registered tensors become invalid
the moment their leases go, so the module must be discarded afterward. It is
the teardown operation for a caller that is done with the state, such as a
qualification host that cannot hold an anonymous model copy beside the full
pinned spill pool. A model the runtime does not own is left unchanged.

### Reading in place

<!-- source-signature: src/shadowspill/pytorch/state/model.py:read_model_state -->
```text
read_model_state(
    model,
    *,
    runtime,
) -> dict[str, torch.Tensor]
```

<!-- source-signature: src/shadowspill/pytorch/state/optimizer.py:read_optimizer_state -->
```text
read_optimizer_state(
    optimizer,
    *,
    runtime,
) -> dict[str, torch.Tensor]
```

`read_model_state()` and `read_optimizer_state()` each return a flat mapping
from the name the state is enumerated under to a host tensor. They answer what
the state currently is without rebinding anything, which is what makes them usable while a plan holds the target --
`export_*` cannot run then, and the runtime refuses it.

The values are ordinary host memory outside the runtime pools, one buffer per
storage root with the target's views laid over it, so entries that shared a
root still share one, and they keep what they held when the call returned.
State crosses a pool's edge by copying, whichever pool holds it: a pool whose
memory is not in this address space cannot be viewed at all, so one path that
always works is worth more than a cheaper one that applies only sometimes.

### Who owns what

Planning takes whichever model it is given. State the caller imported is
adopted and outlives the plan; state that has not been imported is imported
in place by `plan_step()` or `plan_forward()`, which then own it, so closing
the callable releases that state and empties the parameters that viewed it.
Read what you need before the close, or import beforehand to keep it. Only
`build_step_programs()` requires an explicit import, because it returns no
callable that could own the result.

Everything else a plan owns is created in the pools rather than on the host:
gradients, activations and workspaces are runtime objects the plan's actions
move between pools, and the tensors the lowering builds them from are fake, so
they cost nothing while a program is being built. Optimizer state is created in
the spill pool too, in the order a checkpoint is imported: the optimizer
declares what it keeps on meta, which allocates nothing; planning imports those
entries into the spill pool before they hold anything, and writes each one's
start there -- the value the optimizer's own first step gives it, read from how
that step makes the entry (see [the
optimizer](../../architecture/optimizer.md#started-where-the-optimizer-starts-it)).
Where the pool is one this process cannot address, the entries are filled first
and imported after, which costs nothing extra: such a pool keeps a host copy of
its state for as long as it holds it.

The optimizer planning is given is the reference for whose state that is. If
*its* state was already imported, planning adopts it as it stands and it
outlives the plan; if it was not, planning creates it as above and the plan owns
the result. So `import_optimizer_state()` outside planning is always valid and
never changes what planning does, and state imported for an optimizer planning
is never given is invisible to it. `PlannedTrainStep.load_state_dict()` is the
way to resume values into a plan that already owns its state.

## Planning entry points

Building and planning are separate jobs, and the entry points divide along
that line:

| Call | Does | Takes |
|---|---|---|
| `build_step_programs()` | Captures, compiles, profiles and lowers a step, one program per ordering. Runs no search, returns no callable. | build-store arguments only |
| `plan_program()` | Plans a program that has already been built. Captures nothing. Lives in [`shadowspill.planner`](neutral.md), because it needs no frontend. | plan-store arguments only |
| `plan_step()`, `plan_forward()` | Both: build, then plan, then return a live callable. | both sets |
| `plan_step_search()` | Builds every geometry once and plans each under every budget. Executes nothing. | both sets |

Under `plan_program()` sits one layer, on the [framework-neutral
page](neutral.md): the search. Which one runs is
`search_options.algorithm`, and `None` there means the one that ships. A
search takes no store at all.

### Store arguments

Every entry point above reaches one artifact store with two independent trees.
`build/` holds what a run paid for and another run can reuse -- exports,
Inductor caches, graph pairs, optimizer captures, profiles. `planning/` holds
what a run decided -- programs, requests, results, plans. Nothing under
`planning/` is written by a build, and nothing under `build/` by a planning
call, which is what lets one build store serve many runs that each keep their
own plans.

| argument | type | default | meaning |
|---|---|---|---|
| `artifact_store` | path \| `None` | `None` | Roots both trees. `None` uses `~/.cache/shadowspill`. Either way the trees sit under a `v<N>` subdirectory naming the store format version, so one directory survives an update. |
| `build_store` | path \| `None` | `None` | Roots the `build/` tree somewhere of its own, overriding `artifact_store` for it. |
| `plan_store` | path \| `None` | `None` | Roots the `planning/` tree somewhere of its own, overriding `artifact_store` for it. |
| `build_store_mode` | `"contribute"` \| `"reuse"` \| `"require"` \| `"refresh"` | `"contribute"` | What this run does with the build tree. |
| `plan_store_mode` | same four | `"contribute"` | What this run does with the planning tree. |
| `export_bypass_key` | `str` \| `None` | `None` | The caller's name for the code the build is made from (model, objective, optimizer); compiled and profiled artifacts are filed under it, so a lower-level change that leaves the exported graph unchanged is told apart by a new key. |

The four modes are the whole policy, and each tree takes its own:

| mode | Reads hits | Writes misses | On a miss |
|---|---|---|---|
| `contribute` | yes | yes | builds or plans it, and stores it. This is how a store fills up. |
| `reuse` | yes | no | builds or plans it and persists nothing, so a shared store is never changed. |
| `require` | yes | no | refuses, naming what was missing. This is what makes a store a fixed reference: two runs compared against it stood on the same artifacts. |
| `refresh` | no | yes | ignores what is there, rebuilds, and writes over it, replacing a stale entry without discarding the rest of the store. |

The common shape is a build store several runs read and a plan store each run
keeps to itself, so a search reuses another run's captures, profiles and
lowering while still planning every point itself. [The artifact
store](../artifact-store.md) has the layout, what each digest holds, and what
the modes do to PyTorch's own compilation caches.

### Arguments the entry points share

`plan_forward()`, `plan_step()` and `build_step_programs()` take these with
identical meaning. `plan_step_search()` derives most of them per geometry; the
ones it does take are listed in [its own section](#plan_step_search).

| argument | type | default | what it must be |
|---|---|---|---|
| `model` | `nn.Module` | required | The model to plan. Its state is adopted if the caller imported it and imported in place if not; `build_step_programs()` requires the import. |
| `runtime` | `Runtime` | required | Open runtime whose pools and routes are ready. |
| `execution` | `str` | required | Name of the device pool in `runtime.pools`. |
| `spill` | `str` | required | Name of the spill pool in `runtime.pools`. |
| `execution_budget` | `int` \| `None` | `None` | Device bytes the plan may use; the pool's suballocatable capacity when `None`. A value at or below that capacity is taken as given, and the pool's whole `physical_capacity` is accepted as the same thing spelled the way it was configured. A value strictly between the two is ambiguous and refused, as is anything above the physical cap. |
| `share_slab_with` | `PlannedForward` \| `PlannedTrainStep` \| `None` | `None` | `plan_forward()` and `plan_step()` only. An open planned callable whose slab this plan's layout is admitted into, instead of a range of its own; see [sharing a slab](../../architecture/physical-admission.md#sharing-a-slab). `execution_budget` is then at most the slab's size, and is that size when `None`. Close this plan before the one whose slab it shares. |
| `spill_budget` | `int` \| `None` | `None` | Spill bytes the plan may use; the pool's capacity when `None`, and never more than it. |
| `dynamic_scratch_reserve_bytes` | `int` \| `None` | `None` | Device bytes held back for allocations the plan does not own. Measured when `None`; an explicit value can only raise the measured reserve, never lower it, and cannot exceed the execution budget. |
| `execution_device` | `int` \| `str` \| `torch.device` \| `None` | `None` | Accelerator to plan for; PyTorch's current one when `None`. An explicit device must match the execution pool. |
| `partition` | `PartitionSpec` | `"auto"` | `"auto"`, `"whole"`, or a `PartitionPolicy`. Partitioning only creates ordered stage occurrences; it does not choose graph-pair alternatives. See [custom partitioning](../../examples/custom-partitioning.md). |
| `verbose` | `bool` | `True` | Reports each planning phase and unique structural contract as it starts. Diagnostics are retained on the plan report either way. |
| `profiling_metadata` | `Sequence[object]` \| `None` | `None` | One JSON-compatible entry per example microbatch, distinguishing value-sensitive measurements that tensor geometry does not express. It reaches profile and plan identity, and is never passed to the model, objective, or runtime. |
| `allocation_probe_seeds` | `int` | `1` | Independent randomized activation probes per structural contract. |
| `allocation_probe_repetitions` | `int` | `2` | Identical repeats per probe seed, which is what separates a real first-use reservation from noise. |

What any search is told -- the evict-eligibility floor and whether the search
must reproduce exactly at any worker count -- is `search_options.generic`, a
[`GenericPlanningOptions`](neutral.md#searchoptions). These entry points take no
second way to set it.

Budgets of at least one GiB plan at whole-GiB granularity, rounded down, so a
budget that follows a pool's measured capacity gives the same plan identity in
every process; calibrated bandwidths and latencies are rounded the same way, as
[the plan report](../plan-report.md) describes.

### `plan_forward()`

Plans one fixed-shape forward program and returns a `PlannedForward` bound to
the open runtime.

<!-- source-signature: src/shadowspill/pytorch/api.py:plan_forward -->
```text
plan_forward(
    model,
    *,
    example_inputs,
    runtime,
    execution,
    spill,
    execution_budget=None,
    share_slab_with=None,
    spill_budget=None,
    dynamic_scratch_reserve_bytes=None,
    search_options=None,
    execution_device=None,
    partition='auto',
    verbose=True,
    artifact_store=None,
    build_store=None,
    plan_store=None,
    profiling_metadata=None,
    allocation_probe_seeds=1,
    allocation_probe_repetitions=2,
    shared_outputs=(),
    build_store_mode='contribute',
    plan_store_mode='contribute',
    export_bypass_key=None,
    transfer_bandwidths=None,
) -> PlannedForward
```

Beyond the shared and store arguments:

| argument | type | default | what it must be |
|---|---|---|---|
| `example_inputs` | `Sequence[Any]` | required | One fixed example sequence, whose geometry fixes the callable's input signature. A leaf may be wrapped with `shared_input()`. |
| `shared_outputs` | sequence of `SharedOutput` | `()` | Output leaves retained as runtime objects rather than copied out. |
| `transfer_bandwidths` | `TransferBandwidths` \| `None` | `None` | Rates to price every copy at instead of the calibration the runtime measured, as `plan_program()` takes them. A calibration moves from run to run on one machine and the plan is keyed by what it was priced against, so a plan that has to be the one an earlier search chose is planned against the lanes that search planned against. Rates naming no latency keep the calibrated one. |

```text
shared_output(*path, retain_in) -> SharedOutput
```

| argument | type | default | what it must be |
|---|---|---|---|
| `*path` | `str` \| `int` | required | The pytree path to one tensor leaf in the public output: mapping keys and sequence indices in order. |
| `retain_in` | `str` \| `tuple[str, ...]` | required | The pool, or pools, that value is retained in. |

`shared_output()` identifies a tensor leaf in the public output pytree and
retains that value as a runtime object. The corresponding result leaf is a
`TensorRef`, not a copied caller-owned tensor. `TensorRef` records the logical
runtime-object identity (`object`), `generation`, `dtype`, `shape`, `stride`,
`storage_offset`, `requires_grad` and `retained_pools`, without exposing a
backend address.

`TensorRef.close()` releases that public ownership. Closing the planned
callable releases its plan ownership but does not invalidate an outstanding
`TensorRef`; the runtime object is reclaimed after its final owner closes.
One planned shared-output slot holds one generation at a time, so the next
invocation fails clearly until the preceding reference is closed, and then
updates the same logical object in place rather than introducing a second
identity or a value copy.

`SharedInput` is the symmetric input declaration.

```text
shared_input(
    reference,
    *,
    require_in,
    consistency=ObjectConsistency.CAUSAL,
    profiling_value=None,
) -> SharedInput
```

| argument | type | default | what it must be |
|---|---|---|---|
| `reference` | `TensorRef` | required | The open reference the producer handed back. |
| `require_in` | `str` | required | A pool the reference guarantees the value is in. |
| `consistency` | `ObjectConsistency` | `CAUSAL` | Ordering policy for the binding. |
| `profiling_value` | `torch.Tensor` \| `None` | `None` | A CPU tensor standing in for the value while profiling. Required for an integer or Boolean control input, whose value decides what the capture does. |

Wrap a `TensorRef` with it in `example_inputs` when planning the consumer. The
consumer plan binds the same runtime object; it does not create another logical
object or copy the value through caller memory. At invocation time, pass an open
`TensorRef` with the same runtime identity and tensor geometry:

```python
produced = producer(inputs)
state = produced["state"]

consumer = plan_forward(
    consumer_model,
    example_inputs=[shared_input(state, require_in="execution")],
    runtime=runtime,
    execution="execution",
    spill="spill",
)
result = consumer([state])
```

The `ObjectConsistency` enumeration holds the two ordering policies a binding
may take: `ObjectConsistency.CAUSAL`, the default, makes each task acquire the
object's current generation plus its published readiness dependency, while
`ObjectConsistency.UNORDERED` deliberately omits cross-callable value ordering
and retains the object and its lease safely regardless. Floating shared inputs
receive a deterministic task-local profiling representative when
`profiling_value` is omitted.

Several planned callables may remain admitted to one runtime and may bind the
same `TensorRef`.

### `plan_step()`

Plans a fixed accumulated forward/objective/backward/update program and returns
a `PlannedTrainStep`.

<!-- source-signature: src/shadowspill/pytorch/api.py:plan_step -->
```text
plan_step(
    model,
    *,
    objective,
    optimizer,
    hyperparams=(),
    example_inputs,
    runtime,
    execution,
    spill,
    execution_budget=None,
    share_slab_with=None,
    spill_budget=None,
    dynamic_scratch_reserve_bytes=None,
    execution_device=None,
    partition='auto',
    optimizer_ordering='stage_interleaved',
    depth=None,
    breadth=None,
    reverse_breadth=True,
    pair_loss=True,
    search_options=None,
    incumbent=None,
    verbose=True,
    artifact_store=None,
    build_store=None,
    plan_store=None,
    profiling_metadata=None,
    allocation_probe_seeds=1,
    allocation_probe_repetitions=2,
    build_store_mode='contribute',
    plan_store_mode='contribute',
    export_bypass_key=None,
    transfer_bandwidths=None,
    master_dtype=None,
    grad_dtype=None,
) -> PlannedTrainStep
```

Beyond the shared and store arguments:

| argument | type | default | what it must be |
|---|---|---|---|
| `objective` | callable | required | `(model, *microbatch) -> Tensor \| ObjectiveResult`, returning the scalar the step differentiates. |
| `optimizer` | callable | required | Given the model's parameters, returns a `torch.optim.Optimizer`. The class itself does (`torch.optim.AdamW`); so does any partial or lambda over one. |
| `hyperparams` | `Sequence[str]` | `()` | Names of values a step may set later, e.g. `("lr",)` or `("lr", "betas")`. Each must name an entry in a parameter group or a model buffer holding a number, or a sequence of them. Named entries are held in host scalars before capture -- float64 for a float, int64 for an int -- and everything else is left as the optimizer made it. A bool is refused: it selects what the update does, which is what the capture is. |
| `example_inputs` | `Sequence[Sequence[Any]]` | required | One fixed example sequence per microbatch; its length is the step's microbatch count. |
| `optimizer_ordering` | `"stage_interleaved"` \| `"tail"` | `"stage_interleaved"` | Whether each stage updates as its gradients land, or all updates run at the end. |
| `depth` | `int` \| `None` | `None` | Passes over the microbatches. With `breadth`, their product must be `len(example_inputs)`. |
| `breadth` | `int` \| `None` | `None` | Microbatches per pass. Give one of the two and the other follows. |
| `reverse_breadth` | `bool` | `True` | Walks a pass's microbatches in reverse during backward. Vacuous at `breadth=1`. |
| `pair_loss` | `bool` | `True` | Runs each microbatch's last stage forward and backward together. Vacuous at `breadth=1`. |
| `search_options` | `SearchOptions` \| `None` | `None` | What the planner is told about searching: `generic` for what any search understands, `algorithm` for the search itself carrying its own options, and `workers`. `None` runs the search that ships with its defaults. |
| `incumbent` | `AnnotatedProgramPlan` \| `None` | `None` | A plan already in hand for this same program; the search measures it at this budget and never answers with worse. |
| `transfer_bandwidths` | `TransferBandwidths` \| `None` | `None` | As for `plan_forward()`. A step that runs what `plan_step_search()` chose is planned against the report's `planned_lanes`, so it asks the store the search's question and executes the plan the search chose. |
| `master_dtype` | `torch.dtype` \| `None` | `None` | Gives every weight the step trains at another dtype a master copy at this one, and builds `optimizer` over the masters; the update writes each weight from its master. See [the optimizer](../../architecture/optimizer.md#master-copies-and-the-dtype-gradients-are-kept-at). |
| `grad_dtype` | `torch.dtype` \| `None` | `None` | The dtype gradients are created and accumulated at, the weights' own when `None`. The update casts a gradient only where its parameter is at another dtype. |

`depth` and `breadth` say how the step walks those microbatches: `depth`
passes of `breadth` microbatches each, every microbatch of a pass running one
stage before any runs the next, so each stage's parameters are fetched once
per pass rather than once per microbatch. Give neither and the step runs
depth-first, one microbatch after another. `pair_loss` runs each microbatch's
last stage forward and backward together, so the loss's saved state is consumed
as it is produced instead of being held for the pass. The ordering is part of
the plan's identity and is recorded on the report as a `StepDataOrdering`.

`search_options` has three fields: `generic`, what any search understands;
`algorithm`, the search itself carrying its own options, with `None` meaning the
one that ships; and `workers`. `generic` and `algorithm` are part of the plan's
identity in the store, `workers` is not, and the whole of it is recorded on the
report as `search_options`. See
[`SearchOptions`](neutral.md#searchoptions).

`incumbent` is the plan to beat. A step run after a sweep executes the plan the
sweep chose even when this call's calibration or budget differs, because the
answer is held to the incumbent unless something does strictly better.

A value that varies between steps -- a scheduled learning rate, say -- is
passed to the optimizer as a **tensor** rather than a float, and written in
place between steps. A tensor enters the captured update's identity by geometry
alone, so one capture serves every value it takes, while a float enters by
value and would capture again for each one. See [the
optimizer](../../architecture/optimizer.md).

### `build_step_programs()`

Captures, compiles, profiles and lowers a reusable step, and returns one
`StepProgram` per ordering in `orderings`, in that order, from one capture, one
materialization and one profiling; the orderings differ only in the walk the
lowering emits, and `None` builds the depth-first ordering alone. It runs no
search and leaves no active callable: temporary compilation and
materialization state is released before it returns. Because it writes no
plans it takes no plan-store arguments; pass a result to `plan_program()`,
which plans it and does take them, repeatedly and with different budgets and
bandwidths if wanted.

With an `export_bypass_key`, each ordering's program is first looked up in the
build store's step archive under the identity the request has before any
capture -- the key, the model's structure, the inputs' signatures, the
optimizer's type, step code and hyperparameters, the request's own settings,
the machine and the profiling environment -- and only the orderings not found
there are built, from one capture. Without a key every call captures, and the
content-addressed stores serve what they hold as before.

<!-- source-signature: src/shadowspill/pytorch/api.py:build_step_programs -->
```text
build_step_programs(
    model,
    *,
    objective,
    optimizer,
    hyperparams=(),
    example_inputs,
    runtime,
    execution,
    spill,
    execution_budget=None,
    spill_budget=None,
    dynamic_scratch_reserve_bytes=None,
    execution_device=None,
    partition='auto',
    optimizer_ordering='stage_interleaved',
    orderings=None,
    verbose=True,
    artifact_store=None,
    build_store=None,
    profiling_metadata=None,
    allocation_probe_seeds=1,
    allocation_probe_repetitions=2,
    build_store_mode='contribute',
    export_bypass_key=None,
    master_dtype=None,
    grad_dtype=None,
) -> tuple[StepProgram, ...]
```

Every argument means what it means for `plan_step()`, and `orderings` takes
`StepDataOrdering` values, each covering `len(example_inputs)` microbatches.
The model's state must already have been imported, since no callable is
returned that could own it. The budgets and the ordering are recorded in each
program: they describe the machine the step was profiled against and the walk
it was lowered with, and a different walk is a different program.

### `plan_step_search()`

Plans every admitted split of one step's sequence total into microbatches and
accumulation rounds, under every requested budget pair, and executes nothing.
Each geometry pays capture, materialization and profiling once and lowering
once per ordering, through `build_step_programs()`; with an `export_bypass_key`
a geometry whose programs the build store already holds pays only their lookup.
Every geometry-ordering-budget point then asks the planning store for the
summary it keeps beside a certified plan, through `summarize_plan()`, and runs
one search only where the store has no standing answer: a miss, a refusal with
a plan to beat in hand, or a plan to beat that claims to be faster than the
stored answer, which is the store's own rule. Whole plans are read for each
budget's winner, which `winner_plans` hands the run that follows, and for a
plan to beat the moment a later point has to beat it. It returns a
`StepSearchReport`.

<!-- source-signature: src/shadowspill/pytorch/step_search/__init__.py:plan_step_search -->
```text
plan_step_search(
    model,
    *,
    objective,
    optimizer,
    hyperparams=(),
    example_microbatches,
    total_sequences_per_step,
    sequence_length,
    budgets,
    runtime,
    execution,
    spill,
    transfer_bandwidths=None,
    min_tokens_per_microbatch=None,
    max_tokens_per_microbatch=None,
    optimizer_ordering='stage_interleaved',
    orderings=None,
    search_options=None,
    incumbents=True,
    artifact_store=None,
    build_store=None,
    plan_store=None,
    build_store_mode='contribute',
    plan_store_mode='contribute',
    verbose=False,
    progress=None,
    export_bypass_key=None,
    master_dtype=None,
    grad_dtype=None,
) -> StepSearchReport
```

`model`, `objective`, `optimizer`, `hyperparams`,
`runtime`, `execution`, `spill`,
`optimizer_ordering`, `master_dtype`, `grad_dtype` and the store arguments
mean what they mean for `plan_step()`. The rest are:

| argument | type | default | what it must be |
|---|---|---|---|
| `example_microbatches` | `(sequences, accumulation) -> Sequence[Sequence[Any]]` | required | Supplies the example inputs for one geometry. Structure matters; values do not. |
| `total_sequences_per_step` | `int` | required | Sequences one optimizer step consumes. Every divisor pair of it is a candidate geometry. |
| `sequence_length` | `int` | required | Tokens per sequence, which is what makes the token bounds mean the same thing at every length. |
| `budgets` | `Sequence[tuple[int, int]]` | required | The `(execution, spill)` byte pairs every geometry is planned under. At least one. |
| `transfer_bandwidths` | `TransferBandwidths` \| `None` | `None` | Overrides the calibration each step program embeds from the runtime. The report records both, so two searches can be compared or one pinned to another's. |
| `min_tokens_per_microbatch` | `int` \| `None` | `None` | Skips a geometry whose microbatch is smaller, recording the reason. |
| `max_tokens_per_microbatch` | `int` \| `None` | `None` | Skips a geometry whose microbatch is larger, recording the reason. |
| `orderings` | `(accumulation) -> Sequence[StepDataOrdering]` \| `None` | `None` | Which microbatch walks to try for a geometry. `None` tries the default set. |
| `search_options` | `SearchOptions` \| `None` | `None` | As for `plan_step()`; every point is searched under it, worker count included. |
| `incumbents` | `bool` | `True` | Plans each program's budgets ascending and hands every point the best plan found at a smaller budget as the plan to beat, so no program plans worse with more memory. `False` searches every point alone, which is how the two are compared. |
| `verbose` | `bool` | `False` | Forwards each planning call's own phase progress. |
| `progress` | `(str) -> None` \| `None` | `None` | Receives one line per geometry and point boundary, so a caller can tee a live log. |

`StepSearchReport` carries `total_sequences_per_step` and `sequence_length`, the
`budgets` searched, one `StepSearchGeometryBuild` per program built, that is
per ordering of each geometry, carrying the frontend phases that program was
charged -- the shared capture and profiling on a geometry's first ordering, each
lowering on its own -- with the geometry's build wall clock on the first
ordering's entry and zero on the rest, so the entries sum to the build; one
`StepSearchPoint` per geometry-ordering-budget combination, the geometries the token bounds `skipped`
with their reasons, the `search_options` every point was searched under, any
`transfer_bandwidths` override, `planned_lanes` -- the lanes every point was
priced against: the override, else the calibration the first built geometry
planned with, which a caller running a winner hands to `plan_step()` -- and
`winner_plans`, each budget pair's winning `AnnotatedProgramPlan` held in
memory. A point carries its `status`,
`makespan_seconds`, `summary` as a `PlanSummary`, `search_seconds`,
`incumbent_budget_bytes` when it answered with a handed-in plan, and
`graph_pair_selections`: one `GraphPairOutcome` (from `shadowspill.planner`)
per graph-pair selection the search evaluated, not only the one it answered
with.

`orderings` lowers each ordering into its own program, sharing the geometry's
capture and profiles, and plans it under every budget; the report's points and
builds carry the ordering, and the winner at a budget may be any ordering of
any geometry. The default is every `depth x breadth` factor pair of the
accumulation count with the flags at their defaults; the search never toggles
`reverse_breadth` or `pair_loss`.

### `search_geometries()`

The geometry enumeration `plan_step_search()` runs on its own: every divisor
pair of the sequence total, largest microbatch first, with what the token bounds
skipped and why. It is `shadowspill.search`'s, and re-exported here because it is
usually reached alongside `plan_step_search()`; so are `StepSearchReport`,
`StepSearchPoint`, `StepSearchGeometryBuild` and `default_orderings`, all
documented on [the neutral page](neutral.md#shadowspillsearch). Reading a saved
report needs none of this package.

```text
search_geometries(
    total_sequences_per_step,
    *,
    sequence_length,
    min_tokens_per_microbatch=None,
    max_tokens_per_microbatch=None,
) -> tuple[
    tuple[tuple[int, int], ...],
    tuple[tuple[int, int, str], ...],
]
```

| argument | type | default | what it must be |
|---|---|---|---|
| `total_sequences_per_step` | `int` | required | Sequences one optimizer step consumes. |
| `sequence_length` | `int` | required | Tokens per sequence, which is what makes the token bounds mean the same thing at every length. |
| `min_tokens_per_microbatch` | `int` \| `None` | `None` | Skip a geometry whose microbatch is smaller. |
| `max_tokens_per_microbatch` | `int` \| `None` | `None` | Skip a geometry whose microbatch is larger. |

It returns two tuples: the admitted `(sequences_per_microbatch,
accumulation_count)` pairs, and the skipped ones with the reason as a third
element.

Running a winner afterward is one warm `plan_step()` call at the chosen
geometry, taking that budget's `winner_plans` entry as `incumbent` so it
executes what the search chose even when its calibration or facts differ.

Failures are outcomes rather than errors. A point that proves infeasible or
exhausts its search budget carries that status while the search continues.
Because profiling runs a task's real kernels, the largest geometries can
exhaust the device before any plan exists; such a geometry reports every one
of its budgets `infeasible`, with the exhaustion as the point's error, and
contributes no build to the report because it produced no program. Any other
build failure is raised.

## Inputs, objectives, and partitioning

`TensorSpec` is storage-free fixed tensor geometry for planning. It records
shape, dtype, optional stride, `requires_grad`, and layout.

An objective may return a scalar loss tensor or `ObjectiveResult`. A bare
tensor becomes the corresponding `StepResult.objectives` entry and has
`metrics=None`. `ObjectiveResult` explicitly names the differentiable `loss`
and arbitrary nondifferentiated `metrics`; each becomes the corresponding
entry in the two per-round `StepResult` tuples. ShadowSpill validates this
contract during capture rather than inferring a loss from model output names.

| `ObjectiveResult` field | Contract |
|---|---|
| `loss` | One floating-point or complex scalar tensor that participates in backward. |
| `metrics` | Optional nondifferentiated metadata returned to the caller for the same accumulation round. |

`metrics` may be a pytree. Tensor leaves are detached task outputs. Static
leaves must be copyable, and the pytree structure must remain fixed across the
captured workload. Metrics never contribute to backward, are not aggregated
across accumulation rounds, and are unrelated to `PlanReport` or runtime-trace
diagnostics. Applications that do not need auxiliary outputs should return the
loss tensor directly.

`PartitionSpec` accepts `"auto"`, `"whole"`, or a `PartitionPolicy` object.
A custom `PartitionPolicy.assign_stages(graph_module, module)` returns a
complete mapping from executable FX node names to nonnegative contiguous stage
labels. It must not mutate the graph.

## Planned callables

Both callables expose the `plan_report` attribute, `close() -> None`, context
manager support, and a `state_dict()` / `load_state_dict()` pair that takes back
exactly what `state_dict()` produced. `PlannedForward`'s pair is the model's own
CPU state mapping; `PlannedTrainStep`'s is the three-key checkpoint below, which
its `save(path)` writes to a file straight from the pool.
Both also expose `invocation_timings()` and `mark_cycle_end()`, the
invocation's time on the device clock; see [timing](timing.md).

```text
PlannedTrainStep.synchronize() -> None
PlannedForward.synchronize() -> None
```

Returns once the callable's work has finished, its end-of-step writeback
included. A call returns before then, so that whatever the caller does between
calls overlaps the writeback, and the next call waits for it before it stages
anything; wait explicitly only to keep what comes next apart from the step --
timing a step to its end, say, or reading the pool once it has drained.
Raises once the callable is closed.

<!-- source-signature: src/shadowspill/pytorch/callables.py:PlannedForward.__call__ -->
```text
PlannedForward(
    inputs,
    *,
    runtime_trace=False,
    profiler_annotations=False,
) -> object
```

<!-- source-signature: src/shadowspill/pytorch/callables.py:PlannedForward.submit -->
```text
PlannedForward.submit(
    inputs,
    *,
    runtime_trace=False,
    profiler_annotations=False,
) -> InvocationResult[object]
```

<!-- source-signature: src/shadowspill/pytorch/callables.py:PlannedTrainStep.__call__ -->
```text
PlannedTrainStep(
    inputs,
    *,
    hyperparams=None,
    runtime_trace=False,
    profiler_annotations=False,
) -> StepResult
```

<!-- source-signature: src/shadowspill/pytorch/callables.py:PlannedTrainStep.submit -->
```text
PlannedTrainStep.submit(
    inputs,
    *,
    hyperparams=None,
    runtime_trace=False,
    profiler_annotations=False,
) -> InvocationResult[StepResult]
```

| argument | type | default | meaning |
|---|---|---|---|
| `inputs` | `Sequence[Any]`, or `Sequence[Sequence[Any]]` for a step | required | This invocation's values, validated against the fixed template before any input slot is written or task launched. A difference raises `InputGuardError`. |
| `hyperparams` | `Mapping[str, float \| Sequence[float]]` \| `None` | `None` | This step's tunable values. `PlannedTrainStep` only. |
| `runtime_trace` | `bool` | `False` | Records the structured trace of this invocation. `PlannedTrainStep` reaches it through `StepResult.diagnostics`; `PlannedForward` returns the model output and nothing else, so its handle is `PlannedForward.diagnostics`. Resolve one before the next call. |
| `profiler_annotations` | `bool` | `False` | Emits backend profiler ranges for tasks, compiled calls, transfers and allocations. Independent of `runtime_trace`. |

`PlannedForward` returns the model output; `PlannedTrainStep` returns a
`StepResult` holding one detached scalar objective and the reconstructed
objective metrics for each accumulation round, the completed `step_number`,
and an optional `DiagnosticsHandle`. Tensor-valued metrics are detached;
static metric leaves preserve the captured pytree. `DiagnosticsHandle.result()`
and `DiagnosticsHandle.wait()` synchronously resolve the trace once; `resolved`
reports whether that has happened.

### Setting hyperparameters

`hyperparams` sets this step's tunable values: `training(batches,
hyperparams={"lr": rate})`. Names resolve against the optimizer's parameter
groups and the model's named buffers, the two registries of named values that
already exist, so a learning rate and a model temperature are set the same
way. Every optimizer group carrying the name is written, so one schedule
reaches an optimizer with several groups; groups that must differ are written
directly, which is the mechanism underneath. An entry holding several values,
as `betas` does, takes one number for all of them or a sequence in order.

A value can only change if it is held in a tensor, which is what `plan_step`'s
own `hyperparams` argument arranges: it names the values a step may set, and
planning holds exactly those in scalar tensors before the step is captured.
Only the named ones, because an optimizer is entitled to require a number, and
holding one behind its back would change what it computes. The scalars are
float64 and stay on the host, so setting one copies nothing and synchronizes
nothing.

Three things are refused rather than absorbed: an unknown name raises
`KeyError`, because a value going nowhere would look like a schedule that ran;
a name held as a plain number raises `TypeError`, because the capture fixed
that value when it was traced, and the message says to name it in
`plan_step(hyperparams=...)`; and a name that exists in both registries raises
`KeyError` rather than guessing. See
[the optimizer](../../architecture/optimizer.md).

### Submitting without synchronizing

`submit()` performs the normal host dispatch and returns an
`InvocationResult` backed by one cold-created, timing-disabled completion
event. `InvocationResult.result()` and `wait()` synchronize exactly once and
return the public payload; `resolved` reports whether that explicit boundary
has been crossed. Different callables may have outstanding submissions at the
same time. A single callable accepts one outstanding submission because it
reuses one admitted physical layout and task-record set; resolve that result
before submitting the callable again. Callable recurrence and close wait only
for work owned by that plan, never for unrelated callables.

### Checkpoints and closing

Closing copies nothing, and it moves no weights. `import_model_state()` put the
model's parameters in the spill pool -- as the parameters' own storage, where
the pool is one this process can address, and as the authoritative copy behind
them where it is not -- and the pool holds the updated weights throughout: a step both begins and ends with parameters
spill-resident, so each update is already there. Planning points those same
`Parameter` objects at device placeholders for as long as the plan lives; the
last plan over the model to close points them back.
`export_model_state()` is the separate call that copies the values into
ordinary CPU tensors. Closing also takes back whatever the plan's own scopes
left behind, as [above](#what-a-closing-plan-leaves-behind).

A model imported with `import_model_state()` can back several planned
callables at once -- a training step and a forward pass that evaluates it,
say -- planned in either order and closed in either order. They bind the
same runtime objects, so each call sees every update an earlier call made.
State a plan imported for itself, from a model passed to planning without
`import_model_state()`, goes when that plan closes, so it is not shared: the
second plan is refused.

Optimizer state has no equivalent home. `plan_step()` builds the
optimizer from the callable it is given and creates its state in storage the
plan owns, so unless the caller imported that state themselves there is no
caller-owned pool for it to be left in. State the caller imported is untouched
by this, because nothing is created for it.

Releasing the plan therefore releases the state with it: a training callable's
`state_dict()` and `load_state_dict()` answer only while it is open, and both
raise afterwards rather than reporting an empty optimizer. Take the checkpoint
before closing, and resume from one with `load_state_dict()`, which writes the
values into the storage the plan already owns. The checkpoint has to hold every
entry of state the plan keeps -- one from an optimizer that never stepped holds
none -- and one that lacks any is refused before anything changes. Execution failure closes the
same way, and a failed step publishes no optimizer update in any case.

`state_dict()` returns an independent snapshot -- for a training callable, the
three keys `model`, `optimizer` and `step`, which is exactly what
`load_state_dict()` requires back. A weight with a master copy
(`master_dtype`) is written as its master, at the master's dtype, and loading
writes the value to the master and its cast to the weight. Every tensor in it is its own compact host
allocation outside the runtime pools, so it can be serialized while training
continues. The spill pool keeps the authoritative copy throughout and is read
in place, so the snapshot is normally the only copy of the state outside the
pool; an object whose pool copy is not current is read into a buffer first and
costs two until the snapshot is built.

`save(path)` writes that checkpoint to a file without the snapshot: the model's
and the optimizer's state are viewed where they are in the spill pool and
written from there, so saving costs no host copy of the state, however large,
and the callable goes on training on the same state. Resume with
`load_state_dict(torch.load(path, mmap=True))`. A pool this process cannot
address is read out as `state_dict()` reads it.

## Exceptions

Planning failures raise the framework-neutral hierarchy in
`shadowspill.errors`, which [the framework-neutral page](neutral.md#shadowspillerrors)
defines and the [errors, failures, and cleanup guide](../failures.md) puts to
work. The ones a frontend planning call raises are `CaptureError`,
`CompilationError`, `ProfilingError`, `ObjectiveError`, `InputGuardError`,
`PlanInfeasibleError` and `PlanSearchExhaustedError`, all reachable as
`PlanningError` except `InputGuardError`, which is a `ValueError` raised at
invocation rather than at planning. Compiler and profiling errors retain the
structural contract, task kind, and operators when available.

Runtime failures raise `RuntimeConfigurationError` for a configuration a
runtime cannot accept and `RuntimeExecutionError` for a failure during
execution; both retain the first failure the C runtime reported and its task
identity, reached through `Runtime.last_failure` as `RuntimeFailureDiagnostics`.
