# Frontend and lifecycle API

The symbols on this page are exported by `shadowspill.memory` or
`shadowspill.pytorch`: the pool and route configuration a runtime is built
from, the runtime itself, the state-import operations, the planning entry
points, and the callables planning returns.

## Memory pool configuration

`shadowspill.memory` holds the values that describe a machine to ShadowSpill.
They are frozen dataclasses; `device()`, `pinned_host()` and
`transfer_route()` are keyword-only constructors for them, and
`MemoryPoolConfig` is the union `DevicePool | PinnedHostPool`.

`DevicePool` is an execution-device pool, built by `device()`:

| argument | type | default | meaning |
|---|---|---|---|
| `physical_capacity` | `int` | required | The whole process-attributable device memory cap, provider headroom included. |
| `device` | `int` | `0` | Accelerator ordinal the pool allocates on. |
| `provider_headroom` | `int` | `1280 << 20` | Bytes inside `physical_capacity` left to the provider's own allocations. Non-negative and smaller than `physical_capacity`. |

`PinnedHostPool` is a registered pinned-host spill pool, built by
`pinned_host()`:

| argument | type | default | meaning |
|---|---|---|---|
| `capacity` | `int` | required | Pinned host bytes the pool registers. |

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
and calibrates the real directed transfers between their addresses. Construct
it once, before any workload state exists: model and optimizer state are then
created and imported into an initialized runtime, and planning reads the
published `transfer_capabilities` snapshot rather than recalibrating.

<!-- source-signature: src/shadowspill/pytorch/runtime_adapter/runtime.py:Runtime.__init__ -->
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
| `pools` | `Mapping[str, MemoryPoolConfig]` | required | Pool name to configuration. The PyTorch backend takes one `DevicePool` and any number of `PinnedHostPool` entries. |
| `routes` | `Mapping[str, TransferRoute]` | required | Route name to directed pool pair. A plan can only move bytes along a route registered here. |
| `library_path` | `str` \| `Path` \| `None` | `None` | The compiled PyTorch adapter to load; `None` resolves the one installed beside the package. |
| `calibrate` | `bool` | `True` | Measure every registered route at construction. With `False`, planning refuses a route that was never calibrated until `calibrate_transfer_capabilities()` has run. |
| `worker_poll_nanoseconds` | `int` | `1_000` | How long the native transfer worker waits between polls. |
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

Measures all routes, or the `(source, destination)` pairs `routes` names, and
atomically publishes the new matrix, which it also returns. The two copy sizes
separate latency from bandwidth; `warmup_copies` are discarded and
`measured_copies` are kept. This runtime must be locally idle, but ShadowSpill
performs no cross-process barrier: callers may coordinate several processes and
calibrate concurrently to measure contended links.

`Runtime.close()` verifies that no planning, callable, persistent imported
state, public object reference, or caller-owned device output remains, then
tears the runtime down as [memory
runtime](../../architecture/memory-runtime.md#failure-and-teardown) describes.
Release or copy ordinary device outputs before this call. PyTorch's
process-global allocator shim cannot be uninstalled, so it remains in a
permanently closed state and rejects later device allocations.

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
copy: every tensor is allocated in the pool, `reset_parameters()` writes the
values there, and the same module is returned, rebound. No host memory
proportional to the model is ever allocated. The model must satisfy the
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

`export_*` copies the authoritative bytes into ordinary CPU allocations and
rebinds the same registered tensor identities to them, returning the object it
was given. `release_runtime=False`, the default, keeps the runtime objects for
later reuse; `release_runtime=True` releases them after the copy.

`release_model_state()` releases those runtime objects without materializing
any CPU copy and returns `None`: the module's registered tensors become invalid
the moment their leases go, so the module must be discarded afterward. It is
the teardown operation for callers that no longer need the state, such as
qualification hosts that cannot hold an anonymous model copy beside the full
pinned spill arena. A model the runtime does not own is left unchanged.

### Reading in place

<!-- source-signature: src/shadowspill/pytorch/state/model.py:read_model_state -->
```text
read_model_state(
    model,
    *,
    runtime,
    copy=True,
) -> dict[str, torch.Tensor]
```

<!-- source-signature: src/shadowspill/pytorch/state/optimizer.py:read_optimizer_state -->
```text
read_optimizer_state(
    optimizer,
    *,
    runtime,
    copy=True,
) -> dict[str, torch.Tensor]
```

`read_model_state()` and `read_optimizer_state()` each return a flat mapping
from the name the state is enumerated under to a host tensor. They answer what
the state currently is without rebinding anything, which is what makes them usable while a plan holds the target --
`export_*` cannot run then, and the runtime refuses it.

`copy=True`, the default, gives ordinary host memory outside the runtime
pools, one buffer per storage root with the target's views laid over it, so
entries that shared a root still share one, and the values keep what they held
when the call returned. `copy=False` allocates nothing and views the pool's
own bytes instead: ordinary torch operations work on them, but treat them as
read-only, because writing through one changes runtime state behind the
runtime's back, and they stop being current the next time the plan runs. A
storage root whose pool copy is not the authoritative one is copied either
way.

### Who owns what

Planning takes whichever model it is given. State the caller imported is
adopted and outlives the plan; state that has not been imported is imported
in place by `plan_step()` or `plan_forward()`, which then own it, so closing
the callable releases that state and empties the parameters that viewed it.
Read what you need before the close, or import beforehand to keep it. Only
`build_step_program()` requires an explicit import, because it returns no
callable that could own the result.

There are three ways model state comes to live in a pool, and none of them
needs the host to hold it twice. `import_model_state()` takes a model the
caller built and releases the host copy as it goes.
`import_model_state_from_file()` fills the pool from a checkpoint by mapping
the file, so the only host memory involved is reclaimable page cache. A
planning call that returns a callable imports state the caller did not, and
that state belongs to the callable and is released with it. What the host
still holds in the first case is the model the caller constructed there, which
is the caller's own object and outside this boundary.

Everything else a plan owns is created in the pools rather than on the host:
gradients, activations and workspaces are runtime objects the plan's actions
move between pools, and the tensors the lowering builds them from are fake, so
they cost nothing while a program is being built. Optimizer state is created
there too: the optimizer declares what it keeps on meta, which allocates
nothing, planning allocates that in the spill pool, and `optimizer_state_init`
fills it in place.

`import_optimizer_state()` and `export_optimizer_state()` apply the same
storage policy to already materialized optimizer state, as a standalone
ownership operation, and planning is agnostic of it. What planning looks at is
the optimizer it is given, and that object is the reference: if *its* state was
already imported, planning adopts it as it stands and it outlives the plan; if
it was not, planning declares the state on meta, allocates it in the spill
pool, and fills it through `optimizer_state_init`, and the plan owns the
result. So importing optimizer state outside planning is always valid and
never changes what planning does. State imported for an optimizer planning is
never given is invisible to it. `PlannedTrainStep.load_state_dict()` remains
the way to resume values into a plan that already owns its state.

## Planning entry points

Building and planning are separate jobs, and the entry points divide along
that line:

| Call | Does | Takes |
|---|---|---|
| `build_step_program()` | Captures, compiles, profiles and lowers a step. Runs no search, returns no callable. | build-store arguments only |
| `plan_program()` | Plans a program that has already been built. Captures nothing. Lives in [`shadowspill.planner`](neutral.md), because it needs no frontend. | plan-store arguments only |
| `plan_step()`, `plan_forward()` | Both: build, then plan, then return a live callable. | both sets |
| `plan_step_search()` | Builds every geometry once and plans each under every budget. Executes nothing. | both sets |

Under `plan_program()` sits one layer, on the [framework-neutral
page](neutral.md): the search. `plan_program()` chooses which one runs
through its `search` argument, and `pressurefit` is the one that ships --
the name of that search algorithm and nothing else. A search takes no store
at all.

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
| `implementation_revision` | `str` \| `None` | `None` | Names the operation implementations the artifacts were produced against, so compiled and profiled entries -- and the plans measured on them -- are not reused across a kernel change that leaves the exported graph identical. |

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

`plan_forward()`, `plan_step()` and `build_step_program()` take these with
identical meaning; `plan_step_search()` takes the ones its per-geometry
planning does not derive.

| argument | type | default | what it must be |
|---|---|---|---|
| `model` | `nn.Module` | required | The model to plan. Its state is adopted if the caller imported it and imported in place if not; `build_step_program()` requires the import. |
| `runtime` | `Runtime` | required | Open runtime whose pools and routes are ready. |
| `execution` | `str` | required | Name of the device pool in `runtime.pools`. |
| `spill` | `str` | required | Name of the spill pool in `runtime.pools`. |
| `execution_budget` | `int` \| `None` | `None` | Device bytes the plan may use; the pool's suballocatable capacity when `None`. A value at or below that capacity is taken as given, and the pool's whole `physical_capacity` is accepted as the same thing spelled the way it was configured. A value strictly between the two is ambiguous and refused, as is anything above the physical cap. |
| `spill_budget` | `int` \| `None` | `None` | Spill bytes the plan may use; the pool's capacity when `None`, and never more than it. |
| `dynamic_scratch_reserve_bytes` | `int` \| `None` | `None` | Device bytes held back for allocations the plan does not own. Measured when `None`; an explicit value can only raise the measured reserve, never lower it, and cannot exceed the execution budget. |
| `minimum_object_bytes_evict_eligible` | `int` | `1 << 20` | Objects smaller than this stay resident from their first to their last access instead of being evicted and fetched mid-step; their opening fetch, release after the last access, and terminal writeback are unchanged. Default 1 MiB, the size under which a copy is latency-bound; zero makes every object eligible. Part of the plan identity. Not taken by `build_step_program()`, which runs no search. |
| `deterministic` | `bool` | `False` | Reproduces the search exactly at any worker count: a candidate's placement gate consults only its own placed plans rather than the shared best-placed record. It costs wall time, and it is part of the plan's identity, so a plan searched under it is a separate store entry. Not taken by `build_step_program()`. |
| `execution_device` | `int` \| `str` \| `torch.device` \| `None` | `None` | Accelerator to plan for; PyTorch's current one when `None`. An explicit device must match the execution pool. |
| `partition` | `PartitionSpec` | `"auto"` | `"auto"`, `"whole"`, or a `PartitionPolicy`. Partitioning only creates ordered stage occurrences; it does not choose graph-pair alternatives. See [custom partitioning](../../examples/custom-partitioning.md). |
| `verbose` | `bool` | `True` | Reports each planning phase and unique structural contract as it starts. Diagnostics are retained on the plan report either way. |
| `profiling_metadata` | `Sequence[object]` \| `None` | `None` | One JSON-compatible entry per example microbatch, distinguishing value-sensitive measurements that tensor geometry does not express. It reaches profile and plan identity, and is never passed to the model, objective, or runtime. |
| `allocation_probe_seeds` | `int` | `1` | Independent randomized activation probes per structural contract. |
| `allocation_probe_repetitions` | `int` | `2` | Identical repeats per probe seed, which is what separates a real first-use reservation from noise. |

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
    spill_budget=None,
    dynamic_scratch_reserve_bytes=None,
    minimum_object_bytes_evict_eligible=1 << 20,
    deterministic=False,
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
    implementation_revision=None,
) -> PlannedForward
```

Beyond the shared and store arguments:

| argument | type | default | what it must be |
|---|---|---|---|
| `example_inputs` | `Sequence[Any]` | required | One fixed example sequence, whose geometry fixes the callable's input signature. A leaf may be wrapped with `shared_input()`. |
| `shared_outputs` | sequence of `SharedOutput` | `()` | Output leaves retained as runtime objects rather than copied out. |

`shared_output(*path, retain_in=pool_name)` identifies a tensor leaf in the
public output pytree and retains that value as a runtime object in the named
pool. The corresponding result leaf is a `TensorRef`, not a copied
caller-owned tensor. `TensorRef` records the logical runtime-object identity,
residency generation, dtype, shape, stride, and storage offset without
exposing a backend address.

`TensorRef.close()` releases that public ownership. Closing the planned
callable releases its plan ownership but does not invalidate an outstanding
`TensorRef`; the runtime object is reclaimed after its final owner closes.
One planned shared-output slot holds one generation at a time, so the next
invocation fails clearly until the preceding reference is closed. Once
closed, the next invocation updates the same logical object record in place.
Its current physical lease and residency generation may be replaced; no new
public object identity or value copy is introduced.

`SharedInput` is the symmetric input declaration. Wrap a `TensorRef` with
`shared_input(reference, require_in=pool_name)` in `example_inputs` when
planning the consumer. The consumer plan binds the same runtime object; it
does not create another logical object or copy the value through caller
memory. At invocation time, pass an open `TensorRef` with the same runtime
identity and tensor geometry:

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

`require_in` must name a pool guaranteed by the reference. The
`ObjectConsistency` enumeration holds the two ordering policies a binding may
take: `ObjectConsistency.CAUSAL`, the default, makes each task acquire the
object's current generation plus its published readiness dependency, while
`ObjectConsistency.UNORDERED` deliberately omits cross-callable value ordering
and retains the object and its lease safely regardless. Floating shared inputs
receive a deterministic task-local profiling representative; integer and
Boolean control inputs require an explicit CPU `profiling_value` on
`SharedInput`.

Several planned callables may remain admitted to one runtime and may bind the
same `TensorRef`, and distinct callables may be submitted without synchronizing
the dispatcher between them.

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
    optimizer_state_init=None,
    hyperparams=(),
    example_inputs,
    runtime,
    execution,
    spill,
    execution_budget=None,
    spill_budget=None,
    dynamic_scratch_reserve_bytes=None,
    minimum_object_bytes_evict_eligible=1 << 20,
    deterministic=False,
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
    implementation_revision=None,
) -> PlannedTrainStep
```

Beyond the shared and store arguments:

| argument | type | default | what it must be |
|---|---|---|---|
| `objective` | callable | required | `(model, *microbatch) -> Tensor \| ObjectiveResult`, returning the scalar the step differentiates. |
| `optimizer` | callable | required | Given the model's parameters, returns a `torch.optim.Optimizer`. The class itself does (`torch.optim.AdamW`); so does any partial or lambda over one. |
| `optimizer_state_init` | `(name, tensor, parameter) -> None` \| `None` | `None` | Fills one declared state entry in place, given the entry's name, the pool-backed tensor, and the parameter it belongs to. Required unless the optimizer handed back already holds imported state, because a default would be an assumption that fails silently. |
| `hyperparams` | `Sequence[str]` | `()` | Names of values a step may set later, e.g. `("lr",)` or `("lr", "betas")`. Each must name an entry in a parameter group or a model buffer holding a number, or a sequence of them. Named entries are held in host scalars before capture -- float64 for a float, int64 for an int -- and everything else is left as the optimizer made it. A bool is refused: it selects what the update does, which is what the capture is. |
| `example_inputs` | `Sequence[Sequence[Any]]` | required | One fixed example sequence per microbatch; its length is the step's microbatch count. |
| `optimizer_ordering` | `"stage_interleaved"` \| `"tail"` | `"stage_interleaved"` | Whether each stage updates as its gradients land, or all updates run at the end. |
| `depth` | `int` \| `None` | `None` | Passes over the microbatches. With `breadth`, their product must be `len(example_inputs)`. |
| `breadth` | `int` \| `None` | `None` | Microbatches per pass. Give one of the two and the other follows. |
| `reverse_breadth` | `bool` | `True` | Walks a pass's microbatches in reverse during backward. Vacuous at `breadth=1`. |
| `pair_loss` | `bool` | `True` | Runs each microbatch's last stage forward and backward together. Vacuous at `breadth=1`. |
| `search_options` | `SearchOptions` \| `None` | `None` | What the planner is told about searching: `generic` for what any search understands, `algorithm` for the search itself carrying its own options, and `workers`. `None` runs the search that ships with its defaults. |
| `incumbent` | `AnnotatedProgramPlan` \| `None` | `None` | A plan already in hand for this same program; the search measures it at this budget and never answers with worse. |

`depth` and `breadth` say how the step walks those microbatches: `depth`
passes of `breadth` microbatches each, every microbatch of a pass running one
stage before any runs the next, so each stage's parameters are fetched once
per pass rather than once per microbatch. Give neither and the step runs
depth-first, one microbatch after another. `pair_loss` runs each microbatch's
last stage forward and backward together, so the loss's saved state is consumed
as it is produced instead of being held for the pass. The ordering is part of
the plan's identity and is recorded on the report as a `StepDataOrdering`.

`search_options` carries the algorithm and what it may try. For PressureFit that is a
`PressureFitOptions`, whose `resolution_options` name the resolutions it
plans: the shares of flexible groups to recompute, one resolved program each,
as exact fractions such as `("0", "1/2", "1")`; see
[`shadowspill.planner`](neutral.md). The whole record is part of the plan's
identity in the store and is recorded on the report as `search_options`.

`incumbent` is the plan to beat. A step run after a sweep executes the plan the
sweep chose even when this call's calibration or budget differs, because the
answer is held to the incumbent unless something does strictly better.

A value that varies between steps -- a scheduled learning rate, say -- is
passed to the optimizer as a **tensor** rather than a float, and written in
place between steps. A tensor enters the captured update's identity by geometry
alone, so one capture serves every value it takes, while a float enters by
value and would capture again for each one. See [the
optimizer](../../architecture/optimizer.md).

### `build_step_program()`

Captures, compiles, profiles and lowers a reusable step, and returns a
`StepProgram`. It runs no search and leaves no active callable: temporary
compilation and materialization state is released before it returns. Because it
writes no plans it takes no plan-store arguments; pass its result to
`plan_program()`, which plans it and does take them, repeatedly and with
different budgets and bandwidths if wanted.

<!-- source-signature: src/shadowspill/pytorch/api.py:build_step_program -->
```text
build_step_program(
    model,
    *,
    objective,
    optimizer,
    optimizer_state_init=None,
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
    depth=None,
    breadth=None,
    reverse_breadth=True,
    pair_loss=True,
    verbose=True,
    artifact_store=None,
    build_store=None,
    profiling_metadata=None,
    allocation_probe_seeds=1,
    allocation_probe_repetitions=2,
    build_store_mode='contribute',
    implementation_revision=None,
) -> StepProgram
```

Every argument means what it means for `plan_step()`. The model's state must
already have been imported, since no callable is returned that could own it.
The budgets and the ordering are recorded in the program: they describe the
machine the step was profiled against and the walk it was lowered with, and a
different walk is a different program.

### `plan_step_search()`

Plans every admitted split of one step's sequence total into microbatches and
accumulation rounds, under every requested budget pair, and executes nothing.
Each distinct geometry pays capture, profiling and lowering once -- the build
tree deduplicates by structural digest -- and every geometry-budget point then
runs the PressureFit search. It returns a `StepSearchReport`.

<!-- source-signature: src/shadowspill/pytorch/step_search.py:plan_step_search -->
```text
plan_step_search(
    model,
    *,
    objective,
    optimizer,
    optimizer_state_init=None,
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
    implementation_revision=None,
) -> StepSearchReport
```

`model`, `objective`, `optimizer`, `optimizer_state_init`, `hyperparams`,
`runtime`, `execution`, `spill`,
`optimizer_ordering` and the store arguments mean what they mean for
`plan_step()`. The rest are:

| argument | type | default | what it must be |
|---|---|---|---|
| `example_microbatches` | `(sequences, accumulation) -> Sequence[Sequence[Any]]` | required | Supplies the example inputs for one geometry. Structure matters; values do not. |
| `total_sequences_per_step` | `int` | required | Sequences one optimizer step consumes. Every divisor pair of it is a candidate geometry. |
| `sequence_length` | `int` | required | Tokens per sequence, which is what makes the token bounds mean the same thing at every length. |
| `budgets` | `Sequence[tuple[int, int]]` | required | The `(execution, spill)` byte pairs every geometry is planned under. At least one. |
| `transfer_bandwidths` | `TransferBandwidths` \| `None` | `None` | Overrides the calibration each step program embeds from the runtime. The report records both, so two searches can be compared or one pinned to another's. |
| `min_tokens_per_microbatch` | `int` \| `None` | `None` | Skips a geometry whose microbatch is smaller, recording the reason. |
| `max_tokens_per_microbatch` | `int` \| `None` | `None` | Skips a geometry whose microbatch is larger, recording the reason. |
| `options` | `SearchOptions` \| `None` | `None` | What every search is told, unchanged at every point: determinism and the evict-eligibility floor. |
| `workers` | `int` | `0` | How much of the machine each point's search may use; zero for every logical CPU. No part of the plan key. |
| `orderings` | `(accumulation) -> Sequence[StepDataOrdering]` \| `None` | `None` | Which microbatch walks to try for a geometry. `None` is `default_orderings()`. |
| `search_options` | `SearchOptions` \| `None` | `None` | As for `plan_step()`; every point is searched under it. Invalid options are rejected before any geometry is built. |
| `incumbents` | `bool` | `True` | Plans each program's budgets ascending and hands every point the best plan found at a smaller budget as the plan to beat, so no program plans worse with more memory. `False` searches every point alone, which is how the two are compared. |
| `verbose` | `bool` | `False` | Forwards each planning call's own phase progress. |
| `progress` | `(str) -> None` \| `None` | `None` | Receives one line per geometry and point boundary, so a caller can tee a live log. |

`StepSearchReport` carries the budgets and geometries searched, one
`StepSearchGeometryBuild` per built geometry with its build wall clock broken
down by frontend phase, one `StepSearchPoint` per geometry-ordering-budget
combination, the geometries the token bounds skipped with their reasons, the
resolution options and any bandwidth override the search used, and
`winner_plans`, each budget pair's winning `AnnotatedProgramPlan` held in
memory. A point carries its status, simulated makespan, `PlanSummary`,
`incumbent_budget_bytes` when it answered with a handed-in plan, and
`graph_pair_selections`: one `GraphPairOutcome` per graph-pair selection the
search evaluated, not only the one it answered with. `search_geometries()` is
the underlying enumeration -- every divisor pair of the sequence total, largest
microbatch first, with the bounds' skips and reasons -- and returns the
admitted pairs and the skipped ones.

`orderings` lowers each ordering into its own program, sharing the geometry's
capture and profiles, and plans it under every budget; the report's points and
builds carry the ordering, and the winner at a budget may be any ordering of
any geometry. The default, `default_orderings()`, is every `depth x breadth`
factor pair of the accumulation count with the flags at their defaults; the
search never toggles `reverse_breadth` or `pair_loss`.

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

Both callables expose `plan_report`, `state_dict()`, `load_state_dict()`,
`close()`, and context manager support. `PlannedTrainStep` also exposes
`invocation_timings()` and `mark_cycle_end()`, the step's time on the device
clock; see [timing](timing.md).

<!-- source-signature: src/shadowspill/pytorch/callables.py:PlannedForward.__call__ -->
```text
PlannedForward(inputs, *, profiler_annotations=False) -> object
```

<!-- source-signature: src/shadowspill/pytorch/callables.py:PlannedForward.submit -->
```text
PlannedForward.submit(
    inputs,
    *,
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
| `runtime_trace` | `bool` | `False` | Records the structured step trace, reached through `StepResult.diagnostics`. `PlannedTrainStep` only. |
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

Closing copies nothing, and it moves no weights. `import_model_state()` gave
the model's parameters storage in the spill pool, and that one storage holds
the updated weights throughout: a step both begins and ends with parameters
spill-resident, so each update is already there. Running a step points those
same `Parameter` objects at device memory; closing points them back.
`export_model_state()` is the separate call that copies the values into
ordinary CPU tensors.

Optimizer state has no equivalent home today. `plan_step()` builds the
optimizer from the callable it is given and creates its state in storage the
plan owns, and planning refuses an optimizer whose state the caller already
imported, so there is no caller-owned pool for it to be left in. That state is
taken from the spill pool as it is created rather than built on the host and
copied in: while the optimizer initializes, a host allocation large enough to
be worth an object is served from the pool, so the values are written where
they will live. On a model with a state several times its own size that is the
difference between a build that needs the pool and one that needs the pool
again beside it. State the caller imported is untouched by this, because
nothing is created for it.

Releasing the plan therefore releases the state with it: a training callable's
`state_dict()` and `load_state_dict()` answer only while it is open, and both
raise afterwards rather than reporting an empty optimizer. Take the checkpoint
before closing, and resume from one with `load_state_dict()`, which writes the
values into the storage the plan already owns. Execution failure closes the
same way, and a failed step publishes no optimizer update in any case.

`state_dict()` returns an independent snapshot -- for a training callable, the
three keys `model`, `optimizer` and `step`, which is exactly what
`load_state_dict()` requires back. Every tensor in it is its own compact host
allocation outside the runtime pools, so it can be serialized while training
continues. The spill pool keeps the authoritative copy throughout and is read
in place, so the snapshot is normally the only copy of the state outside the
pool; an object whose pool copy is not current is read into a buffer first and
costs two until the snapshot is built. Even one copy of optimizer state is, on
a large model, the largest transient the frontend asks for.

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
execution; both retain the first native failure and its task identity, reported
through `Runtime.last_failure` as `RuntimeFailureDiagnostics`.
