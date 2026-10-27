# PyTorch runtime and build policies

The public workflow is documented in the [phase guide](../../architecture/offline-pipeline.md)
and [Trainer/Executor reference](training.md). Executor installs the allocator
and creates pools in a fresh worker. Build, profiling and planning do not require
runtime pools. The low-level runtime remains useful for adapter integration tests.

## Memory pool configuration

For temporary SSD storage and bounded host staging, see [SSD pools](ssd.md).

`shadowspill.memory` holds the values that describe a machine to ShadowSpill.
They are frozen dataclasses; `device()`, `pinned_host()` and
`transfer_route()` are keyword-only constructors for them, and
`MemoryPoolConfig` is the union `DevicePool | SpillPool`.

`DevicePool` is an execution-device pool, built by `device()`:

| argument | type | default | meaning |
|---|---|---|---|
| `physical_capacity` | `int` | required | The process-attributable device memory budget used to size the pool, external headroom included. |
| `device` | `int` | `0` | Accelerator ordinal the pool allocates on. |
| `external_headroom` | `int` | `512 << 20` | Bytes reserved inside `physical_capacity` for external memory. Non-negative and smaller than `physical_capacity`; `0` reserves no allowance. |
| `reject_overbudget` | `bool` | `False` | Reject external or whole-process memory overruns when `True`. Independent of pool sizing; pool bounds remain enforced in both modes. |

`SpillPool` is what every pool that is not the execution pool answers:

| member | type | meaning |
|---|---|---|
| `capacity` | `int` | Bytes the pool takes at construction and holds for its life. |
| `kind` | `int` | The pool-kind value the runtime looks its memory up by. |
| `kind_name` | `str` | What the pool registry reports for it. |
| `addressable` | `bool` | Whether CPU setup code can directly access the pool payload. `False` for remote and SSD pools; their tensor metadata retains no complete host payload. |
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

`SSDPool`, built by `shadowspill.ssd.ssd()`, configures temporary direct-I/O
storage and bounded host staging. See [SSD spill storage](ssd.md) for its
capacity, directory, chunk-size and queue-depth arguments.

`TransferRoute` is one directed relationship between two named pools, built
by `transfer_route()`:

| argument | type | default | meaning |
|---|---|---|---|
| `source` | `str` | required | Name of the pool copied from. Must be a Python identifier. |
| `destination` | `str` | required | Name of the pool copied into. Must differ from `source`. |

Direction is immutable: a route is never handed a copy direction later. The
lane behind it is resolved from the endpoint pools' kinds when the runtime is
constructed.

External memory is process-attributable device memory outside ShadowSpill's
allocation pool, excluding the baseline measured before pool creation. Pool
sizing subtracts `external_headroom` (512 MiB by default) and the initial
baseline from `physical_capacity`. Zero headroom reserves no allowance.

Enforcement is independent: `reject_overbudget=False` (the default) reports
external and whole-process memory overruns without rejecting them. Set it to
`True` to enforce both limits during bootstrap, sealing, and execution. In
both modes the pool keeps its configured size, pool bounds remain enforced,
and actual device allocation failures still fail.

```python
from shadowspill.memory import device

# Default: reserve 512 MiB, report overruns without rejecting them.
pool = device(physical_capacity=5 << 30)
# Same pool sizing, strict enforcement.
strict_pool = device(physical_capacity=5 << 30, reject_overbudget=True)
# Reserve no external allowance, report overruns without rejecting them.
large_pool = device(physical_capacity=5 << 30, external_headroom=0)
```

Persistent allocations *inside* the pool are accounted for separately through
`fixed_execution_bytes`; they do not consume external headroom.

## Runtime

`Runtime` installs the allocator, registers the configured pools and routes,
and calibrates the real directed transfers between their addresses. It is
`shadowspill.runtime.Runtime` with PyTorch supplied as its frontend -- [the
neutral page](neutral.md#runtime) documents the runtime itself -- and takes the
same arguments minus that one. Executor constructs it once in a fresh worker.
Application preparation uses the public build, measurement and planning APIs;
those phases do not construct this runtime.

<!-- source-signature: src/shadowspill/pytorch/runtime.py:Runtime.__init__ -->
```text
Runtime(
    *,
    pools,
    routes,
    library_path=None,
    calibrate=True,
    numa_binding=True,
    worker_poll_nanoseconds=1000,
    background_transfer_window_bytes=DEFAULT_BACKGROUND_WINDOW_BYTES,
    backend=None,
    control_group=None,
    host_headroom_bytes=2 << 30,
    preparation_timeout=1800.0,
)
```

| argument | type | default | meaning |
|---|---|---|---|
| `pools` | `Mapping[str, MemoryPoolConfig]` | required | Pool name to configuration. Exactly one `DevicePool`, and at least one `SpillPool` beside it. |
| `routes` | `Mapping[str, TransferRoute]` | required | Route name to directed pool pair. A plan can only move bytes along a route registered here. |
| `library_path` | `str` \| `Path` \| `None` | `None` | The PyTorch adapter library to load; `None` resolves the one installed beside the package. |
| `calibrate` | `bool` | `True` | Measure every registered route at construction. With `False`, no route probes run; saved-Plan execution uses its supplied CommunicationInfo. |
| `numa_binding` | `bool` | `True` | Discover device-local host placement during initialization; fallback warns. Set `False` to preserve caller placement. |
| `worker_poll_nanoseconds` | `int` | `1_000` | How long the C transfer worker waits between polls. |
| `background_transfer_window_bytes` | `int` | `64 << 20` | How far a lane may run ahead with transfers the plan did not schedule. |
| `backend` | `str` \| `None` | `None` | Which backend shared object the adapter loads: `None` the one accelerator backend installed beside the libraries, a name resolves to `libshadowspill_backend_<name>.so` there, and a path is used as given. |

`background_transfer_window_bytes` bounds unscheduled work such as model materialization
or state reconciliation: a transfer the plan did schedule never
waits behind more than this many background bytes. Zero removes the bound. See
[transfers](../../architecture/transfers.md#dispatch).

Pool and route names are user-defined identities. Runtime admission binds
`execution` and `spill` pool names and resolves the matching directed
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

Runtime owns this measurement for every lane, including SSD. It initializes
non-addressable probe sources, measures solo and concurrent directions, and
publishes fixed effective rates for planning. Smaller probes reduce setup I/O
but may change the estimate; they do not change lane chunk sizes. See
[calibration](../../architecture/transfers.md#calibration).

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

## Inputs, objectives, and partitioning

`TensorSpec` is storage-free fixed tensor geometry for planning. It records
shape, dtype, optional stride, `requires_grad`, and layout.

An objective may return a scalar loss tensor, `(loss, metrics)`, or `ObjectiveResult`. A bare
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
across accumulation rounds, and are unrelated to planning reports or runtime-trace
diagnostics. Applications that do not need auxiliary outputs should return the
loss tensor directly.

## PartitionPolicy

`PartitionSpec` accepts `"auto"`, `"whole"`, or an object implementing the
public `shadowspill.pytorch.PartitionPolicy` protocol:

```python
class MyPartition(PartitionPolicy):
    def assign_stages(self, graph_module, module) -> Mapping[str, int]:
        ...
```

Return every executable FX node name exactly once, excluding `placeholder`,
`get_attr` and `output`. Labels must be nonnegative integers, not booleans;
labels need not be consecutive, but each occupies one contiguous graph interval.
The callback must not modify the graph/model/metadata or execute model kernels.
Use deterministic structural rules and a stable configuration representation.
Subclassing the protocol is optional: implementing the method is sufficient.

Pass the same policy as `partition=policy` to `build_forward`, `build_forward_backward`,
`build_step`. A quickstart factory supplies it
in `build_options={"partition": policy}`. Direct API calls do not infer policy from a model name.

The policy defines forward-stage boundaries. ShadowSpill derives object inputs,
outputs, explicit mutations and aliases, then constructs backward alternatives
and profiles workspace. It does not let the policy split an opaque custom op,
choose residency, supply byte counts, or select save/recompute behavior.
Malformed mappings fail with `CaptureError` before compilation; later capture
may also reject unsupported differentiation or mutation boundaries.

See the [architecture contract](../../architecture/partitioning.md),
[standalone example](../../examples/custom-partitioning.md), and
[GLM example](../../../workloads/models/glm53_flash/README.md#partitioning).


## Invocation handle

`InvocationResult` is the handle returned by Executor. `result()` / `wait()`
waits once and returns caller-owned outputs, preserving completion errors for
later reads. `resolved` reports completion and `diagnostics` exposes the optional
runtime trace. See [execution](training.md).

## Build and measurement policies

`PartitionPolicy` supplies custom stage boundaries to the public builders;
`PartitionSpec` describes the resulting stages. `ProfilingOptions` controls
warmup and timing in `measure_program`. These options do not create runtime
pools. See [the build phase](../../architecture/offline-pipeline.md).
