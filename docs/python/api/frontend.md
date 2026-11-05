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
| `capacity` | `int` | required | Bytes the daemon is asked for. Declared, not discovered: runtime construction fails if the daemon cannot serve them, so a plan stays reproducible from its configuration. |
| `host` | `str` | required | Where the daemon is. |
| `port` | `int` | required | Its TCP port. |
| `selector` | `str` | `"host"` | Which memory the daemon should serve, in the daemon's own vocabulary. |

It needs `libshadowspill_network.so`, which a build without it does not
produce; without that library, a runtime or calibration that opens the pool
raises.

`SSDPool`, built by `shadowspill.ssd.ssd()`, configures direct-I/O storage,
temporary unless it is opened on a pool image, and bounded host staging. See
[SSD spill storage](ssd.md) for its capacity, directory, staging, chunk-size,
queue-depth and image arguments.

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
route registries, transfer calibration, pending and admitted plan handles, and
registered runtime objects. Its public properties are `frontend` (the PyTorch
`RuntimeFrontend` it was opened with), `pools` (name to `MemoryPool`), `routes`
(name to `RuntimeRoute`) and `transfer_capabilities` (a `TransferCapabilities`
matrix of `TransferProfile` entries).

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

`Runtime.close()` verifies that no pending plan setup, admitted plan, public
object reference, or caller-owned device output remains, then
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
a task of the program, and `no scope` for an unscoped allocation. Low-level
allocator probes and runtime objects also have named synthetic scopes
(`profiling scope`, `runtime object`, `initial actions`). These support native
allocator tests and direct runtime callers; ordinary `measure_program` does
not use the execution allocator.

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

The runtime's frontend says what a range is in PyTorch's terms.
`runtime.frontend.occupants(allocations, locate)` maps each range to the
framework objects whose storage lies inside it, where `locate(address)` names
the allocation that owns an address, as `allocation_id_for_address(runtime,
address)` in `shadowspill.runtime.occupancy` does. A range with no
occupant is held by something the framework does not own, and that is itself the
answer: there is no reference for a caller to drop.
`runtime.frontend.retainers(held)` then names where each of those objects is
*referenced from*, since a reference can only be dropped where it is held.
Pass `ignore` the containers the question itself built, so the answer names
holders rather than the query. Both compare addresses and keep no reference of
their own, which would extend the lifetime of what is being investigated;
`retainers` returns descriptions rather than references.

## Inputs, objectives, and partitioning

`TensorSpec` is storage-free fixed tensor geometry for planning. It records
shape, dtype, optional stride, `requires_grad`, and layout.

An objective may return a scalar loss tensor, `(loss, metrics)`, or `ObjectiveResult`. A bare
tensor becomes the corresponding `StepResult.losses` entry and has
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

## Graph-pair policies

Pass `graph_pair_policy=policy` to `build_forward_backward` or `build_step` to
control each checkpoint stage's saved-state cuts independently. `None` means
`GraphPairPolicy()`. Import `GraphPairPolicy`, `RetentionPolicy`, `RetentionRule`
and `OperatorClass` from `shadowspill.pytorch`. Forward-only builds do not take
this argument because they have no backward saved state.

`GraphPairPolicy(save=RetentionPolicy(), recompute=RetentionPolicy(default="recompute"))`
returns an immutable pair of capture policies:

| Argument | Type | Contract |
| --- | --- | --- |
| `save` | `RetentionPolicy` | Policy for the `save` graph pair. The default cost policy permits removals from backward-required forward state. Set `default="retain"` to keep that state without cost-based removals. |
| `recompute` | `RetentionPolicy` | Policy for the `recompute` graph pair. The default retains checkpoint inputs and supported routing choices, regenerating other intermediates. Add retain rules to preserve selected operator results and required auxiliary outputs. |

The policies are independent, with no requirement that one cut save more bytes
than the other. They describe state needed by backward, not every forward
intermediate: unused results are removed. Required inputs, effects and mutations
still constrain valid cuts. Retained values are eligible for planner-controlled
host spill; retention does not require device residency.

`RetentionPolicy(memory_bound_flops_per_byte=16.0, rules=(), default="cost")`
returns an immutable policy for either graph pair:

| Argument | Type | Contract |
| --- | --- | --- |
| `memory_bound_flops_per_byte` | finite nonnegative `float` | Under the cost default, operations at or below this arithmetic intensity may be regenerated. This is not a memory budget or measured execution time. |
| `rules` | `tuple[RetentionRule, ...]` | Exact operator overrides, optionally scoped to source modules. Each target/scope pair may appear once; conflicting matches fail capture. |
| `default` | `"cost"`, `"retain"` or `"recompute"` | `cost` retains compute-bound and unpriced custom operations, permitting memory-bound replay. `retain` keeps backward-required results. `recompute` regenerates unselected intermediates from checkpoint inputs and selected retained results. Correctness-required state can also remain saved. |

`RetentionRule(target, action, *, module_pattern=None)` identifies one exact forward operator overload
name, such as `"aten.mm.default"`. `target` is a nonempty `str`; `action` is
`"retain"` or `"recompute"`. The rule includes tuple extraction, so backward can
receive the selected operation's required auxiliary outputs as well as its
visible result. Unused outputs are removed. The rule does not request saving the
operation's inputs: those can be regenerated from earlier saved state.
Targets absent from an individual stage have no effect there, allowing one
policy to apply across different stages. Inspect captured graphs and the
graph-pair manifest to verify the intended operators and regenerated work.

`retain` prevents the matching forward computation from being replayed in
backward; it does not require a separate saved port containing its raw result.
The cut can save that result, save a sufficient derived value instead, or omit
local backward state when the result is needed only by forward consumers.
For example, a compact count statistic can cross the cut as integer counts or
normalized floating-point frequencies. An existing forward output does not
automatically require a duplicate backward save. Inspect the actual saved
ports and their storage backings to determine retained bytes; the rule name
alone does not determine memory use. Profiling and residency planning still
apply to the resulting graph pair.

`module_pattern` is an optional case-sensitive shell pattern matched against
the operation's source module paths. Any matching ancestor includes its
descendants. For example, `"model.blocks.*.experts"` selects operations in each
block's `experts` module and its children; `*` can match dots. `None` applies
the rule everywhere. Module paths come from export, survive differentiation
through structural source indices, and also identify operations introduced by
decomposing that source operation. The target still names the exact overload
in the differentiated forward graph. Missing required source provenance fails
capture rather than silently turning a scoped rule into a global rule.

Scope avoids removing useful saved cuts in unrelated operations. For example,
if an expert implementation produces chunk-local state through an add:

```python
save_policy = RetentionPolicy(
    rules=(RetentionRule(
        "aten.add.Tensor", "recompute", module_pattern="model.blocks.*.experts"
    ),)
)
```

Other modules keep the cost-based default. A global add-removal rule can force
both large projection operands to remain saved instead of their single sum.
Choose scopes from the captured source paths, and verify the resulting saved
tensors and chunk-local legality. This control does not make an otherwise
invalid saved output legal. Rules have no ordering precedence: overlapping
matches with different actions are errors. Use the policy's `default` for a
broad choice and explicit rules for exceptions.

Explicit rules override the default classification. Model checkpoint annotations
take precedence over the default; contradictory explicit rules fail capture.
Capture also refuses a requested rule the resulting saved cut cannot satisfy,
for example when correctness requires saving a value requested for replay.
Selective cuts retain results of operations tagged as using RNG state or ordered
effects. Explicit replay rules for those operations fail capture: use retained
results or expose replayable state through ordinary tensor inputs. This avoids
implicit generator objects in saved graph signatures.
Rules select computations, not arbitrary views of their storage. An aliased
saved output can retain its complete backing allocation.

For example, compare two candidates with the same save policy and toggle
retention of selected sequence operators in recompute. `attention_operators`
contains the exact dense-attention overload names supplied by the provider:

```python
from shadowspill.pytorch import GraphPairPolicy, RetentionPolicy, RetentionRule

save_policy = RetentionPolicy()
graph_pair_policies = {
    "inputs_only": GraphPairPolicy(save=save_policy),
    "attention_retained": GraphPairPolicy(
        save=save_policy,
        recompute=RetentionPolicy(
            default="recompute",
            rules=tuple(RetentionRule(op, "retain") for op in attention_operators),
        ),
    ),
}
```

Each candidate offers its two graph pairs to planning. PressureFit chooses
save/recompute per stage and plans residency for that candidate; program search
compares the resulting admitted Plans across candidates. The chunk policy's
`variants` can restrict which outer variants are offered, independently of the
retention policies. Forward sinks require a `save` choice, so changing only
`recompute` leaves their paired-loss save policy intact.

To let PressureFit choose whether to retain dense attention **within one
Program**, put the selective policy on `save` and leave `recompute` at its
default. Both variants also preserve supported routing choices:

```python
attention_only_save = GraphPairPolicy(
    save=RetentionPolicy(
        default="recompute",
        rules=tuple(RetentionRule(op, "retain") for op in attention_operators),
    ),
    recompute=RetentionPolicy(default="recompute"),
)
```

Here `attention_operators` contains only the dense-attention provider overloads
whose results should be retained. Both variants must remain enabled. The
`save` name does not force the cost policy: its default is overridden by this
explicit policy. Required auxiliary outputs, checkpoint inputs and effect state
still constrain the cut. Attention inputs such as projected Q/K/V may be
regenerated; inspect the captured backward to verify what replay is avoided.

The three candidate pairs offer different per-stage choices:

| Candidate | SAVE cut | RECOMPUTE cut |
| --- | --- | --- |
| Cost / inputs | Cost-based retention | Checkpoint inputs |
| Cost / attention | Cost-based retention | Selected attention state and checkpoint inputs |
| Attention / inputs | Selected attention state and checkpoint inputs | Checkpoint inputs |

All three pairs additionally retain supported [routing choices](../../architecture/routing-retention.md)
by default, including in cuts labeled "inputs" above. An explicit `RetentionRule`
can override a routing operator independently for each variant. These saved
tensors use normal placement and spill planning.

The last pair removes other cost-retained results from its SAVE alternative.
In stages without selected attention operators its two cuts may coincide. It
can reduce saved state, but may repeat more non-attention work than the cost
alternatives. Compare their measured admitted Plans rather than assuming an
ordering. Changing SAVE also applies to the mandatory terminal SAVE choice;
terminal chunking and backward requirements still need to validate. A separate
Program candidate does not add a third per-stage variant to another Program.

| Phase/control | Responsibility |
| --- | --- |
| `partition` | Chooses checkpoint boundaries and their inputs. |
| `graph_pair_policy` during build | Constructs each variant's saved-state cut and backward graph. Both policy identities, operator classifications, resolved source scopes and graph contracts key capture reuse. |
| `chunking` during build | Chooses execution regions, axes, sizes and permitted outer variants. It does not change the selected outer saved-state cut. |
| Profiling | Measures each final complete task, including its chunk loop, assembly and provider workspace. |
| Planning/PressureFit | Chooses among measured allowed stage variants and decides residency, transfers and execution order under device and spill budgets. Retained does not mean device-resident. |
| Candidate search | Plans the declared measured candidates at each budget and selects the lowest admitted simulated update time. |
| Execution | Uses the exact selected saved Plan without invoking retention callbacks or resizing chunks. |

Changing either policy requires building and profiling the resulting task forms;
CPU replanning can reuse saved candidates for different budgets and transfer
costs. Equal final task contracts can share code/profiles; a changed saved cut
cannot borrow another cut's memory evidence. See
[graph-pair construction](../../architecture/graph-pair-construction.md#save),
[artifact metadata](artifacts.md#aot-graph-pair-cache), and the
[candidate example](../../../benchmarking/quickstart.md#searching-chunk-and-retention-policies).

For unusual cost classification, subclass `RetentionPolicy.classify(node)`.
It receives a traced FX node and returns `OperatorClass.MEMORY_BOUND`,
`OperatorClass.COMPUTE_BOUND`, `OperatorClass.UNKNOWN`, or `None` for a nonoperator.
Explicit rules and the retain/recompute defaults resolve before cost classification.
Override `identity()` with a deterministic JSON-compatible mapping containing
every extra configuration value; include `super().identity()`. Classification
must depend only on traced metadata and that configuration, never live payloads
or runtime memory pressure. `digest(graph_module)` combines this identity with
custom-operator classifications for capture reuse.

## Custom gradient destinations

`build_step` and `build_forward_backward` accept `gradient_destination_rules`,
an empty tuple by default. These rules let an opaque backward operator add its
contribution directly into existing parameter-gradient storage. The first
contribution creates the gradient; later uses across checkpoint regions,
execution regions, chunks and microbatches reuse that destination. One replacement
call handles its selected outputs together; it must not repeat the backward
computation for each output.

The public types are exported from `shadowspill.pytorch`:

| Type | Contract |
| --- | --- |
| `GradientDestinationRule` | Names `target` and `replacement` dispatcher overloads, eligible `output_indices`, a positive `version`, and an optional build-time `mapper`. The replacement receives the original arguments plus mutable `destinations`, `output_indices`, and `contract_version`. |
| `GradientDestinationContext` | The mapper's input: the FX `producer`, eligible `destinations`, and the `fuse_narrow_gradient_accumulation` precision permission. |
| `GradientDestination` | One eligible producer `output_index`, its final `gradient`, existing `destination`, and `operations` from that gradient back to the raw producer output. |
| `GradientDestinationMapping` | The mapper's result: selected `output_indices`, corresponding sequences of destination `views`, and provider-specific `keywords` passed as real operator arguments. Returning `None` declines fusion. |
| `DestinationView` | A storage-preserving `target` operation and its `arguments`, applied to a supplied destination. It must remain writable without a copying reshape or overlapping elements. |

Without a mapper, the rule handles reversible views at unchanged dtype. A custom
mapper must preserve layout, scaling, casts, rounding and padding, honor the
precision permission, and be deterministic without editing the supplied FX graph.
Unmatched or unsupported outputs keep ordinary accumulation. The accumulating
overload preserves unconsumed outputs and may return an empty tensor, `None`, or
the destination at a consumed position, subject to its dispatcher and
functionalization contract. Fixed Tensor schemas with empty consumed outputs
are supported by automatic functionalization.

Rules are build inputs, independent of checkpoint retention and execution axes
or chunk sizes. Replacement calls and contract versions enter artifact identity;
callbacks are not consulted during saved execution. Their kernels still require
normal compilation, profiling and physical admission, including any internal
temporaries. Chunked custom overloads also need `OperatorRule` axis contracts.
See [custom gradient destinations](../../architecture/custom-gradient-destinations.md)
for mutation schemas, ownership restrictions, examples and provider qualification.

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


### IsolateOperations

`IsolateOperations(targets, base="auto")` is a `PartitionPolicy` that gives each
selected operation and its tuple-unpacking nodes a separate stage. Use it to
separate sequence operations from surrounding independent computation. `targets` is a
`tuple[str, ...]` of exact operator overload names in the exported graph; `base`
is a `PartitionSpec` supplying the other cuts. Overlapping intervals merge.
Other nodes inside an interval stay with it to preserve execution order.
`assign_stages(graph_module, module)` returns the normal node-name mapping.

## Axis chunking policies

Pass `chunking=policy` to `build_forward`, `build_forward_backward` or
`build_step` to run selected effective stages in axis slices. The default `None`
leaves stage execution unchanged. The public names below are exported from
`shadowspill.pytorch`, as well as `shadowspill.pytorch.chunking`.

Each region chunks one logical axis at a time. Its tensors can map that axis
to different physical dimensions, including factored dimensions after a
reshape. The axis can represent tokens, independent heads, batch items, or
another independent domain. Operations must preserve chunk independence or
provide a supported additive reduction; this is not unrestricted multidimensional
tiling. Each execution region chooses its own axis and chunk size.

`ChunkRegions(partition, chunking, variants=("save", "recompute"), lifetimes=None)` selects
execution regions **inside** each checkpoint stage. The builder's `partition=`
still defines checkpoint boundaries. The arguments are:

| argument | type | meaning |
|---|---|---|
| `partition` | `PartitionSpec` | Assigns contiguous execution regions on each explicit forward or backward graph. Its `module` argument is that task's graph module; use graph metadata and operator targets, rather than source-model children. |
| `chunking` | `ChunkPolicy` | Receives each region's physical tensor ports and chooses its own chunk size and chunk axes, or `None` for whole-region execution. Backward regions receive separate callbacks. |
| `variants` | `tuple[str, ...]` | Distinct outer training choices from `"save"` and `"recompute"`. Each selected child contract must permit the outer choice. The terminal training stage remains unchanged with its mandatory save choice. |
| `lifetimes` | `ExecutionLifetimes \| None` | Optional operation scheduling on the complete explicit task, before gradient destination routing and execution partitioning. Default `None` preserves the captured schedule. Only outer `ChunkRegions` accepts it. |

The callback can choose different sizes by operator, module path, direction,
input geometry, or `structural_contract`. Equivalent child contracts share
compilation; each different resolved composition receives its own profile key.
Execution-region intermediates are local task workspace. They do not add outer
checkpoint inputs, saved values, or independently scheduled tasks. Cuts must
pass flat tensor ports: keep tuple unpacking with its producer.

```python
from shadowspill.pytorch import ChunkRegions, IsolateOperations

# The caller's axis_policy can return a different AxisChunks for every region.
def execution_policy(axis_policy, sequence_operator_targets):
    return ChunkRegions(
        partition=IsolateOperations(sequence_operator_targets, base="whole"),
        chunking=axis_policy,
        variants=("recompute",),
    )
```

### Execution lifetimes

`ExecutionLifetimes(rules=(), readonly=(), select=None, *, layout=())` controls
operation placement and internal view compaction inside a checkpoint task. It
can move readonly work once, copy selected views into smaller backing storage,
and explicitly repeat selected primal work near consumers. Replay has no default
selections: expensive work repeats only when explicitly selected. These controls
do not change the saved-state cut or request device/host residency. Axis selection
and chunk sizes still belong to `AxisChunks`; retention belongs to
`GraphPairPolicy`; PressureFit chooses variants and residency from measured tasks.

`LifetimeRule(target, *, move=None, barrier=None, compact=None, replay=None, source_target=None,
module_pattern=None)` selects one exact explicit operator overload. Optional
`source_target` matches the original source operation; `module_pattern` matches
any attributed source module path with a case-sensitive shell pattern. An
unattributed operation does not match a source-scoped rule. Each rule must set
at least one of these choices:

| choice | meaning |
|---|---|
| `move=True` | Permit the readonly operation to move once toward its first consumer, subject to effects, alias writes and protected lifetimes. `False` explicitly disables this permission. |
| `barrier=True` | Keep this operation in place and require that every proposed move does not increase the internal backing bytes live immediately before it. `False` leaves it unprotected. A barrier takes precedence over movement and view preparation. |
| `compact=True` | Prepare this internal alias view near its producer and consider a contiguous copy for later execution-region consumers. Related selected views are considered jointly. At least one protected backing set must shrink and none may grow. Unsupported candidates remain un-copied with recorded reasons. |
| `replay=True` | Permit repeating this primal operation in later consumer regions. Inputs must still have their original versions; readonly and value/alias checks must pass. `False` excludes it, including implicit tuple extraction from a selected producer. |
| `None` | Leave that choice unspecified. Conflicting explicit values from matching rules or the callback fail capture. Rule order does not resolve conflicts. |

```python
from shadowspill.pytorch import (
    ChunkRegions, ExecutionLifetimes, IsolateOperations, LifetimeRule,
)

def scheduled_regions(axis_policy, sequence_targets):
    return ChunkRegions(
        IsolateOperations(sequence_targets, base="whole"),
        axis_policy,
        lifetimes=ExecutionLifetimes(rules=(
            LifetimeRule("aten.sin.default", move=True,
                         module_pattern="model.blocks.*"),
            *(LifetimeRule(target, barrier=True) for target in sequence_targets),
        )),
    )
```

For unusual selection logic, `select(operation)` returns
`LifetimeSelection(move=None, barrier=None, compact=None, replay=None)` or `None`. The frozen
`LifetimeOperation` contains `index`, exact `target`, optional `source_target`,
`module_paths`, `effect_fence`, task `kind`, `variant`, `stage_id`, `checkpoint_stage_id`,
`phase` (`"primal"`, `"derivative"` or `"unknown"`), and
flattened `inputs`/`outputs` tensor geometries (`TensorGeometry | None`). It
contains no tensors or mutable graph nodes. The callback must be deterministic
and read-only; do not run kernels or mutate global capture state from it. It
runs once per operation when that task occurrence is built. Profile, plan and
cold execution use the saved result without calling it again.
`effect_fence` explains a mutation, RNG or uncertified opaque effect, or is
`None` for certified readonly work. Absence of an effect fence is not permission
to cross an aliased writer or increase a protected backing set.

Unknown opaque operations are effect fences. A versioned
`ReadonlyOperator(target, version)` in `readonly` certifies deterministic,
readonly execution for that exact overload. It cannot override mutation,
RNG/effect registrations, or unproved alias ownership. An `OperatorRule` for
axis independence or an operator's cost classification is not this effects
contract. Conditional or unknown alias returns are conservatively refused
when lifetime analysis is needed. Parameter-gradient conversion barriers remain
protected regardless of user movement choices.


Compaction changes storage layout and therefore needs a separate proof from
readonly execution. `LayoutIndependentOperator(target, version, inputs)` in
`layout` certifies value semantics for specified dispatcher argument indices
(including keyword arguments and tensor lists). Replacing those arguments by
equal logical tensors with different strides, offsets, backing spans or alias
relationships must preserve every initialized logical output value, shape and
dtype. The result layout may differ; the pass follows layout changes through
fresh computations as well as alias views. Bitwise equality across different
kernel layouts is not promised by this contract; ordinary numerical qualification
still applies. A provider needs both readonly and layout contracts. Neither can
override mutation, RNG or unsupported ATen storage semantics.

For example, add `LifetimeRule("aten.slice.Tensor", compact=True)` to consider
escaping slices, while marking sequence operators with `barrier=True`. A callback
can select other view/tuple operations. Selected views may be prepared earlier
even when `move=False`; use `barrier=True` to pin an operation. Input/parameter
backings, outer aliases, and written backings cannot compact. With no protected
barriers there is no qualifying memory reduction, so copies are not applied.

The pass conservatively qualifies known value-based ATen operations. Raw
`as_strided`, dtype-reinterpreting views and unknown layout-sensitive consumers
are refused when their input layout changes. An unchanged complete dense fresh
result ends propagation; a fresh result with changed strides does not. A failed
candidate is discarded as a group, and the task's outer storage contract must
remain identical. Execution partition rules are resolved before considering
cross-region uses, then resolved again after transformation and gradient routing;
they must cover inserted `aten.clone.default` operations too.

Replay runs after movement and before view preparation/compaction. It uses the
captured differentiation provenance; operations with missing provenance in a
backward task cannot replay. Explicit forward/inference tasks identify primal
work when no differentiation tag is present. Structural tuple extraction follows
a selected producer automatically unless explicitly excluded. Each selected
operation is copied at most once per consumer region, and barrier-containing
regions never receive copies. `replay=True` does not authorize a barrier itself,
a gradient conversion, an effect, or an externally observed/written result.

Readonly execution alone also cannot justify splitting an alias relationship.
The same input-value contracts protect replay consumers, including an operation
that is itself being replayed with copied dependencies. Dependencies must still
represent the original values at the copied computation; disjoint known tensor
writes may be crossed, but overlapping writes, RNG and unknown effects block it.
Every proposal must preserve the outer storage contract and avoid increases at
all protected memory points. Intermediate proposals need not shrink a live set:
redirecting a later consumer may release the last use of an original backing.
With no selected barriers this local memory check is empty, so physical admission
of the final measured composition remains essential. The profiler measures every
extra computation. A replayed view inherits its original compaction permission;
selectors are never called again for generated nodes.

Resolved operation indices, permissions, used contract versions and scheduling
decisions enter `execution_regions["lifetimes"]` in task metadata and executable
identity. `schedule`, `replay`, `views` and `compaction` records use indices from their
respective input order; movement/view records include their resulting order,
and replay records map original source indices to final copy indices and regions.
Occurrence explanations appear separately in the
`execution_lifetimes` list, including source module paths and semantic backing
counts. These counts exclude task inputs, opaque workspace and allocator
overhead; the complete finalized composition still needs profiling and physical
admission. Different module names with identical resolved behavior can reuse
compilation. Callbacks and occurrence names are not serialized into executable
identity.

Only the outer `ChunkRegions` may specify lifetimes. Setting it inside
`AxisChunks.regions` is refused so scheduling always sees the complete
checkpoint task once. Replay permission is independent of ordinary graph-pair
recomputation and is empty by default.

`ChunkPolicy` implements `configure_stage(stage: StageInfo) -> AxisChunks | None`
to select an
execution contract after partitioning and control-stage folding. Return `None`
to leave a stage intact. The callback must be deterministic and read-only;
it must not run model kernels or mutate capture metadata. Unknown operations
in a selected region fail capture rather than silently running unchunked.

<!-- source-signature: src/shadowspill/pytorch/chunking/policy.py:ChunkPolicy.configure_stage -->
```text
configure_stage(stage: StageInfo) -> AxisChunks | None
```

`StageInfo` provides these read-only fields:

| field | type | meaning |
|---|---|---|
| `stage_id` | `str` | This effective stage occurrence's identifier. |
| `module_paths` | `tuple[str, ...]` | Source module paths present in the current occurrence. Explicit forward/backward tasks resolve these through source-operation provenance; generated operations without provenance add no path. |
| `inputs` | tuple of tensor geometries or `None` | Full input shapes, strides, dtype and storage offsets, indexed by effective stage input position. Static values have no tensor geometry. |
| `input_roles` | `tuple[TaskInputRole, ...]` | Provenance roles such as parameter, buffer, activation or control. |
| `operators` | `tuple[str, ...]` | Exact operator overload names appearing in this stage. |
| `terminal` | `bool` | Whether this is the terminal training stage, which generic axis chunking excludes. Forward-only stages have `False`. |
| `kind` | `str \| None` | For an execution region, the explicit task direction, including `"forward"`, `"backward"` or `"inference"`. A source-stage callback receives `None`. |
| `checkpoint_stage_id` | `str \| None` | The owning checkpoint occurrence for an execution-region callback. |
| `structural_contract` | `str \| None` | Unchunked execution-region structural digest, usable for exact per-contract overrides. It excludes the subsequently selected chunk size. |
| `input_domains` | `tuple[Mapping[str, ChunkAxis], ...]` | Known named axis mappings for each execution-region input, including flattened saved tensors and backward cotangents. Empty for source-stage callbacks. |

`AxisChunks(chunk_size, inputs, variants=("save", "recompute"), rules=(), domain=None, regions=None)` returns
an immutable chunk specification:

| argument | type | meaning |
|---|---|---|
| `chunk_size` | positive `int` | Maximum logical axis positions per invocation. An uneven tail gets its own compiled geometry. |
| `inputs` | `Mapping[int, int \| ChunkAxis]` | Effective stage input positions and their chunk axes. All selected inputs must describe the same positive logical axis extent. Remaining inputs are shared. |
| `variants` | `tuple[str, ...]` | Nonempty distinct training retention choices from `"save"` and `"recompute"`. This restricts stage alternatives, independently of residency. Mandatory endpoint choices must remain available. There is no backward retention choice in forward-only builds. |
| `rules` | `tuple[OperatorRule, ...]` | Caller-certified contracts for opaque operators in this stage. Targets must be unique. |
| `domain` | `str \| None` | Optional nonempty name for the logical coordinates, such as `"tokens"` or `"heads"`. Matching inherited input mappings are included automatically; an explicit conflicting mapping is refused. `None` leaves mappings local to this region. |
| `regions` | `ChunkRegions \| None` | Optional execution partition and per-region policies inside each bounded slice. Requires a named domain and must permit every outer variant. Child callbacks receive slice-sized input geometries and inherited axes. `None` compiles one graph per slice. |

Use nested regions when consecutive computations need different chunk bounds
without a full-axis temporary between them. For example, an outer bound of
32768 can contain projection regions bounded by 8192 and another region bounded
by 32768. The intermediate storage then spans at most the outer bound. Each
inner loop handles its own tail; sizes need not divide one another. Checkpoint
ports and saved values retain their complete shapes. The whole group must pass
the outer axis-separability checks, including opaque operator rules.

```python
from shadowspill.pytorch import AxisChunks, ChunkRegions

def bounded_regions(input_axes, execution_partition, inner_policy):
    return AxisChunks(
        32768, input_axes, domain="positions",
        regions=ChunkRegions(execution_partition, inner_policy),
    )
```

An integer input axis is shorthand for `ChunkAxis(dim)`. Negative dimensions are
resolved against the input rank. `ChunkAxis(dim: int, before: int = 1,
after: int = 1)` describes an ordered dimension with length
`before * logical_extent * after`; both factors must be positive. For example,
flattening a batch and sequence into one dimension uses `ChunkAxis(0, before=B)`
when each invocation should take the same sequence slice from every batch item.
The factors specify axis coordinates, not independent microbatches.

Policies should prefer `stage.input_domains[i][domain]` when available over
inferring an axis from equal dimension lengths. For example, a saved statistic
can flatten multiple dimensions and still carry the same logical axis. Named
mappings propagate through chunked children, whole regions with supported
operator contracts, and the outer graph pair's saved values. For example, a
whole normalization can produce flattened `[batch * sequence * heads]`
statistics that a later chunked consumer must slice with its token inputs.
Executing the normalization whole does not make those statistics shared.
Named domains carry their declared `OperatorRule` contracts from forward into
backward and into nested compositions. Conflicting contracts for the same
overload and domain are refused. This propagation does not select chunking for
the whole region or change checkpoint retention. An operation with unknown
semantics does not establish an output mapping; the policy must declare it
where execution resumes.

```python
from shadowspill.pytorch import AxisChunks
from shadowspill.task.inputs import TaskInputRole

class FeedForwardChunks:
    def configure_stage(self, stage):
        if stage.terminal or not any(p.endswith(".ffn") for p in stage.module_paths):
            return None
        axes = {
            i: 1
            for i, (value, role) in enumerate(zip(stage.inputs, stage.input_roles))
            if value is not None and len(value.shape) == 3
            and role not in {TaskInputRole.PARAMETER, TaskInputRole.BUFFER}
        }
        return AxisChunks(8192, axes, variants=("recompute",)) if axes else None

# Supply this alongside the partition policy to any public builder.
chunking = FeedForwardChunks()
```

`OperatorRule(target, version, output_axes, reductions=())` certifies one opaque
operator overload's semantics. `target: str` names that exact overload;
`version: str` is a nonempty semantic version included in executable identity.
`output_axes` is a tuple of `int | ChunkAxis | AxisFrom | ChunkLocal | None`,
with one entry per flattened
output: an integer declares a dimension of exactly the logical axis extent,
`ChunkAxis` declares a factored dimension, `AxisFrom` derives its layout, and
`None` declares a shared output.
`reductions: tuple[int, ...]` instead marks listed `None` outputs as additive
chunk contributions. Shared outputs must be invariant across chunks. Reduced
outputs must sum to the full result; an average or normalized statistic does
not satisfy that requirement. Separate forward and backward overloads need
separate contracts.

`AxisFrom(input_index: int, trailing_dims: int = 0)` derives an output's ordered
chunk factor from an operator's positional input geometry. `input_index` is
nonnegative; `trailing_dims` removes that many trailing feature dimensions
before reshaping to the output. Normalization statistics flattened from
`[batch, rows, features]` use `AxisFrom(0, 1)`. This declares a shape relation;
it cannot certify a data-dependent permutation or global routing statistic.

`ChunkLocal(dim: int | None = None)` declares opaque scratch whose coordinates
belong to one child invocation, such as a dispatch permutation. An integer
`dim` names an extent proportional to the logical axis extent; `None` means fixed
geometry. These values may flow through the child, but cannot escape as saved
values or public outputs. Mixing local coordinates with ordered axis values
requires an explicit opaque restoration rule. A save variant that needs local
scratch is refused; select recompute only when its captured cut does not retain
that scratch. This contract does not make capacity-limited routing or a global
auxiliary loss independent along the selected axis. Additive sufficient statistics may escape as
reductions, with nonlinear aggregation in a separate stage.

Supported built-in rules validate axis propagation, additive parameter VJPs and
explicit backward gradient mutations. Opaque rules remain the caller's semantic
responsibility. Generic terminal training stages, RNG operations, wrapped tensor
components and unsupported mutations are refused. Attention, recurrence and
convolution must retain their complete semantics outside regions chunked along dependent axes. The [architecture contract](../../architecture/axis-chunking.md)
explains retention, complete-task profiling and automatic boundary residency.


## Invocation handle

`InvocationResult` is the handle returned by Executor. `result()` / `wait()`
waits once and returns caller-owned outputs, preserving completion errors for
later reads. `resolved` reports completion and `diagnostics` exposes the optional
runtime trace. See [execution](training.md).

## Build and measurement policies

`PartitionPolicy` supplies custom stage boundaries to the public builders;
`PartitionSpec` is the accepted partition policy type. `measure_program` accepts
`warmup_seconds` and `measure_seconds` directly. These phases do not create
runtime pools. See [the build phase](../../architecture/offline-pipeline.md).
