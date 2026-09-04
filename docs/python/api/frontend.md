# Frontend and lifecycle API

The symbols on this page are exported by `shadowspill.memory` or
`shadowspill.pytorch`.

## Memory pool configuration

| Symbol | Purpose |
|---|---|
| `DevicePool` | Immutable execution-device pool configuration. |
| `PinnedHostPool` | Immutable registered pinned-host spill-pool configuration. |
| `MemoryPoolConfig` | Union of supported pool configurations. |
| `TransferRoute` | Immutable directed relationship between two named pools. |
| `device()` | Construct a `DevicePool` from `physical_capacity`, device ordinal, and optional provider headroom. |
| `pinned_host()` | Construct a `PinnedHostPool` from a byte capacity. |
| `transfer_route()` | Construct a directed route from source and destination pool names. |

Provider headroom is inside `DevicePool.physical_capacity`. The runtime reports
the derived suballocatable capacity after initialization.

## Runtime

<!-- source-signature: src/shadowspill/pytorch/runtime_adapter/runtime.py:Runtime.__init__ -->
```text
Runtime(
    *,
    pools: Mapping[str, MemoryPoolConfig],
    routes: Mapping[str, TransferRoute],
    library_path: str | Path | None = None,
    calibrate: bool = True,
    worker_poll_nanoseconds: int = 1_000,
    backend: str | None = None,
)
```

`Runtime` owns the installed allocator, initialized `MemoryPool` and
`RuntimeRoute` registries, transfer calibration, active callable count,
persistent state registry, and latest failure. Public properties are `pools`,
`routes`, `transfer_capabilities`, and `last_failure`.

Create `Runtime` before constructing or loading workload state. Construction
registers the configured physical pools and calibrates the real directed
routes between their addresses. Model and optimizer state are then created and
imported into that initialized runtime; planning reads the published
`transfer_capabilities` snapshot rather than recalibrating independently.

Pool and route names are user-defined identities. A planning call binds its
own `execution` and `spill` pool names and resolves the matching directed
fetch/evict routes; those roles are not global properties of `Runtime`. The
current PyTorch backend supports one device pool and any number of registered
pinned-host pools. Unsupported pool or route combinations fail during runtime
construction or plan resolution.

`calibrate_transfer_capabilities()` measures all or selected source/destination
routes and atomically publishes a new `TransferCapabilities` matrix. Callers
may coordinate several processes and invoke calibration concurrently to
measure contended links.

`Runtime.close()` verifies that no planning, callable, persistent imported
state, public object reference, or caller-owned device output remains, then
tears the runtime down as [memory runtime](../../architecture/memory-runtime.md#failure-and-teardown)
describes. Release or copy ordinary device outputs before this call. PyTorch's process-global
allocator shim cannot be uninstalled, so it remains in a permanently closed
state and rejects subsequent device allocations.

The immutable runtime values are:

- `MemoryPool`
- `RuntimeRoute`
- `TransferProfile`
- `TransferCapabilities`
- `ExecutionTaskIdentity`
- `RuntimeFailureDiagnostics`

Configuration and execution failures use `RuntimeConfigurationError` and
`RuntimeExecutionError`.

## Persistent state

<!-- source-signature: src/shadowspill/pytorch/state/model.py:import_model_state -->
```text
import_model_state(model, *, runtime, pool, release_source=True)
```

<!-- source-signature: src/shadowspill/pytorch/state/model.py:import_model_state_from_file -->
```text
import_model_state_from_file(model, path, *, runtime, pool)
```

<!-- source-signature: src/shadowspill/pytorch/state/model.py:export_model_state -->
```text
export_model_state(model, *, runtime, release_runtime=False)
```

<!-- source-signature: src/shadowspill/pytorch/state/model.py:release_model_state -->
```text
release_model_state(model, *, runtime)
```

<!-- source-signature: src/shadowspill/pytorch/state/model.py:read_model_state -->
```text
read_model_state(model, *, runtime, copy=True)
```

<!-- source-signature: src/shadowspill/pytorch/state/optimizer.py:import_optimizer_state -->
```text
import_optimizer_state(optimizer, *, runtime, pool, release_source=True)
```

<!-- source-signature: src/shadowspill/pytorch/state/optimizer.py:import_optimizer_state_from_file -->
```text
import_optimizer_state_from_file(optimizer, path, *, runtime, pool)
```

<!-- source-signature: src/shadowspill/pytorch/state/optimizer.py:export_optimizer_state -->
```text
export_optimizer_state(optimizer, *, runtime, release_runtime=False)
```

<!-- source-signature: src/shadowspill/pytorch/state/optimizer.py:read_optimizer_state -->
```text
read_optimizer_state(optimizer, *, runtime, copy=True)
```

Planning takes whichever model it is given. State the caller imported is
adopted and outlives the plan; state that has not been imported is imported
in place by `plan_step()` or `plan_forward()`, which then own it, so closing
the callable releases that state and empties the parameters that viewed it.
Read what you need before the close, or import beforehand to keep it. Only
`make_step_program()` still requires an explicit import, because it returns
no callable that could own the result.

`import_model_state()` returns a copied module hierarchy whose registered
tensors point at runtime spill leases. `release_source=True` means ShadowSpill
does not retain the input model; Python releases it when no caller reference
remains. `export_model_state()` rebinds the same registered tensor
identities to ordinary CPU storages and optionally releases runtime objects.
`release_model_state()` releases those runtime objects without materializing
any CPU copy: the module's registered tensors become invalid, so the module
must be discarded afterward. It is the teardown operation for callers that no
longer need the state, such as qualification hosts that cannot hold an
anonymous model copy beside the full pinned spill arena.

`import_model_state_from_file()` and `import_optimizer_state_from_file()`
fill pool state from a checkpoint without building the checkpoint in ordinary
host memory first. The file is mapped rather than read, so its pages are
reclaimable cache, and the import happens before the copy, so the values land
in pool memory directly. The checkpoint must name every tensor the target
enumerates and agree with each on dtype and shape; raw bytes cannot be
converted, so a disagreement is refused rather than reinterpreted, and extra
names in the file are ignored. One file per call: a checkpoint sharded across
several files is refused. The optimizer form is keyed by the paths
`import_optimizer_state()` enumerates, which is what `read_optimizer_state()`
writes, so a checkpoint saved from one reads back through the other.

A model need never occupy ordinary host memory on its way into a pool.
Construct it under `torch.device("meta")`, so its parameters have no storage;
assign a mapped checkpoint onto it with
`model.load_state_dict(torch.load(path, mmap=True, weights_only=True),
assign=True)`, so its parameters become file-backed pages rather than
anonymous allocations; then `import_model_state()` copies those into the pool
and releases the source. Only the pool copy is anonymous host memory at any
point. This needs no ShadowSpill-specific call: `import_model_state_from_file()`
is the shorter form when the model is already built.

`read_model_state()` and `read_optimizer_state()` answer what the state
currently is without rebinding anything, which is what makes them usable while
a plan holds the target -- `export_*` cannot run then, and the runtime refuses
it. Each returns a flat mapping from the name the state is enumerated under to
a host tensor.

`copy=True`, the default, gives ordinary host memory outside the runtime
pools, one buffer per storage root with the target's views laid over it, so
entries that shared a root still share one, and the values keep what they held
when the call returned. `copy=False` allocates nothing and views the pool's
own bytes instead: ordinary torch operations work on them, but treat them as
read-only, because writing through one changes runtime state behind the
runtime's back, and they stop being current the next time the plan runs. A
storage root whose pool copy is not the authoritative one is copied either
way.

`import_optimizer_state()` and `export_optimizer_state()` apply the same
storage policy to already materialized optimizer state, as a standalone
ownership operation. They are not a planning input: `plan_step()` constructs
and manages its own optimizer from the supplied factory and imports that
optimizer's state itself, so handing it one whose state is already imported
raises `RuntimeConfigurationError`. Resume prior state through
`PlannedTrainStep.load_state_dict()` instead.

## Planning entrypoints

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
    execution_device=None,
    partition="auto",
    verbose=True,
    artifact_store_dir=None,
    profiling_metadata=None,
    allocation_probe_seeds=1,
    allocation_probe_repetitions=2,
    shared_outputs=(),
    save_plan=True,
    force_fresh=False,
    overwrite_plan=False,
    implementation_revision=None,
) -> PlannedForward
```

`plan_forward()` accepts one fixed example-input sequence and returns
`PlannedForward`.

The optional `shared_outputs` sequence contains `SharedOutput` declarations.
Use `shared_output(*path, retain_in=pool_name)` to identify a tensor leaf in
the public output pytree and retain that value as a runtime object in the
named pool. The corresponding result leaf is a `TensorRef`, not a copied
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

`ObjectConsistency.CAUSAL` is the default and makes each task acquire the
object's current generation plus its published readiness dependency.
`ObjectConsistency.UNORDERED` deliberately omits cross-callable value
ordering while retaining the object and lease safely. `require_in` must name
a pool guaranteed by the reference. Floating shared inputs receive a
deterministic task-local profiling representative; integer and Boolean
control inputs require an explicit CPU `profiling_value` on `SharedInput`.
The `ObjectConsistency` enumeration contains these two policies.

Several planned callables may remain admitted to one runtime and may bind the
same `TensorRef`. Distinct callables may be submitted without synchronizing the
dispatcher between them. Causal shared inputs consume the current published
generation; unordered shared inputs deliberately omit value ordering while
retaining lease safety.

<!-- source-signature: src/shadowspill/pytorch/api.py:plan_step -->
```text
plan_step(
    model,
    *,
    objective,
    opt,
    example_inputs,
    runtime,
    execution,
    spill,
    execution_budget=None,
    spill_budget=None,
    dynamic_scratch_reserve_bytes=None,
    minimum_object_bytes_evict_eligible=1 << 20,
    execution_device=None,
    partition="auto",
    optimizer_ordering="stage_interleaved",
    verbose=True,
    artifact_store_dir=None,
    profiling_metadata=None,
    allocation_probe_seeds=1,
    allocation_probe_repetitions=2,
    save_plan=True,
    force_fresh=False,
    overwrite_plan=False,
    implementation_revision=None,
) -> PlannedTrainStep
```

`plan_step()` accepts one fixed example sequence per accumulation round. The
`optimizer_ordering` value is `"stage_interleaved"` or `"tail"`.

Shared planning arguments have these meanings:

| Argument | Contract |
|---|---|
| `runtime` | Open runtime that owns the imported model. |
| `execution`, `spill` | Keys in `runtime.pools`. |
| `execution_budget`, `spill_budget` | Optional byte budgets no larger than configured pool limits; `None` means the pool's capacity. Budgets of at least one GiB plan at whole-GiB granularity (rounded down), so a budget that follows a pool's measured capacity gives the same plan identity in every process; calibrated bandwidths and latencies are rounded the same way, as [the plan report](../plan-report.md) describes. |
| `dynamic_scratch_reserve_bytes` | Optional lower bound for bounded dynamic scratch; cannot reduce the measured requirement. |
| `minimum_object_bytes_evict_eligible` | Objects smaller than this stay resident from their first to their last access instead of being evicted and fetched mid-step; their opening fetch, release after the last access, and terminal writeback are unchanged. Default 1 MiB, the size under which a copy is latency-bound; zero makes every object eligible. Part of the plan identity. |
| `execution_device` | Accelerator ordinal or `torch.device`; `None` uses the current PyTorch device. |
| `partition` | `"auto"`, `"whole"`, or `PartitionPolicy`. |
| `artifact_store_dir` | Shared content-addressed artifact root. |
| `profiling_metadata` | JSON-compatible identity for data-sensitive task measurement. |
| `allocation_probe_seeds` | Independent randomized activation probes per structural contract. |
| `allocation_probe_repetitions` | Identical repeats per probe seed. |
| `SharedInput` / `shared_input()` | Zero-copy binding of an existing runtime-owned `TensorRef` in `example_inputs`. |
| `shared_outputs` | Forward-output leaves retained as runtime-owned `TensorRef` values. `plan_forward` only. |
| `save_plan`, `force_fresh`, `overwrite_plan` | Artifact cache policy. |
| `implementation_revision` | Explicit implementation identity for compiler/profile invalidation. |

`make_step_program()` performs capture, compilation, profiling, and canonical
lowering but does not run PressureFit or leave an active callable.
Planning a saved `PressureFitProgram` needs no frontend, so
`pressurefit_program()` lives in
[`shadowspill.planner`](neutral.md) rather than here.

<!-- source-signature: src/shadowspill/pytorch/api.py:make_step_program -->
```text
make_step_program(
    model,
    *,
    objective,
    opt,
    example_inputs,
    runtime,
    execution,
    spill,
    execution_budget=None,
    spill_budget=None,
    dynamic_scratch_reserve_bytes=None,
    execution_device=None,
    partition="auto",
    optimizer_ordering="stage_interleaved",
    verbose=True,
    artifact_store_dir=None,
    profiling_metadata=None,
    allocation_probe_seeds=1,
    allocation_probe_repetitions=2,
    save_plan=True,
    force_fresh=False,
    overwrite_plan=False,
    implementation_revision=None,
) -> StepProgram
```


Budgets default to the selected runtime pool capacities and cannot exceed
them. `execution_device=None` uses PyTorch's current accelerator; an explicit
device must match the execution pool.

Planning diagnostics remain present when verbose output is disabled.

`plan_step_search()` plans every admitted split of one step's sequence total
into microbatches and accumulation rounds, under every requested budget
pair, and executes nothing. Each distinct geometry pays capture, profiling,
and lowering once — the artifact store deduplicates by structural digest —
and every geometry-budget point then runs the PressureFit search. The
returned `StepSearchReport` carries per-geometry build phase timings, one
`StepSearchPoint` per geometry and budget with its status, simulated makespan,
and `PlanSummary`, the bound-skipped geometries with reasons, and derived
winners per budget. Running a winner afterward is one warm `plan_step()`
call at the chosen geometry.

Failures are outcomes rather than errors. A point that proves infeasible or
exhausts its search budget carries that status while the search continues.
Because profiling runs a task's real kernels, the largest geometries can
exhaust the device before any plan exists; such a geometry reports every one
of its budgets `infeasible`, with the exhaustion as the point's error, and
contributes no build to the report because it produced no program. Any other
build failure is raised.

<!-- source-signature: src/shadowspill/pytorch/step_search.py:plan_step_search -->
```text
plan_step_search(
    model,
    *,
    objective,
    opt,
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
    options=None,
    minimum_object_bytes_evict_eligible=1 << 20,
    optimizer_ordering="stage_interleaved",
    artifact_store_dir=None,
    verbose=False,
    progress=None,
    force_fresh=False,
    implementation_revision=None,
) -> StepSearchReport
```

`example_microbatches(sequences, accumulation)` supplies example inputs for
one geometry; structure matters, values do not. `transfer_bandwidths`
overrides the calibration each step program embeds from the runtime.
`verbose=True` forwards each planning call's phase progress, and `progress`
receives one line per geometry and point boundary so a caller can tee a live
log.
Infeasible or search-exhausted points are reported outcomes, not raised
errors. `search_geometries()` is the underlying enumeration — every divisor
pair of the sequence total, largest microbatch first, with the bounds'
skips and reasons — and each `StepSearchGeometryBuild` records the shared
capture/profile/lowering wall behind one geometry with its per-phase
seconds.

## Inputs, objectives, and partitioning

`TensorSpec` is storage-free fixed tensor geometry for planning. It records
shape, dtype, optional stride, `requires_grad`, and layout.

An objective may return a scalar loss tensor or `ObjectiveResult`. A bare
tensor becomes the corresponding `StepResult.objectives` entry and has
`metrics=None`. `ObjectiveResult` explicitly names the differentiable `loss`
and arbitrary nondifferentiated `metrics`; each becomes the corresponding
entry in the two per-round `StepResult` tuples. ShadowSpill validates this
contract during capture rather than inferring a loss from model output names.

### ObjectiveResult

| Field | Contract |
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

<!-- source-signature: src/shadowspill/pytorch/callables.py:PlannedForward.__call__ -->
```text
PlannedForward(inputs, *, profiler_annotations=False) -> object
```

`PlannedForward` validates the fixed input signature and returns the model
output.

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
    runtime_trace=False,
    profiler_annotations=False,
) -> StepResult
```

<!-- source-signature: src/shadowspill/pytorch/callables.py:PlannedTrainStep.submit -->
```text
PlannedTrainStep.submit(
    inputs,
    *,
    runtime_trace=False,
    profiler_annotations=False,
) -> InvocationResult[StepResult]
```

`PlannedTrainStep` returns `StepResult`. Both callables expose `plan_report`,
`state_dict()`, `load_state_dict()`, `close()`, and context manager support.

Closing copies nothing, and it moves no weights. `import_model_state()` gave
the model's parameters storage in the spill pool, and that one storage holds
the updated weights throughout: a step both begins and ends with parameters
spill-resident, so each update is already there. Running a step points those
same `Parameter` objects at device memory; closing points them back.
`export_model_state()` is the separate call that copies the values into
ordinary CPU tensors.

Optimizer state has no equivalent home today. `plan_step()` builds the
optimizer from the factory it is given and imports its state into storage the
plan owns, and planning refuses an optimizer whose state the caller already
imported, so there is no caller-owned pool for it to be left in. Releasing the
plan therefore releases the state with it: a training callable's
`state_dict()` and `load_state_dict()` answer only while it is open, and both
raise afterwards rather than reporting an empty optimizer. Take the checkpoint
before closing, and resume from one with `load_state_dict()`, which writes the
values into the storage the plan already owns. Execution failure closes the
same way, and a failed step publishes no optimizer update in any case.

`state_dict()` returns an independent snapshot: every tensor is its own
compact host allocation outside the runtime pools, so it can be serialized
while training continues. The spill pool keeps the authoritative copy
throughout and is read in place, so the snapshot is normally the only copy of
the state outside the pool; an object whose pool copy is not current is read
into a buffer first and costs two until the snapshot is built. Even one copy
of optimizer state is, on a large model, the largest transient the frontend
asks for.

`submit()` performs the normal host dispatch and returns an
`InvocationResult` backed by one cold-created, timing-disabled completion
event. `InvocationResult.result()` and `wait()` synchronize exactly once and
return the public payload; `resolved` reports whether that explicit boundary
has been crossed. Different callables may have outstanding submissions at the
same time. A single callable accepts one outstanding submission because it
reuses one admitted physical layout and task-record set; resolve that result
before submitting the callable again. Callable recurrence and close wait only
for work owned by that plan, never for unrelated callables.

`StepResult` contains one detached scalar objective and the reconstructed
objective metrics for each accumulation round, the completed `step_number`,
and an optional `DiagnosticsHandle`. Tensor-valued metrics are detached;
static metric leaves preserve the captured pytree. `DiagnosticsHandle.result()`
and `DiagnosticsHandle.wait()` synchronously resolve the trace once; `resolved`
reports whether that has happened.

## Exceptions

The [errors, failures, and cleanup guide](../failures.md) explains propagation,
structured runtime evidence, automatic rollback, and ownership-safe teardown.
This section is the compact exported-type reference.

Planning exceptions preserve phase-specific meaning:

- `PlanningError` — base class for frontend planning failures.
- `CaptureError` — Export/AOT capture cannot represent the graph.
- `CompilationError` — a structural task cannot compile.
- `ProfilingError` — isolated measurement or allocation audit fails.
- `AdmissionError` — physical memory resources cannot be admitted.
- `PlanInfeasibleError` — no schedule satisfies declared constraints.
- `PlanSearchExhaustedError` — bounded search ends without a feasibility proof.
- `ObjectiveError` — the training objective violates its contract.
- `InputGuardError` — runtime inputs differ from the fixed template.

Compiler and profiling errors retain structural contract, task kind, and operator
problem when available. Runtime exceptions retain the first native failure and
task identity.
