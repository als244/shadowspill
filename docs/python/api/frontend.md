# Frontend and lifecycle API

The symbols on this page are exported by `shadowspill.memory` or
`shadowspill.pytorch`.

## Memory pool configuration

| Symbol | Purpose |
|---|---|
| `DevicePool` | Immutable execution-device pool configuration. |
| `PinnedHostPool` | Immutable registered pinned-host spill-pool configuration. |
| `MemoryPoolConfig` | Union of supported pool configurations. |
| `device()` | Construct a `DevicePool` from `physical_capacity`, device ordinal, and optional provider headroom. |
| `pinned_host()` | Construct a `PinnedHostPool` from a byte capacity. |

Provider headroom is inside `DevicePool.physical_capacity`. The runtime reports
the derived suballocatable capacity after initialization.

## Runtime

<!-- source-signature: src/shadowspill/pytorch/runtime_adapter/runtime.py:Runtime.__init__ -->
```text
Runtime(
    *,
    pools: Mapping[str, MemoryPoolConfig],
    library_path: str | Path | None = None,
    calibrate: bool = True,
    worker_poll_nanoseconds: int = 1_000,
)
```

`Runtime` owns the installed allocator, initialized `MemoryPool` registry,
transfer calibration, active callable count, persistent state registry, and
latest failure. Public properties are `pools`, `transfer_capabilities`, and
`last_failure`.

`calibrate_transfer_capabilities()` measures all or selected source/destination
routes and atomically publishes a new `TransferCapabilities` matrix. Callers
may coordinate several processes and invoke calibration concurrently to
measure contended links.

`Runtime.close()` rejects new frontend work after verifying that no planning,
callable, or persistent imported state remains. The installed process
allocator itself remains process-owned because PyTorch cannot uninstall it.

The immutable runtime values are:

- `MemoryPool`
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

<!-- source-signature: src/shadowspill/pytorch/state/model.py:export_model_state -->
```text
export_model_state(model, *, runtime, release_runtime=False)
```

<!-- source-signature: src/shadowspill/pytorch/state/optimizer.py:import_optimizer_state -->
```text
import_optimizer_state(optimizer, *, runtime, pool, release_source=True)
```

<!-- source-signature: src/shadowspill/pytorch/state/optimizer.py:export_optimizer_state -->
```text
export_optimizer_state(optimizer, *, runtime, release_runtime=False)
```

`import_model_state()` returns a copied module hierarchy whose registered
tensors point at runtime spill leases. `release_source=True` means ShadowSpill
does not retain the input model; Python releases it when no caller reference
remains. `export_model_state()` rebinds the same registered tensor
identities to ordinary CPU storages and optionally releases runtime objects.

`import_optimizer_state()` and `export_optimizer_state()` apply the same
storage policy to already materialized optimizer state. `plan_step()` normally
constructs and manages its optimizer from the supplied factory, so direct
optimizer import is an advanced lifecycle operation.

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
    execution_device=None,
    partition="auto",
    verbose=True,
    planning_cachedir=None,
    profiling_metadata=None,
    allocation_probe_seeds=1,
    allocation_probe_repetitions=2,
    save_plan=True,
    force_fresh=False,
    overwrite_plan=False,
    implementation_revision=None,
) -> PlannedForward
```

`plan_forward()` accepts one fixed example-input sequence and returns
`PlannedForward`.

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
    execution_device=None,
    partition="auto",
    optimizer_ordering="stage_interleaved",
    verbose=True,
    planning_cachedir=None,
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
| `execution_budget`, `spill_budget` | Optional byte budgets no larger than configured pool limits. |
| `dynamic_scratch_reserve_bytes` | Optional lower bound for bounded dynamic scratch; cannot reduce the measured requirement. |
| `execution_device` | Accelerator ordinal or `torch.device`; `None` uses the current PyTorch device. |
| `partition` | `"auto"`, `"whole"`, or `PartitionPolicy`. |
| `planning_cachedir` | Shared content-addressed artifact root. |
| `profiling_metadata` | JSON-compatible identity for data-sensitive task measurement. |
| `allocation_probe_seeds` | Independent randomized activation probes per structural ABI. |
| `allocation_probe_repetitions` | Identical repeats per probe seed. |
| `save_plan`, `force_fresh`, `overwrite_plan` | Artifact cache policy. |
| `implementation_revision` | Explicit implementation identity for compiler/profile invalidation. |

`make_step_program()` performs capture, compilation, profiling, and canonical
lowering but does not run PressureFit or leave an active callable.
`pressurefit_program()` independently selects and physically admits a saved
`PressureFitProgram` under requested budgets and `TransferBandwidths`.

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
    planning_cachedir=None,
    profiling_metadata=None,
    allocation_probe_seeds=1,
    allocation_probe_repetitions=2,
    save_plan=True,
    force_fresh=False,
    overwrite_plan=False,
    implementation_revision=None,
) -> StepProgram
```

<!-- source-signature: src/shadowspill/pytorch/api.py:pressurefit_program -->
```text
pressurefit_program(
    program,
    *,
    execution_budget=None,
    spill_budget=None,
    transfer_bandwidths=None,
    options=None,
    planning_cachedir=None,
    verbose=True,
    save_plan=True,
    force_fresh=False,
    overwrite_plan=False,
    implementation_revision=None,
) -> AnnotatedProgramPlan
```

Budgets default to the selected runtime pool capacities and cannot exceed
them. `execution_device=None` uses PyTorch's current accelerator; an explicit
device must match the execution pool.

`verbose=True` prints phase progress. Planning diagnostics remain present when
verbose output is disabled.

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

<!-- source-signature: src/shadowspill/pytorch/callables.py:PlannedTrainStep.__call__ -->
```text
PlannedTrainStep(
    inputs,
    *,
    runtime_trace=False,
    profiler_annotations=False,
) -> StepResult
```

`PlannedTrainStep` returns `StepResult`. Both callables expose `plan_report`,
`state_dict()`, `load_state_dict()`, `close()`, and context manager support.

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

Compiler and profiling errors retain structural ABI, task kind, and operator
context when available. Runtime exceptions retain the first native failure and
task identity.
