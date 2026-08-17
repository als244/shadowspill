# Errors, failures, and cleanup

ShadowSpill fails closed at the boundary that can still explain the problem.
Capture, compilation, profiling, planning, admission, input validation, and
execution therefore have distinct exception types. A failed allocator request
is never returned to a kernel as a null pointer.

Normal applications do not need an exception wrapper merely to keep
ShadowSpill safe. Planning rolls back partial state before it raises, and a
planned callable closes itself when execution fails. Applications may still
catch an exception when they have an operational recovery policy, but cleanup
does not depend on doing so.

## Exception taxonomy

All public exceptions below are exported by `shadowspill.pytorch`.

| Boundary | Exception | Meaning |
|---|---|---|
| Runtime construction or lifecycle | `RuntimeConfigurationError` | Pool configuration, allocator installation, calibration, ownership, or close is invalid. |
| Planning | `PlanningError` | Base class for failures before a callable is published. |
| Capture | `CaptureError` | Export, AOTAutograd, partitioning, or semantic-contract extraction cannot represent the requested fixed graph. |
| Compilation | `CompilationError` | A captured structural task cannot be compiled into the required executable ABI. |
| Profiling | `ProfilingError` | Isolated task timing, workspace measurement, or allocation-path validation fails. |
| Physical admission | `AdmissionError` | Runtime pools cannot admit the selected execution plan. |
| Plan feasibility | `PlanInfeasibleError` | No schedule satisfies a declared capacity or another planning constraint. |
| Bounded search | `PlanSearchExhaustedError` | Search ended without finding a plan or proving infeasibility. |
| Objective capture | `ObjectiveError` | A training objective violates the scalar-loss and result contract. |
| Call input validation | `InputGuardError` | Runtime inputs differ from the fixed planning template. No task has run and no state was mutated. |
| Planned execution | `RuntimeExecutionError` | The runtime, allocator, worker, or a task-specific execution contract rejected the step. |

`CompilationError` and `ProfilingError` retain `structural_abi`, `task_kind`,
and `operators` when that context is available. `PlanInfeasibleError` retains
the failure `kind`, device, boundary task, required bytes, and capacity bytes.
The original PyTorch exception remains the cause, so its traceback identifies
the operator and model code that led to a capture or compilation failure.

## Planning failures

The public planning functions classify a failure according to the phase that
owned the rejected contract:

```text
runtime resolution
  -> capture and partition
  -> structural compilation
  -> isolated profiling
  -> Program construction and PressureFit
  -> physical admission
  -> callable publication
```

If any phase fails, `plan_step()`, `plan_forward()`, or `make_step_program()`:

1. retains the original exception and traceback;
2. records any allocator failure already latched by the C adapter;
3. aborts a partially admitted runtime plan;
4. restores persistent object identities; and
5. attaches independent cleanup failures as exception notes instead of
   replacing the primary cause.

A planning failure therefore never returns a partial callable. The same
`Runtime` can be reused only when rollback succeeds and the runtime has not
marked itself unusable.

`PlanInfeasibleError` and `PlanSearchExhaustedError` are intentionally
different. The former is a negative feasibility result; the latter means the
configured bounded search did not establish either result.

## Runtime failure diagnostics

`RuntimeExecutionError.diagnostics` is either `None` or an immutable
`RuntimeFailureDiagnostics`. `Runtime.last_failure` retains the latest
structured frontend failure independently of the exception object's lifetime.
`RuntimeFailureDiagnostics.as_dict()` produces a JSON-compatible mapping for
logs and test artifacts.

The record is organized into four groups:

| Group | Representative fields |
|---|---|
| Failure identity | `operation`, `status`, `status_name`, `device_ordinal` |
| Memory state | `requested_bytes`, `free_bytes`, `largest_free_range_bytes`, `object_id`, `allocation_id` |
| Task identity | `execution_task_id`, semantic name, canonical task ID, internal task ID |
| Allocation contract | task live bytes and limits, maximum-request limits, ABI operation index, and expected/actual operation, ordinal, size, charge, and alignment |

The runtime status names are:

| Status | Interpretation |
|---|---|
| `invalid_argument` | A runtime API argument violated its contract. |
| `allocation_failure` | A backend or pool allocation failed without a stronger classification. |
| `out_of_memory` | The requested physical memory is unavailable. |
| `no_progress` | No free range exists and no known pending transition can satisfy the request. |
| `invalid_state` | An object, lease, action, or lifecycle transition is invalid. |
| `plan_violation` | Real execution departed from an admitted plan invariant. |
| `backend_failure` | The selected device or transfer backend failed. |
| `worker_failure` | The C worker latched its first asynchronous failure. |
| `closed` | Work targeted an owner that has already closed. |
| `task_allocation_envelope_exceeded` | A request or task-local live total exceeded the admitted profiled envelope. |
| `task_allocation_abi_mismatch` | Allocation operation, order, geometry, or ownership differed from the admitted task ABI. |

`is_allocator_oom`, `is_recoverable_no_progress`, and
`is_shadowspill_contract_failure` provide stable classifications without
requiring applications to parse exception text.

## No-progress OOM

A no-progress OOM means the allocator has neither a compatible free range nor
a causally pending retirement, eviction, or transfer that can make one
available. It is different from waiting briefly for known progress. Its
message is deliberately task-attributed:

```text
ShadowSpill no-progress OOM
execution_task: execution_000017
semantic_task: microbatch_0000.stage_0017.backward.recompute
canonical_task: task_42
device: 0
requested: 117440512
free: 39845888
largest_free_range: 25165824
```

The requested, total-free, and largest-contiguous-range values distinguish
capacity exhaustion from fragmentation. Object and allocation IDs are added
when the failing request already has those identities.

## Contract failures versus provider failures

ShadowSpill replaces a secondary provider error only when its own allocator or
execution contract already identified the cause. Examples are a no-progress
OOM, an allocation-envelope violation, and an allocation-ABI mismatch. This
prevents a later invalid-address or null-pointer device error from hiding the
actionable failure.

An unrelated kernel, compiler, device-backend, or provider failure remains the primary
exception. ShadowSpill records cleanup problems as notes but does not relabel a
bad kernel as an allocator OOM.

## Execution rollback

An exception from `PlannedForward` or `PlannedTrainStep` triggers callable
cleanup before the exception reaches the caller. Cleanup:

- records and, when safe, synchronizes the first native failure;
- resolves or cancels pending diagnostics;
- stops profiler annotations;
- clears transient gradients for training;
- restores optimizer and model bindings to their pre-plan ownership state;
- releases the compiled executor and admitted runtime plan; and
- restores persistent object identities.

Every independent cleanup action is attempted. A cleanup failure is appended
to the original exception as a note; it does not mask the task, allocator, or
kernel failure that caused cleanup.

No-progress recovery exists only to make synchronized teardown possible. It
does not silently retry the failed task, relax a budget, or publish a partial
optimizer update.

## Normal close order

Successful workflows close owners in reverse ownership order:

```text
planned callable
  -> export imported state with release_runtime=True
  -> Runtime
```

`Runtime.close()` rejects an active callable, an in-progress plan, or
persistent imported state. This makes ownership leaks visible. The selected
PyTorch allocator is process-global and cannot be uninstalled, so the neutral
C runtime remains process-owned until registered exit cleanup stops and joins
the worker and closes each memory-pool backend. Explicit close is still the
supported path because it provides timely validation and deterministic release
of pinned and device memory.

See the [frontend API](api/frontend.md) for exported exception classes, the
[allocator guide](allocator.md) for callback behavior, and the [memory runtime
architecture](../architecture/memory-runtime.md) for native ownership and
teardown.
