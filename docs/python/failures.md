# Errors, failures, and cleanup

The detailed callable/store lifecycle below covers the model-based frontend
still used by some integrations. For saved-plan execution and its ownership
boundaries, see [Trainer and Executor](api/training.md). The saved-phase behavior
is summarized first.

## Saved phase failures

The [phase APIs](../architecture/offline-pipeline.md) separate preparation from
execution. Build and measurement use ordinary allocations, while planning uses
only saved artifacts. None of these phases constructs runtime pools.

- `build_forward`, `build_forward_backward` and `build_step` report unsupported
  capture as `CaptureError` and task compilation/finalization as `CompilationError`.
  A missing device implementation therefore fails during build, before profiling.
- `measure_program` reports failed task measurement as `ProfilingError`, with
  the task's structural identity, kind and operators. Provider failures, including
  out-of-memory errors, remain attached as the original cause.
- `plan_program` distinguishes infeasibility from an exhausted search. It does
  not modify the supplied profiles, so callers can retry with different budgets.
- A preparation failure does not publish a partial result. Provider resource
  scopes close on failure. Ordinary temporary tensors are released when their
  references, including any retained exception tracebacks, are released.

A valid build/profile/plan can follow a recoverable failure in the same process.
Execution still starts in a fresh worker through `Executor`; a failed device
context is not made reusable merely by catching an exception. Preparation does
not transfer ownership of the caller's model tensors to an execution pool.

## Model-based frontend lifecycle

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

The planning exceptions are framework-free and live in `shadowspill.errors`,
because the planner raises and catches them without importing a framework.
`InputGuardError` lives there too, though it is raised at a call rather than
during planning. The runtime exceptions come from `shadowspill.runtime`.

| Boundary | Exception | Meaning |
|---|---|---|
| Runtime construction or lifecycle | `RuntimeConfigurationError` | Pool configuration, allocator installation, calibration, ownership, or close is invalid. |
| Preparation | `PlanningError` | Base class for build, measurement and planning failures. |
| Capture | `CaptureError` | Export, AOTAutograd, partitioning, or semantic-contract extraction cannot represent the requested fixed graph. |
| One task in one phase | `TaskPhaseError` | Base class for the two failures that name the task they happened on. |
| Compilation | `CompilationError` | A captured structural task cannot be compiled into the required executable contract. |
| Profiling | `ProfilingError` | Isolated task timing, workspace measurement, or allocation-path validation fails. |
| Physical admission | `AdmissionError` | Runtime pools cannot admit the selected execution plan. |
| Plan feasibility | `PlanInfeasibleError` | No schedule satisfies a declared capacity or another planning constraint. |
| Bounded search | `PlanSearchExhaustedError` | Search ended without finding a plan or proving infeasibility. |
| A store in `require` mode | `LookupError` | The store lacks the artifact and the mode refuses to produce it. The message names what missed, for a plan the program, capacity and lanes the key was made of, and the modes that would allow producing it. |
| Objective capture | `ObjectiveError` | A training objective violates the scalar-loss and result contract. |
| Call input validation | `InputGuardError` | Runtime inputs differ from the fixed planning template. No task has run and no state was mutated. |
| Planned execution | `RuntimeExecutionError` | The runtime, allocator, worker, or a task-specific execution contract rejected the step. |

Everything from `CaptureError` down to `ObjectiveError` derives from
`PlanningError`, so a caller that only wants "planning did not produce a plan"
catches that one. Two nestings inside it are worth knowing: `CompilationError`
and `ProfilingError` share `TaskPhaseError`, and `PlanInfeasibleError` derives
from `AdmissionError`, so catching admission also catches infeasibility.
One admission failure is outside this table: `shadowspill.planner.admission.physical` raises `PhysicalAdmissionError`, a `ValueError`, when a budget's headroom leaves nothing to admit. That is the standalone helper documented on [the neutral API page](api/neutral.md#physical-admission); planning calls it and reports what it found as the `AdmissionError` above.

`InputGuardError` is deliberately outside the hierarchy — it is a `ValueError`
raised at a call, not during planning. So is the `require` store's refusal: a
plain `LookupError`, because nothing about planning failed; the run asked for a
store it was not allowed to fill.

A `TaskPhaseError` retains `structural_contract`, `task_kind`, and `operators`
when the failing task is known. `PlanInfeasibleError` retains the failure
`kind`, `device_id`, `boundary_task_id`, `required_bytes`, and
`capacity_bytes`; it and `PlanSearchExhaustedError` both retain `diagnostics`,
the search evidence behind the refusal. Both refusals are recorded in the
planning store under the key the plan would have had, and the next request for
that plan, made with no plan to beat in hand, raises the recorded refusal again
without searching; see [the artifact store](artifact-store.md#internal-content-addressed-store). The original PyTorch exception remains
the cause, so its traceback identifies the operator and model code that led to
a capture or compilation failure.

## Planning failures

The phases have independent ownership:

```text
build -> Program
measure_program -> Profiles
apply_profiles -> ProfiledProgram
plan_program -> Plan
Executor -> admitted runtime resources
```

Build and profiling use no runtime pools. CPU planning cannot leave a partially
admitted runtime behind. Each function either returns its complete artifact or
raises; an unsuccessful sweep case records its diagnostics and continues.
Executor construction creates and admits resources in its fresh worker and
cleans up partial setup on failure.

`PlanInfeasibleError` is a negative feasibility result. `PlanSearchExhaustedError`
means a bounded search did not establish a result. Neither returns a partial
executable Plan. Estimated memory permits simulation only, not execution.

## Runtime failure diagnostics

`RuntimeExecutionError.diagnostics` is either `None` or an immutable
`RuntimeFailureDiagnostics`. `Runtime.last_failure` retains the latest
structured frontend failure independently of the exception object's lifetime.
`RuntimeFailureDiagnostics.as_dict()` produces a JSON-compatible mapping for
logs and test artifacts.

The record is organized into five groups:

| Group | Representative fields |
|---|---|
| Failure identity | `operation`, `status`, `status_name`, `device_ordinal` |
| Why it refused | `reason`, `reason_name`, `refused_action`, `object_state` |
| Memory state | `requested_bytes`, `free_bytes`, `largest_free_range_bytes`, `object_id`, `allocation_id` |
| Task identity | `execution_task_id`, semantic name, canonical task ID, internal task ID |
| Allocation contract | task live bytes and limits, maximum-request limits, contract operation index, and expected/actual operation, ordinal, size, charge, and alignment |

The runtime status names are:

| Status | Interpretation |
|---|---|
| `invalid_argument` | A runtime API argument violated its contract. |
| `internal_failure` | A backend or pool allocation failed without a stronger classification. |
| `out_of_memory` | The requested physical memory is unavailable. |
| `no_progress` | No free range exists and no known pending transition can satisfy the request. |
| `invalid_state` | An object, lease, action, or lifecycle transition is invalid. |
| `plan_violation` | Real execution departed from an admitted plan invariant. |
| `backend_failure` | The selected device or transfer backend failed. |
| `worker_failure` | The C worker latched its first asynchronous failure. |
| `closed` | Work targeted an owner that has already closed. |
| `task_allocation_envelope_exceeded` | A request or task-local live total exceeded the admitted profiled envelope. |
| `task_allocation_contract_mismatch` | Allocation operation, order, geometry, or ownership differed from the admitted task contract. |

`RuntimeFailureDiagnostics.is_allocator_oom`, `.is_recoverable_no_progress`,
and `.is_shadowspill_contract_failure` classify one of those statuses without
requiring applications to parse exception text.

## No-progress OOM

A no-progress OOM means the allocator has neither a compatible free range nor
a causally pending retirement, eviction, or transfer that can make one
available. It is different from waiting briefly for known progress. Its
message is deliberately task-attributed:

```text
ShadowSpill out of memory, with nothing left to release
execution_task: execution_000017
semantic_task: microbatch_0000.stage_0017.backward.recompute
canonical_task: task_42
device: 0
requested: 112.00 MiB (117440512 bytes)
pool free: 38.00 MiB (39845888 bytes)
largest free range: 24.00 MiB (25165824 bytes)
```

The title's second clause is what separates the two cases: an exhaustion with
something still to release reads `ShadowSpill out of memory` alone. Sizes carry
the readable form and the exact count, because neither alone says whether a
request exceeded a budget. The requested, pool-free, and largest-contiguous-range
values distinguish capacity exhaustion from fragmentation. Object and allocation
ids are added when the failing request already has those identities. Where no
task is known, an `operation:` line names the callback instead of the three task
lines.

## Contract failures versus provider failures

ShadowSpill replaces a secondary provider error only when its own allocator or
execution contract already identified the cause. Examples are a no-progress
OOM, an allocation-envelope violation, and an allocation-contract mismatch. This
prevents a later invalid-address or null-pointer device error from hiding the
actionable failure.

An unrelated kernel, compiler, device-backend, or provider failure remains the primary
exception. ShadowSpill records cleanup problems as notes but does not relabel a
bad kernel as an allocator OOM.

## Execution rollback

A dispatch or completion failure makes Executor unusable for subsequent calls.
The original exception remains available from the result handle. This does not
roll back partially executed weight updates; restart from a known checkpoint.

`close()` attempts to release admitted paths, provider resources, completion
markers and distributed bindings before closing the runtime. A context manager
adds cleanup failures as exception notes instead of masking the original error.
Allocator failures are not silently retried with a relaxed budget.

## Normal close order

Wait for the invocation, export any required checkpoint, release caller-owned GPU
outputs and close Executor or Trainer. Closing copies no model tensors. The
runtime rejects shutdown while live outputs still own device storage; release
them and retry close. Completed outputs remain valid until explicitly released.

The allocator shim is process-global. After runtime teardown it stays closed,
so further GPU work belongs in another fresh worker. See the
[execution reference](api/training.md), [allocator guide](allocator.md) and
[memory runtime](../architecture/memory-runtime.md#failure-and-teardown).
