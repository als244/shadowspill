# Errors, failures, and cleanup

Build, measurement, planning and execution have separate ownership and failure
boundaries. See [Trainer and Executor](api/training.md) for normal usage.

## Preparation failures

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

## Exception taxonomy

The planning exceptions are framework-free and live in `shadowspill.errors`,
because the planner raises and catches them without importing a framework.
`InputGuardError` lives there too, though it is raised at a call rather than
during planning. The runtime exceptions come from `shadowspill.runtime`, except
`RuntimeInstallError`, which `shadowspill.runtime.bootstrap` defines.

| Boundary | Exception | Meaning |
|---|---|---|
| Runtime construction or lifecycle | `RuntimeConfigurationError` | Pool configuration, calibration, ownership, or close is invalid. |
| Allocator installation | `RuntimeInstallError` | The process allocator or the framework adapter cannot be installed, or, with `reject_overbudget=True`, the bootstrapped process already exceeds its device budget. |
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
| Allocation inside a task | `torch.OutOfMemoryError`, or `RuntimeError` | ShadowSpill's allocator refused a request inside the provider's allocation call: an out-of-memory refusal raises the first, any other refusal the second. The message is ShadowSpill's diagnosis. |
| Planned execution | `RuntimeExecutionError` | The runtime, worker, or a task-specific execution contract rejected a runtime call. |

`PlanningError` covers capture, compilation, measurement, admission, search
and objective failures. `CompilationError` and `ProfilingError` share
`TaskPhaseError`; `PlanInfeasibleError` derives from `AdmissionError`.
Runtime construction, input validation, store misses and execution failures are
outside this hierarchy.
The low-level `shadowspill.runtime.admission_replay` helper raises
`PhysicalAdmissionError`, a `ValueError` with the failed operation's capacity,
free-range and live-lease evidence. It directly probes the production memory-pool
policy; see [the neutral API page](api/neutral.md#physical-admission).

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
without searching, rebuilt from its message and kind alone; see [the artifact store](artifact-store.md#internal-content-addressed-store). Capture and task-phase failures preserve the original provider exception as
the cause, so its traceback identifies the failing operation.

## Planning and worker admission

Each phase returns its complete artifact or raises. An unsuccessful sweep case
records its diagnostics and continues. CPU planning admits no runtime resources;
Executor construction creates and admits them in a fresh worker and cleans up
partial setup on failure. Memory the process acquires outside the pool beyond
the planned headroom is reported on stderr and execution continues; with
`reject_overbudget=True` on Executor or Trainer it is refused, as
`RuntimeInstallError` while the runtime is installed and `AdmissionError`
afterwards. Pool bounds and device allocation failures are enforced either way.
Distributed workers whose Programs disagree on what they exchange, or whose
Plans would issue their collectives in different orders, are refused with a
`ValueError` before any collective runs; see
[fresh execution workers](api/distributed.md#fresh-execution-workers).

`PlanInfeasibleError` is a negative feasibility result. `PlanSearchExhaustedError`
means a bounded search did not establish a result. Neither returns a partial
executable Plan. Estimated memory permits simulation only, not execution.

## Runtime failure diagnostics

`RuntimeExecutionError.diagnostics` is either `None` or an immutable
`RuntimeFailureDiagnostics`, the runtime's first latched failure read when a
runtime call is refused.
`RuntimeFailureDiagnostics.as_dict()` produces a JSON-compatible mapping for
logs and test artifacts.

The record is organized into five groups:

| Group | Representative fields |
|---|---|
| Failure identity | `operation`, `status`, `status_name`, `device_ordinal` |
| Why it refused | `reason`, `reason_name`, `refused_action`, `object_state` |
| Memory state | `pool_id`, `requested_bytes`, `free_bytes`, `largest_free_range_bytes`, `object_id`, `allocation_id` |
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
available. It is different from waiting briefly for known progress. The
allocator raises it as the provider's out-of-memory exception, from inside the
allocation call, with a task-attributed message:

```text
ShadowSpill out of memory in the execution pool (device 0), with nothing left to release
reason: no range is large enough and nothing remains to release for one
execution_task: example_0000
semantic_task: stage_0017.recompute.backward
canonical_task: task_000042
requested: 112.00 MiB (117440512 bytes)
pool free: 38.00 MiB (39845888 bytes)
largest free range: 24.00 MiB (25165824 bytes)
```

The title names the pool and the device, and its last clause is what separates
the two cases: an exhaustion with something still to release ends at the
device. A `reason:` line follows when the runtime latched one, and a `note:`
line when an earlier failure had already stopped the runtime. Sizes carry
the readable form and the exact count, because neither alone says whether a
request exceeded a budget. The requested, pool-free, and largest-contiguous-range
values distinguish capacity exhaustion from fragmentation; the two pool lines
are left out when the runtime recorded neither. Where no task is known, the
task lines are absent, or one line names the runtime scope that made the
request.

## Contract failures versus provider failures

ShadowSpill's allocator refuses a request inside the provider's allocation
call, so no compiled code receives a null pointer. A no-progress OOM raises the
provider's out-of-memory exception; an allocation-envelope violation or an
allocation-contract mismatch raises its generic error. Either carries
ShadowSpill's diagnosis as its message, so a later invalid-address or
null-pointer device error cannot hide the actionable failure.

An unrelated kernel, compiler, device-backend, or provider failure remains the primary
exception. ShadowSpill records cleanup problems as notes but does not relabel a
bad kernel as an allocator OOM.

## Execution rollback

A dispatch or completion failure makes Executor unusable for subsequent calls,
which raise with the original exception as their cause. A completion failure
also stays on the result handle, which raises it again. This does not
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
