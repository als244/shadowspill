# Task boundaries

A task boundary connects the planned storage layout to one compiled task call.
`before_task` acquires inputs and opens an allocation scope; `after_task`
publishes results and mutations, closes the scope and submits planned actions.
The worker continues transfers and retirement while the frontend enqueues
compute.

This page describes that lifecycle. See [memory runtime](memory-runtime.md) for
pools and leases, [physical admission](physical-admission.md) for layout checks,
and the [Runtime C API](../c/runtime.md) for exact signatures.

## Where the boundary begins and ends

The neutral runtime returns addresses. The frontend binds them to its own tensor
storages and publishes the results. Boundary timing includes both layers:

```text
opening boundary
    validate input devices and resolve the compute stream
    shadowspill_before_task_handle(...) -> borrowed input bindings
    validate bindings and rebind frontend storages
    shadowspill_wait_task_allocations_handle(...)

compiled task: enqueue kernels and allocate through the active scope

closing boundary
    adopt result storage and rebind replacement views
    dematerialize released objects
    shadowspill_after_task_handle(...)
```

Every frontend action between tasks belongs to a boundary. The
[diagnostic timestamps](../python/advanced/step-diagnostics.md) cover the whole opening
or closing operation, rather than only the nested C call. If any operation
raises after the scope opens, the frontend must close or abort it.

## What each call does

### `shadowspill_before_task_handle`

1. Validate the runtime/task handle and claim the invocation. Concurrent use of
   the same task handle is refused. Record the opening trace event and establish
   fixed-layout task-start dependencies.
2. Check the runtime failure latch and acquire each distinct input: residency,
   version and lease must match the Plan.
3. Make the compute stream wait for in-flight fetches' readiness events, take
   owner references and return bindings.
4. Open the thread-local task scope.

Bindings borrow the task record's array. Do not free them or keep them past
`after_task`. A worker failure prevents the next object acquisition and therefore
stops new task submission.

### `shadowspill_wait_task_allocations_handle`

Inputs have been acquired, but future workspace allocations may reuse ranges
still held by a transfer or earlier task. This call establishes those dependencies
before the first kernel:

- For a transfer, wait on the host until the worker has submitted it and
  published a completion event, then insert a compute-stream wait on that event.
- For an earlier task, insert a stream wait on its published completion event.

The host rendezvous waits for event publication, not copy completion. It can
still limit frontend run-ahead: a transfer may be submitted only after its
trigger task finishes. `before_task_entered_at_seconds` through
`before_task_exited_at_seconds` includes the host wait;
`allocation_reuse_wait_seconds` measures the subsequent compute-stream wait.
These are different intervals and should not be added as independent stalls.

Establishing all range dependencies here can delay the first kernel for a range
used only later in the task. It also makes that wait observable at the boundary;
the allocation itself finds its dependency already published.

### `shadowspill_after_task_handle`

1. Check the latched status and completed allocation sequence.
2. Publish mutations by advancing authoritative object versions.
3. When actions, retirements or range successors need a fence, record task
   completion on the compute stream and attach it to retired leases.
4. Reserve action destinations, instantiate the action batch against current
   object state, publish it and wake the worker.
5. Wait until the worker has issued this batch's fetches, then leave the scope.

Completion events mean the preceding kernels have been enqueued, not that they
have finished. On failure, discard unpublished actions, clear pending handoffs
and latch the error. Retirement still receives a fence so released allocations
can be reclaimed safely. The scope is closed on either path.

### `shadowspill_abort_task_handle`

Abort closes an opened scope when frontend rebinding, argument assembly, kernel
launch, result publication or interruption prevents normal completion. It is
valid only from the thread inside the task. It finalizes retirements and clears
handoffs without publishing mutations or actions.

Propagating an exception without cleanup would leave the invocation active and
leases awaiting a completion event that will never arrive. Abort fixes that
incomplete lifecycle; it does not clear a latched runtime failure or make the
failed task's outputs valid.

## Asynchronous work and blocking points

At return from `after_task`:

| Work | State |
|---|---|
| Task kernels | Enqueued, possibly still running |
| Triggered fetches | Issued; their readiness events are available |
| Triggered evictions/releases | May still be waiting for the worker |
| Leases freed during the task | Retiring against its completion event |
| Returning retired capacity | Worker responsibility after completion |

Fetch publication is the closing boundary's worker handshake. Opening a task
can also wait for allocation-range event publication, and capacity reservations
can wait for pending releases. Thus asynchronous execution does not mean every
host boundary is nonblocking. `shadowspill_plan_wait_idle` drains one Plan;
`shadowspill_runtime_wait_idle` drains the runtime.

| Responsibility | Dispatching thread | Worker |
|---|---|---|
| Task scope | Enters, allocates, closes/aborts | Never enters |
| Input acquisition | Validates and inserts stream waits | Does not acquire task inputs |
| Actions | Reserves destinations and publishes batch | Issues copies and publishes readiness |
| Retirement | Links freed leases to completion fence | Polls completion and returns ranges |
| Failure | Latches errors and refuses new work | Latches errors seen by the next boundary |

The dispatcher does not take over retirement when capacity is unavailable. It
waits for the worker's capacity epoch to advance or for a no-progress result.
The fetch handshake and Plan idle wait use active polling on the critical path;
the runtime-wide idle wait uses a condition variable.

## Allocation scopes and validation

A thread-local task scope identifies the invocation, Plan, pool, allocation
ordinal and freed leases. Framework allocator callbacks use it without an
explicit task ID. Two threads may run different task handles; one handle cannot
host concurrent invocations.

Each request is checked against the measured envelope and allocation sequence.
Planned allocations use fixed offsets. Within an uninterrupted group of frees,
order may differ if the same identities and geometry are released exactly once.
Frees cannot move across an allocation, and unresolved frees block both the next
invariant allocation and task completion.

Work outside a task can use an explicit allocation scope, as isolated profiling
does before a Plan exists. The caller reserves a
[Plan identity](plan-identity.md#the-id) and associates the scope with it.
Unscoped allocations instead take the dynamic, unattributed path. Failed scopes
are aborted rather than published.

### What a scope owes when it ends

Every scoped lease records its originating Plan and scope. Closing the scope
accounts for all three outcomes:

| Outcome | Ownership after close |
|---|---|
| Freed during the scope | Retired against its completion fence; reusable only after safe completion |
| Explicitly promoted | Held by the caller, such as a returned objective value |
| Retained by the framework/library | Still attributed to the originating scope until its owner releases it |

Scope closure does not revoke a live frontend value. Audits report retained
allocations' offsets, sizes and originating scopes so unexpected retention is
diagnosable.
Compiled libraries may retain state, but that state must have explicit accounting
and placement. A provider's own stateless interface may impose stricter rules.

### What actually releases a lease

For a PyTorch tensor, the release chain is:

```text
last owning reference disappears
    -> tensor storage frees its address
    -> pluggable allocator maps address to allocation
    -> shadowspill_memory_pool_free releases or defers the lease
    -> worker returns the range after its completion fence
```

Reference cycles can delay this chain until collection. A scope cannot forcibly
reclaim storage still reachable through a tensor: a later use would access freed
memory, and a later free could release a newly assigned allocation at the same
address. Forced release is only an explicit teardown escape hatch.

Small retained allocations can split a large free range and prevent a later
contiguous reservation despite ample aggregate free bytes. Attribute them and
place expected long-lived state before the reservation; do not free it behind
its owner's back. [Causal reuse](memory-runtime.md#causal-reuse) governs when a
released range becomes available.

## Planned actions and capacity

Fetch, eviction, writeback and release actions are instantiated at the closing
boundary of their trigger task. Destination leases are reserved before the batch
is published. If pending releases can make room, the reservation waits and
retries with its priority preserved; the simulator models the same delay.
`after_task` returns `NO_PROGRESS` when the destination cannot fit even after all
pending releases. The worker cannot repair an impossible destination later.

## Frontend storage bindings

CPU-valued outputs, controls and literals remain ordinary CPU bindings. Their
device object/alias extents are zero, and mixed-device results transfer ownership
only for unique device roots. CPU values receive neither device allocation
ordinals nor automatic spill-pool placement. Training republishes CPU root
bindings each step, including Export-lifted literals absent from the original
module.

Captured location describes where a value is used, not where a template was
initialized. An opaque optimizer can start with CPU/meta templates while its
spillable state has fake device geometry; CPU counters remain CPU bindings.
[Profiling](../reference/engine/profiling.rst) materializes authentic
control producers on each value's captured device. Generated representatives
for proven arithmetic-only statistics do not change runtime values.

A backward result unused by the outer program still occupies storage during
its call. Lowering accounts for that root as transient workspace, except when
it shares a root already charged to a retained result. Activation-gradient
layout conversions occur inside the compiled backward so their cost is measured;
see [cotangent layouts](graph-pair-construction.md#what-a-cotangent-is-laid-out-like).

### Retained views after state mutation

A functionalized mutation replaces an object's device allocation. Retained
frontend views share its canonical PyTorch storage handle while keeping their
own shapes, strides and offsets. Fetch, release and checkpoint restore can then
rebind every view through one handle. Separate storage handles holding the same
pointer would leave some views stale after the representative was rebound.
Unifying handles does not copy device data.

## Failure

Both boundaries return `ShadowSpillStatus`. A failure latched on either thread
reaches a subsequent boundary; `shadowspill_runtime_failure()` preserves its
first cause separately from the current call's status. Always finish cleanup
through the closing or abort path after a scope has opened.

See [failure, abort and process exit](failure-and-exit.md) for scope cleanup and
why a terminating process may abandon the runtime rather than close it.

---

Previous: [Events](events.md) ·
[Architecture index](README.md) ·
Next: [Step boundaries](step-boundaries.md)
