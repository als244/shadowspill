# Memory runtime

The runtime is framework-neutral. It owns bounded memory pools,
logical objects, physical leases, transfer lanes, completion events, tracing,
and failure propagation. The PyTorch adapter translates allocator callbacks
and storage operations into this contract.

## Pools, budgets, and leases

`MemoryPool` is a generic range owner registered by identity with the runtime,
and directed transfer routes are registered separately; how pools get their
arenas and routes their lanes is in [memory pools](memory-pools.md), and why
runtime construction precedes workload-state construction is in [its
construction order](memory-pools.md#construction-order). Each admitted plan
selects its execution pool, spill pool, fetch route, and evict route, so the
roles are the plan's rather than the runtime's. The PyTorch adapter registers
one device pool and any number of pinned-host pools.

A `DevicePool`'s `physical_capacity` is the complete process-attributable
accelerator cap. Provider headroom and the driver's own baseline lie inside
that cap, and the runtime reports the suballocatable pool capacity it derives
after initialization. Planning budgets may reduce configured capacities but
cannot exceed them.

A `MemoryLease` owns one range for one residency generation. Objects keep a
lease per pool location; aliases and views share the same object and lease.
Generations prevent stale events, frees, bindings, or worker completions from
modifying a successor.

Runtime-global shared leases are physically charged once and retained outside
any one callable's movable-object schedule; which mutations and orderings each
shared-residency policy permits is in [the program](program.md).

Every callable uses its own plan-local alias IDs and fixed-layout slice. A
shared-input binding maps one of those local aliases to an existing
runtime-global object handle. Physical admission validates that the alias is
externally resident but assigns it no plan-owned offset. Closing either plan
releases only that plan's ownership; the object remains until its final plan
or public reference closes. Recurrent producers preserve the logical object
and update its current residency generation in place, so already-admitted
consumers keep the same object binding. The predecessor lease retires behind
its completion dependency; replacing a generation never copies the value just
to preserve frontend identity.

Lease states have one meaning across execution and spill pools:

| State | Meaning |
|---|---|
| `FREE` | The record owns no pool range. |
| `IN_USE` | An object, task allocation, or caller actively owns the range. |
| `RETIRE_PENDING` | Logical ownership ended; a completion dependency still protects the bytes. |
| `RESERVED` | An action owns immediately allocated destination capacity but has not acquired it for use. |
| `SUCCESSOR_RESERVED` | A successor owns a pending claim on a retiring predecessor's complete charged range. |
| `PREDECESSOR_TRANSFERRED` | Range ownership moved atomically to the successor; the detached predecessor record awaits release. |

## Fixed layout with bounded dynamic allocation

An admitted plan uses one complete step-level physical layout for
schedule-managed allocations:

- initial object generations;
- strict task-allocation contract core slots;
- persistent outputs and mutation replacements;
- fetch and evict destinations.

Offsets are relative to the callable's admitted slice of the runtime pool.
The layout models actual allocation/free geometry and causal overlap; it does
not collapse a task into one synthetic workspace block. Runtime callbacks
validate the allocation contract before returning a planned range.

Two cases remain dynamic by design:

- bounded optional anonymous/provider allocations use the admitted dynamic
  scratch reserve;
- terminal caller-owned outputs use dynamic leases so they may outlive a
  later callable invocation.

The scratch reserve is derived from profiling. A user may raise it with
`dynamic_scratch_reserve_bytes`, but cannot reduce the measured requirement.
Runtime fixed-service headroom, provider/problem headroom, task workspace, and
dynamic scratch are distinct accounting categories.

The full admission formulation, placement algorithm, offset coordinate
systems, causal certificate, capacity refinement, and report fields are in
[Physical admission and offset handling](physical-admission.md). This page
focuses on how the runtime consumes that certificate.

## Causal reuse

A released range is not reusable merely because its logical owner is done.
The backend stream that last used it must establish completion. A successor
can reserve a pending range without blocking the host only when its consuming
stream can wait on the predecessor event before accessing the address.

Completion processing atomically advances the predecessor and any reserved
successor generation. Once completion is known, the predecessor is removed
from pending ownership rather than remaining in a special "complete but not
free" state.

## Transfers

What happens between an action's trigger and its completion -- routes and their
lanes, destination reservation, the two queues a lane serves, and how a
write-back differs from an eviction -- is in [transfers](transfers.md#dispatch).

One consequence belongs here, because it is about ranges rather than copies: a
release scheduled behind a pending write-back of the same object does not
retire its source at the trigger, since the copy is still reading it. The
worker retires and frees the range when it reaches the release, after the copy
has landed, which is when the simulator frees it too.

## Worker

One C-owned worker services completions, releases, and both transfer lanes.
It is named `shadowspill_worker` in profiler traces, and `shadowspill.wkr` at
the OS level, where the name is shorter. The hot loop visits each completion
frontier, drains immediately completed FIFO successors, handles retirements,
and dispatches queued actions. A queued transfer is dispatched when it is the
head of its lane queue and its preconditions hold. The default incomplete-head
query cadence is `worker_poll_nanoseconds`, one microsecond; an
already-complete head is followed immediately without an artificial delay.

Steady-state execution performs no host allocation and creates or destroys no
backend event, because cold plan adoption reserves every inventory the hot path
draws on: event leases with their backend events ([events](events.md)),
retirement queue entries, `MemoryLease` records, and lease-use records. A later
callable sharing the runtime may grow these only at the same cold boundary.

Two of those inventories exist for reasons worth naming. Adoption sizes one
pool-owned release-frontier workspace from the sealed lease inventory, which
dry-runs pending-range coalescing inside a bounded borrowed range-node arena,
so destination reservation never builds a heap array or clones heap-owned range
nodes while holding the pool. And a lease-use record names one distinct stream
while its lease is live, so an asynchronous free records the completion event
into that same record and hands the immutable list to the retirement queue:
there is no stream snapshot and no copied event-wrapper list. Generation-tagged
leases return records and handles to their respective owners, and the worker
queries only FIFO heads and calls the backend outside data-structure locks.

## Dispatcher, streams, and worker timeline

```text
Python dispatcher       compute stream         C worker          transfer lane
       |                      |                     |                    |
       | before_task()        |                     |                    |
       | acquire generation   |                     |                    |
       |--------------------->| wait(readiness)     |                    |
       | launch callable      | queued kernels      |                    |
       | after_task()         |                     |                    |
       |--------------------->| record task fence   |                    |
       | reserve destination  |                     |                    |
       | queue action ---------------------------->|                    |
       | wait submission ack  |                     |------------------->| submit copy
       |<--------------------------------------------| ack submitted      |
       | return / next task   |                     |                    |
       |                      |                     | query FIFO heads   | record event
       |                      |                     | publish generation | completion
       | next before_task()   |                     |                    |
       |--------------------->| wait(copy event)    |                    |
```

`before_task()` inserts a device-stream dependency; it does not synchronize
the Python thread on ordinary readiness. `after_task()` reserves transfer
capacity, publishes its predecoded action batch, and spins only until the
worker acknowledges that every fetch in the batch has been issued and carries
a readiness event. It never waits for copy completion, and never for the
batch's evictions or releases.

A dispatcher allocation or destination reservation that cannot be served waits
only while the pool still has a release source -- a pending retirement or a
queued capacity action. It leaves the wait as soon as the pool's monotonic
capacity epoch moves, which is what says capacity was actually returned, and
gives up the moment nothing is left to wait for, which is what turns a pool
that can never satisfy the request into a no-progress failure rather than a
spin. Neither the waiter nor the worker sleeps.

## Task boundaries

Task boundaries have a page of their own: [task
boundaries](task-boundaries.md) covers what `before_task` and `after_task` are
each responsible for, how allocations find their task, which of a plan's
actions run where, and exactly what is still in flight when the dispatching
thread returns.

What matters here is the concurrency they permit. Multiple plan and task
handles may coexist in the runtime. Distinct callables own distinct admitted
task records and may remain active together; one callable permits one
outstanding submitted invocation, because its physical layout and preallocated
validation and action records are reused. A second concurrent invocation of the
same mutable handle fails closed rather than sharing that state.

The public PyTorch frontend exposes `submit()` for explicit invocation
ownership. Dispatch runs immediately and returns an `InvocationResult`;
`result()` is the single synchronization point for that invocation's public
result.

Reusing or closing a callable waits only for that plan's claimed task scopes,
actions, and task-owned retirements, never for unrelated plans.

The Python `_before_task()` and `_after_task()` add the framework half:
rebinding storages, assembling arguments, classifying outputs, dematerializing
releases, and recording timing. Forward and training share that skeleton and
the same default-off profiler-annotation policy. The neutral
`shadowspill_before_task_handle()` and `shadowspill_after_task_handle()`
contain no PyTorch storage logic.

## Failure and teardown

Nonzero allocation failures become typed exceptions in the PyTorch adapter;
they are never returned as a null pointer to a kernel. Runtime failures retain
the execution ID, semantic task, request, pool state, and first cause in the
library, so no-progress OOM, allocation-contract mismatch, worker failure, and
backend failure remain distinguishable.

Planned callables close their admitted execution state. Python
`Runtime.close()` requires no active callable, persistent imported state,
public object reference, or caller-owned device output. It then calls the C
close/destroy path, which stops and joins the worker, closes every route and
pool backend, unregisters pinned memory, and releases device memory. PyTorch
cannot uninstall its selected process allocator, so only the allocator shim
remains; it rejects future allocations as closed.

A process that is exiting takes a different path, because waiting there
prevents the exit rather than delaying it. See
[failure, abort, and process exit](failure-and-exit.md).

The Python-facing taxonomy, structured diagnostic fields, automatic rollback,
and normal close order are documented in [Errors, failures, and
cleanup](../python/failures.md).

## Optional tracing

`runtime_trace=False` has no trace-buffer work on the critical path.
`profiler_annotations=False` independently controls the backend's
profiler. When runtime tracing is enabled, bounded preallocated buffers record
task, allocator, transfer, and failure events and are converted to Python only
when diagnostics are resolved.

See [Interpreting StepResult diagnostics](../python/step-diagnostics.md) for
allocator/lease evidence, runtime counters, transfer frontiers, task-boundary
timing, and overflow handling.

Previous: [Simulation](simulation.md). Continue with the [Python allocator
guide](../python/allocator.md) or [Runtime C API](../c/runtime.md).
