# Memory runtime

The runtime is framework-neutral. It owns bounded memory pools,
logical objects, physical leases, transfer lanes, completion events, tracing,
and failure propagation. The PyTorch adapter translates allocator callbacks
and storage operations into this contract.

## Pools, budgets, and leases

`MemoryPool` is a generic range owner registered by identity with the runtime,
and directed transfer routes are registered separately; how pools get their
arenas and routes their lanes is in [memory pools](memory-pools.md). Each
admitted plan selects its execution pool, spill pool, fetch route, and evict
route; the runtime has no single global execution/spill role pair. The PyTorch
adapter registers one device pool and any number of pinned-host pools.

The execution pool's `physical_capacity` is the complete process-attributable
device cap. Provider headroom and the driver's own baseline lie inside that cap. Planning budgets
may reduce configured capacities but cannot exceed them.

Runtime construction precedes workload-state construction, so calibration
runs on the real arenas before the model claims host memory; the ordering and
its reason are in [memory pools](memory-pools.md#construction-order).

A `MemoryLease` owns one range for one residency generation. Objects keep a
lease per pool location; aliases and views share the same object and lease.
Generations prevent stale events, frees, bindings, or worker completions from
modifying a successor.

Runtime-global shared leases are physically charged once and retained outside
any one callable's movable-object schedule. `SHARED_READ_ONLY` accepts only
existing inputs and rejects all writes. `SHARED_WRITABLE_CAUSAL` orders each
generation and allows producers to publish a replacement lease.
`SHARED_WRITABLE_UNORDERED` permits stable-address in-place mutation without
introducing cross-callable ordering; readers may therefore observe an older,
newer, or concurrently changing value.

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

The memory pool knows ownership and dependencies, not transfer meaning.
Transfer components create, acquire, cancel, and publish reservations through
the pool API.

## Fixed layout with bounded dynamic allocation

Production plans use one complete step-level physical layout for
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

Fetch and evict are separate routes, each with its own lane, a stream the
runtime owns ([transfers](transfers.md)). At an
action trigger:

1. the dispatcher reserves destination capacity in directive order;
2. the action owns that reservation while queued;
3. at lane head, the worker submits the copy and records completion;
4. on completion, the object publishes the ready residency generation.

An eviction source is not freed when the action is queued. It becomes
reusable only through its transfer-completion dependency. Reserving at trigger
time prevents later task allocations from overtaking planned transfer
capacity; deferring copy submission preserves FIFO lane behavior.

## Worker

One C-owned worker services completions, releases, and both transfer lanes. It
is named `shadowspill_worker` in profiler traces (`shadowspill.wkr` is the
OS-level shortened name). The hot loop visits each
completion frontier, drains immediately completed FIFO successors, handles
retirements, and dispatches queued actions. The default incomplete-head query
cadence is one microsecond; an already-complete head is followed immediately
without an artificial delay.

Cold plan adoption reserves event leases with their backend events
([events](events.md)), retirement queue entries, `MemoryLease` records, and
lease-use records. It also sizes one pool-owned release-frontier workspace from the
sealed lease inventory. This workspace dry-runs pending-range coalescing
inside a bounded borrowed range-node arena; destination reservation never
builds a heap array or clones heap-owned range nodes while holding the pool.
A lease-use record names one distinct stream while its lease is live;
an asynchronous free records the completion event directly into that same
record and gives the immutable list to the retirement queue. There is no
stream snapshot or copied event-wrapper list. A later callable sharing the
runtime may grow these inventories only at the same cold boundary.
Generation-tagged leases return records and handles to their respective
owners; the worker queries only FIFO heads and calls the backend outside
data-structure locks. Steady-state execution performs no host allocation and
creates or destroys no backend event.

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
worker acknowledges submission of every causally eligible route operation.
It never waits for copy completion. Completion publication belongs to the
worker. A dispatcher allocation may wait only when a known pending transition
can satisfy it; otherwise it fails with no progress.

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
actions, and task-owned retirements. It never waits for unrelated plans. The
wait is an active atomic poll; neither dispatcher nor worker enters a condition
wait, sleep, or scheduler yield.

The Python `_before_task()` and `_after_task()` add the framework half:
rebinding storages, assembling arguments, classifying outputs, dematerializing
releases, and recording timing. Forward and training share that skeleton and
the same default-off profiler-annotation policy. The neutral
`shadowspill_before_task_handle()` and `shadowspill_after_task_handle()`
contain no PyTorch storage logic.

## Failure and teardown

Nonzero allocation failures become typed exceptions in the PyTorch adapter;
they are never returned as a null pointer to a kernel. Runtime failures retain
the execution ID, semantic task, request, pool state, and first cause in
the library.
No-progress OOM, allocation-contract mismatch, worker failure, and backend failure
remain distinguishable.

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
