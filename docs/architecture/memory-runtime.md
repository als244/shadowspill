# Memory runtime

The compiled runtime is framework-neutral. It owns bounded memory pools,
logical objects, physical leases, transfer lanes, completion events, tracing,
and failure propagation. The PyTorch adapter translates allocator callbacks
and storage operations into this contract.

## Pools, budgets, and leases

`MemoryPool` is a generic range owner instantiated for the execution and spill
pools. The supported backends are an accelerator-device slab and a registered
pinned-host slab. Host memory is obtained with ordinary allocation followed by
provider registration, and is unregistered before it is freed.

The execution pool's `physical_capacity` is the complete process-attributable
device cap. Provider/context headroom lies inside that cap. Planning budgets
may reduce configured capacities but cannot exceed them.

A `MemoryLease` owns one range for one residency generation. Objects keep a
lease per pool location; aliases and views share the same object and lease.
Generations prevent stale events, frees, bindings, or worker completions from
modifying a successor.

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
- strict task-allocation ABI core slots;
- persistent outputs and mutation replacements;
- fetch and evict destinations.

Offsets are relative to the callable's admitted slice of the runtime pool.
The layout models actual allocation/free geometry and causal overlap; it does
not collapse a task into one synthetic workspace block. Runtime callbacks
validate the allocation ABI before returning a planned range.

Two cases remain dynamic by design:

- bounded optional anonymous/provider allocations use the admitted dynamic
  scratch reserve;
- terminal caller-owned outputs use dynamic leases so they may outlive a
  later callable invocation.

The scratch reserve is derived from profiling. A user may raise it with
`dynamic_scratch_reserve_bytes`, but cannot reduce the measured requirement.
Runtime fixed-service headroom, provider/context headroom, task workspace, and
dynamic scratch are distinct accounting categories.

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

Fetch and evict are separate lanes with independent locks and streams. At an
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
is named `shadowspill.wkr` in profiler traces. The hot loop visits each
completion frontier, drains immediately completed FIFO successors, handles
retirements, and dispatches queued actions. The default incomplete-head query
cadence is one microsecond; an already-complete head is followed immediately
without an artificial delay.

Events are precreated and reused through generation-tagged leases. The worker
queries only FIFO heads and calls the backend outside data-structure locks.
Steady-state execution creates or destroys no events.

## Task boundaries

The Python `_before_task()` boundary resolves the immutable execution record,
calls the neutral runtime, acquires input generations, inserts stream waits,
rebinds storages, assembles arguments, and records timing. `_after_task()`
classifies outputs, publishes mutations, dematerializes releases, records the
completion fence, submits actions, and performs terminal cleanup.

The neutral `shadowspill_before_task()` and `shadowspill_after_task()` remain
small object/lease/action orchestrators. They contain no PyTorch storage logic.

## Failure and teardown

Nonzero allocation failures become typed exceptions in the PyTorch adapter;
they are never returned as a null pointer to a kernel. Runtime failures retain
the execution ID, semantic task, request, pool state, and first native cause.
No-progress OOM, allocation-ABI mismatch, worker failure, and backend failure
remain distinguishable.

Planned callables close their admitted execution state. Python
`Runtime.close()` requires no active callable or persistent relocated state and
closes the frontend handle. PyTorch cannot uninstall a selected process
allocator, so the adapter retains the neutral runtime until its registered
process-exit cleanup calls the C close/destroy path. That path stops and joins
the worker, closes every pool backend, unregisters pinned memory, and releases
device memory. Explicit callable and Python-runtime close remain required for
timely ownership validation and error reporting.

## Optional tracing

`runtime_trace=False` has no trace-buffer work on the critical path.
`profiler_annotations=False` independently controls NVTX or another backend
profiler. When runtime tracing is enabled, bounded preallocated buffers record
task, allocator, transfer, and failure events and are converted to Python only
when diagnostics are resolved.
