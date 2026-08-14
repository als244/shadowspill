# Framework-neutral runtime

`libshadowspill_runtime.so` consumes admitted task boundaries and owns memory
pools, leases, object residency, transfers, readiness, retirement, failures,
and teardown. It contains no framework or accelerator-provider type.

## Central owners

One runtime contains:

- a registry of bounded `MemoryPool` arenas;
- a hash-indexed object table, where one object is one alias bundle;
- one `MemoryLease` for each reserved range and residency generation;
- directed transfer routes and independent fetch/evict lane queues;
- an allocation/retirement owner;
- FIFO completion frontiers and a reusable event pool;
- one worker thread;
- bounded, default-off trace buffers and a first-failure record.

The initial provider instantiates one execution pool and one spill pool. A pool
does not encode whether storage is local, accelerator, host, peer, remote, or
persistent. That meaning belongs to its pool backend and directed routes.

## Allocation and leases

Framework allocation synchronously leases a coalescing range from the selected
execution pool. Logical free occurs immediately. Same-stream use is ordered by
the stream itself; distinct recorded streams attach retirement fences and
delay physical reuse until completion. Allocation waits only when a known
retirement or memory action can produce a sufficiently large range. Otherwise
it returns a diagnostic no-progress OOM.

One stable `MemoryLease` moves through reserved, transferring, active,
retiring, and released states. Reservation and transfer do not create separate
allocation identities. A generation prevents stale transfer or retirement
completion from publishing state for a recycled range.

Object locations are an indexed array over runtime pools. Each entry stores its
lease and version state. The current frontend selects execution and spill roles
but the object representation does not contain fixed host/device fields.

## Task protocol

`shadowspill_before_task` resolves the immutable execution record, acquires
each distinct input generation, validates execution residency, and inserts a
compute-lane wait for each unfinished fetch event. It returns current addresses
without synchronizing the host.

`shadowspill_after_task` publishes declared mutations, records one task fence,
and submits the plan's exact ordered release, evict, and fetch actions. It does
not move triggers, substitute actions, or launch framework work.

The worker repeatedly:

1. handles FIFO event completions;
2. handles stream-safe retirements;
3. dispatches ready memory actions on the appropriate route lane;
4. publishes failures and wakes dependent waiters;
5. polls or waits according to the configured latency policy.

Backend calls occur outside unrelated data-structure locks. The worker is
named `shadowspill_worker`. Provider profiler callbacks name route streams
`shadowspill_fetch` and `shadowspill_evict`.

## Transfer capabilities

Each `(source_pool, destination_pool)` route owns asynchronous copy and lane
lifecycle behavior. Runtime initialization calibrates supported directions and
publishes a dense N-by-N latency/bandwidth matrix. Diagonal entries explicitly
represent identity movement. Recalibration may target all or selected routes,
requires local idleness, and atomically publishes a new generation. Existing
plans retain their earlier immutable snapshot.

The runtime performs no cross-process barrier. Users can coordinate separate
processes and invoke calibration concurrently to measure shared-link
contention.

## Tracing and profiling

Runtime tracing is bounded and off by default. Preparation allocates record
capacity; begin/end only enable and disable append. Records cover task
boundaries, waits, action queueing, lease reservation, transfer dispatch and
completion, allocation pressure, retirement, and first failure. Allocation
lifetimes have a separate ordered ledger.

Profiler integration is another independent interface. The neutral runtime
uses a no-op-capable `ShadowSpillProfiler` vtable for thread/stream names and
ranges. NVTX exists only in the NVIDIA provider implementation; a ROCm provider
can supply rocTX without changing neutral runtime code.

## Failure semantics

The runtime latches the first native failure and wakes every dependent waiter.
A no-progress allocation record includes the active task when one exists,
requested bytes, total free bytes, largest free range, object/allocation IDs,
and pool/device identity. The PyTorch frontend translates only the allocator
out-of-memory statuses; other backend errors remain provider exceptions.

`shadowspill_runtime_recover_no_progress` is a narrow rollback primitive. It
is legal only after the failed allocator caller has returned, participating
compute work is synchronized, and no allocator waiter remains. It clears only
`NO_PROGRESS`; it cannot clear backend, worker, state, plan, or device faults.
Recovery permits teardown and a later independent planning attempt. It never
retries the failed allocation or makes an infeasible plan feasible.

Allocator failure does not cancel stream causality. Tensor destruction while a
failed task unwinds may still issue logical frees. The failed `after_task`
boundary records one compute-stream fence for those task-local retirements;
the abort path records their known stream uses when no `after_task` boundary is
reached. Recovery refuses to clear the failure latch unless every pending
retirement has a fully published queue record. This fail-closed check prevents
an idle wait from accepting an orphaned retirement with no possible completion
source.

## Lifecycle

`shadowspill_runtime_wait_idle` and close are explicitly synchronizing.
Ordinary task boundaries, allocation, result construction, and trace append are
not device-wide synchronization points. Close rejects new work, drains owned
work, joins the worker, and releases pools, routes, events, queues, and tables
while preserving the first failure.

Each `MemoryPool` backend owns its close operation. Runtime teardown iterates
the pool registry rather than naming execution/spill implementations. The
CUDA device-pool close frees its conventional slab; the pinned-host close uses
`cuMemHostUnregister` followed by ordinary `free`. A C process-exit handler
also performs final native teardown, stopping and joining the worker before
closing pools, transfer streams, event leases, queues, and lookup tables.
