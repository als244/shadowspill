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

One stable `MemoryLease` describes pool ownership only. Transfer actions and
object residency have separate state machines; the pool never records
`FETCHING`, `EVICTING`, source/destination, or route state.

| Lease state | Pool meaning |
|---|---|
| `FREE` | The record owns no range. |
| `IN_USE` | A consumer owns the range and may use its pointer subject to dependencies it already accepted. |
| `RETIRE_PENDING` | The current consumer still owns the range until its completion dependency is committed. |
| `RESERVED` | The range is exclusively held for a future consumer but has not been handed to it. |
| `SUCCESSOR_RESERVED` | A future consumer has claimed a `RETIRE_PENDING` predecessor's range; the predecessor still owns the bytes. |
| `PREDECESSOR_TRANSFERRED` | Range ownership has moved to the successor; this record retains only historical metadata until predecessor completion is processed. |

The private pool transition surface is intentionally small:

```text
reserve lease
mark lease reserved
begin / cancel retirement
publish retirement dependency
reserve causal successor
acquire reserved lease
cancel reservation
release lease
snapshot geometry
```

`acquire_reserved_lease()` changes `RESERVED` to `IN_USE`. For a
`SUCCESSOR_RESERVED` lease it also returns the predecessor dependency that the
consumer must honor before touching the address. A generation prevents stale
completion, release, or acquisition from publishing state for a recycled
range.

Object locations are an indexed array over runtime pools. Each entry stores its
lease and version state. The current frontend selects execution and spill roles
but the object representation does not contain fixed host/device fields.

### Causal reuse rules

Backend event completion and pool availability are deliberately different
facts. An event's `backend_complete` bit means only that its stream reached the
event. A range is allocatable only after its `MemoryPool` has committed the
owning lease transition under that pool's lock.

There are two nonblocking reuse paths:

1. An ordinary framework allocation may recycle an anonymous logically freed
   lease immediately only when every prior use and the new consumer use the
   same execution stream. Stream order itself prevents the new work from
   overtaking the old work. A cross-stream ordinary allocation does not receive
   a pending address: it waits for a committed free range or fails with a
   no-progress OOM.
2. A memory action may reserve a pending predecessor range at its directive
   trigger. The resulting lease is action-owned and is not exposed to an
   arbitrary framework allocation. When a transfer action reaches its lane
   head, that separate component acquires the reserved lease and makes its
   stream wait on the dependency returned by the pool. For a fetch, the
   consuming compute stream subsequently waits on the fetch-completion event
   before the frontend binds and launches work using that generation.

The planned fetch dependency chain is therefore:

```text
predecessor stream reaches completion event
    -> fetch lane may write the reserved range
    -> fetch completion event
    -> consuming compute may read the object generation
```

Completion commits exactly one pool transition:

```text
no successor:
    pending predecessor -> FREE

reserved successor:
    pending predecessor + SUCCESSOR_RESERVED
        -> predecessor metadata retired + successor RESERVED
```

The second transition hands ownership off directly. The bytes never enter the
free-range tree, so an unrelated allocation cannot take them between
predecessor completion and consumer acquisition. If the consumer acquires
first, ownership moves directly to `IN_USE` and the pool returns the retained
predecessor dependency. The later predecessor completion then retires only its
metadata. Generation checks reject every stale completion in either order.

Transfer terminology belongs above this API. For example, an object may be
`OFFLOADING` while its source lease is generically `RETIRE_PENDING`; completing
the transfer changes object residency, while completing the dependency commits
the pool handoff or free. A zero-copy object handoff changes logical object
ownership without retiring the shared physical lease at all.

Immediately free compatible ranges are always preferred over causal
predecessors. Among pending candidates, ShadowSpill prefers a published
dependency, then least charged-range waste, oldest logical release, and lowest
address. It never interprets an observed-but-uncommitted event as free memory.

Free ranges are coalesced with adjacent free neighbors. ShadowSpill does not
move live leases to compact the arena; planning-time allocator replay validates
the expected fragmentation and derives any required reserve.

## Planning-time AdmissionReplay

Replay and simulation have deliberately different responsibilities.

`admission_replay.c` is a timing-free interpreter over the production
`MemoryPool` policy. It receives only causal ownership boundaries—lease,
retirement, reservation, acquisition, completion, and release—and answers:

- whether the slab geometry is feasible;
- which range each lease decision selects;
- how much physical capacity is live or reserved;
- how much fragmentation occurs;
- which pending predecessor a successor reuses; and
- which completion dependency the successor must honor.

The replay source, public C ABI, and Python wrapper are isolated in:

```text
runtime/src/admission_replay.c
runtime/include/shadowspill/admission_replay.h
src/shadowspill/runtime/admission_replay.py
```

`memory_pool.c` remains the small reusable allocator/state-machine component.
It knows neither about replay scripts nor simulator time.

The simulator takes the selected task and transfer schedule plus replay's
dependency edges. It assigns nanosecond timestamps from compute profiles,
route latency/bandwidth, resource lanes, and queue backlog. A slow eviction
therefore delays a successor fetch; a fast eviction may make the dependency
wait zero. Neither outcome changes replay safety or range ownership.

Replay must not free a retirement merely because a predicted timestamp says it
should have completed. If two events have no happens-before relationship, the
retirement remains causally pending. A planned successor can reserve its range
only with an explicit dependency edge; an unrelated allocation cannot consume
it. This makes admission deterministic across timing noise and transfer
recalibration.

Planning composes the two components as follows:

```text
candidate memory schedule
    -> causal MemoryPool replay
    -> memory-reuse dependency edges
    -> timed simulator
    -> validated ExecutionPlan
```

If the added edges change scheduling boundaries that affect the ownership
script, planning repeats until both the replay decision digest and dependency
edge set are stable. Timing affects makespan and plan quality, never the proof
that memory exists.

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
