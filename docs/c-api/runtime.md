# Runtime C API

Public declarations live in:

- `runtime/include/shadowspill/backend.h`;
- `runtime/include/shadowspill/admission_replay.h`;
- `runtime/include/shadowspill/profiler.h`;
- `runtime/include/shadowspill/runtime.h`;
- `runtime/backends/mock/include/shadowspill/backend_mock.h`.

The runtime ABI is version 23. Public functions return
`ShadowSpillRuntimeStatus` except documented idempotent destroy/read-only
operations. Call `shadowspill_runtime_abi_version()` before constructing a
runtime and validate every supplied vtable ABI.

`admission_replay.h` has its own versioned, timing-free batch ABI. Call
`shadowspill_admission_replay_abi_version()` before passing a replay program to
`shadowspill_admission_replay_run()`. Replay borrows every input/output buffer for
the duration of the call and retains no caller storage.

## Backend boundaries

`ShadowSpillMemoryPoolBackend` owns one opaque byte-addressable arena.
`ShadowSpillTransferRoute` owns one directed pool-pair copy capability and its
lanes. `ShadowSpillProfiler` optionally owns names and diagnostic ranges. The
neutral ABI contains no vendor stream, event, pointer, or profiler type.

`ShadowSpillBackend` remains the temporary two-pool compatibility construction
bundle while runtime creation migrates to a fully dynamic pool/route registry.
New policy belongs in the pool, route, event, or profiler interfaces—not in
that bundle.

## Thread safety and blocking

The runtime has focused synchronization owners: pool geometry, object table,
individual objects, route lanes, completions, event leases, trace buffers, and
lifecycle/failure state. No backend operation may execute while an unrelated
owner lock is held.

Allocation can wait only when a known retirement or memory action can make a
suitable range available. Impossible requests return
`SHADOWSPILL_RUNTIME_NO_PROGRESS` with a failure snapshot. Task readiness uses
stream-event dependencies; it does not synchronize the host merely because a
fetch is unfinished.

`shadowspill_runtime_wait_idle`, transfer recalibration, pool reconfiguration,
and close are explicit idle/lifecycle boundaries. Close drains work, joins the
worker, and releases owned resources while preserving the first failure.

`shadowspill_runtime_recover_no_progress` is a narrow rollback operation. It
may clear only a latched `SHADOWSPILL_RUNTIME_NO_PROGRESS`, after the failed
allocator callback has returned, device work has been quiesced, and no
allocator waiter remains. It never retries an allocation and cannot clear a
backend, worker, plan, state, or closed-runtime failure.

The call also requires every logically freed lease to have a completely
published retirement record. Failed task boundaries preserve their compute
fence, and explicit task abort records known stream uses. If either path leaves
an orphaned retirement, recovery returns `SHADOWSPILL_RUNTIME_INVALID_STATE`
without clearing the original latch; it never converts an impossible idle wait
into an unbounded block.

## Ownership

- Runtime and vtable values are copied; vtable contexts are borrowed and must
  outlive the runtime.
- Pool ranges are represented by stable generation-tagged `MemoryLease`
  records internally.
- Returned allocation/binding values remain valid only for their generation.
- Input description arrays are borrowed only for the call.
- Diagnostics/statistics are caller-owned snapshots with no internal pointer.

An event becoming complete does not make an associated range allocatable.
Completion observation is backend state; `MemoryPool` ownership is the memory
authority. Under the pool lock, completion either coalesces an unclaimed range
into the free tree or hands it directly to its already-reserved causal
successor. A successor handoff never exposes an intermediate free range.

Ordinary allocator callbacks can return pending storage without blocking only
for verified same-stream anonymous reuse. Planned transfer reservations are
not returned to arbitrary callers: the transfer lane waits on the predecessor,
and consuming compute waits on fetch completion. Cross-stream ordinary reuse
waits for a committed pool transition.

## Internal MemoryPool boundary

`MemoryPool` is a framework-, backend-, and transfer-agnostic C component. Its
private API operates on ranges, `MemoryLease` records, generations, and opaque
completion dependencies. It does not accept action kinds, routes, streams,
objects, or fetch/evict terminology.

The action layer may ask the pool to reserve a lease at a trigger, reserve a
causal successor, or acquire an existing reservation. Acquisition returns an
opaque predecessor event when one must be honored. The action/route layer—not
the pool—decides which stream waits on that event and what copy is submitted.
Likewise, object readiness and `PREFETCHING`/`OFFLOADING` transitions are never
written by pool code.

All `_locked` pool functions require ownership of that pool's mutex. Backend
operations and route submission are forbidden under it. Pool state changes are
constant-time except range-tree allocation/free and deterministic scans for a
compatible causal predecessor; completion polling occurs outside the pool.

## AdmissionReplay boundary

AdmissionReplay is a separate runtime component implemented in
`runtime/src/admission_replay.c`; it is not part of `memory_pool.c`. It translates
an ordered, timing-free ownership script into calls to the exact production
`MemoryPool` transitions. Its result contains allocator decisions, physical
charge deltas, fragmentation, infeasibility geometry, and any causal
predecessor-to-successor dependencies introduced by range reuse.

Replay never predicts a transfer duration or decides when a task runs. An
unfinished retirement stays pending unless the script contains a causal
completion boundary. A reserved successor may share that pending range only
by carrying the predecessor dependency. Therefore a faster or slower backend
cannot make an otherwise unsafe replay feasible.

The simulator consumes replay's dependency edges and assigns timestamps using
task profiles, route calibration, and lane backlog. In short:

```text
AdmissionReplay: can this ownership schedule fit safely?
Simulator:     when will the admitted work run, and how long will it take?
```

## Route calibration

`shadowspill_runtime_calibrate_transfer_capabilities` calibrates every
configured off-diagonal route when no keys are supplied, or selected routes
otherwise. The runtime must be locally idle. There is deliberately no
cross-process coordination, allowing applications to barrier separate runtimes
and calibrate concurrently under shared-link load.

`shadowspill_runtime_transfer_profiles` returns one lock-consistent row-major
N-by-N matrix plus its generation. Profiles contain measured latency,
bandwidth, timestamp, sample geometry/count, availability, calibration state,
and provenance. Successful recalibration atomically publishes a new matrix.

## Bounded telemetry

Allocation profiling and step tracing preallocate bounded record buffers.
Tracing is disabled by default and never grows a hot-path buffer. Runtime trace
records cover task boundaries, readiness dependencies, action queueing, lease
reservation, fetch/evict dispatch and completion, allocation waits,
retirements, and first failure. The separate allocation ledger records request,
charge, range, generation, task origin, promotion, logical free, and release.

`ShadowSpillProfiler` is independent of these structured records. It maps
thread/stream naming and ranges onto provider tools for interactive traces; a
missing profiler is a valid no-op configuration and cannot affect correctness.
