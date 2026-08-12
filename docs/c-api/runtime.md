# Runtime C API

Public declarations live in:

- `runtime/include/shadowspill/backend.h`;
- `runtime/include/shadowspill/profiler.h`;
- `runtime/include/shadowspill/runtime.h`;
- `runtime/backends/mock/include/shadowspill/backend_mock.h`.

The runtime ABI is version 13. Public functions return
`ShadowSpillRuntimeStatus` except documented idempotent destroy/read-only
operations. Call `shadowspill_runtime_abi_version()` before constructing a
runtime and validate every supplied vtable ABI.

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

## Ownership

- Runtime and vtable values are copied; vtable contexts are borrowed and must
  outlive the runtime.
- Pool ranges are represented by stable generation-tagged `MemoryLease`
  records internally.
- Returned allocation/binding values remain valid only for their generation.
- Input description arrays are borrowed only for the call.
- Diagnostics/statistics are caller-owned snapshots with no internal pointer.

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
