# Runtime backend and profiler APIs

The neutral runtime depends on opaque pool, route, stream, event, and profiler
operations. Provider implementations live under
`csrc/runtime/backends/<provider>/`; they do not change object or schedule
policy.

## Memory pools

`ShadowSpillMemoryPoolBackend` contains:

- an ABI field and borrowed context;
- `allocate_arena()` for one contiguous byte-addressable arena;
- idempotent `close()` for the arena.

An execution pool returns addresses accepted by the selected framework
accelerator. A spill pool may return any address token understood by its
directed transfer routes.

## Transfer routes

`ShadowSpillTransferRoute` represents one immutable source/destination pool
pair. It creates and destroys ordered lanes, submits `copy_async()`, and
synchronizes only at explicit lifecycle boundaries. Direction is a property of
the route, not a runtime switch interpreted by the copy implementation.

`ShadowSpillBackendStream` and `ShadowSpillBackendEvent` are opaque tokens.
The runtime stores and returns them without interpreting provider fields.

## Execution events

`ShadowSpillBackend` supplies execution allocation/free, spill allocation/free,
stream creation/destruction, event creation/destruction, event record/query/
wait, asynchronous copy, and explicit stream synchronization.

`query_event()` is nonblocking. `wait_event()` enqueues a dependency and does
not block the host. The backend must accept calls from the frontend and worker
threads and must keep submitted memory valid through the recorded completion.

## Profiler

`ShadowSpillProfiler` is an optional observability table:

- `name_current_thread()`
- `name_stream()`
- `set_enabled()`
- `range_begin()`
- `range_end()`

Missing callbacks are no-ops and cannot affect execution semantics. Runtime
code uses this interface rather than importing NVTX or another provider API.

The CUDA backend implements device and registered-host pools, fetch and evict
streams, events, copies, and NVTX annotations. The mock backend implements the
same neutral contracts for accelerator-free unit and sanitizer tests.
