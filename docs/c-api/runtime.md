# Runtime C API

Public declarations live in:

- `runtime/include/shadowspill/backend.h`;
- `runtime/include/shadowspill/runtime.h`;
- `runtime/backends/mock/include/shadowspill/backend_mock.h`.

The ABI is version 8. All public functions return an enum status except
idempotent destroy functions and read-only mock controls. Call
`shadowspill_runtime_abi_version()` and validate the backend ABI before creating
a runtime.

## Thread safety and blocking

Runtime calls are serialized by one internal mutex and condition variable.
Calls may come from framework and progress threads. `shadowspill_allocate` may
block on the condition variable only when recorded pending work can release a
range; impossible requests return `SHADOWSPILL_RUNTIME_NO_PROGRESS` with a
latched `ShadowSpillRuntimeFailure` snapshot.

`shadowspill_before_task` performs no host synchronization for an admitted
prefetch. It inserts backend stream waits and returns the destination pointer.
It may briefly wait for the progress worker to admit an already-queued
prefetch destination under allocator pressure.

`shadowspill_runtime_wait_idle` is explicitly synchronizing and intended for
tests, checkpoints, and lifecycle boundaries. `shadowspill_runtime_close` is
also synchronizing and idempotent; it rejects new work, drains or preserves the
first failure, joins the worker, and frees all runtime-owned resources.

`shadowspill_runtime_resize_host_arena` is a planning-only, explicitly idle
operation. It grows the pinned-host arena while preserving all object offsets
and payloads. Shrinkage is rejected. The backend briefly owns both old and new
arenas, so the frontend must admit that overlap under its public host budget;
the PyTorch adapter forbids growth after physical sealing.

## Ownership

- Runtime and backend configurations are copied during creation.
- Allocation and binding outputs are values; their pointers remain valid only
  for the reported device residency generation.
- Input, update, and action arrays are borrowed only for the call.
- The backend context must outlive the runtime using its vtable.
- Destroy the runtime before destroying its backend context.

Diagnostics and statistics are snapshots. They expose no internal record or
backend handle and are safe to inspect concurrently.

`shadowspill_transfer_object_to_caller` removes one ready logical object and
returns its still-live allocation. The allocation becomes ordinary rather than
plan-owned and must eventually pass through `shadowspill_free`; the transition
does not wait for device work or change the address. It rejects queued actions,
missing generations, and non-ready residency.

`shadowspill_unregister_object` removes a host-only or released logical object
after all queued actions have drained and returns its pinned-host range to the
arena. `shadowspill_write_host_object` and `shadowspill_read_host_object` copy
between caller-owned CPU storage and retained spill storage; both validate the
exact object extent and authoritative version.

A completed release or offload retains one retired address/generation token in
the object record. This token exists only so a framework may safely replace a
just-retired non-owning storage pointer after asynchronous progress wins the
race with frontend dematerialization. It does not restore residency, extend the
allocation lifetime, or make that address usable by numerical work.

## Allocation profiling

`shadowspill_allocation_telemetry_start` preallocates one bounded event buffer
before profiling. Between a successful `shadowspill_before_task` and its
matching `shadowspill_after_task`, allocation callbacks are tagged with that
task identity. The trace records requested and charged bytes, slab offset,
generation, physical create/release order, and promotion from anonymous output
allocation to planned-object ownership. `shadowspill_abort_task` clears only
the calling thread's task tag when frontend execution raises.

Capture never grows its buffer. Exhaustion latches
`SHADOWSPILL_RUNTIME_ALLOCATION_FAILURE`, so an incomplete workspace profile
cannot be admitted. `shadowspill_allocation_telemetry_read` copies records into
caller-owned storage and may first be called with a null destination to query
the exact count. Physical releases delayed by a distinct recorded stream retain
the task identity of their logical free; plan actions retain their trigger task
identity even when the progress thread performs the allocation or release.

The mock backend supports directional copy delays, event delay, operation
counts, and exact failure injection. `shadowspill_mock_fail_next_operation` is
atomic with respect to backend activity and is intended for deterministic
waiter/failure tests. It is a qualification backend, not part of production
execution.

## Bounded execution tracing

Tracing is disabled by default. In that state, runtime hot paths perform only
one predictable disabled-flag check and append no records. The frontend may
prepare reusable caller-sized buffers with `shadowspill_trace_prepare`; this
allocates CPU memory but does not begin capture. A prepared session is started
with `shadowspill_trace_begin(step_id)` and stopped with
`shadowspill_trace_end()`. Neither end nor read synchronizes a backend stream;
the caller establishes the desired boundary, normally with its compute-stream
completion plus `shadowspill_runtime_wait_idle`, before ending the trace.

`shadowspill_trace_read` first accepts null record arrays to return exact
counts in `ShadowSpillTraceSummary`, then copies runtime and allocation records
into caller-owned arrays. Capacity never grows during capture. Overflow is
reported in the summary and does not change numerical execution or latch an
OOM; tracing is diagnostic evidence rather than an admission prerequisite.
The existing allocation-profiling API retains its stricter behavior and
latches failure on overflow because an incomplete workspace profile is unsafe.

Runtime trace timestamps use `CLOCK_MONOTONIC`. Events cover session bounds,
task entry/exit, readiness dependencies, action queueing, destination
reservation, transfer dispatch/completion, allocator blocking/wakeup,
retirement completion, and first-failure latching. Event-specific details are:

| Event | `detail_0` | `detail_1` |
|---|---:|---:|
| `BEFORE_TASK` | input count | queued actions |
| `AFTER_TASK` | status | submitted action count |
| `READINESS_WAIT` | 0 host condition, 1 stream event | queue/wait count |
| `ACTION_QUEUED` | action kind | queued actions |
| `DESTINATION_RESERVED` | action kind | slab offset |
| `TRANSFER_*` | 0 H2D, 1 D2H | queued actions |
| `ALLOCATION_WAIT_*` | free bytes | largest free range |
| `RETIREMENT_COMPLETED` | slab offset | charged bytes |
| `FAILURE_LATCHED` | status | free bytes |

IDs that do not apply are `SHADOWSPILL_RUNTIME_NO_ID`. Public records contain
no pointer, stream, event, framework, or vendor type.
