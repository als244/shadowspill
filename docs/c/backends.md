# Backend contract

`<shadowspill/backend.h>` is the whole contract between ShadowSpill and an
accelerator provider: one header, one version, `SHADOWSPILL_BACKEND_ABI_VERSION`.
A backend is a flat table of driver-level calls that one shared object per
provider implements and exports through two symbols. Everything built from
those calls -- pools, routes, lanes, event pools, calibration -- is
ShadowSpill's, so a backend carries no policy and owns nothing beyond the
provider context.
[Backends](../architecture/backends.md) explains the boundary; this page is the
reference.

## Tokens and records

- `ShadowSpillBackendStream`, `ShadowSpillBackendEvent`: opaque two-word
  provider tokens the runtime stores and returns unread.
- `ShadowSpillProfilerRange`: a range identifier from `range_begin`.
- `ShadowSpillBackendConfig`: `abi_version`, `device_ordinal`.
- `ShadowSpillBackendCapabilities`: `device_ordinal`, `minimum_alignment` for
  device allocations, and `provider`, the platform's short lowercase name for
  diagnostics (`"mock"` for the mock), in a fixed
  `SHADOWSPILL_BACKEND_PROVIDER_NAME_CAPACITY` buffer.
- `ShadowSpillBackendPhysicalMemory`: `process_bytes`, `device_used_bytes`,
  `device_total_bytes` as the platform accounts for them now.
- `ShadowSpillBackendStatistics`: counters of the calls made through the
  table: device allocations and frees with their byte totals, pinned-host
  registrations and unregistrations with theirs, streams and events created
  and destroyed, copies and bytes per direction, event queries, stream waits
  and synchronizations, and `provider_activations`, the times the provider's
  context had to be made current on a calling thread. A backend without a
  notion of a counter reports zero.

## The table

`ShadowSpillBackend` carries `abi_version` -- still 1, because no backend
outside this tree implements the contract yet, so it is settled rather than
kept; the check exists so a library and a header from different builds cannot
be paired -- the provider object `state` that
every entry receives, and these entries. Each returns 0 on success and
nonzero on failure unless noted.

| group | entries |
|---|---|
| memory | `allocate_device(bytes, &address)`, `free_device(address, bytes)`, `register_host_memory(address, bytes)`, `unregister_host_memory(address, bytes)` |
| signals | `allocate_signals(count, &signals, &host)`, `free_signals(signals)`, `wait_value(stream, signals, index, generation)` |
| streams | `create_stream(&stream)`, `destroy_stream(stream)`, `synchronize_stream(stream)`, `resolve_stream(stream_handle)` returning the word this backend knows that stream by |
| copies | `copy_host_to_device(device, host, bytes, stream)`, `copy_device_to_host(host, device, bytes, stream)`, `copy_device_to_device(destination, source, bytes, stream)` |
| events | `create_event(&event, timing)`, `destroy_event(event)`, `record_event(event, stream)`, `query_event(event, &complete)`, `wait_event(stream, event)`, `synchronize_event(event)`, `elapsed_nanoseconds(from, to, &nanoseconds)` |
| facts | `capabilities(&out)`, `physical_memory(&out)`, and `statistics(&out)`, the one entry here that returns nothing |
| profiler, optional | `name_thread(name)`, `name_stream(stream, name)`, `profiler_enable(enabled)`, `range_begin(name)` returning a range, `range_end(range)` |

Memory: device memory is the backend's to allocate; host memory is
ShadowSpill's, mapped by the pinned-host pool and registered here so the
provider can copy from it asynchronously. Frees and unregistrations carry the
byte count so the backend keeps no size bookkeeping.

A stream and an event are each **one opaque word**, `uint64_t`, exactly as a
profiler range is. Only the backend reads it. A backend over a driver keeps the driver's own
stream there, which is typically a pointer; the mock keeps a pointer to a
record of its own. Neither keeps a lookup table -- the word *is* the handle,
cast back on use -- and a backend that names streams some other way, by index
or by ticket, puts that in the same word instead. Nothing outside a backend may
construct or inspect one.

Zero means *none*: for an event, no event; for a stream, the backend's default
stream -- the one a driver runs work on when the caller names none. The mock
has one too, so a caller with no streams of its own can still drive the table.

Streams are ordered queues of copies and events, and copies are asynchronous
and ordered on their stream. `resolve_stream` answers with the word this
backend knows a stream by, given the integer its owner knows it by. It is how a
stream the backend did not create -- the framework's compute stream -- enters
the runtime. Where the two name a stream the same way it is the identity; where they do
not, as with the mock's default stream, the backend maps it.

Events: a dependency event (`timing` clear) is the fast kind that record,
query, and wait work with. A timing event carries a device timestamp when
recorded, and `elapsed_nanoseconds` reads the device-clock interval between
two of them: 0 with the interval, 1 while either is still pending, -1 when the
## Signals, and waiting on one

A signal block is words a stream can wait on and a host thread can store to.
`allocate_signals` returns an opaque handle and the host address of the first
word; `wait_value` holds a stream until the word at an index reaches a
generation, comparing greater-or-equal so a value already past it does not
stall.

They exist for a lane whose bytes do not move on a stream. Such a lane cannot
make a stream wait for it by recording an event, because the consumer's
`wait_event` may be enqueued before the transfer finishes and a wait on an event
not yet recorded does not wait. The runtime enqueues the value wait and the
record together at dispatch; the lane stores the generation when the bytes have
landed; everything downstream sees an ordinary event.

**A word may need two addresses**, and only the backend should know it. A
provider whose streams read host memory through a mapping requires the block to
be allocated so that mapping exists, and then the caller stores through one
address while the stream reads another. That is why `wait_value` names a word by
index rather than by pointer: a caller holding only the address it stores to
could not supply the other, and has no reason to know it exists.

pair cannot be measured. `record_event` and `wait_event` enqueue without
blocking the host -- `wait_event` orders one stream behind an event on
another -- and `query_event` is a nonblocking poll. The two calls that do block
the calling thread are `synchronize_event`, which returns once the device has
reached one event, and `synchronize_stream`, which returns once it has finished
a whole stream.

Profiler entries are best-effort diagnostics and never change execution
semantics; a NULL entry is a no-op.

## The two symbols

```c
int shadowspill_backend_create(const ShadowSpillBackendConfig *config,
                               ShadowSpillBackend *backend);
void shadowspill_backend_destroy(ShadowSpillBackend *backend);
```

`shadowspill_backend_create()` fills the table and returns 0, or returns
nonzero leaving nothing to destroy. `shadowspill_backend_destroy()` releases the
provider object and zeroes the table; it runs after the runtime it served is
gone, so every stream, event, mapping, and allocation has already been returned
through the table. `SHADOWSPILL_BACKEND_CREATE_SYMBOL` and
`SHADOWSPILL_BACKEND_DESTROY_SYMBOL` name them for `dlsym()`, and
`ShadowSpillBackendCreate` and `ShadowSpillBackendDestroy` are the
function-pointer types to cast the results to. `shadowspill_backend_is_valid()`
in `<shadowspill/runtime.h>` is the check both the runtime and the adapter apply
to a table before using it.

## Threading and lifetime

The runtime borrows `state` for its lifetime. Entries are called from the
runtime worker and from the framework's threads; the backend serializes what
its provider requires. Calibration drives the two copy directions at once on
separate streams, so streams must be independent.

## Adding a backend

A new provider is a directory `csrc/backends/<provider>/` compiled against
this header alone, built as `libshadowspill_backend_<provider>.so` beside the
runtime library, exporting the two symbols. `csrc/backends/CMakeLists.txt`
builds every provider whose toolchain is installed, or the ones named in
`SHADOWSPILL_BACKENDS`. `Runtime(backend="<provider>")` selects it by name;
`Runtime(backend=None)` selects the one accelerator backend installed. The
tree holds one accelerator backend and the accelerator-free mock backend
(`mock/`), which the C canaries and sanitizer tests use, and whose extra test
hooks live in `<shadowspill/backend_mock.h>`.
