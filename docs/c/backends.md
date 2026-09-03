# Backend contract

`<shadowspill/backend.h>` is the whole contract between ShadowSpill and an
accelerator provider: one header, one version, `SHADOWSPILL_BACKEND_ABI_VERSION`.
A backend is a flat table of driver-level calls that one shared object per
provider implements and exports through two symbols. Everything built from
those calls, pools, routes, lanes, event pools, calibration, is ShadowSpill's,
so a backend carries no policy and owns nothing beyond the provider context.
[Backends](../architecture/backends.md) explains the boundary; this page is the
reference.

## Tokens and records

- `ShadowSpillBackendStream`, `ShadowSpillBackendEvent`: opaque two-word
  provider tokens the runtime stores and returns unread.
- `ShadowSpillProfilerRange`: a range identifier from `range_begin`.
- `ShadowSpillBackendConfig`: `abi_version`, `device_ordinal`.
- `ShadowSpillBackendCapabilities`: `device_ordinal`, `minimum_alignment` for
  device allocations, and `provider`, the platform's short lowercase name for
  diagnostics (`"mock"` for the mock).
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

`ShadowSpillBackend` carries `abi_version`, the provider object `state` that
every entry receives, and these entries. Each returns 0 on success and
nonzero on failure unless noted.

| group | entries |
|---|---|
| memory | `allocate_device(bytes, &address)`, `free_device(address, bytes)`, `register_host_memory(address, bytes)`, `unregister_host_memory(address, bytes)` |
| streams | `create_stream(&stream)`, `destroy_stream(stream)`, `synchronize_stream(stream)`, `wrap_stream(framework_stream_handle)` returning a token |
| copies | `copy_host_to_device(device, host, bytes, stream)`, `copy_device_to_host(host, device, bytes, stream)`, `copy_device_to_device(destination, source, bytes, stream)` |
| events | `create_event(&event, timing)`, `destroy_event(event)`, `record_event(event, stream)`, `query_event(event, &complete)`, `wait_event(stream, event)`, `elapsed_nanoseconds(from, to, &nanoseconds)` |
| facts | `capabilities(&out)`, `physical_memory(&out)`, `statistics(&out)` |
| profiler, optional | `name_thread(name)`, `name_stream(stream, name)`, `profiler_enable(enabled)`, `range_begin(name)`, `range_end(range)` |

Memory: device memory is the backend's to allocate; host memory is
ShadowSpill's, mapped by the pinned-host pool and registered here so the
provider can copy from it asynchronously. Frees and unregistrations carry the
byte count so the backend keeps no size bookkeeping.

Streams are ordered queues of copies and events. Copies are asynchronous and
ordered on their stream. `wrap_stream` turns the integer handle the framework
exposes for one of its own streams into a token.

Events: a dependency event (`timing` clear) is the fast kind that record,
query, and wait work with. A timing event carries a device timestamp when
recorded, and `elapsed_nanoseconds` reads the device-clock interval between
two of them: 0 with the interval, 1 while either is still pending, -1 when the
pair cannot be measured. `record_event` and `wait_event` enqueue without
blocking the host; `query_event` is a nonblocking poll.

Profiler entries are best-effort diagnostics and never change execution
semantics; a NULL entry is a no-op.

## The two symbols

```c
int shadowspill_backend_create(const ShadowSpillBackendConfig *config,
                               ShadowSpillBackend *backend);
void shadowspill_backend_destroy(ShadowSpillBackend *backend);
```

`shadowspill_backend_create()` fills the table and returns 0, or returns
nonzero leaving nothing to destroy. `shadowspill_backend_destroy()` releases
the provider object and zeroes the table; it
runs after the runtime it served is gone, so every stream, event, mapping, and
arena has already been returned through the table.
`SHADOWSPILL_BACKEND_CREATE_SYMBOL` and `SHADOWSPILL_BACKEND_DESTROY_SYMBOL`
name them for `dlsym()`. `shadowspill_backend_is_valid()` in
`<shadowspill/runtime.h>` is the check both the runtime and the adapter apply
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
