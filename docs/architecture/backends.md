# Backends

A backend is the one component that knows an accelerator platform, and it
knows nothing else: it is a flat table of driver-level calls. Everything with
a lifetime or a policy, pools, routes, transfer lanes, event pools, calibration,
tracing, is ShadowSpill's and is built from those calls. This page is about
where that boundary sits and why; the reference for every entry is the
[backend contract](../c/backends.md).

## The boundary

```text
PyTorch  ->  adapter (libshadowspill_pytorch.so)
                 |  dlopen("libshadowspill_backend_<provider>.so")
                 |  shadowspill_backend_create() -> ShadowSpillBackend
                 v
             neutral runtime (libshadowspill.so): pools, routes, lanes,
             event pools, calibration, tracing, the worker
                 |  allocate/free device, register host memory, streams,
                 |  copies, events, facts, profiler names and ranges
                 v
             backend (libshadowspill_backend_<provider>.so, ..._mock.so)
```

One header, one version, and two exported symbols. The runtime is handed the
table at create and copies it; the adapter obtains the table by opening a
library by name, so no compiled component above the backend links a provider
or includes a provider header.

## What ShadowSpill builds from the table

Each has its own page: [memory pools](memory-pools.md), [transfers](transfers.md),
[events](events.md).

- **Pools** own their arenas. A device pool's arena comes from
  `allocate_device`; a pinned-host pool's arena is an anonymous mapping the
  pool makes and hands to `register_host_memory`, so the C allocator never
  touches it and the provider only pins it in place.
- **Routes** are a source pool, a destination pool, and the copy direction
  the two pools' kinds imply. The runtime creates one lane per route with
  `create_stream`, the worker dispatches every copy onto it, and calibration
  measures each route alone and against its reverse on those lanes.
- **Event pools** keep backend events across leases. Reserving a pool creates
  its events up front and seals it, so a steady-state step makes no driver
  calls; the runtime's statistics count creates after sealing. Timing events
  for traced transfers come from a second pool reserved when a trace is
  prepared.
- **Profiling** goes through the optional profiler entries; a backend without
  a profiler leaves them NULL and the runtime treats them as no-ops.

## Why the boundary is here

A driver-level table is what every platform has in common, so it is the
largest contract that stays honest across providers, and it is the smallest
one that lets ShadowSpill own every decision that matters for planning:
what is allocated where, which stream carries which copy, when events are
created, and what gets measured. A backend that created lanes or pooled events
would be making those decisions twice.

## Choosing a backend

`Runtime(backend=None)`, the default, selects the one accelerator backend
installed beside the ShadowSpill libraries and refuses to guess when there are
several. A name resolves to `libshadowspill_backend_<name>.so` there, using the
same lookup as the runtime library itself, and a path is used as given. The
build selects which providers to compile the same way: every provider whose
toolchain is installed, or the ones named in `SHADOWSPILL_BACKENDS`.

## Adding a provider

1. Create `csrc/backends/<provider>/` and compile it against
   `<shadowspill/backend.h>` alone.
2. Implement every entry of the table; report zero for counters the platform
   has no notion of, and leave the profiler entries NULL if it has none.
3. Export the two symbols and register the provider in
   `csrc/backends/CMakeLists.txt` so it builds as
   `libshadowspill_backend_<provider>.so` next to the others.
4. Run the contract canary against the library, then the runtime canaries and
   the PyTorch canaries with `Runtime(backend="<provider>")`.
