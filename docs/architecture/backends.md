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
                 |  copies, events, capabilities/physical memory/statistics,
                 |  profiler names and ranges
                 v
             backend (libshadowspill_backend_<provider>.so, ..._mock.so)
```

One header, one version, and two exported symbols. The runtime is handed the
table at create and copies it; the adapter obtains the table by opening a
library by name, so nothing above the backend links a provider or includes a
provider header.

## What ShadowSpill builds from the table

Each has its own page: [memory pools](memory-pools.md), [lanes](lanes.md),
[transfers](transfers.md), [events](events.md).

- **Pools** own their memory, and a pool's kind decides where it comes from.
  Two kinds get theirs from here: a device pool's from `allocate_device`, and a
  pinned-host pool's from an anonymous mapping the pool makes and hands to
  `register_host_memory`, so the C allocator never touches it and the provider
  only pins it in place. A kind served from somewhere else reaches the backend
  not at all -- see [memory pools](memory-pools.md#a-pools-memory-is-found-by-kind).
- **Routes** are a source pool and a destination pool. Their two kinds select
  the route's **lane**, which is what actually moves the bytes; the built-in one
  is a thin table over `copy_host_to_device` and `copy_device_to_host` on a
  stream from `create_stream`. The worker dispatches every transfer through the
  lane and makes no backend call of its own, and calibration goes the same way.
- **Signal words** are memory a stream can wait on and a host thread can store
  to, from `allocate_signals`; `wait_value` holds a stream until a word reaches a
  generation. They exist for a lane whose bytes do not move on a stream, which
  cannot make a stream wait by recording an event -- a consumer's wait may be
  enqueued before the transfer finishes, and a wait on an unrecorded event does
  not wait. A provider whose streams read host memory through a mapping needs the
  block allocated so that mapping exists, which is why the entry names a word by
  index rather than by pointer.
- **Event pools** keep backend events across leases. Reserving a pool creates
  its events up front and seals it, so a steady-state step makes no driver
  calls; the runtime's statistics count creates after sealing. Events that
  carry a device timestamp come from a second pool, reserved when a trace is
  prepared: the worker brackets traced copies with them, and a caller timing
  its own work takes markers from the same pool, so a step and the transfers
  inside it are measured on one clock.
- **Profiling** goes through the optional profiler entries. The runtime owns
  the ranges and the flag that turns them on, so a frontend opens a range on
  the runtime rather than keeping a profiler of its own; a backend without one
  leaves the entries NULL and every range is a no-op.

## Why the boundary is here

A driver-level table is what every platform has in common, so it is the
largest contract that stays honest across providers, and it is the smallest
one that lets ShadowSpill own every decision that matters for planning:
what is allocated where, which stream carries which copy, when events are
created, and what gets measured. A backend that created lanes or pooled events
would be making those decisions twice.

## Choosing a backend

`Runtime(backend=None)`, the default, requires exactly one accelerator backend
installed beside the ShadowSpill libraries and refuses to guess otherwise; the
mock is never a candidate, so it has to be named. A name resolves to
`libshadowspill_backend_<name>.so` there, using the same lookup as the runtime
library itself, and a path is used as given. The build selects which providers
to compile the same way: every provider whose toolchain is installed, or the
ones named in `SHADOWSPILL_BACKENDS`.

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

Previous: [Shared objects](shared-objects.md). Next: [Memory
pools](memory-pools.md).
