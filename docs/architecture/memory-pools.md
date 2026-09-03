# Memory pools

The runtime's memory objects live in `csrc/src/runtime/memory/`. They are
ShadowSpill's, built on the [backend contract](../c/backends.md): a backend
allocates device memory and pins host memory; everything with a lifetime or a
policy is here. Routes and lanes are in [transfers](transfers.md), and the
events that protect a range in [events](events.md).

## Pools and arenas

A `MemoryPool` is a range owner registered by identity, with one arena and a
kind. A device pool's arena comes from the backend's `allocate_device`. A
pinned-host pool's arena is an anonymous private mapping the pool makes
itself, page-aligned and untouched by the C allocator, which the backend then
registers with `register_host_memory` so the provider can copy from it
asynchronously; release unregisters and unmaps in that order. Frees and
unregistrations carry the byte count, so the backend keeps no size table.

The pool knows ownership and dependencies, not transfer meaning: a
`MemoryLease` owns one range for one residency generation, and transfer
components create, acquire, cancel, and publish reservations through the pool
API. Budgets, lease states, and shared leases are described in
[memory runtime](memory-runtime.md).

A pool can grow at an idle boundary (`shadowspill_memory_pool_grow()`): the
runtime takes a larger arena of the same kind, copies the live bytes, releases
the old arena, and rebases every lease.

## Construction order

Runtime construction precedes workload-state construction: the runtime first
creates every configured pool and calibrates each route using ranges from
those actual arenas, then workload state is constructed and imported. That
keeps the physical pages and DMA mapping of a large pinned spill arena
independent of earlier anonymous model allocations, and gives planning a
transfer profile measured on the memory the step will use.
