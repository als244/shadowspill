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
the old arena, and rebases every lease. Both arenas are held while the copy
runs, so a caller has to budget for that transient; the refusals are in the
[runtime C API](../c/runtime.md).

## Memory a plan will own comes from the pool that will own it

Planning creates state the plan will keep, and the largest of it is an
optimizer's: several times the model for an ordinary adaptive optimizer. That
state is taken from the spill pool as it is created rather than built in
ordinary host memory and copied in, so the host is never asked for the whole
of it beside the pool that is about to hold it. A host allocation made while
that state is being created, and large enough to be worth an object, is served
from the pool; anything smaller, and anything a plan does not keep, is an
ordinary host allocation and is given back.

The rest of what a plan owns obeys the same rule. Gradients, activations and
workspaces are runtime objects created in a pool, and the tensors a program is
lowered from are fake and cost nothing. Model and optimizer state reaches a
pool without a second copy of itself by one of the paths in [importing
state](state-import.md).

## Construction order

Runtime construction precedes workload-state construction: the runtime first
creates every configured pool and calibrates each route using ranges from
those actual arenas, then workload state is constructed and imported. That
keeps the physical pages and DMA mapping of a large pinned spill arena
independent of earlier anonymous model allocations, and gives planning a
transfer profile measured on the memory the step will use.
