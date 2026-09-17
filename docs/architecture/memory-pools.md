# Memory pools

The runtime's memory objects live in `csrc/src/runtime/memory/`. They are
ShadowSpill's: everything with a lifetime or a policy is here. Where a pool's
region *comes from* is a kind's business, and only some kinds ask the
[backend](../c/backends.md) for it. Routes are in
[transfers](transfers.md), what moves bytes between two pools in
[lanes](lanes.md), and the events that protect a range in [events](events.md).

## Pools and their memory

A `MemoryPool` is a range owner registered by identity, with one region of
memory and a kind. What varies between kinds is only how that region is
obtained and what an address in it means; ranges, leases, retirement and the
release frontier are indifferent to both.

The pool knows ownership and dependencies, not transfer meaning: a
`MemoryLease` owns one range for one residency generation, and transfer
components create, acquire, cancel, and publish reservations through the pool
API. Budgets, lease states, and shared leases are described in
[memory runtime](memory-runtime.md).

A pool's memory is fixed for its lifetime. It is taken at create and released at
close, and nothing resizes it in between -- so the address a lease resolves to
never moves, and no code outside the pool has to cope with it moving. Nothing
grows a pool either, and that is not an omission: growth is the one operation
that would have to copy a pool's payload, which is the one thing nothing here
can do.

## A pool's memory is found by kind

A pool's kind selects an `acquire`/`release` pair from the `pool_memory` list
on `ShadowSpillRuntimeConfig`. That is the whole of the mechanism: **there is no
branch anywhere asking what kind of memory a pool has**, and the two kinds the
runtime implements are registered exactly as a kind from a loaded library is,
so one lookup resolves both.

Three kinds exist today. The runtime implements the first two; the third comes
from a library the runtime does not load and does not link, and is reached by
the same lookup as the others.

| kind | `acquire` | `release` |
|---|---|---|
| device | the backend's `allocate_device` | `free_device` |
| pinned host | an anonymous private mapping the pool makes itself, page-aligned and untouched by the C allocator, which the backend then registers with `register_host_memory` so the provider can copy from it asynchronously | unregisters, then unmaps, in that order |
| remote | a peer allocates and registers the region; what comes back is an address in *its* space | the peer frees it |

Frees and unregistrations carry the byte count, so the backend keeps no size
table.

Two entries claiming one kind fails `shadowspill_runtime_create()`, and so does
a pool whose kind no entry serves. **That lookup is the only place a kind is
judged**; there is no range check on the enum, which is a list of the kinds
shipped here rather than a bound.

Whatever a kind needs to remember in order to give the region back -- a
connection, a key, a handle -- it returns from `acquire` as an opaque state
that comes back to `release` untouched. And whatever distinguishes two pools of
one kind -- which machine, which device -- reaches `acquire` as the *pool's*
configuration, not the kind's, so no pool description field ever has to name
one.

### Nothing reads through a pool address

An allocation's pointer is assigned, compared, handed out, and **never
dereferenced** by anything in the memory subsystem. Offsets are computed
against the base; the base itself is never read through.

That is what makes a region on another machine the same object as a local one,
and it is the reason the kind lookup is affordable rather than a layer of
indirection over a switch. It also stopped being a happy accident: the one
routine that violated it -- pool growth, which copied the payload into a larger
region -- was deleted rather than left to be the exception that would make a
remote pool crash.

Inside the runtime, a [lane](lanes.md) is the only thing that reads through a
pool address, and only for the two pools it was made to connect. An address
handed *out* is another matter: a plan's execution pool exists precisely so a
caller can read and write through what it is given. That is a difference
between the two *roles* a plan assigns, not between kinds, and it is why a
pool's kind constrains who may hold its addresses without changing anything
about how the pool works.

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

## A pool answers for itself

`shadowspill_memory_pool_statistics` reports one pool's numbers: capacity, what
is allocated and free, the largest free range and the external fragmentation
that follows from it, the live allocation count, and the lease-record reserves.
The runtime reports only `pool_count` and what no pool knows, because a runtime
may own any number of pools and a pool has no role of its own -- which pools
serve as a given plan's execution and spill pools is that plan's choice, so
numbers named for those roles would belong to the plan, not here.

## Construction order

Runtime construction precedes workload-state construction: the runtime first
creates every configured pool and calibrates each route using ranges from
those actual pools, then workload state is constructed and imported. That
keeps a large spill pool's physical pages, and the DMA mapping a pinned one
takes, independent of earlier anonymous model allocations, and gives planning a
transfer profile measured on the memory the step will use.

Previous: [Backends](backends.md). Next: [Lanes](lanes.md).
