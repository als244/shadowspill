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

Four kinds ship today. The runtime implements the first two; remote and SSD
come from extension libraries registered by the caller. The runtime does not
load or link those libraries; the same lookup reaches all four.

| kind | `acquire` | `release` |
|---|---|---|
| device | the backend's `allocate_device` | `free_device` |
| pinned host | an anonymous private mapping the pool makes itself, page-aligned and untouched by the C allocator, which the backend then registers with `register_host_memory` so the provider can copy from it asynchronously | unregisters, then unmaps, in that order |
| remote | a peer allocates and registers the region; what comes back is an address in *its* space | the peer frees it |
| SSD | a preallocated, unlinked direct-I/O file and a non-dereferenceable address token | closes the file and releases the token and bounded state-I/O scratch |

SSD setup, staging limits and lifecycle are described in the
[C extension guide](../c/ssd.md) and [Python API](../python/api/ssd.md).

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

Transfer [lanes](lanes.md) access the two pool kinds they connect. Bounded object
read/write calls use the [pool-memory contract](../c/pool-memory.md), which routes
non-addressable storage through the kind's callbacks. Executor's `SpillStorage`
uses these calls for initialization and checkpoint I/O. The object registry
tracks registrations and leases; it does not stage tensor payloads.

Execution-pool addresses are deliberately exposed to compiled tasks. Their
lifetimes follow the Plan's allocation and completion dependencies. A remote or
SSD address is never exposed as a local tensor pointer.

## Execution owns tensor values

Build and planning record geometry, dependencies and memory requirements without
creating runtime pools. Executor creates the pools, reserves the declared initial
storage, then fills it through bounded tensor requests. Checkpoint loading and
random initialization use the same interface; see [initialization](state-import.md).

Addressable spill pools are filled in place. For remote memory and SSD, each
request uses a bounded host buffer and publishes it through the backend's write
operation. Scratch is reused for the next chunk. No model-sized staging copy is
required, including for optimizer tensors and distributed shards.

Export uses bounded read buffers. Checkpoint helpers stream those buffers to a
separate archive; the temporary spill pool itself is not a checkpoint. Releasing
storage does not implicitly export its values. Gradients, activations and
workspaces acquire their planned runtime storage when produced.

## A pool answers for itself

`shadowspill_memory_pool_statistics` reports one pool's numbers: capacity, what
is allocated and free, the largest free range and the external fragmentation
that follows from it, the live allocation count, and the lease-record reserves.
The runtime reports only `pool_count` and what no pool knows, because a runtime
may own any number of pools and a pool has no role of its own -- which pools
serve as a given plan's execution and spill pools is that plan's choice, so
numbers named for those roles would belong to the plan, not here.

## Construction order

Communication calibration is an optional phase before planning. It measures the
selected spill backend using temporary pools and saves `CommunicationInfo`;
callers can also supply rates directly. Model/task profiling uses the ordinary
framework allocator and needs no runtime pool.

A fresh execution worker installs the allocator, admits the saved Plan and
creates its pools before initializing tensor values. It uses the saved costs and
layout without recalibrating or replanning. This keeps large spill allocations
independent of anonymous full-model initialization and preserves the phase boundary.

Previous: [Backends](backends.md). Next: [Lanes](lanes.md).
