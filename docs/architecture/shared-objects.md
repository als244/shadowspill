# Shared objects

One value, several plans, charged once. A model's parameters do not want a copy
per plan, and a tensor one plan produces is sometimes the tensor another
consumes. A shared object is how a value outlives the plan that created or
imported it, and how more than one plan reaches the same bytes.

## Object, location, lease

Three nouns, often conflated, and the distinction is what makes the rest legible.

An **object** is a logical value the runtime owns: an id, a size, a generation,
an authoritative version, and a residency state. It is not storage.

A **location** is that object's copy in one pool — one per pool. It holds the
lease backing that copy, the version that copy carries, whether the copy is
current, and whether the object owns the lease or is using one placed for it:

```c
typedef struct ShadowSpillObjectLocation {
    ShadowSpillMemoryLease *lease;
    uint64_t version;
    uint8_t current;
    uint8_t owns_lease;
} ShadowSpillObjectLocation;
```

A **lease** is the allocation: bytes at an offset in one pool, with the origin of
the work that made it ([plan identity](plan-identity.md)).

So an object can exist with no storage at all, with a copy in spill only, with a
copy in both pools where one is stale, or with a copy in the execution pool whose
lease it does not own. `shadowspill_object_location_snapshot` reads that state;
[the boundary-state model](memory-runtime.md) is where those bits are reasoned
about.

## Binding names an object; it does not allocate

`shadowspill_plan_bind_object(plan, plan_object_id, handle, consistency)` maps a
**plan-local** object id — the alias the program numbered — to a runtime object,
and takes an independent reference to it. That is all it does. No lease is
created, no bytes are charged, and the plan does not gain storage by binding.

Two plans binding the same object therefore name one set of locations under two
plan-local ids, which is exactly the point.

Leases arrive by one of three routes:

- **Resident registration.** `shadowspill_register_object` with
  `initially_resident` reserves storage in the pool `initial_pool_id` names, and
  the object owns that lease. One that is not resident holds no lease until a
  task publishes into it.
- **A fetch.** A plan's fetch action reserves the destination lease in the
  execution pool at the task boundary, so that lease records the plan whose
  action moved the bytes.
- **A task publication.** A task allocates, writes, and publishes into a
  non-resident object, promoting its own allocation into that object's location.

The middle route is why a shared object's device copy is not anonymous: it is
attributable to the plan that asked for it, even though the object is not.

## Lifetime

A handle is a reference. `shadowspill_object_handle_acquire` takes one and
`shadowspill_object_handle_release` gives it back;
`shadowspill_object_release_generation` releases a particular generation and
refuses a generation that is not current, so a release cannot silently discard a
value a later writer produced.

`shadowspill_unregister_object` may be called while plans are still bound. The
object survives until its final owner closes — both plans destroyed and the public
reference released — which is what
`runtime_objects_survive_until_their_final_owner_closes` in
`tests/csrc/runtime/runtime_plan_canary.c` covers. `retain_spill_copy` keeps the
spill copy authoritative while a device copy also exists.

## Ordering between plans

`ShadowSpillObjectConsistency` is per binding: `CAUSAL`, where every reader and
writer acquires the current generation and its readiness dependency, or
`UNORDERED`, where cross-plan visibility is intentionally not ordered. The
residency policies an alias group may declare — `SHARED_READ_ONLY`,
`SHARED_WRITABLE_CAUSAL`, `SHARED_WRITABLE_UNORDERED` — and which mutations each
permits are in [the program](program.md#what-it-holds).

## Declaring it from PyTorch

`shared_input(reference, require_in=..., consistency=..., profiling_value=...)`
binds an existing runtime-backed `TensorRef` at one input position instead of
copying a tensor in, and names the pool that reference must be resident in.
`shared_output(*path, retain_in=...)` names an output pytree leaf that stays
runtime-owned, with the pool or pools it must be retained in. A `TensorRef` is
public view metadata over an `ObjectRef` — dtype, shape, stride, offset,
generation — captured without retaining a tensor address, and a `StateRef` is a
mapping of them. [Concurrent callables](../examples/concurrent-callables.md) is
the worked example.

## What it means for cleanup

A closing plan releases the ranges its own scopes made. A shared object's
storage is the case that must survive that, and how it is recognised depends on
the route the lease came by:

- Storage the runtime reserved at registration records its own scope,
  `SHADOWSPILL_RUNTIME_OBJECT_SCOPE_ID`, and **no plan**. No plan may reclaim it,
  because no plan made it.
- A fetch destination records the plan whose action moved the bytes, so a device
  copy of a shared object is attributable to that plan even though the object is
  not.

## Which pool, and why that is a frontend choice

Nothing in the runtime ties a registered object to one pool. Pools are a flat
array with ids; *execution* and *spill* are **roles a plan assigns** in its
`ShadowSpillPlanDescription`, and `shadowspill_register_object` takes any
`initial_pool_id`. A runtime object resident in the execution pool is a legal
configuration the C layer supports today.

The frontend nonetheless puts every registration in the spill pool --
the bridge (`runtime_adapter/bridge/objects.py`) registers host objects and
placeholders with `pool_id=spill_pool_id`,
and state adoption passes the spill pool -- and that is the right policy rather
than a missing generalization. The execution pool is the scarce device arena a
fixed layout reserves in one contiguous span. Storage resident there that the
plan did not place occupies the arena outside the layout, which is precisely what
costs the largest free range. Imported state belongs in the spill pool and is
fetched in on the schedule the plan priced.

So today no shared object's own lease sits in the execution arena, by choice.

## One genuine asymmetry, and it is in the runtime

Per-lease bookkeeping is attached to the lease *creation* path rather than to "a
lease was reserved in some pool". `publish_lease_record_locked` is what links a
lease into `active_leases`, indexes it by id and by pointer, counts it in
`live_allocations`, and emits an allocation event -- and only
`own_and_publish_lease_locked` and `publish_successor_lease_locked` call it. A
spill copy is reserved and never published.

Three consequences, all from that one fact:

- `shadowspill_memory_pool_live_allocations` asked of a spill pool succeeds and
  reports nothing.
- `allocation_for_pointer` cannot resolve a spill address, because the pointer
  index is built at publication.
- A registered object's storage is absent from the enumeration in *either* pool.
  Its `SHADOWSPILL_RUNTIME_OBJECT_SCOPE_ID` is recorded on the lease, and read
  through the object registry rather than through the pool's lease list.

Byte accounting is unaffected and already generic: a pool's allocated and peak
bytes come from its range allocator, which every reservation updates. So
`shadowspill_memory_pool_statistics()` asked of the spill pool reports its
`allocated_bytes` and `peak_allocated_bytes` correctly while that pool's lease
list is empty. The fields publication feeds -- `live_allocations` and
`requested_allocated_bytes` -- are the ones that read as zero there.

Previous: [Plan identity](plan-identity.md). Next: [Memory
runtime](memory-runtime.md).
