# Pool memory contract

`include/shadowspill/runtime/pool_memory.h` — where a pool's memory comes from,
and how the runtime finds out. A kind of memory implements it; the runtime
calls it and never asks what kind it is.

The contract is **runtime-owned**, the way the [backend contract](backends.md)
and the [lane contract](lanes.md) are: the type is a field on
`ShadowSpillRuntimeConfig`, and a kind implemented elsewhere is built against
these headers with nothing else in view.

For why a pool's memory is a contract at all, see
[memory pools](../architecture/memory-pools.md#a-pools-memory-is-found-by-kind).

## Types

| type | |
|---|---|
| `ShadowSpillPoolMemoryDescription` | one entry in `ShadowSpillRuntimeConfig.pool_memory`: a kind, an `acquire`, a `release`, an optional `write`/`read` pair, and a `configuration` |
| `ShadowSpillPoolKind` | the kinds shipped here. **A list, not a bound** — nothing validates a kind by comparing against the last value |

## Operations

```c
int (*acquire)(void *configuration, uint64_t capacity,
               void **base, void **state);
int (*release)(void *state, void *base, uint64_t capacity);
```

- **`acquire`** obtains `capacity` bytes and reports where they start. Called
  **once per pool at create**, on the thread that creates the runtime, and
  reports failure by returning non-zero. Both entries are required.
- **`state`** is whatever the kind must remember in order to give the region
  back — a connection, a key, a handle — handed to `release` untouched. A kind
  with nothing to remember leaves it NULL.
- **`release`** gives the region back, receiving exactly what `acquire`
  produced.

**`base` need not be dereferenceable by this process.** It is an address in the
pool's own space, and the only arithmetic anything does on it is adding an
offset. Whether that yields something this machine can read is the kind's
business; see [what the runtime never does with a pool
address](../architecture/memory-pools.md#nothing-reads-through-a-pool-address).

## Crossing the pool's edge

```c
int (*write)(void *state, uint64_t offset,
             const void *source, uint64_t bytes);
int (*read)(void *state, uint64_t offset,
            void *destination, uint64_t bytes);
```

**Optional, and optional together.** Leaving both NULL says this process can
dereference the region, so the runtime moves bytes with an ordinary copy; both
built-in kinds do that. A kind whose region is not in this address space
implements both, and must, because the alternative is a fault on the first
byte.

`source` and `destination` are pointers **in the runtime process**. That is the
honest statement of it: not "host memory", which would promise something about
the machine, but "an address this process can use". Every caller already holds
one, because every caller is inside this process. Serving a client that is not
would need a different entry than either of these, and nothing here pretends
to.

`offset` is measured from the pool's base, so neither entry needs to know what
a pool address means -- the same property that lets `acquire` report a base
this process cannot read.

They exist because moving bytes across a pool's edge is a property of the
memory, not of a transfer. Importing state and reading a checkpoint back are
not scheduled transfers on any route: they happen outside a plan, against
ordinary memory the caller owns. Routing either through a [lane](lanes.md)
would need a lane for a pair of kinds that is not a route. Both are called from
the thread importing or exporting state, never from the worker, and are
synchronous: when one returns 0 the bytes have landed.

## Registering one

```c
typedef struct ShadowSpillPoolMemoryDescription {
    uint8_t kind;                /* a ShadowSpillPoolKind value */
    int (*acquire)(void *configuration, uint64_t capacity,
                   void **base, void **state);
    int (*release)(void *state, void *base, uint64_t capacity);
    int (*write)(void *state, uint64_t offset,        /* optional, */
                 const void *source, uint64_t bytes); /* with read */
    int (*read)(void *state, uint64_t offset,
                void *destination, uint64_t bytes);
    void *configuration;
} ShadowSpillPoolMemoryDescription;
```

Put entries in `ShadowSpillRuntimeConfig.pool_memory` with
`pool_memory_count`. The runtime seeds its built-ins first and appends these,
so **one lookup by kind resolves both**. Two entries claiming one kind fails
`shadowspill_runtime_create()` with `SHADOWSPILL_STATUS_INVALID_ARGUMENT` —
order must never decide it silently, because whichever lost would be invisible
and every allocation in that pool would quietly come from the other. Create
returns a status and carries no message, so which kind collided is not
reported; a caller that registers kinds knows which it supplied.

## Configuration is per pool, not per kind

`configuration` appears twice, and the pool's wins:

| field | what it is |
|---|---|
| `ShadowSpillPoolMemoryDescription.configuration` | the kind's default, from whoever registered the entry |
| `ShadowSpillMemoryPoolDescription.configuration` | **this pool's**, and what `acquire` receives when it is non-NULL |

Two pools of one kind that differ — two remote pools on different machines,
say — share the entry and differ only here. That is why no description field
ever has to name a machine, a device or an address: the kind is told what it
needs through a pointer only it and its caller understand.

## Validity

`shadowspill_runtime_create()` refuses an entry with a NULL `acquire` or
`release`, one that supplies exactly one of `write` and `read`, or a kind
another entry already claims. The pair is checked here rather than left to
whichever direction is used first: a kind that could take state and not give it
back would import a model and then fail to export it, at the point the values
were wanted rather than at the create that registered it. It refuses a **pool** whose
kind no entry serves — which is the only place a kind is judged. There is no
range check on `ShadowSpillPoolKind`, deliberately: a bound would have to widen
for every kind a library adds, and it would be a second opinion about a
question the lookup already answers.
