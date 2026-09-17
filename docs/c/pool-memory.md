# Pool memory contract

`include/shadowspill/runtime/pool_memory.h` — where a pool's memory comes from,
and how the runtime finds out. A kind of memory implements this pair; the
runtime calls it and never asks what kind it is.

The contract is **runtime-owned**, the way the [backend contract](backends.md)
and the [lane contract](lanes.md) are: the type is a field on
`ShadowSpillRuntimeConfig`, and a kind implemented elsewhere is built against
these headers with nothing else in view.

For why a pool's memory is a contract at all, see
[memory pools](../architecture/memory-pools.md#a-pools-memory-is-found-by-kind).

## Types

| type | |
|---|---|
| `ShadowSpillPoolMemoryDescription` | one entry in `ShadowSpillRuntimeConfig.pool_memory`: a kind, an `acquire`, a `release`, and a `configuration` |
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

## Registering one

```c
typedef struct ShadowSpillPoolMemoryDescription {
    uint8_t kind;                /* a ShadowSpillPoolKind value */
    int (*acquire)(void *configuration, uint64_t capacity,
                   void **base, void **state);
    int (*release)(void *state, void *base, uint64_t capacity);
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
`release`, or a kind another entry already claims. It refuses a **pool** whose
kind no entry serves — which is the only place a kind is judged. There is no
range check on `ShadowSpillPoolKind`, deliberately: a bound would have to widen
for every kind a library adds, and it would be a second opinion about a
question the lookup already answers.
