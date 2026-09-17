# Lane contract

`include/shadowspill/runtime/lane.h` — what moves bytes between two pools, and
how the runtime finds one. A transport implements this table; the runtime calls
it and never asks what kind of transport it is.

The contract is **runtime-owned**, the way the [backend contract](backends.md)
is: the types are fields on `ShadowSpillRuntimeConfig`, and a lane written
elsewhere is built against these headers with nothing else in view.

For what a lane is and why the contract has the shape it does, see
[lanes](../architecture/lanes.md).

## Types

| type | |
|---|---|
| `ShadowSpillLane` | one lane, made per route at create. Incomplete: an implementation casts to its own type |
| `ShadowSpillLaneOperations` | the table below |
| `ShadowSpillLaneDescription` | one entry in `ShadowSpillRuntimeConfig.lanes`: a kind pair, a table, a `create`, and a `configuration` |
| `ShadowSpillStreamInterval` | runtime-internal; a lane passes one through the interval entries and never looks inside |
| `ShadowSpillLaneStatistics` | what a lane has moved, filled by the optional `statistics` entry below |

## Operations

```c
int  (*wait)(ShadowSpillLane *lane, ShadowSpillBackendEvent event);
int  (*copy)(ShadowSpillLane *lane, void *destination, const void *source,
             uint64_t bytes);
int  (*signal)(ShadowSpillLane *lane, ShadowSpillBackendEvent event);
int  (*synchronize)(ShadowSpillLane *lane);
int  (*interval_open)(ShadowSpillLane *lane, ShadowSpillStreamInterval *interval);
int  (*interval_close)(ShadowSpillLane *lane, ShadowSpillStreamInterval *interval);
int  (*statistics)(const ShadowSpillLane *lane,
                   ShadowSpillLaneStatistics *statistics);
void (*destroy)(ShadowSpillLane *lane);
```

- **`wait`** orders this lane's work behind `event`. Returns 0 enqueued or
  already satisfied, **1 to be retried at the next poll**, -1 failed. A lane
  whose waits are device-side never returns 1. On 1 the action returns to the
  head of its queue, so a retry does not reorder what the task boundaries
  triggered.
- **`copy`** moves bytes between addresses in the two pools this lane connects.
  Nothing but the lane dereferences either.
- **`signal`** makes `event` complete once everything issued on this lane so far
  has landed. Downstream sees an ordinary backend event whatever the lane did,
  which is what keeps completion tracking and retirement in one form.
- **`interval_open` / `interval_close`** are NULL-able, and go **together** —
  one without the other cannot produce a readable interval. NULL means this
  lane's transfers are recorded untimed.

An event is a `ShadowSpillBackendEvent`, one opaque word, and never the
runtime's event lease.

**There is no entry for the runtime to drive a lane with.** A lane makes the
event given to `signal` complete when the bytes have landed, by whatever means
suits it — the device ordering it behind a stream, or a thread of the lane's own
watching its transport. See [lanes](../architecture/lanes.md) for why the table
once had a `poll` and no longer does.

## Registering one

```c
typedef struct ShadowSpillLaneDescription {
    uint8_t from_kind;                 /* ShadowSpillPoolKind, source */
    uint8_t to_kind;                   /* ShadowSpillPoolKind, destination */
    const ShadowSpillLaneOperations *operations;
    int (*create)(ShadowSpillRuntime *runtime,
                  const ShadowSpillBackend *backend,
                  ShadowSpillBackendStream stream,
                  void *configuration,
                  ShadowSpillLane **lane);
    void *configuration;
} ShadowSpillLaneDescription;
```

Put entries in `ShadowSpillRuntimeConfig.lanes` with `lane_count`. The runtime
seeds its built-ins first and appends these, so one lookup resolves both. **Two
entries claiming one directional kind pair fails create** with
`SHADOWSPILL_STATUS_INVALID_ARGUMENT`; create returns a status and carries no
message, so which pair collided is not reported.

`create` is called once per route, released in reverse at close, and receives:

- `runtime` — the lane's way back in. A lane that discovers a failure on a
  thread of its own has no return value to fail through, so it latches the
  failure itself. The latch is built for concurrent callers and **first writer
  wins**, so the original cause survives; the worker already reads the latched
  status twice a turn, so nothing is added to its loop.
- `backend` — usable, but only through the stream below.
- `stream` — created by the runtime **for this lane**, and not the route's
  stream, which has one writer. For the built-in lane the two are the same,
  because it is acting as the runtime's own copy mechanism.
- `configuration` — passed through untouched. Whoever registers the entry
  decides what it means.

It is also where a lane **probes**: nothing a loaded library holds runs at load,
so anything that depends on what the hardware can do belongs here, where there
is a failure path and an unwind.

## What a lane has moved

```c
typedef struct ShadowSpillLaneStatistics {
    uint64_t copies;      /* transfers accepted */
    uint64_t chunks;      /* pieces the hardware was handed */
    uint64_t bytes;       /* bytes accepted, summed over copies */
    uint64_t signals;     /* completion signals issued */
    uint64_t waits;       /* dependency waits enqueued */
    uint64_t retries;     /* waits that asked to be retried */
    uint64_t failures;    /* transfers that did not land */
    uint8_t  timed;       /* whether the durations below mean anything */
    double   posted_to_completion_seconds;
    double   longest_completion_seconds;
} ShadowSpillLaneStatistics;
```

**The `statistics` entry is optional**, on the same rule as the interval pair:
required when the runtime cannot proceed without it, optional when its absence
only costs observability. A lane that keeps no count leaves it NULL, and
`shadowspill_route_lane_statistics()` answers `SHADOWSPILL_STATUS_UNSUPPORTED`
rather than zeroes -- which a reader could not tell from a lane that moved
nothing.

Counters are maintained **unconditionally**. A lane that counts only when asked
cannot explain the run that went wrong, and the cost is a few relaxed atomic
adds against a transfer measured in milliseconds.

The two durations are the exception and are governed by `timed`. They describe
the interval a lane can actually observe -- from handing the hardware a piece of
work to seeing its completion -- which is not time on the wire, and separating
those is the point of reporting it. A lane that hands a transfer onward and
returns never sees the completion instant at all; it reports `timed = 0` and
leaves the durations zero, because a zero duration otherwise reads as "instant".
A lane that can time cheaply may still do so only while a trace is running, and
`shadowspill_lane_trace_active()` is how it asks, being outside the runtime and
unable to read that state itself.

`chunks` differs from `copies` only for a lane that splits a transfer; where it
does not, the two are equal by construction. That is the difference between
"how many transfers" and "how many times the hardware was asked".

One lane serves one directional pair of pool kinds, so these are per route and
per direction, and they accumulate for the life of the lane. The
[step diagnostics](../python/step-diagnostics.md) report them beside the
per-step view the trace gives, which is a different question and not a
substitute.

## Validity

`shadowspill_runtime_create()` refuses a config whose entry has a NULL
`operations` or `create`, a NULL required entry, one interval entry without the
other, or a kind pair another entry already claims. It also refuses a route
whose two pools' kinds no entry serves. `statistics` is optional and is never
a reason to refuse.
