# Lane contract

`include/shadowspill/runtime/lane.h` — what moves bytes between two pools, and
how the runtime finds one. A transport implements this table; the runtime calls
it and never asks what kind of transport it is.
`include/shadowspill/runtime/lane_base.h` is its companion: the struct a
transport embeds. Only an implementation needs it, which is why it is a second
header rather than part of the umbrella.

The contract is **runtime-owned**, the way the [backend contract](backends.md)
is: the types are fields on `ShadowSpillRuntimeConfig`, and a lane written
elsewhere is built against these headers with nothing else in view.

For what a lane is and why the contract has the shape it does, see
[lanes](../architecture/lanes.md).

## Types

| type | |
|---|---|
| `ShadowSpillLane` | one lane, made per route at create. Opaque in `lane.h`; defined in `lane_base.h`, which an implementation embeds as its first member |
| `ShadowSpillLaneRange` | where one of a lane's two pools lives: `address` and `bytes`, filled by the runtime |
| `ShadowSpillLaneOperations` | the table below |
| `ShadowSpillLaneDescription` | one entry in `ShadowSpillRuntimeConfig.lanes`: a kind pair, a table, a `create`, and a `configuration` |
| `ShadowSpillLaneTransfer` | what one transfer did, filled by the optional `transfer` entry below |
| `ShadowSpillLaneTiming` | what transfers cost a lane, filled by the optional `timing` entry below |
| `ShadowSpillLaneStatistics` | what a lane has moved: the counters the runtime reads out of the base, plus whatever `timing` reported |

## The struct a transport embeds

```c
struct ShadowSpillLane {
    ShadowSpillRuntime *runtime;
    const ShadowSpillBackend *backend;
    ShadowSpillBackendStream stream;
    uint8_t from_kind;
    uint8_t to_kind;
    ShadowSpillLaneRange from_range, to_range;
    _Atomic uint64_t copies, chunks, bytes, signals, waits, retries, failures;
};

static inline void shadowspill_lane_counted(_Atomic uint64_t *counter, uint64_t by);
static inline uint64_t shadowspill_lane_count(const _Atomic uint64_t *counter);
```

A transport embeds it first and casts between the two:

```c
typedef struct { ShadowSpillLane base; /* ... */ } MyLane;
```

**The runtime fills every field**, including the pair of kinds and the two
pools' ranges, and hands it to `create` to copy in — so a transport writes none
of it and a field added here changes no transport. `from_kind` and `to_kind`
are where a transport reads its direction; there is no direction to configure.
`from_range` and `to_range` are where a transport whose hardware must be made
able to reach a pool — a NIC registering it — finds the pool, at create and
once, rather than learning it from the addresses transfers hand it.

The counters are maintained through `shadowspill_lane_counted()`, which is
relaxed in one place so no transport picks an ordering by accident, and the
runtime reads them out of the base. Nothing copies them anywhere.

## Operations

```c
int  (*wait)(ShadowSpillLane *lane, ShadowSpillBackendEvent event);
int  (*copy)(ShadowSpillLane *lane, void *destination, const void *source,
             uint64_t bytes, uint64_t *handle);
int  (*signal)(ShadowSpillLane *lane, uint64_t handle,
               ShadowSpillBackendEvent event);
int  (*synchronize)(ShadowSpillLane *lane);
int  (*transfer)(ShadowSpillLane *lane, uint64_t handle,
                 ShadowSpillLaneTransfer *transfer);
void (*destroy)(ShadowSpillLane *lane);
int  (*timing)(const ShadowSpillLane *lane, ShadowSpillLaneTiming *timing);
```

- **`wait`** orders this lane's work behind `event`. Returns 0 enqueued or
  already satisfied, **1 to be retried at the next poll**, -1 failed. A lane
  whose waits are device-side never returns 1. On 1 the action returns to the
  head of its queue, so a retry does not reorder what the task boundaries
  triggered.
- **`copy`** moves bytes between addresses in the two pools this lane connects.
  Nothing but the lane dereferences either. It writes `handle`, naming the
  transfer for the two entries below; **0 means the lane kept nothing about
  it**, which is what a lane answers when no trace is running.
  `shadowspill_lane_trace_active()` is how it asks, being outside the runtime
  and unable to read that state itself.
- **`signal`** makes `event` complete once everything issued on this lane so far
  has landed. Downstream sees an ordinary backend event whatever the lane did,
  which is what keeps completion tracking and retirement in one form. A lane
  whose ordering already covers everything issued needs nothing from `handle`.
- **`transfer`** is NULL-able, and reports what one transfer did. It is asked
  **once**, after that transfer's event has completed, and only for a handle
  `copy` returned nonzero. **The query retires the handle**: a lane may release
  whatever it kept the moment it answers, and the runtime will not ask again —
  which is why there is no release entry beside it.

`synchronize` and `destroy` are what they say. `timing` is with the statistics
it fills, under [what a lane has moved](#what-a-lane-has-moved).

An event is a `ShadowSpillBackendEvent`, one opaque word, and never the
runtime's event lease.

**There is no entry for the runtime to drive a lane with.** A lane makes the
event given to `signal` complete when the bytes have landed, by whatever means
suits it — the device ordering it behind a stream, or a thread of the lane's own
watching its transport. See [lanes](../architecture/lanes.md) for why there is
no `poll`.

## Registering one

```c
typedef struct ShadowSpillLaneDescription {
    uint8_t from_kind;                 /* ShadowSpillPoolKind, source */
    uint8_t to_kind;                   /* ShadowSpillPoolKind, destination */
    const ShadowSpillLaneOperations *operations;
    int (*create)(const ShadowSpillLane *base,
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

- `base` — the common struct the runtime has already filled. A transport
  allocates its own struct and copies it into the first member,
  `created->base = *base;`, before anything else, so the rest of create — and
  every failure path that unwinds through `destroy` — can reach the backend
  through it. Its fields:
  - `runtime` — the lane's way back in. A lane that discovers a failure on a
    thread of its own has no return value to fail through, so it latches the
    failure itself. The latch is built for concurrent callers and **first
    writer wins**, so the original cause survives; the worker already reads the
    latched status twice a turn, so nothing is added to its loop.
  - `backend` — usable, but only through the stream below.
  - `stream` — the route's stream. The runtime writes it, and a lane writes it
    only from the thread that called into the lane: the built-in lane's copies
    go there, and a lane whose bytes move elsewhere puts its ordering there.
  - `from_kind`, `to_kind` — the pair this entry was registered under, and
    where a transport reads its direction.
- `configuration` — passed through untouched. Whoever registers the entry
  decides what it means. Neither built-in lane needs one.

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

`shadowspill_route_lane_statistics()` fills this. **The seven counters come from
the lane's base**, so they are the counts it kept rather than a copy it made,
and they are maintained **unconditionally** — a lane that counts only when asked
cannot explain the run that went wrong, and the cost is a few relaxed atomic
adds against a transfer measured in milliseconds.

**The `timing` entry is optional**, on the same rule as `transfer`: required
when the runtime cannot proceed without it, optional when its absence only costs
observability. A lane that reads no clock on the transfer path leaves it NULL,
or refuses the call, and `timed` stays 0.

The two durations are governed by `timed`. They describe the interval a lane
can actually observe -- from handing the hardware a piece of work to seeing its
completion -- which is not time on the wire, and separating those is the point
of reporting it. A lane that hands a transfer onward and
returns never sees the completion instant at all; it reports no `timing` at all,
because a zero duration otherwise reads as "instant". A lane that can time
cheaply may still do so only while a trace is running.

`chunks` differs from `copies` only for a lane that splits a transfer; where it
does not, the two are equal by construction. That is the difference between
"how many transfers" and "how many times the hardware was asked".

One lane serves one directional pair of pool kinds, so these are per route and
per direction, and they accumulate for the life of the lane. The
[step diagnostics](../python/step-diagnostics.md) report them beside the
per-step view the trace gives, which is a different question and not a
substitute.

## What one transfer did

```c
#define SHADOWSPILL_LANE_NO_TIME UINT64_MAX

typedef struct ShadowSpillLaneTransfer {
    uint64_t issued_at_nanoseconds;
    uint64_t started_at_nanoseconds;
    uint64_t finished_at_nanoseconds;
    uint64_t bytes;
    uint64_t chunks;
} ShadowSpillLaneTransfer;
```

All three instants are nanoseconds from the trace's origin, the axis the rest of
a step is placed on, and `SHADOWSPILL_LANE_NO_TIME` where a lane has nothing to
report — beside which `bytes` and `chunks` are still worth having, and are what
a lane always knows.

`issued_at` is when the runtime handed the transfer over, before any dependency
the lane was given had cleared, and `started_at` is when its bytes began moving.
**The gap between them is the wait.** Folded together, a transfer held behind an
event reads as a slow one rather than a late one, which are different problems.

Two clocks reach this axis. A lane whose bytes move on a stream reads instants
off timing events it recorded around the copy, already on the origin's axis. A
lane whose bytes move elsewhere reads a host clock and converts:

```c
SHADOWSPILL_API uint64_t shadowspill_lane_origin_instant(
    ShadowSpillRuntime *runtime, uint64_t monotonic_nanoseconds);
```

`monotonic_nanoseconds` is read from `CLOCK_MONOTONIC`, the same clock the
runtime stamps its own trace events with — so the two axes can be cross-checked
rather than found to disagree. The anchor between it and the origin is sampled once, where
the origin event is recorded, and `shadowspill_trace_begin()` states what a
caller owes for it to mean anything: **the origin marker is recorded on an idle
stream.** The result is `SHADOWSPILL_LANE_NO_TIME` when no trace is running,
when the trace was begun with no origin, or for an instant before the origin.

A lane may use both clocks, and the pinned-host lane does: its `issued_at` is a
converted host instant and its other two come off the stream. That costs it a
clock read per transfer where a third timing event would have cost more. The
remote lane reads all three on the host clock — before posting the first chunk's
verb, and when the last chunk's completion is reaped.

`chunks` is a transport's own granularity, not a unit shared between lanes: one
enqueued copy is one chunk on the pinned-host lane, while a remote transfer
reports however many pieces the NIC was handed.


## Validity

`shadowspill_runtime_create()` refuses a config whose entry has a NULL
`operations` or `create`, a NULL required entry, or a kind pair another entry
already claims. It also refuses a route whose two pools' kinds no entry serves.
`transfer` and `timing` are optional and are never a reason to refuse.
