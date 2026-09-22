# Lanes

A lane moves bytes between two pools and says when they have landed. It is the
one thing in the runtime that knows how a transfer is actually performed, and
the only contract a new kind of transport has to implement.

`csrc/include/shadowspill/runtime/lane.h` declares it, and
`csrc/include/shadowspill/runtime/lane_base.h` the struct an implementation
embeds; the built-in implementation is
`csrc/src/runtime/transfers/pinned_host_device_lane.c`, and
`csrc/src/runtime/transfers/lanes.c` is how the runtime finds one.

## A lane, a queue, and a stream are three things

The three are easy to conflate:

| | what it is | what it never does |
|---|---|---|
| **queue** | the ordering in front of a lane: the actions a route has been given, and those in flight on it | touch a backend |
| **lane** | what moves the bytes and reports completion | write to the route's stream |
| **route's stream** | where the events that order a transfer against compute are recorded | carry bytes, for a lane that moves them elsewhere |

The route's stream has **exactly one writer, the runtime**. Completion tracking,
retirement and readiness publication all depend on that, so it is the invariant
a lane is written around rather than a detail of the current implementation.

## A lane is chosen by the pool kinds it connects

A route names a source and a destination pool. Their kinds are a directional
pair, and the pair selects a lane from the `lanes` list on
`ShadowSpillRuntimeConfig`. That is the whole of the mechanism: **there is no
branch anywhere asking what kind of transport a route has.**

The runtime seeds that list with its built-ins and appends whatever the config
registered, so a lane a separate library provides is found by the same lookup as
a built-in one. Two entries claiming the same pair fails
`shadowspill_runtime_create()` — order must never decide it silently, because
whichever lost would be invisible and every transfer on that route would quietly
use the other. Create returns a status and has nowhere to put a message, so the
refusal does not say which pair collided; a caller that registers lanes knows
which it supplied.

A lane is therefore **named for the kinds it connects**, never for an argument it
takes. The built-in takes a stream; so would a lane between two device pools.
The stream distinguishes nothing.

## Every lane is the same struct, extended

A lane holds the same fields and counts the same seven whatever it is, so
those live in one struct that every implementation embeds as its **first
member** and casts between:

```c
typedef struct { ShadowSpillLane base; /* ... */ } MyLane;
```

That is what makes `ShadowSpillLane *` mean one thing everywhere, in the runtime
and in a library the runtime loaded alike. The runtime fills the base — runtime
handle, backend, the route's stream, the two pool kinds, where each pool's
memory lives, counters at zero — and hands it to `create` to copy in, so an
implementation writes none of it and cannot write it wrong, and a field added
to the base changes no implementation at all. The two ranges are for a lane
whose hardware has to be made able to reach a pool before it can move a byte
of it: it does that at create, once, from what the runtime already holds.

**The counters are the runtime's to read.** They are maintained through
`shadowspill_lane_counted()`, and the runtime reports them straight out of the
base. Nothing copies them anywhere, so no implementation can report a count
that disagrees with the one it kept.

The split across two headers is deliberate: the base is a layout for code that
*implements* a lane, and everyone else — anyone merely declaring a runtime —
sees an opaque pointer. It also keeps that layout out of the umbrella header
the framework adapter pulls into its C++ translation units, which is why the
base is free to use `_Atomic` and the declaration header is not.

## The contract

**The obligation, and the whole of it:** a lane makes the event it was given
complete when the bytes have landed, and **the runtime does not drive it**. How
it arranges that is its own business. `transfer` and `timing` may be absent;
everything else is required.

| entry | what it must do |
|---|---|
| `wait(lane, event)` | order this lane's work behind `event` |
| `copy(lane, destination, source, bytes, handle)` | move the bytes, and name the transfer |
| `signal(lane, handle, event)` | make `event` complete once everything issued so far has landed |
| `synchronize(lane)` | block until everything issued has landed |
| `transfer(lane, handle, out)` | what one transfer did; **optional** |
| `destroy(lane)` | release what `create` took |
| `timing(lane, out)` | what transfers cost this lane; **optional** |

### The handle, and what it is for

`copy` hands back a handle naming the transfer, and **0 means the lane is
keeping nothing about it** — which is what every lane answers when no trace is
running, so nothing is recorded that nothing will read. The handle rides the
action to completion, where the runtime asks `transfer` once. **That query
retires the handle**: a lane may release whatever it kept the moment it answers,
and the runtime will not ask again. That is the whole lifetime rule, and it is
why there is no release entry beside it.

A lane reports what it actually observed, in whatever terms it has.

An event here is a `ShadowSpillBackendEvent` — one opaque word — and never the
runtime's event lease, which is reference counting and pool links a lane has no
business seeing. That is what lets a lane in a separately loaded library
implement this against the runtime's headers alone.

### Why `transfer` and `timing` may be absent

The rule is the one the backend table already follows for its profiler entries:
**required when the runtime cannot proceed without it, optional when its absence
only costs observability.** A lane that reports neither moves bytes exactly as
well as one that reports both; its transfers are recorded untimed, which the
trace and every reader already handle.

A `ShadowSpillLaneTransfer` carries two instants, bytes and chunks. The instants
are nanoseconds from the trace's origin, the axis the rest of a step is placed
on, and `SHADOWSPILL_LANE_NO_TIME` where a lane has nothing to say — which is
better than a time that means nothing. A lane whose bytes move on a stream reads
them off timing events it recorded around the copy. A lane whose bytes move
elsewhere has only its own clock, and **nothing anchors that clock to this
origin**: the origin is a device event, and no host instant is recorded beside
it. Such a lane reports the bytes and the chunks, which need no anchor, and the
ratio between them is the number worth having anyway — it says whether a slow
transfer was one long wait or many short ones.

### Why `wait` may ask to be retried

`wait` returns 0 when the dependency is enqueued or already satisfied, **1 when
it cannot be enqueued yet**, and -1 on failure.

A lane whose waits are device-side always enqueues and never returns 1. A lane
that cannot express a device dependency at all has to observe the event
completing instead, and 1 is how it says so. The action then goes back to the
**head** of its queue, not the tail, so a retry cannot reorder what the task
boundaries triggered, and the existing poll cadence
(`worker_poll_nanoseconds`) revisits it.

### How a lane knows its bytes have landed

Its own business, and deliberately so. A lane whose copies run on a stream lets
the device order the event behind them — the built-in does exactly that, and
needs nothing else. A lane that completes on its own schedule watches for that
however suits it: a thread of its own, blocking on whatever its transport
offers, and then releasing the event. Nothing downstream can tell which, because
everything downstream reads an event.

**There is no entry for the runtime to poke a lane with.** The worker's loop
gates every transfer, so work added there is paid at whatever rate the loop
happens to turn, and a poll per active route would be paid on every turn while
doing nothing on most. The obligation above is what makes it unnecessary: the
lane watches, the runtime does not ask.

A lane that runs a thread chooses between blocking — no core while idle, but a
wake-up in front of every transfer — and watching, which is the reverse. The
remote lane watches by default, and `SHADOWSPILL_NETWORK_SPIN_NANOSECONDS`
bounds that or turns it off. Either way its thread's life is bounded by
`create` and `destroy`.

## What a lane may do with the backend

A lane is given a backend and the route's stream at create, and may copy,
record and query on it. The constraint is **single-writer ordering**: one
thread's worth of work, in one order, on that stream.

Both lanes use it, and neither is a second writer, because every call into the
lane's table arrives on the thread that called it — the worker for a planned
transfer, the caller for calibration. The built-in lane's copies and completion
event go there because they belong in the same order; it is acting as the
runtime's own copy mechanism. A lane whose bytes move elsewhere puts its
ordering there too — the value waits its `signal` enqueues and the event
recorded behind them, and the copies staging needs — all of it from the thread
that called it, none of it from the thread that watches its hardware.

### A lane must not wait on itself

The rule that matters is not about threads, because a lane need not have one:

> **A lane's completion path must not depend on anything the lane has made
> wait.**

The built-in satisfies it without trying. Its completion path is the stream,
the driver advances it, and it makes nothing wait.

A lane that completes on its own schedule has to be deliberate. A value wait
it enqueues is satisfied only by its own completion path, so a device call on
that path can be blocked by the very wait it exists to satisfy — and the driver
couples threads that share nothing else: a launch that finds its stream's queue
full holds the context lock until the device makes progress, so a watcher that
needed the driver in order to make that progress waits for the lock the launch
holds. The remote lane's thread therefore makes no device call: the device's
side of every transfer is enqueued at dispatch on the route stream, behind
value waits the thread satisfies with plain stores, and on the direct path,
where the NIC touches the pool with no stream in between, the route stream
stores a word behind the runtime's waits and the thread posts nothing before it
reads it.

The same rule reaches inside such a lane, and is easy to miss there. A lane that
keeps work in flight across transfers has two halves — issuing and retiring —
and **issuing must never block on something only retiring can release.** The
next transfer's device work sits behind the previous transfer's `signal` on the
stream, so a thread waiting for the device to reach it is waiting for itself.
Every readiness question on the issuing side is therefore asked, not waited on:
if the answer is no, the thread goes and retires, which is what makes the answer
become yes. Stated as a rule rather than as an account of one lane, because the
deadlock it prevents does not look like a deadlock in the code that causes it.

## Create, probe, and teardown

A lane is made once per route at `shadowspill_runtime_create()` and destroyed in
reverse at close. `create` receives **the base the runtime has already filled**
and its own configuration; it allocates its own struct, copies the base into the
first member, and everything the base holds is available for the rest of create
— which is why it arrives this way rather than being filled afterwards, since a
lane that fails halfway needs the backend to unwind through.

Direction is not configured. A lane that copies one way for a fetch and the
other for an evict reads `from_kind` and `to_kind` out of its base, so the pair
of kinds a description is registered under is the only place that says which way
a route goes.

**The runtime handle is how a lane reports a failure it finds on its own
thread.** With no entry for the runtime to call, a lane that fails has no return
value to fail through — so it latches the failure itself. That needs no new
mechanism: the latch is built for concurrent callers, **first writer wins** so
the original cause is preserved, and the worker already reads the latched status
twice per turn. A lane failure therefore costs nothing in the loop and arrives
by the path every other failure already takes.

`create` is also **where a lane probes**. Nothing a loaded library holds runs
when the library is loaded, so everything that depends on what the hardware can
actually do belongs here, where there is a failure path and an unwind. A lane
that cannot come up fails create, and the runtime unwinds rather than starting
with a route that cannot move a byte.

## What the contract does not reach

Event leases are one form, the event pools know no transport, the completion
tracker keys its FIFOs by stream and queries the event like any other,
retirement is the same for every lane, and nothing in `memory/` learns that a
transport exists. A lane produces an ordinary backend event, whatever it did to
get there, and everything downstream reads it as one; that is what keeps the
contract affordable.

See [transfers](transfers.md) for routes, queues and calibration, and the
[lane contract](../c/lanes.md) for the C declarations.

Previous: [Memory pools](memory-pools.md). Next: [Transfers](transfers.md).
