# Lanes

A lane moves bytes between two pools and says when they have landed. It is the
one thing in the runtime that knows how a transfer is actually performed, and
the only contract a new kind of transport has to implement.

`csrc/include/shadowspill/runtime/lane.h` declares it; the built-in
implementation is `csrc/src/runtime/transfers/pinned_host_device_lane.c`, and
`csrc/src/runtime/transfers/lanes.c` is how the runtime finds one.

## A lane, a queue, and a stream are three things

They were once two, and the confusion is worth naming so it is not
reintroduced.

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

## The contract

**The obligation, and the whole of it:** a lane makes the event it was given
complete when the bytes have landed, and **the runtime does not drive it**. How
it arranges that is its own business. The two intervals may be absent;
everything else is required.

| entry | what it must do |
|---|---|
| `wait(lane, event)` | order this lane's work behind `event` |
| `copy(lane, destination, source, bytes)` | move the bytes |
| `signal(lane, event)` | make `event` complete once everything issued so far has landed |
| `synchronize(lane)` | block until everything issued has landed |
| `interval_open` / `interval_close` | bracket the next copy so it can be timed; **optional, as a pair** |
| `destroy(lane)` | release what `create` took |

An event here is a `ShadowSpillBackendEvent` — one opaque word — and never the
runtime's event lease, which is reference counting and pool links a lane has no
business seeing. That is what lets a lane in a separately loaded library
implement this against the runtime's headers alone.

### Why the intervals may be absent

The rule is the one the backend table already follows for its profiler entries:
**required when the runtime cannot proceed without it, optional when its absence
only costs observability.** A lane with no intervals moves bytes exactly as well
as one with them; its transfers are recorded untimed, which the trace and every
reader already handle. A lane that cannot place an instant on the trace's clock —
because it completes on a clock the trace does not share — leaves both NULL
rather than reporting a time that means nothing.

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

**There is no entry for the runtime to poke a lane with, and there was.** The
0902 design gave the table a `poll` the worker called once per active route per
turn, on the assumption that a lane could not watch for itself. It cost **1.7 %
of the shortest step in the qualification matrix** while doing nothing at all,
because the worker's loop gates every transfer and work added there is paid at
whatever rate the loop happens to turn. The obligation above replaces it: the
lane watches, the runtime does not ask.

A lane that runs a thread should **block rather than spin** — otherwise a thread
per lane becomes a core per lane — and its thread's life is bounded by `create`
and `destroy`.

## What a lane may do with the backend

A lane is given a backend and **a stream of its own** at create. It may copy,
record and query on that stream. It may not touch the route's stream, for the
reason above.

For the built-in lane the granted stream *is* the route's stream, because its
copies and its completion event belong in the same order — it is acting as the
runtime's own copy mechanism, not as a second writer. A lane whose bytes move
elsewhere takes a separate stream and leaves the route's alone, reaching it only
through the value wait the runtime enqueues on its behalf.

That is the narrow version of a rule that was once broader. The constraint worth
keeping is single-writer ordering on one stream, not an embargo on the backend;
stating it the wide way cost real machinery for no invariant.

## Create, probe, and teardown

A lane is made once per route at `shadowspill_runtime_create()` and destroyed in
reverse at close. `create` receives the runtime, the backend, its granted stream,
and its own configuration.

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

## What this leaves untouched

Deliberately, because it is what makes the contract affordable: event leases stay
one form, the event pools are unchanged, the completion tracker keys its FIFOs by
stream and makes no backend call, retirement is unchanged, and nothing in
`memory/` learns that a transport exists. A lane produces an ordinary backend
event, whatever it did to get there, and everything downstream reads it as one.

See [transfers](transfers.md) for routes, queues and calibration, and the
[lane contract](../c/lanes.md) for the C declarations.

Previous: [Memory pools](memory-pools.md). Next: [Transfers](transfers.md).
