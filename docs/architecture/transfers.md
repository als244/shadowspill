# Transfers

Routes, queues, lanes, and calibration live in `csrc/src/runtime/transfers/`.
A backend supplies streams and directional copies; the runtime decides which
copy goes on which lane, in what order, and measures what it built.

## Routes, queues, and lanes

A route is a directed copy path: a source pool and a destination pool, whose
two kinds imply the direction -- with the built-in kinds, pinned host to device
is a fetch and device to pinned host an evict. Routes are declared to
`shadowspill_runtime_create()` by pool ids, and it refuses three things: a
route whose two pools share a kind, a second route over a pair it already has,
and **a route whose directional kind pair no lane serves**. The third is what a
route to a pool kind nothing can yet copy to runs into, and it is refused at
create rather than at the first transfer.

Each route owns a **queue** and a **lane**. The queue is the ordering: three
intrusive FIFOs behind one mutex, holding the actions a route has been given and
the ones in flight on it. It touches no backend. The lane is what moves the
bytes, and it has a page of its own — [lanes](lanes.md) — covering the contract,
how one is chosen from the pool kinds a route connects, and the stream ordering
it must not disturb.

What matters here is what a route holds: a queue, a stream, and the lane
resolved from its two pools' kinds at create. The worker dispatches every
transfer through that lane and makes no backend call of its own. The route's
stream is where the events that order a transfer against compute are recorded,
and the runtime is its only writer.

## Calibration

Calibration is the runtime's measurement of its own routes, through the same
lane operations a transfer uses, on the memory it owns. It reserves probe ranges in the real pools, measures each
route alone with warm-up and repeated copies, then measures a route against
its reverse at the same time on their two lanes and publishes the concurrent
per-direction rates as each route's effective bandwidth, keeping the solo
figures beside them. The simultaneous pass issues the two directions' copies
alternately rather than one direction's batch and then the other's, because a
lane whose `copy` enqueues work per chunk makes issuing a batch cost real time:
drained in turn, the first direction's measurement window would contain the
second's dispatch and report a rate that low by however long that took. Planning consumes that immutable profile and never
benchmarks a route itself; see the [runtime C API](../c/runtime.md).

## Dispatch

The worker owns both queues; it does not own the lanes and does not drive
them. At an action trigger the dispatcher reserves destination capacity in
directive order and the action holds that reservation while queued; at queue
head the worker submits the copy **through the route's lane** and asks the lane
to signal the completion event; on completion the object publishes the ready
residency generation. An eviction's source is not freed when the action is
queued: it becomes reusable only through that completion, which is what keeps
later task allocations from overtaking planned transfer capacity while the queue
stays FIFO. The worker loop itself is described in
[memory runtime](memory-runtime.md#worker). A write-back travels the evict
route exactly as an eviction does, in the same order and against the same
window; only what its completion publishes differs: the spill copy becomes
current and the execution copy stays.

Each queue serves two orders. Transfers the plan scheduled are dispatched in
the order their boundaries triggered them. Transfers the plan did not
schedule — the opening restore of an initial device set, a reconciliation —
are background transfers: they are dispatched in their own order, and only
while the queue holds fewer than the configured window of their bytes in
flight, so a scheduled transfer never waits behind more than that window
however large the background batch is. A single background copy larger than
the window runs alone. Once dispatched, a queue is one FIFO in dispatch
order, which is the stream's order and what completion follows. The window
is `ShadowSpillRuntimeConfig.background_transfer_window_bytes`; zero removes
the bound.

Previous: [Lanes](lanes.md). Next: [Events](events.md).
