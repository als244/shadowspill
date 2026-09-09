# Transfers

Routes, lanes, and calibration live in `csrc/src/runtime/transfers/`. A
backend supplies streams and directional copies; the runtime decides which
copy goes on which stream, in what order, and measures what it built.

## Routes and lanes

A route is a directed copy path: a source pool, a destination pool, and the
direction their kinds imply (pinned host to device is a fetch, device to
pinned host an evict). Routes are declared to `shadowspill_runtime_create()`
by pool ids; the runtime refuses a route between pools of the same kind or a
duplicate pair.

Each route owns one lane: a backend stream the runtime creates at start,
names through the profiler, and destroys at close. The worker dispatches every
copy of a route onto its lane in FIFO order and records the completion event
there, so ordering within a direction is the stream's, and the two directions
never share a stream. Which copy the backend performs follows the route's
direction, `copy_host_to_device` or `copy_device_to_host`.

## Calibration

Calibration is the runtime's measurement of its own routes, on the lanes and
arenas it owns. It reserves probe ranges in the real pools, measures each
route alone with warm-up and repeated copies, then measures a route against
its reverse at the same time on their two lanes and publishes the concurrent
per-direction rates as each route's effective bandwidth, keeping the solo
figures beside them. Planning consumes that immutable profile and never
benchmarks a route itself; see the [runtime C API](../c/runtime.md).

## Dispatch

The worker owns both lanes. At an action trigger the dispatcher reserves
destination capacity in directive order and the action holds that reservation
while queued; at lane head the worker submits the copy on the lane and records
its completion event; on completion the object publishes the ready residency
generation. An eviction's source is not freed when the action is queued: it
becomes reusable only through that completion, which is what keeps later task
allocations from overtaking planned transfer capacity while the lane stays
FIFO. The worker loop itself is described in
[memory runtime](memory-runtime.md#worker). A write-back travels the evict
lane exactly as an eviction does, in the same order and against the same
window; only what its completion publishes differs: the spill copy becomes
current and the execution copy stays.

Each lane serves two queues. Transfers the plan scheduled are dispatched in
the order their boundaries triggered them. Transfers the plan did not
schedule — the opening restore of an initial device set, a reconciliation —
are background transfers: they are dispatched in their own order, and only
while the lane holds fewer than the configured window of their bytes in
flight, so a scheduled transfer never waits behind more than that window
however large the background batch is. A single background copy larger than
the window runs alone. Once dispatched, a lane is one FIFO in dispatch
order, which is the stream's order and what completion follows. The window
is `ShadowSpillRuntimeConfig.background_transfer_window_bytes`; zero removes
the bound.
