# Timelines: how a step is measured

A traced step answers one question: where did the real step's time go,
against where the plan said it would go? The answer is only readable if
every measurement sits on one timeline, so this page says what is measured,
with which clock, from what origin, and what an untraced step pays for it.

## Two clocks, one origin

Everything that happens on the device is measured with **timing events**: a
backend event that records the device clock when the stream it is recorded
on reaches it. The difference between two completed timing events is a
device-clock interval, exact to the event's resolution and independent of
when the host enqueued them.

The **origin event** is recorded on the compute stream at the start of the
call, before the step's first task. Every device measurement is read as the
elapsed time from that one event, so the compute stream and both transfer
lanes share a zero without any clock fitting.

Host work uses `CLOCK_MONOTONIC`, counted from a timestamp taken immediately
before recording the invocation origin. Host and device durations use their
own clocks; enqueue latency remains visible rather than fitted away.

## The compute lane

The frontend records four timing events per selected task on the compute
stream:

- **reached**, when the stream arrives at the task's readiness marker;
- **inputs ready**, after waiting for input residency;
- **started**, after waiting for allocation ranges to become reusable;
- **finished**, after the task's kernels.

Their differences are the task's two waits and its duration, and their
positions from the origin are its place on the timeline. Around them the
frontend stamps the host clock at the entry and exit of both task
boundaries, which is where the dispatch costs come from; see
[task boundaries](task-boundaries.md).

## The transfer lanes

Transfers are dispatched by the runtime's worker thread, but **the worker does
not measure them** -- the lane does, and the worker only asks. While a trace is
active `copy` hands back a handle naming the transfer, and when the worker's
nonblocking poll later sees the completion event it asks the lane, once, what
that transfer did. The answer goes into the transfer's completion record, and
the question retires the handle.

It is arranged that way because a lane is the only thing that knows how its
bytes move. The built-in lane brackets its copy with a **stream interval**: one
timing event recorded immediately before the copy, one immediately after it,
both ahead of the completion event the copy already carries for dependencies.
Because a stream executes in order, the first event marks the moment the lane
finished everything ahead of the copy and began it, and the second the moment
the copy finished; once the completion is observed, both stamps are guaranteed
readable.

A lane whose bytes do not move on a stream cannot be asked for that, and asking
was what the worker used to do. It **reports no instants** -- there is no anchor
from such a lane's own clock to this origin, the origin being a device event
with no host stamp recorded beside it -- and its transfers are placed on the
host timeline only. What it can report is how many bytes it moved and in how
many pieces, which needs no clock; see
[lanes](lanes.md#why-transfer-and-timing-may-be-absent).

The interval the built-in lane uses is a generic runtime type,
`ShadowSpillStreamInterval` in the synchronization layer: open on a stream,
close on the stream, read from an origin, discard. It goes through the
backend's event calls -- timing events and `elapsed_nanoseconds` -- and knows
nothing about transfers, so anything else the runtime wants placed on the
device timeline can use it.

The host clock still records the worker's own observations of each
transfer: when the action was queued, when its destination was reserved,
when the copy was handed to the lane, and when the poll observed completion.
Read against the instants the lane reported, those say how long a copy waited
behind its predecessors and how far the poll lagged the device -- neither of
which is transfer time.

## Simulated time

Simulation and execution both start at invocation entry. The zero-duration
start task triggers ordinary fetches before the first computation. Diagnostics
retain these coordinates: a start delta is measured start minus simulated
start, including any accumulated entry or dispatch delay. Every transfer must
match a scheduled transfer by lane order, trigger, object, and byte count.

The trace's invocation duration ends at the latest computational or required
transfer completion. This matches the simulated makespan's boundary. Entry
delay, the computational task window, and terminal duration sum to this total.
Control tasks remain in the task inventory but not in model-compute totals.

## The step: origin to origin

Every invocation records three timing events on the compute stream: origin,
first computational task start, and last computational task end. The next
origin is recorded after the previous invocation's terminal work has drained.
The cycle runs from one origin to the next, so repeated cycles account for
required transfers once and also include caller work between invocations.
`mark_cycle_end()` waits for the same terminal work and records an end marker
when no next invocation follows.

```text
cycle = entry delay + selected span + exposed tail
```

Cycle time measures throughput. The traced invocation's terminal duration
measures required work; the cycle's exposed tail also includes caller and
instrumentation delays. These quantities are kept separate.

Markers are reused round-robin. Reading a closed cycle waits for its closing
event. See the [timing API](../python/api/timing.md) and
[step boundaries](step-boundaries.md).

## What an untraced step pays

Three event records: the origin, the span start and the span end above, each
one enqueue on the stream and one host call. The task markers and the stream
intervals are recorded only while a trace is armed. The one instruction an
untraced transfer dispatch spends on any of this is the acquire load of the
trace's active flag, which is the same gate the runtime's trace appends
already pay. Timing events come from the runtime's timing pool, reserved when
the trace is prepared and kept apart from the dependency pool
([events](events.md#the-timing-pool)), so a traced step cannot exhaust the
events an untraced step relies on.

The step-level result of all this is described field by field in
[StepResult diagnostics](../python/step-diagnostics.md); the event and stream
calls the intervals rest on are in the [backend
contract](../c/backends.md).

Previous: [PyTorch adapter](adapter.md).
