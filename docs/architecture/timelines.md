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

Host work is measured on `CLOCK_MONOTONIC`, counted from the runtime
trace's beginning. The runtime records that beginning at the same point the
origin event is recorded, so host and device times are two views of the
same instant, though never subtracted from each other: durations are
compared within a clock, and only orderings across them.

## The compute lane

The frontend records three timing events per selected task on the compute
stream:

- **reached**, when the stream arrives at the task's readiness marker;
- **started**, after the stream has waited for the task's inputs to be
  resident and for the ranges its allocations reuse to be released;
- **finished**, after the task's kernels.

Their differences are the task's two waits and its duration, and their
positions from the origin are its place on the timeline. Around them the
frontend stamps the host clock at the entry and exit of both task
boundaries, which is where the dispatch costs come from; see
[task boundaries](task-boundaries.md).

## The transfer lanes

Transfers are dispatched by the runtime's worker thread, so the worker is
what measures them. For every copy it submits while a trace is active it
opens a **stream interval** on the lane: one timing event recorded
immediately before the copy, one immediately after it, both ahead of the
completion event the copy already carries for dependencies. Because a
stream executes in order, the first event marks the moment the lane
finished everything ahead of the copy and began it, and the second the
moment the copy finished. When the worker's nonblocking poll later sees the
completion event, both stamps are guaranteed readable, and the worker reads
them from the origin and writes the interval into the transfer's completion
record.

The interval is a generic runtime type, `ShadowSpillStreamInterval` in the
synchronization layer: open on a stream, close on the stream, read from an
origin, discard. It goes through the backend's event calls -- timing events
and `elapsed_nanoseconds` -- and knows nothing about transfers, so anything
else the runtime wants placed on the device timeline can use it.

The host clock still records the worker's own observations of each
transfer: when the action was queued, when its destination was reserved,
when the copy was handed to the lane, and when the poll observed completion.
Read against the stream interval, those say how long a copy waited behind
its predecessors and how far the poll lagged the device -- neither of which
is transfer time.

## Simulated time, and alignment

The simulator has its own clock, which starts when the first task starts.
The step diagnostics shift it so that the first selected task starts at
zero, and report when that task's kernels actually started on the device as
`first_task_started_at_seconds`: the step's prologue. Every invocation pays it
— the previous invocation's drain, input staging, and the opening restore of
the first task's inputs — and the simulator prices none of it, because its
clock begins where that work ends. Every delta between a simulated and a
measured time is taken after the shift, so it reads as drift within the step
and the prologue is read once rather than repeated in every delta; see [step
boundaries](step-boundaries.md) for what the boundary regions contain and
which of them the makespan does charge for.

## The step: origin to origin

Every invocation, traced or not, records three timing events on the compute
stream: its **origin** where it begins, before its first task; its **span
start** where its first task's compute starts; and its **span end** where
its last task's compute ends. The stream is in order, so the next
invocation's origin is reached only after everything this one enqueued, and
the **cycle** of an invocation is its origin to the next origin. That is the
step's time: what a repeated step costs and what throughput divides by. A
loop that stops records an end marker in the same place a next origin would
sit, so its last step reads like every other.

The cycle partitions exactly into three parts, each a difference of two of
the events:

```text
cycle = opening delay + selected span + exposed tail
```

The opening delay is origin to span start: the first task's readiness waits and
whatever the opening still held the stream for. The selected span is the
tasks. The exposed tail is span end to the next origin: terminal work the
stream itself still did. Transfers that drained on the lanes meanwhile are
not in the cycle, because they cost the step nothing; a later invocation
that has to wait for them pays in its own opening delay, where the cost
belongs.

The events are created once per callable and reused round-robin, so an
invocation creates nothing; reading a cycle waits for the closing event and
for nothing else. The Python side of this is
[timing](../python/api/timing.md).

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
