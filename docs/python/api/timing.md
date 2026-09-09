# Timing: the step on the device clock

Every invocation of a planned training callable records three timing
events on the compute stream, without being asked: its origin before its
first task, its span start at its first task's compute, and its span end at
its last task's compute. The step's time is the **cycle**, origin to the next
invocation's origin, which is what a repeated step costs and what throughput
divides by; the architecture page [timelines](../../architecture/timelines.md#the-step-origin-to-origin)
defines it. This page is the API that reads it.

## `PlannedTrainStep.invocation_timings()`

```text
PlannedTrainStep.invocation_timings() -> tuple[InvocationTiming, ...]
```

Takes no arguments and returns one `InvocationTiming` for every invocation
whose cycle is complete, once each, oldest first. A cycle is complete once a
later invocation began or `mark_cycle_end()` closed it.
Reading waits for the device to reach the closing event and for nothing
else, so a loop may read the previous step's timing after each call without
draining the stream. The callable keeps the timelines of the sixteen most
recent invocations; a loop that never reads loses the oldest completed ones
past that, never a running one. Reading a closed callable raises.

## `PlannedTrainStep.mark_cycle_end()`

```text
PlannedTrainStep.mark_cycle_end() -> None
```

Takes no arguments and returns `None`. It closes the last invocation's cycle
where the next one would begin: the marker is recorded on the compute stream
behind everything the invocation enqueued, before any wait the caller
performs. A loop that measures its last step calls this after that step's call
returns and before it synchronizes, so the step reads like every other rather
than including the drain.

## `InvocationTiming`

| Field | Meaning |
|---|---|
| `step_number` | The completed step's number, as `StepResult.step_number`. |
| `cycle_seconds` | Origin to the next origin or the end marker: the step's time. |
| `opening_delay_seconds` | Origin to the first task's compute start: the first task's readiness waits and whatever the opening still held the stream for. |
| `selected_span_seconds` | First task's compute start to the last task's compute end. |
| `exposed_tail_seconds` | Last task's compute end to the cycle's end: terminal work the stream itself still did. |

`cycle_seconds == opening_delay_seconds + selected_span_seconds + exposed_tail_seconds`
exactly, since each is a difference of two of the same events. Transfers
that drained on the lanes after the last task are not in the cycle: they
cost the step nothing, and an invocation that has to wait for them pays in
its own opening delay.

A traced step's `StepTimingSummary` carries the same three parts as
`cycle_seconds`, `opening_delay_seconds` and `exposed_tail_seconds`, beside the
selected span it already reported; `cycle_seconds` is `None` when the trace
is resolved before anything closed the cycle, which is why the quickstart
marks the cycle's end before resolving its traced step. See
[StepResult diagnostics](../step-diagnostics.md#summary).

## Where it is used

The quickstart's `run_budgets.csv` and throughput figures report the median
cycle of the steps after the first (the first pays the plan's opening); the
performance gate's `median_step_seconds` is the median cycle over its
measured steps and its simulator error compares that cycle with the
predicted step. Both report the host's own wall time beside it, which
decides nothing.
