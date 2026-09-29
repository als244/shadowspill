# Timing: the step on the device clock

Every invocation of a planned callable, training or forward, records three
timing events on the compute stream, without being asked: its origin before its
first task, its span start at its first task's compute, and its span end at
its last task's compute. The step's time is the **cycle**, origin to the next
invocation's origin, which is what a repeated step costs and what throughput
divides by; the architecture page [timelines](../../architecture/timelines.md#the-step-origin-to-origin)
defines it. This page is the API that reads it.

The instants are the runtime's own, on the same clock it times its transfers
with, so a step's cycle and the transfers inside it are comparable without
correcting between two clocks.

## `invocation_timings()`

```text
PlannedTrainStep.invocation_timings() -> tuple[InvocationTiming, ...]
PlannedForward.invocation_timings() -> tuple[InvocationTiming, ...]
```

Takes no arguments and returns one `InvocationTiming` for every invocation
whose cycle is complete, once each, oldest first. A cycle is complete once a
later invocation began or `mark_cycle_end()` closed it.
Reading waits for the device to reach the closing event and for nothing
else, so a loop may read the previous step's timing after each call without
draining the stream. The callable keeps the timelines of the sixteen most
recent invocations; a loop that never reads loses the oldest completed ones
past that, never a running one. Reading a closed callable raises.

## `mark_cycle_end()`

```text
PlannedTrainStep.mark_cycle_end() -> None
PlannedForward.mark_cycle_end() -> None
```

Takes no arguments and returns `None`. Waits for this plan's required terminal
work, then records the final cycle marker on the compute stream. A loop calls
this after its final invocation, where the next invocation would otherwise
wait and record its origin.

## `prepare_runtime_trace()`

```text
PlannedTrainStep.prepare_runtime_trace() -> None
PlannedForward.prepare_runtime_trace() -> None
```

Allocates reusable detailed-trace buffers and events. Call before a measured
loop so preparing its final traced step cannot inflate the preceding cycle.
Repeated calls do nothing; calling a closed callable raises. The first
`runtime_trace=True` call still prepares lazily if necessary.

## `InvocationTiming`

| field | type | meaning |
|---|---|---|
| `step_number` | `int` | The completed step's number. |
| `cycle_seconds` | `float` | Origin to the next origin or final marker, including caller work. |
| `entry_delay_seconds` | `float` | Origin to first computational task start. |
| `selected_span_seconds` | `float` | First computational task start to last computational task end. |
| `exposed_tail_seconds` | `float` | Last computation to cycle end, including terminal transfers and caller work. |

`cycle_seconds == entry_delay_seconds + selected_span_seconds + exposed_tail_seconds`
by construction. The next origin follows the prior terminal drain, so required
writeback is included exactly once.

A traced step also reports `real_invocation_seconds` through the last required
compute or transfer completion, and `real_terminal_tail_seconds` from last
compute to that completion. Its cycle is optional until a successor or final
marker exists. See [StepResult diagnostics](../step-diagnostics.md#summary).

## Where it is used

The benchmarking quickstart's `run_budgets.csv` and throughput figures report
the median cycle of the steps after the first, excluding first-call setup and the final traced call; the performance gate's `median_step_seconds` is the median cycle over
its measured steps, and its simulator error compares that cycle with the
predicted step. Both report the host's own wall time beside it, which decides
nothing.
