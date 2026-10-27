# Timing an invocation

`Executor.run` returns after dispatch. Measure completion through its result
handle; timing only `run()` measures host submission, not the training step.

```python
from time import perf_counter

start = perf_counter()
pending = executor.run(args=examples)
outputs = pending.result()
elapsed_seconds = perf_counter() - start
```

This end-to-end interval includes required terminal writeback and any enabled
observations. Warm up before measuring steady state. Trainer's `StepResult.seconds`
uses the completed invocation; benchmark runners retain their measurement policy
with the results.

## Device trace

Pass `runtime_trace=True` to record task and transfer events on the runtime's
clock. `pending.diagnostics` waits for the call and provides `StepDiagnostics`.
Tracing defaults off; its collection and buffer setup can affect wall time.
External profiler annotations are separately controlled by `profiler_annotations`.

The summary distinguishes the first-task entry delay, selected compute span,
required transfer tail, device invocation time and host submission work.
`InvocationTiming` is an internal recorder value with `step_number`,
`cycle_seconds`, `entry_delay_seconds`, `selected_span_seconds` and
`exposed_tail_seconds`. An isolated traced invocation closes its cycle at
completion; it does not measure caller work between separate invocations.

See [trace fields](../step-diagnostics.md#summary) and
[timeline clocks](../../architecture/timelines.md) for the definitions.
