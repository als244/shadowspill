# Diagnostics API

Saved `Plan` objects expose `summary`, `schedule`, `timeline`, `memory_layout`
and `diagnostics`. These are inspectable on CPU. `write_report(plan, directory)`
produces simulated timelines and tables; a `SweepResult` retains each successful
Plan and the diagnostics of unsuccessful cases. See [phase artifacts](artifacts.md).

## Execution traces

Both forward and training calls use the same result handle:

```python
pending = executor.run(args=examples, runtime_trace=True)
outputs = pending.result()
step = pending.diagnostics
```

`pending.diagnostics` waits for completion and returns `StepDiagnostics`; it is
`None` when tracing is disabled. Repeated reads reuse the completed result.
The records contain seven views: `summary`, `tasks`, `transfers`, `timelines`,
`allocator`, `runtime` and `collectives`. The [field guide](../step-diagnostics.md)
describes their clocks, byte counts and completeness flags.

`Executor.run(..., profiler_annotations=True)` emits semantic task and transfer
ranges for external GPU captures. This is independent of runtime-event tracing.
Both flags default off. See the [saved-winner capture tool](../../../benchmarking/quickstart.md#nsight-captures-of-saved-winners)
for Nsight captures in fresh workers.

## Reports

`write_report` consumes a saved Plan or SweepResult without importing PyTorch or
opening runtime pools. Report generation and artifact persistence are separate:
use `plan.save(path)` for a portable executable archive. Reports do not replace
allocation evidence or make an estimated-memory Plan executable.

[Interpreting a Plan](../plan-report.md) and the
[worked trace example](../../examples/diagnostics.md) show how to compare
simulated costs with measured tasks and transfers.

PressureFit's optional [cut-by-cut traces](../pressurefit-trace.md) retain search
alternatives and memory-pressure evidence. These describe search decisions,
not measured GPU execution.

## Failures

`RuntimeExecutionError` can carry `RuntimeFailureDiagnostics` with the failed
task, allocation, capacity and contract operation. The runtime retains the
first failure; see [failures and cleanup](../failures.md). Failed invocations
make Executor unusable for further work. Closing it still releases its resources.
