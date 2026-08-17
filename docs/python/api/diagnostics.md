# Diagnostics API

Planning diagnostics are always produced. Step diagnostics are opt-in through
`runtime_trace=True`. This page inventories the public values; use
[Interpreting a PlanReport](../plan-report.md) and [Interpreting StepResult
diagnostics](../step-diagnostics.md) for field-by-field workflows.

## Plan report

Every planned callable exposes `PlanReport` through `.plan_report`. The report
contains a `PlanDiagnostics` tree keyed primarily by chronological execution
ID.

```python
report = train_step.plan_report

print(report.program.digest)
print(report.predicted_makespan_ns)
print(report.predicted_device_peak_bytes)
print(report.fetch_profile)

task = report.diagnostics.task("execution_000017")
print(task.semantic_name, task.chosen_graph_pair_variant)
```

`PlanReport` also exposes the selected `execution_plan`, optional
`initial_execution_plan`, recurrent and initialization PressureFit results,
configured pool names/budgets, transfer capabilities, task profiles, transfer
actions, and aggregate transfer bytes.

The public planning diagnostic records are:

- `PlanPhaseTiming`
- `PlanCompilerProfile`
- `PlanCacheArtifact`
- `PlanProfilingMetadata`
- `PlanObjectFootprint`
- `PlanAllocationEvent`
- `PlanStorageRoot`
- `PlanOutputView`
- `PlanMutationBinding`
- `PlanCompiledRoot`
- `PlanCompiledOutputView`
- `PlanGraphProfile`
- `PlanGraphPair`
- `PlanUniqueStage`
- `PlanTaskStage`

Together they expose stage-to-structural-contract mapping, every legal graph pair,
chosen variant per execution, semantic storage roots and views, compiled
physical layout, input/output/mutation/workspace sizes, task timings,
allocation behavior, profiling metadata, cache artifacts, and phase wall time.

Additional nested report records describe allocation-contract operations,
representative inputs, task memory envelopes, physical layouts, and admission
attempts. They are intentionally reached through the report rather than added
to the top-level `shadowspill.pytorch` import surface.

## Step result and handle

`StepResult` contains detached objectives, reconstructed objective metrics,
the completed step number, and an optional diagnostics handle. Tensor-valued
metric leaves remain tensors; static leaves preserve their captured values.
`StepResult.diagnostics` is `None` for an ordinary step and a
`DiagnosticsHandle` for a traced step. Resolving the handle returns
`StepDiagnostics` and may wait for recorded events.

Each execution has `ExecutionTiming` and exactly seven boundary timestamps:

- host `before_task` entry and exit;
- host `after_task` entry and exit;
- compute-stream timestamp before readiness waits;
- compute-stream `before_task_compute`;
- compute-stream `after_task_compute`.

These distinguish host orchestration, stream readiness stalls, task compute,
run-ahead, and inter-task dispatch gaps.

`StepTimingSummary` aggregates profiled task time, real task-event time,
simulated and real inter-task gaps, simulated and real selected-task span, and
startup/cooldown evidence. `PhaseTimingComparison` summarizes differences by
phase. `SimulatorTransferComparison` compares each real transfer with its
simulated trigger, start, end, and duration.

Allocator diagnostics separate zero-byte requests, fixed-layout core slots,
dynamic scratch, caller-owned allocations, pending releases, waits, failures,
and task-envelope usage. Transfer diagnostics retain object identity,
direction, bytes, queue/wire timestamps, and completion.

All public diagnostic records are immutable. `PlanDiagnostics.as_dict()` and
`StepDiagnostics.as_dict()` return JSON-friendly nested dictionaries for
artifact storage or analysis.

The public summary/comparison values are `ExecutionTiming`,
`StepTimingSummary`, `PhaseTimingComparison`, and
`SimulatorTransferComparison`. Detailed task, allocator, runtime, and transfer
records are reached through `StepDiagnostics`.

## Defaults and overhead

Tracing uses bounded preallocated native buffers. No trace record is appended
when `runtime_trace=False`. Profiler ranges are controlled independently by
`profiler_annotations`; enabling one does not enable the other.
