# Diagnostics API

Saved Plans expose `summary`, `schedule`, `timeline`, `memory_layout` and
`diagnostics`. These are inspectable on CPU; `write_report(plan, directory)`
produces simulated timelines and tables without execution. Search results retain
each successful Plan and failure diagnostics. See the
[phase API](../../architecture/offline-pipeline.md) for artifact boundaries.

For external GPU captures, `Executor.run(..., profiler_annotations=True)` emits
semantic task and transfer ranges until invocation completion. It defaults off
and does not replan or collect a runtime-event ledger. The
[saved-winner capture tool](../../../benchmarking/quickstart.md#nsight-captures-of-saved-winners)
uses the existing Plan in fresh workers and checks trace completeness.

The remaining sections document the older callable report/event interfaces,
which are retained while their diagnostic consumers migrate. In that path,
step diagnostics are opt-in through `runtime_trace=True`. For
field-by-field workflows use [Interpreting a PlanReport](../plan-report.md) and
its [field reference](../plan-report-fields.md), and [Interpreting StepResult
diagnostics](../step-diagnostics.md).

PressureFit also supports opt-in [cut-by-cut search traces](../pressurefit-trace.md)
through `SearchOptions.trace_directory`. The reader in
`shadowspill.diagnostics.pressurefit_trace` exposes reduction replay, ranked
alternatives, the top three over-budget boundaries, and semantic label joins.

## Plan report

Every planned callable exposes a `PlanReport` through `.plan_report`. It is the
whole record of what planning decided and why, and it exists whether or not
`verbose` printed anything.

```python
report = train_step.plan_report

print(report.program.digest)
print(report.predicted_makespan_ns)
print(report.predicted_device_peak_bytes)
print(report.fetch_profile)

task = report.diagnostics.task("execution_000017")
print(task.semantic_name, task.chosen_graph_pair_variant)
```

Besides the derived `summary` (a `PlanSummary`, described in [the field
reference](../plan-report-fields.md#plansummary)), the report exposes the
selected `execution_plan`, its `search_result`, the configured pool names and budgets, the transfer capabilities, task
profiles, transfer actions, aggregate transfer bytes, the `StepDataOrdering`
the step walked its microbatches with, and the `search_options` it was searched
under. It also carries a `PlanDiagnostics` tree, keyed primarily by
chronological execution id.

The public planning diagnostic records inside that tree are:

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

Together they expose stage-to-structural-contract mapping, every legal graph
pair, the chosen variant per execution, semantic storage roots and views,
compiled physical layout, input/output/mutation/workspace sizes, task timings,
allocation behavior, profiling metadata, store artifacts, and phase wall time.

Additional nested report records describe allocation-contract operations,
representative inputs, task memory envelopes, physical layouts, and admission
attempts. They are intentionally reached through the report rather than added
to the top-level `shadowspill.pytorch` import surface.

## Step result and handle

`StepResult` contains the detached objectives, the reconstructed objective
metrics, the completed step number, and an optional diagnostics handle.
Tensor-valued metric leaves remain tensors; static leaves preserve their
captured values. A planned forward call is traced the same way and reaches
its handle through `PlannedForward.diagnostics`, because a forward call
returns the model's own output and carries nothing else back.
`StepResult.diagnostics` is `None` for an ordinary step and a
`DiagnosticsHandle` for a traced one; resolving the handle returns
`StepDiagnostics` and may wait for recorded events.

`StepDiagnostics` has seven views: `summary`, `tasks`, `transfers`, `timelines`,
`allocator`, `runtime`, and `collectives`.

`StepTimingSummary` is the reconciliation, and it carries the traced step's
cycle, head and exposed tail beside the selected span ([timing](timing.md)):
profiled task time against real task-event time, simulated against real
waiting, the selected span, the simulator's makespan, and the call-level host
totals. `PhaseTimingComparison` breaks the task time down by phase.
`InvocationTiming` is the same cycle measured for every invocation, traced or
not, returned by `PlannedTrainStep.invocation_timings()` and defined on the
[timing](timing.md) page.

`tasks` maps execution task ids to a `TaskRecord` each, and `transfers` is a
`TransferRecords` pair of mappings, `fetch` and `evict`, from transfer id to
`TransferRecord`. Every record places simulated beside measured -- start, end,
duration, and their deltas from the common invocation origin -- with a host group for boundary
entry and exit, dispatch costs, and the worker's queueing and completion
observations. Each transfer names the tasks it sits between (`previous_access`,
`next_access`) and the task whose result it carries (`modified_by`), all of
them keys into `tasks`.

`Timelines` is the order on three lanes sharing one device-clock zero: the
execution task ids in compute-stream order, and for `fetch` and `evict` a
`TransferQueue` holding the transfer ids in FIFO order and a `LaneSummary`.

`allocator` is an `AllocatorTrace`, the ordered allocation and free ledger with
the allocator's own pool read before and after the step; `runtime` is a
`RuntimeTrace`,
the runtime's counter deltas, terminal queue state, trace capacity and overflow
flags, and the raw runtime event records. Both are reached through
`StepDiagnostics` rather than imported from `shadowspill.pytorch`.

`collectives` is a `CollectiveTrace` with a bounded sequence of `CollectiveRecord`
values. A distributed GPU trace automatically captures NCCL collectives issued
by tasks and by sharded transfers, including optimizer gradient reduction and
weight publication. Each record includes:

- `issued_at_ns`: host submission time mapped to the invocation origin;
- `started_at_ns` and `finished_at_ns`: NCCL GPU event times on that same axis;
- operation, group identity, global membership, local rank, and collective sequence;
- per-rank input/output bytes, task/execution identity, and, for transfers, the
  object, alias, action ordinal, and fetch/evict source.

Start follows the collective stream's prior work and input waits. The measured
interval includes peer waits inside the collective. Host completion polling is
not used as GPU completion time. Input/output bytes describe payloads, not total
network traffic. Collective durations are already included in their enclosing
task or transfer and must not be added again.

Tracing retains NCCL event references until diagnostics are resolved. The
callback is removed after the invocation; PyTorch keeps event timing enabled
on process groups that have been traced. `dropped`, `errors`, and
`missing_timings` make incomplete capture explicit and also clear
`summary.trace_complete`. Missing timestamps are `None`. Records are included
in serialized step JSON, including when the allocator/runtime ledgers are saved
separately. CPU control-group collectives are outside this GPU timeline.

Every record on this page lives in `shadowspill.diagnostics`, which names no
framework: reading a stored step's diagnostics costs no framework import.

All public diagnostic records are immutable. `PlanDiagnostics.as_dict()` and
`StepDiagnostics.as_dict()` return JSON-friendly nested dictionaries for
storage or analysis; the [StepResult diagnostics
guide](../step-diagnostics.md) defines every field.
`shadowspill.diagnostics.step_file` stores a traced step as that dictionary
without its two event ledgers, which `write_step_diagnostics` writes beside it
as columns and `read_step_events` reads back
([exporting diagnostics](../step-diagnostics.md#exporting-diagnostics)).

## Reports

`write_report(plan_or_sweep, directory, plots=True)` consumes an existing `Plan`
or `SweepResult` on CPU. It writes an HTML index, numeric task/transfer tables,
summaries and simulated timeline figures. Each sweep case keeps its own Plan or
failure; winners remain inspectable through `SweepResult.best_per_budget`.

Artifact saving is separate from report generation. See the
[phase API guide](../../architecture/offline-pipeline.md) for build, profile,
planning and execution examples.

## Runtime failures

`RuntimeExecutionError.diagnostics` retains structured failure details when
the native runtime supplied them. `pool_id` identifies the pool whose allocation
failed; it can differ from the task's execution pool when an eviction needs
spill storage. `requested_bytes`, `free_bytes`, and `largest_free_range_bytes`
refer to that pool. A transfer with a sharded spill representation reports the
shard's requested bytes when its spill reservation fails.

Out-of-memory messages include the refused action and object state when the
frontend can resolve them, along with the semantic task and pool identifier.
`diagnostics.as_dict()` retains these fields for programmatic analysis.

## Defaults and overhead

Tracing uses bounded preallocated buffers inside the C runtime. No trace record
is appended when `runtime_trace=False`. Profiler ranges are controlled
independently by `profiler_annotations`; enabling one does not enable the
other.
