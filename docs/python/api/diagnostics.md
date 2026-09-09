# Diagnostics API

Planning diagnostics are always produced; step diagnostics are opt-in through
`runtime_trace=True`. This page inventories the public values. For
field-by-field workflows use [Interpreting a PlanReport](../plan-report.md) and
its [field reference](../plan-report-fields.md), and [Interpreting StepResult
diagnostics](../step-diagnostics.md).

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
selected `execution_plan` and optional `initial_execution_plan`, the
`ProgramPlanResult` for the recurrent step and for an initialization step when
there was one, the configured pool names and budgets, the transfer
capabilities, task profiles, transfer actions, aggregate transfer bytes, the
`StepDataOrdering` the step walked its microbatches with, and the
`resolution_options` it was searched over. It also carries a `PlanDiagnostics`
tree, keyed primarily by chronological execution id.

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
captured values. `StepResult.diagnostics` is `None` for an ordinary step and a
`DiagnosticsHandle` for a traced one; resolving the handle returns
`StepDiagnostics` and may wait for recorded events.

`StepDiagnostics` has six views.

`StepTimingSummary` is the reconciliation, and it carries the traced step's
cycle, head and exposed tail beside the selected span ([timing](timing.md)):
profiled task time against real task-event time, simulated against real
waiting, the selected span, the simulator's makespan, and the call-level host
totals. `PhaseTimingComparison` breaks the task time down by phase.

`tasks` maps execution task ids to a `TaskRecord` each, and `transfers` is a
`TransferRecords` pair of mappings, `fetch` and `evict`, from transfer id to
`TransferRecord`. Every record places simulated beside measured -- start, end,
duration, and their deltas after alignment -- with a host group for boundary
entry and exit, dispatch costs, and the worker's queueing and completion
observations. Each transfer names the tasks it sits between (`previous_access`,
`next_access`) and the task whose result it carries (`modified_by`), all of
them keys into `tasks`.

`Timelines` is the order on three lanes sharing one device-clock zero: the
execution task ids in compute-stream order, and for `fetch` and `evict` a
`TransferLane` holding the transfer ids in FIFO order and a `LaneSummary`.

`AllocatorTrace` is the ordered allocation and free ledger with the pool's
geometry before and after the step. `RuntimeTrace` is the runtime's counter
deltas, terminal queue state, trace capacity and overflow flags, and the raw
runtime event records.

All public diagnostic records are immutable. `PlanDiagnostics.as_dict()` and
`StepDiagnostics.as_dict()` return JSON-friendly nested dictionaries for
storage or analysis; the [StepResult diagnostics
guide](../step-diagnostics.md) defines every field.

## Figures

`shadowspill.plots` draws figures from artifacts that already exist. It plans
nothing and executes nothing, so it never needs the device that produced its
inputs, and it writes into the directory it is given without keying the tree
itself. Both functions return the paths they wrote.

```text
plot_step_search(report, directory) -> tuple[Path, ...]
plot_step_run(entries, directory, *, tokens_per_step) -> tuple[Path, ...]
```

| argument | type | meaning |
|---|---|---|
| `report` | `StepSearchReport` | What `plan_step_search()` answered with. Must hold a single spill budget; the execution budget is the x axis throughout, and a budget with no winning geometry is omitted from every line. |
| `entries` | `Sequence[RunBudgetOutcome]` | The executed budgets to draw, at least one. |
| `directory` | `str` \| `Path` | Where the tree is written. The caller owns it and what distinguishes it. |
| `tokens_per_step` | `int` | Tokens one optimizer step consumes, which is what turns a step time into throughput. |

`plot_step_search()` writes the plan-side tree: throughput, overheads,
transfers, and distance from the unconstrained floor, plus `raw_data/` from
which the whole tree can be drawn again. `plot_step_run()` writes the measured
tree from executed budgets: throughput against the simulation, and where the
prediction fell short. `RunBudgetOutcome` is one executed budget -- what was
planned for it, what the hardware then did, and the same split of the step on
both clocks, so a difference can be attributed rather than only reported.

The [figures guide](../plots.md) describes the tree, what each figure
represents, and the conventions they share.

## Defaults and overhead

Tracing uses bounded preallocated native buffers. No trace record is appended
when `runtime_trace=False`. Profiler ranges are controlled independently by
`profiler_annotations`; enabling one does not enable the other.
