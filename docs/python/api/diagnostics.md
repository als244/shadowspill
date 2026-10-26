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

## Figures

`shadowspill.plots` draws figures from artifacts that already exist. Every
function returns the paths it wrote.

```text
plot_step_search(report, directory) -> tuple[Path, ...]
plot_step_run(entries, directory, *, tokens_per_step) -> tuple[Path, ...]
plot_bandwidth_frontier(path, budgets_gib, lines, *, title, subtitle=None,
                        measured=(), unconstrained_tokens_per_second=None,
                        faded_lines=False) -> Path
```

| argument | type | default | meaning |
|---|---|---|---|
| `report` | `StepSearchReport` | required | What `plan_step_search()` answered with. Must hold a single spill budget; the execution budget is the x axis throughout, and a budget with no winning geometry is omitted from every line. |
| `entries` | `Sequence[RunBudgetOutcome]` | required | The executed budgets to draw, at least one. |
| `directory` | `str` \| `Path` | required | Where the tree is written. The caller owns it and what distinguishes it. |
| `tokens_per_step` | `int` | required | Tokens one optimizer step consumes, which is what turns a step time into throughput. |

`plot_step_search()` writes the plan-side tree; `plot_step_run()` writes the
measured one. `RunBudgetOutcome` is one executed budget: its
`execution_budget_bytes`, the simulated and measured step seconds, the same
split of the step on both clocks -- task compute, idle, and what falls outside
the task window -- and `step_seconds`, every measured step in order, so a
difference can be attributed rather than only reported. `recomputation_seconds`
is the part of task compute the chosen recomputation costs over the save-only
floor; it is a counterfactual rather than a measurement, so the same figure
stands on both clocks. Entry and terminal work are also measured on both clocks:
`simulated_entry_delay_seconds`, `real_entry_delay_seconds`,
`terminal_tail_seconds`, and `real_terminal_tail_seconds`. Their five disjoint
components sum to the corresponding invocation. The fidelity plot uses the
traced invocation; throughput uses the median untraced cycle, which also
includes caller work.

`write_run_tables(entries, directory, *, tokens_per_step)` writes the raw-data
tables alone, without drawing anything. `plot_step_run()` calls it as its last
step, and a caller that runs budgets one at a time calls it after each so the
record stays current: figures are worth rendering once at the end, but the
tables behind them are worth having after every budget, because a run that stops
early should still leave what it measured and the figures can be redrawn from
the tables.

`plot_bandwidth_frontier()` draws one figure rather than a tree: throughput
against execution budget with a line per transfer calibration, for comparing
searches of one program under interconnects a machine does not have. A
`FrontierLine` is one calibration's answer -- its legend label, the two
bandwidths, and the tokens per second it reached at each budget, holding
`None` where no plan existed, so a line keeps its place on the axis. The axis
begins at the first budget any line could answer, because below that no
residency fits at any bandwidth.

`unconstrained_tokens_per_second` draws the compute ceiling, the step with
every task-alternative group charged its cheapest option and nothing waiting,
annotated on the line rather than listed in the legend. `measured` is a
sequence of `MeasuredPoints`, each a label, the budgets a machine actually
reached, and the `line` whose colour they take, so a measurement sits on the
calibration it ran under. `faded_lines` puts the simulated lines into the
background and joins the measured points instead, for the figure whose
subject is what a machine did.

The [figures guide](../plots.md) describes the tree, what each figure
represents, and the conventions they share.

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
