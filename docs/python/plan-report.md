# Interpreting a Plan

`plan_program` returns a `Plan`. Load it with `Plan.load(path)` to inspect it on
CPU; use `write_report(plan, directory)` for simulated timelines and tables.
The archive includes its Program and applied profiles. It requires no source
artifact folders or live model for inspection.

## Structure

| Field | Purpose |
| --- | --- |
| `program` | Task definitions, objects, alternatives and semantic source metadata |
| `profiled_program` | Applied task durations and memory evidence |
| `memory_budgets` | Total GPU allowance and spill capacity per worker |
| `communication_info` | Topology, supplied rates and measured communication evidence |
| `schedule`, `selections` | Selected task alternatives and transfer actions |
| `timeline` | Simulated tasks, transfers, makespan and memory peaks |
| `memory_layout` | Certified physical allocation layout, when available |
| `diagnostics` | Search evidence, candidates and counters |
| `summary` | Concise timing, memory and traffic totals |

```python
from shadowspill import Plan, write_report

plan = Plan.load("training.ssp")
print(plan.summary)
print(plan.simulation_only_reasons)
write_report(plan, "reports/training")
```

## Transfer assumptions and budgets

`summary["duration_seconds"]` is the simulated makespan.
`execution_peak_bytes` includes fixed device memory; `spill_peak_bytes` describes
spill occupancy. `fetch_bytes` and `evict_bytes` include host traffic and
collective payloads, which also appear separately as `*_host_bytes` and
`*_collective_bytes`. Inspect `communication_info` before comparing predictions
made for different backends or machines.

A Plan with estimated memory remains simulation-only. `simulation_only_reasons`
also identifies missing executable code, allocation evidence or communication
qualification. An aggregate workspace peak cannot certify allocation lifetimes.

## Unique stages and graph pairs

`program.tasks` carries the built task alternatives, including save/recompute
forms. `profiled_program.task_profiles` holds their applied timing and workspace
records. Task compatibility identities allow reuse across stage occurrences;
semantic names and source metadata connect those identities to model modules.
Planning chooses among the built alternatives; it does not compile or measure.

## Search diagnostics

`diagnostics` is a `PlanningDiagnostics` tree. It records the candidates visited,
search counters, resolution outcomes and failures. PressureFit may also produce
[cut traces](pressurefit-trace.md) for stepwise replay of its decisions.
The [field reference](plan-report-fields.md) describes these search records.

A feasible result and an exhausted search are different: failure diagnostics
say whether a constraint was violated or the search stopped without an answer.
A sweep retains each case's Plan or failure and a best-per-budget lookup.

## Physical-layout diagnostics

`memory_layout` records fixed offsets, charged extents, lifetimes and the replay
certificate. Its identity must match the costed program, selected schedule and
admission evidence. Executor validates these again before execution.

Logical simulated peaks alone do not establish a physical layout. Fragmentation,
alignment, overlapping lifetimes, outputs and provider scratch all matter. See
[physical admission](../architecture/physical-admission.md).

## Execution comparison

`executor.run(..., runtime_trace=True)` returns a result handle whose
`diagnostics` exposes measured tasks and transfers. Compare those records with
`timeline`; [the worked example](../examples/diagnostics.md) shows the join.
Keep simulated prediction, measured device intervals and host dispatch time
separate. Diagnostic collection is opt-in and can add measured overhead.
