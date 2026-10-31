# Occupancy diagnostics

`python -m shadowspill.diagnostics.occupancy` and this package's public imports
walk a stored plan's step and attribute what occupies the execution and spill
pools at every moment to objects and their categories, beside the fetch,
compute and evict lanes. The [occupancy guide](../../../../docs/python/occupancy.md)
covers the command, the categories and the rules the walk follows. The
implementation has these responsibilities:

| Module | Responsibility |
| --- | --- |
| `intervals.py`, `facts.py` | Object roles and categories, selected tasks, byte intervals and aggregation |
| `clock.py` | Simulated/device timestamps and interval unions |
| `spill.py`, `execution.py` | Spill residency and physical execution leases |
| `plan.py`, `summary.py` | Assemble a plan view or a floor and compute its summary |
| `page.py`, `html.py`, `occupancy_page.html` | Numeric page data, HTML tables and the interactive timeline |
| `stored.py` | Locate the program a stored selection was made for |
| `cli.py` | Argument parsing and command orchestration |

Spill residency follows the simulator's transfer/release events. Execution
occupancy follows the admitted layout's leases, including storage handoffs.
Host-copy payload and full execution-object bytes remain separate; collective
time is not counted as host-copy traffic.

The command reads a plan store's records: a selection under
`planning/results/` and the program the store keeps for it, or the one
`--program` names. A saved Plan's report comes from the public `write_report`
API in `shadowspill.reporting`, which needs no store.

## Views

`attribute(selection, program, diagnostics=None)` returns a `PlanOccupancy`,
one view of a plan: `simulated` on the simulator's clock, or `traced` on the
device's when a traced step's diagnostics are given; a selection without a
`simulation_result` is refused. `unconstrained(program, selections)` and
`all_save(program)` return the two floors: every task back to back at its
profiled time, every object resident and each task's profiled workspace while
it runs, with nothing spilled and no transfers. `clock_label` names a view's
clock; an `all_save` view whose cheapest alternatives include a recompute says
that it is not a measured all-save baseline.

| Field | Meaning |
| --- | --- |
| `view` | `simulated`, `traced`, `unconstrained` or `all_save`. |
| `execution`, `spill` | One `Occupancy` per pool. `execution` is `None` when the selection has no admitted layout, and a floor's `spill` is empty. `pools` lists them in page order, execution first. |
| `tasks` | A `TaskSpan` per executing task: `task_id`, `name` (the traced step's name for it, else its id), `phase`, `start_ns`, `end_ns` and `overhead_ns`. |
| `transfers` | A `TransferSpan` per fetch or evict the walk saw: `direction`, `start_ns`, `end_ns`, `bytes`, `execution_bytes`, `alias_group_id`, `object_id` and `category` (the slot's occupant when the transfer starts) and `trigger_task_id`. |
| `facts` | The `ProgramFacts` the attribution rests on: the executing tasks in order, each object's role, producer, consumers and category, and each alias group's generations. |
| `clock` | The view's `Clock`: task starts and ends, transfer intervals, the makespan and, on a traced view, `untimed_transfers`. |
| `assumed` | What the plan was priced against: the simulation config's solo and concurrent bandwidths and latencies per lane, and the blended rate the simulated transfers achieved (`fetch_blended_bandwidth_bytes_per_second`, `evict_blended_bandwidth_bytes_per_second`), taken from the simulation on every view. |
| `interpolated_leases` | On a traced view, the lease instants no traced event named. |
| `collectives`, `transfer_phases` | On a traced view, the step's collective record and its transfers' phase regions with payload and rate summaries. |

An `Occupancy` holds one pool's `intervals` (`Interval`: `start_ns`, `end_ns`,
`bytes`, `object_id`, `category`, `role`), `reported_peak_bytes` (the planner's
own figure: the simulation's spill peak, or its first device peak),
`capacity_bytes` (the simulation's spill capacity, or the layout's pool
capacity) and, for the execution pool, `required_bytes` (what the admitted
layout needs with every lease at its fixed offset). `at`, `peaks_by` and
`series` read it by category or by role, `peak` as a whole.

## Summary, tables and pages

`summarize(view, tokens_per_step=None)` gives the summary a page shows:

| Key | Meaning |
| --- | --- |
| `seconds` | The step's span on the view's clock: from the step's origin to its makespan or its last task or transfer end, whichever is later. |
| `tokens_per_second` | `tokens_per_step` over `seconds`, when the tokens are given. |
| `execution_peak_gib`, `spill_peak_gib` | Each pool's peak by the walk. |
| `idle_percent` | The share of the span with no task computing: the span less the tasks' summed time. |
| `recompute_percent` | The plan's regeneration overhead, what its recompute alternatives cost beyond their `save` options by the program's profiles, as a share of the span; `None` when a profile is missing. |
| `fetch_utilization_percent`, `evict_utilization_percent` | The share of the span the lane is busy, over the union of its intervals. |
| `fetch_gib`, `evict_gib` | Full execution-object bytes the lane served. |
| `fetch_host_gib`, `evict_host_gib` | Host-copy payload the lane moved. |
| `fetch_gbps`, `evict_gbps` | Execution-object bytes over the lane's busy time: the rate this view achieved. |
| `assumed_fetch_gbps`, `assumed_evict_gbps` | The blended rate the simulated transfers achieved. |
| `assumed_*_solo_gbps`, `assumed_*_concurrent_gbps`, `assumed_*_latency_us` | The simulation config's bandwidths and latencies. |

`table` prints one pool: its peak and when, the planner's reported peak, the
capacity and the layout's requirement, then each category's (or role's) bytes
and share of the peak at the peak and at every `--at` time, and its own peak.
`--json` prints, per view and pool, `peak_bytes`, `peak_ns`,
`reported_peak_bytes`, `capacity_bytes`, `at_peak` (bytes by name at the
peak), `at` (the same at each `--at` time, keyed by nanoseconds) and
`own_peaks` (each name's `[bytes, ns]`). `page_data` is the JSON a page
inlines: `view`, `clock`, `plan` (the line under the title),
`untimed_transfers`, `end_seconds`, `summary`, `pools` (each sampled in GiB by
name over `times`, with its peak, reported peak, capacity and a `note`),
`compute` rows `[start, end, phase, name, overhead]`, `fetch` and `evict` rows
`[start, end, execution bytes, category, alias group, host bytes]`, times in
seconds, and on a traced view `collectives` and `transfer_phases`.

## Reading the numbers

- The execution pool has three peaks. The walk's sums the live leases as if
  they packed perfectly. The reported one is the simulation's device peak,
  object bytes plus each running task's profiled workspace, where the layout
  leases a task's allocations one by one. `required_bytes` is the peak with
  every lease at its fixed offset, at or above the walk's.
- The spill walk applies the simulator's rules, so on the simulated view its
  peak matches the reported one; a traced view moves the same transfers to
  the device's times.
- `bytes` counts the copy phases of an alias group's transfer recipe (the
  whole group when it has none) and `execution_bytes` the object the transfer
  serves; they differ when the recipe copies less than the object and timed
  phases, such as a collective, do the rest. Neither measures collective
  fabric traffic, and a collective's intervals already lie inside their
  parent task or transfer.
- A simulated view achieves its assumed rates by construction; a traced view
  shows what the device did against them. A transfer interval runs from its
  start on the lane to its finish, never from its issue.
- On a traced view, `untimed_transfers` counts simulated transfers the trace
  did not time: they are off the lanes, and a walk that needs their time takes
  their trigger task's end. The execution pool's leases sit at the device
  times of the events that open and close them, and `interpolated_leases`
  counts the instants placed between task boundaries instead.
- `overhead_ns` is a recompute alternative's profiled time beyond its `save`
  option's, carried by its backward task (its last task when it has none):
  zero for other tasks, `None` when either profile is missing, and never a
  measured kernel boundary.
- A slot with no executing occupant counts as `unattributed`, and a task's
  workspace lease as `task workspace`.
