# Occupancy diagnostics

`python -m shadowspill.diagnostics.occupancy` and this package's public imports
walk a stored plan's step and attribute what occupies the execution and spill
pools at every moment to objects and their categories, beside the fetch,
compute and evict lanes. The [occupancy guide](../../../../../../docs/python/advanced/occupancy.md)
covers the command, the categories and the rules the walk follows. The
implementation has these responsibilities:

| Module | Responsibility |
| --- | --- |
| `intervals.py`, `facts.py` | Object roles and categories, selected tasks, byte intervals and aggregation |
| `clock.py` | Simulated/device timestamps and interval unions |
| `spill.py`, `execution.py` | Spill residency and physical execution leases |
| `plan.py`, `summary.py` | Assemble a plan view or a floor and compute its summary |
| `page.py`, `html.py`, `occupancy_page.html` | Numeric page data, HTML tables and the interactive timeline |
| `alternatives.py` | Every task alternative beside what the selection chose: forms, counts, sizes and observed time |
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
`simulation_result` is refused, and so is an object whose role `categorize`
has no category for. `unconstrained(program, selections)` and
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
| `clock` | The view's `Clock`: task starts and ends, each transfer's issue, start and end, the makespan and, on a traced view, `untimed_transfers`. |
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

## Task alternatives

`alternatives_data(program, selections, positions=None, labels=None,
covers=None, measured_ns=None)` describes every alternative group of a
program, chosen or not, for a page's task view. A step repeats its stages, so
groups whose options activate the same profiles in the same phases are one
*form*, described once. A form is a kind of stage and is compiled once for
every stage of its kind; a stage is one place in the model. A form therefore
cannot say where a stage is: the caller gives the parts of the model each task
reaches, by task id, as `covers`.

| Field | Meaning |
| --- | --- |
| `options` | The option ids the program's groups offer, in first-seen order. |
| `forms` | One entry per form: `index`, `kind` (a letter, `A`, `B`, ... `Z`, `AA`, in the order the program first reaches the form, so a letter belongs to one program and is not an identity across programs), `covers` (every part of the model the form's occurrences cover), `instances` (groups sharing the form), `sizes_vary` (an occurrence's sizes differ from the first, which is the one described) and `options`. |
| `forms[].options` | Per option: `option`, `chosen` (occurrences that selected it), `tasks`, `runtime_ns` (the tasks' profiled time, `None` if any is unprofiled), and `retained_groups` and `retained_bytes`, the alias groups the option holds across the group's boundary. |
| `tasks` rows | `phase`, `profile` (the task's profile id), `label` (what `labels` says the profile's tasks do, else empty), `runtime_ns` and `workspace_bytes` from the task's profile, `input_bytes`, `output_bytes`, `mutated_bytes`, `total_bytes` and `measured`. |
| `map` | When `positions` gives every group a `(column, row)`: `columns`, and per row its `cells`, each `[chosen option, form index]` or `None` where no group sits, and its `covers`. `None` when any group has no position. |
| `single` | The tasks outside every group, one row per phase and profile, with `count`, `covers` and the same columns. |
| `covers` | Parts of the model, in the order first met, each `[path, number]` as the caller gave them: a module's path and, for a numbered member of a repeated part, its number (`["layers", 3]`, `["embedding", None]`). Empty without the `covers` argument, and for tasks it places nowhere. |

A size is the device storage a set of objects reaches, each alias group's
spans merged, so two views of one allocation count once. `total_bytes` is what
a task's inputs, outputs and mutated objects reach together plus its profiled
workspace, the storage that must be resident for it to run; without a profiled
workspace it is `None`, never the smaller number. `measured` summarises the
durations `measured_ns` holds for a task over the occurrences that selected its
option (`count`, `median_ns`, `min_ns`, `max_ns`); an option nothing selected
never ran and has none. A saved Plan's report attaches this as each view's
`alternatives`, with a group's microbatch and stage as its position, and with
`labels` and `covers` from the Plan's program: what each profile's tasks do
(`forward`, `backward (accumulating)`, `all_gather`) and the parameters and
buffers each task reads or changes. Progress lines and task errors name a task
from the same source, beside the profile key a row carries as `profile`.

## Reading the numbers

- The execution pool has three peaks. The walk's sums the live leases as if
  they packed perfectly. The reported one is the simulation's device peak,
  object bytes plus each running task's profiled workspace, where the layout
  leases a task's allocations one by one. `required_bytes` is the peak with
  every lease at its fixed offset, at or above the walk's.
- The spill walk applies the simulator's rules, so on the simulated view its
  peak equals the reported one. An evict's or write-back's spill copy counts
  from the transfer's issue, the simulator's submission (its `ready_ns`),
  not from its start on the lane, and a released copy is freed at the
  release's trigger task's end, even where the simulator holds the release
  back behind a pending write-back or an earlier waiting action. A traced
  view moves the same transfers to the device's times and issues each at its
  trigger task's end there.
- `bytes` counts the copy phases of an alias group's transfer recipe (the
  whole group when it has none) and `execution_bytes` the object the transfer
  serves; they differ when the recipe copies less than the object and timed
  phases, such as a collective, do the rest. Neither measures collective
  fabric traffic, and a collective's intervals already lie inside their
  parent task or transfer.
- `transfer_phases` splits each traced transfer's lane interval into regions.
  A `copy` or `collective` region is an event the trace timed. A transfer
  whose recipe is one copy of its whole payload, alone or after the device's
  priced host latency, has no separate events: its interval is one `copy`
  region timed by `copy-only lane events`. Lane time outside a transfer's
  measured events is an `overhead` region. A transformed transfer without a
  copy event is one `unresolved` region and counts in
  `unresolved_copy_transfers` and `unresolved_host_bytes`; neither kind of
  gap carries bytes or a rate. A page draws these regions and its collective
  sections when the step recorded a collective or left a transfer unsplit.
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
