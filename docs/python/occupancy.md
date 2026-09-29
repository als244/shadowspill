# Pool occupancy over a step

`python -m tools.diagnostics.occupancy` reads a plan the store holds and says
what occupies the spill pool and the execution pool at every moment of the
step, attributed to objects and to what those objects are for, as tables and
as pages. It runs on the machine that planned the step or on any other: the
store's files are all it needs. With the traced step's diagnostics it also
puts both pools and the transfer lanes on the device's clock.

## Inputs

A plan store keeps each planning result under
`<store>/v1/planning/results/<xx>/<key>/selection.json`, and the program it
was made for under `<store>/v1/planning/programs/<yy>/<digest>/program.json`,
where the digest is the selection's `program_digest`. The tool finds the
program from the selection; `--program` names one explicitly. The quickstart
writes both under a run's `plan_store/`, and `steps/<budget>gib.json` beside
them is the traced step for `--step`.

Some entries under `results/` name a plan without carrying its evidence (no
`simulation_result`); the tool refuses those and says so.

```text
python -m tools.diagnostics.occupancy <selection.json>
    [--program program.json] [--step steps/12gib.json] [--tokens-per-step N]
    [--by category|role] [--at 5.46,11.05] [--html DIRECTORY] [--json]

python -m tools.diagnostics.occupancy --run <run directory> [--html DIRECTORY]
```

| argument | meaning |
|---|---|
| `selection` | The stored planning result to walk. |
| `--program` | Its program, when it is not beside it in the store. |
| `--step` | The traced step's diagnostics JSON. Adds a second view on the device's clock: the spill pool and the lanes from the traced tasks and transfers, and the execution pool with each lease placed at the device times of the events that open and close it. |
| `--unconstrained` | Also the plan's own floor, as `unconstrained.html` (below). |
| `--all-save` | Also the program at its cheapest alternatives, as `all_save.html` (below); with `--program` and no selection, only that view. |
| `--by` | Group by the tool's categories (the default) or by the program's object roles. |
| `--at` | Seconds into the step to add as columns beside the peak. |
| `--tokens-per-step` | The step's tokens, for the tokens-per-second figure in the page's summary. |
| `--html` | A directory to write the pages into: one page per view, `simulated.html` and `traced.html`, and an `index.html` that links them and carries the tables. Each page names its pool capacities under the title; a run's pages (below) name the whole plan. |
| `--json` | The snapshots as JSON instead of tables. |
| `--run` | A quickstart run directory instead of one selection: pages for every plan it made, under its `timelines/` (or `--html`). See below. |

## What the tables say

One table per pool and view. The first column is the pool at its peak, one
column per `--at` time follows, and the last is each category's own peak and
when it happens. The header names the peak the walk found beside the peak the
planner reported for the same plan; for the spill pool the two agree, because
the walk applies the simulator's rules, and the execution pool's capacity is
the admitted layout's.

```text
spill pool: peak 145.31 GiB at 5.19 s (planner reported 145.31 GiB), capacity 150.00 GiB
  category                     peak          5.46 s        own peak
  saved activations     99.88 GiB  69%     99.75 GiB  69%     99.88 GiB   4.3s
  optimizer state       29.92 GiB  21%     29.92 GiB  21%     29.92 GiB   0.0s
  weights               14.96 GiB  10%     14.96 GiB  10%     14.96 GiB   0.0s
  ...
```

## Categories

Every byte is attributed to the object holding it, and from the object to a
category, from its role and the phases of the tasks that make and read it:

| category | what it is |
|---|---|
| weights | Parameters. |
| optimizer state | The optimizer's state, moments and the like. |
| model gradients | Gradient objects a task accumulates into or the optimizer reads: the gradient of a weight. |
| tangents | Gradient objects handed from one backward stage to the next. |
| saved activations | Activations the forward made, held for a later task: the next stage of the same microbatch after the walk has run the other microbatches through this one, or the backward. Whatever the value means to the model, it is what the forward saved: a weight gradient a fused forward computes early, because a loss's own gradient needs no tangent, is a saved activation until the backward makes the gradient object from it, by copying it or by taking its storage over. |
| recomputed activations | Activations a backward-phase task made, by re-running a forward. |
| inputs | Floating-point values no executing task makes: what the step was given. |
| control | Integer and boolean values: tokens, targets, sequence lengths, indices and counters, which steer kernels rather than carry the model's arithmetic. The lowering gives every non-floating value this role. |
| task workspace | A task's scratch in the execution pool, the layout's own lease. |
| buffers, outputs | The program's roles of those names: registered buffers such as rotary tables, and the step's outputs. |

`--by role` groups by the program's roles instead: `parameter`,
`optimizer_state`, `gradient`, `activation`, `buffer`, `control`, `output`.

## The rules

**Attribution is by object, never by storage slot.** An alias group is one
storage slot, and several objects occupy it over a step: a slot that held a
saved activation during the forward can hold a tangent during the backward.
The walk names, for every byte, the object whose producing task has started
most recently, so the slot counts for whichever value is live. An object no
task produces in a slot that some task produces into is a view of that
task's output, the way a backward reads the forward's saved value under a
second object id, and is not a value of its own. Only the tasks the
selection executes count, so an output of an alternative that was not chosen
never appears.

**The spill pool follows the simulator.** A retained alias group -- checkpoint
state -- holds a spill copy for the whole step. Any other group holds one from
the moment its evict or write-back is issued until the fetch that brings it
back completes, or until it is released, and a group the schedule starts on
the host holds one from the start. That is why the walk's peak equals the
planner's.

**The execution pool follows the admitted layout.** Every lease occupies its
bytes from its predicted start to its predicted end: fetch destinations, task
outputs, initial objects, and task workspace as its own category. The header
shows three numbers. The walk's peak is the sum of the leases live at once.
The planner's reported peak is the simulator's: the same object bytes -- the
two agree to the byte -- plus each task's declared workspace, where the
layout leases the task's allocations one by one, aligned and held for their
lifetimes, so the walk's workspace is somewhat larger. What the layout
requires is that peak with every lease at its fixed offset, which the walk's
perfectly packed sum can only reach or undercut.

**A handed-off lease is the output's from the handoff.** A task's output
may take over the storage of one of its inputs -- a head backward adopts
the head forward's weight-shaped output as the weight gradient, say -- and
the layout records that as no lease of the output's own: the input's lease
runs on and is retired by the output's evict or release. The store keeps no
other record of it, so the walk infers it: an executing task's output
without a `task_output` lease, beside exactly one lease of an input of the
output's size that outlives the task, is that lease's next occupant, from
the task's start. Without this the gradient reads as a saved activation
until its first evict.

**On the device's clock, a lease sits at its events.** A traced step
records no leases, and the device's clock drifts from the simulator's over
a step, so the leases cannot be drawn at their predicted times against
lanes on the device's clock. Every predicted instant is an event's -- a
task's start or end, a fetch's issue at its trigger task's end, an evict's
or write-back's completion, a release at its trigger task's end, or the
step's end -- and the traced page places each lease at those events'
traced times. An instant no event names is interpolated between the task
boundaries around it, and the panel says how many were.

## The pages

`--html` writes one page per view, self-contained with the data inline, plus
an index carrying the tables. A page stacks, top to bottom, the summary, the
lanes, the execution pool and the spill pool, and every panel shares one
zoom, so a burst on the evict lane and the step in the spill pool it causes
are read against one time axis.

- **summary**: the step's span on this clock, tokens per second when the
  tokens are known, the two peaks, the share of the span with nothing
  computing, the plan's regeneration overhead as a share of the span (what
  its recompute alternatives cost beyond their save alternatives, by the
  program's profiles, the number the planner reports), each lane's
  utilisation, the share of the span the lane is busy, the bytes each lane
  moved over the step, and the rate that comes to, beside the bandwidth the
  plan was priced at; the page's subtitle names the assumed fetch and evict
  bandwidths and latency. A simulated page achieves what it assumed by
  construction; a traced page shows what the device did against it. The
  line under the title names the step: for a run's pages the model, the
  geometry (sequences per microbatch, microbatches, ordering), the tokens
  a step, the execution budget and the pool it resolved to, and the spill
  capacity; for one selection given by hand, the two capacities.
- **pools**: one stacked area per pool, the categories in a fixed order with
  state at the bottom, the pool's capacity as a dashed red line and its peak
  as a dotted amber one. Hovering gives the values at that instant. On the
  traced page the execution pool's leases sit at their events' device times
  (the rules above), and its header says so.
- **lanes**: three rows, `fetch`, `compute` and `evict`, every interval a
  bar. Transfer bars are coloured by the category of the object moved and
  compute bars by the task's phase, and the legend keeps the two apart; the
  backward task of a group chosen as `recompute` opens in a recompute colour
  for the length of its regeneration, the group's profiled cost beyond its
  `save` option, then continues in the backward colour. A transfer bar runs
  from the transfer's start on its lane to its finish, never from its issue,
  which can come long before. Hovering names the task or the object, its
  bytes, its times and the rate the copy achieved. A transfer the trace
  could not time -- a record whose lane reported no start or finish, or
  none at all -- is left off the lanes, its trigger task's end stands in
  where a walk needs its time, and the subtitle counts them.

Dragging across a span zooms to it, shift-dragging pans, scrolling zooms
around the cursor, `fit` resets, and the buttons zoom in steps. A lane is rasterised by pixel column: a column is
filled when any bar covers part of it and takes the colour of the category
covering most of it, so a burst of transfers far narrower than a pixel is a
visible block rather than a hairline at any zoom.

## The floors

A floor is the step with nothing to plan around: every task back to back at
its profiled time, every object resident from its production, or from the
step's start for checkpoint state, to its last use, and each task's
profiled workspace while it runs. Nothing spills, the lanes are empty, and
the execution pool's peak is what the step would need to run this way: the
contrast a budgeted plan's page is read against. Two floors differ in
whose alternatives they are taken at.

- **`--unconstrained`**, the plan's own: its alternatives as the plan fixed
  them, recomputation included, so the compute is what that plan does. A
  run's `timelines/` carries one beside every plan and every kept
  resolution.
- **`--all-save`**, the program's: every alternative at its cheapest option
  by the profiles, which for a save-or-recompute choice is `save`. This is
  the compute floor the planner reports as the unconstrained step and the
  figures draw as the ceiling. A run's `timelines/` carries one per
  geometry, under `all_save/`.

## A whole run

`--run` takes a quickstart run directory (the one holding `search.json`,
`steps/` and `plan_store/`) and writes its `timelines/`, which every
quickstart run also writes as it closes:

```text
timelines/
  index.html                       the table of contents for everything
  summary.csv                      one row per page: the summary its cards show
  <budget>/index.html              the budget: each geometry's choice and resolutions
  <budget>/traced.html             the step that ran at this budget, on the device's
                                   clock: a copy of the one in its plan's folder
  <budget>/<geometry>_<walk>/index.html     the plan the search chose there, and what it kept
  <budget>/<geometry>_<walk>/recompute_<share>/   one resolution: simulated.html,
                                            unconstrained.html (its own floor), and
                                            traced.html in the chosen one's, when it ran
  all_save/index.html              every geometry at its cheapest alternatives
  all_save/<geometry>_<walk>/all_save.html  that floor, which no budget bounds
```

Budget first, since a budget is what a run is asked for and geometry and
resolution are what the search chose under it. A plan's folder is named by
the share of the program's flexible groups it recomputes, by the planner's
own rule (`recompute_share`), as the plan store names a kept resolution:
`recompute_0` is all save, `recompute_3of4` three groups in four. The
search's choice sits under its share like the others kept beside it, and is
marked as the choice in every index; when the search kept no resolutions
the choice is the only folder. Every folder's `index.html` links its own
pages, carries their tables, and points to the folders around it: the
choice's names the resolutions kept beside it, its geometry's all-save
floor, its budget and the root; a resolution's names the choice. The root
index lists every budget that ran with its traced step, then every budget
with a row per geometry, the share chosen, its pages and the resolutions
beside it, then the all-save floors.

A stored plan names its program and its capacities but not the geometry
the search called it, so the two are matched by the simulated makespan a
search point and its stored plan share, and, where several budgets share
one plan, by the program digest, which is one geometry walked one way. A
budget that ran is matched to its plan by the makespan its traced step
records, and the traced page lands in that plan's folder, with a copy at
the budget's own level, `<budget>/traced.html`, since one geometry runs per
budget. Tokens per second come from the run's `request.json`.

`summary.csv` has one row per page, the summary the page's cards show
beside what identifies the plan, in the order `SUMMARY_COLUMNS` names:
`kind` (`chosen`, `resolution`, `all_save`), `geometry`, `budget`,
`resolution` (the share recomputed), `selected` (whether the search chose
it), `view`, `page` (the path under
`timelines/`), then `step_seconds`, `tokens_per_second`, `spill_peak_gib`,
`execution_peak_gib`, `idle_percent`, `recompute_percent`, the two lanes'
utilisation, bytes and achieved rate, the assumed rates, and
`untimed_transfers`, `interpolated_leases`, `pool_capacity_gib` and
`spill_capacity_gib`. A traced page's row holds the device's numbers, the
simulated page's the simulator's, so the two rows of one plan read
against each other; the budget-level copy of a traced page has no row of
its own.


## From Python

`tools.diagnostics.occupancy.attribute(selection, program, diagnostics=None)`
takes the three files' contents as mappings (`unconstrained(program,
selections)` takes the program and a plan's choices, `all_save(program)` the
program alone) and returns a `PlanOccupancy`
with `spill` and `execution` (`Occupancy` objects with `at`, `peak`,
`peaks_by` and `series`), the `tasks` and `transfers` as spans, the
`facts` the attribution rests on and, on the device's clock,
`interpolated_leases`, the lease instants no traced event named;
`write_pages` and `table` take those. `write_run_timelines(run_root,
out=None, progress=None)` writes a run's tree and tells `progress` when it
starts and as each geometry finishes.
