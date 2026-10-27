# Pool occupancy over a step

`python -m shadowspill.diagnostics.occupancy` reads a plan the store holds and says
what occupies the spill pool and the execution pool at every moment of the
step, attributed to objects and to what those objects are for, as tables and
as pages. It runs on the machine that planned the step or on any other: the
store's files are all it needs. With the traced step's diagnostics it also
puts both pools and the transfer lanes on the device's clock.

## Inputs

A plan store keeps each planning result under
`<store>/v1/planning/results/<xx>/<key>/selection.json`, and the program it
was made for under `<store>/v1/planning/programs/<yy>/<digest>/program.json`,
where the digest is the selection's `program_digest`. A certified plan's
admitted layout is in `certificate.npz` beside its selection, and the tool
reads the two together; a selection with no certificate beside it is read as
it stands. The tool finds the program from the selection; `--program` names
one explicitly. These are low-level planner artifacts. For current phase archives and
quickstart results, use `write_report(plan_or_sweep, path)`.

Some entries under `results/` name a plan without carrying its evidence (no
`simulation_result`); the tool refuses those and says so.

```text
python -m shadowspill.diagnostics.occupancy <selection.json>
    [--program program.json] [--step steps/12gib.json] [--tokens-per-step N]
    [--by category|role] [--at 5.46,11.05] [--html DIRECTORY] [--json]
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
| `--html` | A directory to write the pages into: one page per view, `simulated.html` and `traced.html`, and an `index.html` that links them and carries the tables. Each page names its pool capacities under the title. |
| `--json` | The snapshots as JSON instead of tables. |

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
  bandwidths and recorded latency. Latency is metadata and is excluded from
  simulated transfer timing. A simulated page achieves what it assumed by
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
  them, recomputation included, so the compute is what that plan does.
- **`--all-save`**, the program's: every alternative at its cheapest option
  by the profiles, which for a save-or-recompute choice is `save`. This is
  the compute floor the planner reports as the unconstrained step and the
  figures draw as the ceiling.

## From Python

For animated views of these saved simulated timelines, including transfer
backlog and physical placement construction, see the source-tree
[plan video tools](../../src/tools/pressurefit_trace/plan/README.md). They reuse this
module's role attribution and interval accounting and accept the annotated
plans written by the collection workflow.

`shadowspill.diagnostics.occupancy.attribute(selection, program, diagnostics=None)`
takes the three files' contents as mappings (`unconstrained(program,
selections)` takes the program and a plan's choices, `all_save(program)` the
program alone); `shadowspill.planner.plan_store.read_stored_selection(path)`
gives a stored selection with its certificate merged in under
`admission_certificate`, which is the mapping `attribute` reads. `attribute`
returns a `PlanOccupancy` with `spill` and `execution` (`Occupancy` objects
with `at`, `peak`, `peaks_by` and `series`), the `tasks` and `transfers` as
spans, the `facts` the attribution rests on and, on the device's clock,
`interpolated_leases`, the lease instants no traced event named;
`write_pages` and `table` take those.
