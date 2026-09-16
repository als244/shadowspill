# Figures over a step search

`shadowspill.plots` turns the artifacts a search or a run already produced into
figures. Nothing here executes a model, and nothing here plans: every value is
read from a `StepSearchReport` or from executed-run outcomes, so a figure can
always be redrawn without the GPU that produced it.

```python
from shadowspill.plots import plot_step_search, plot_step_run

plot_step_search(report, "figures")
plot_step_run(outcomes, "figures", tokens_per_step=65536)
```

`StepSearchReport` is `shadowspill.search`'s, so drawing a figure from a saved
report imports no framework: `benchmarking/replot.py` runs on the plots package
and the search package alone.

`plot_step_search()` takes a `StepSearchReport` and writes `sim/`, which needs
only a plan. `plot_step_run()` takes one `RunBudgetOutcome` per executed budget,
the prediction and the measurement side by side, and writes `real/`. Each also
writes its half of `raw_data/`. Both write into the directory they are given and
key nothing themselves, so what distinguishes one run from another is the
caller's to choose: point a second run at a second directory.

A `RunBudgetOutcome` splits the task window the same way on both clocks --
the tasks' own compute, the recomputation the plan chose to pay for, and the
stall between tasks -- so each side's three parts sum to that side's task
window and a difference can be attributed rather than only reported. What falls
outside that window is kept in the fields and written to `raw_data/` but drawn
nowhere: the opening restore, which the simulator prices at zero because it
assumes the step's initial objects are resident, and the writeback after the
last task, which the simulator prices whole and the stream exposes only as far
as the next step fails to absorb it.

## The tree

```text
figures/
  sim/
    geometry_table.png                  which geometry won each budget
    throughput/
      winners.png                       tokens per second of each winner
      winners_step_time.png             the same in seconds per step
      by_geometry.png                   every geometry, two panels
    overheads/
      winners.png                       recompute and stall, seconds
      winners_shares.png                the same as a share of the step
      by_geometry.png                   what one bar wastes, per geometry
      by_geometry_shares.png            the same, as a share of its step
      by_geometry_with_compute.png      the whole step, compute included
      by_geometry_with_compute_shares.png   the same, as shares
      by_graph_pair_selection/
        <micro>x<accum>.png             one figure per geometry, seconds
        <micro>x<accum>_shares.png      the same, as shares
    transfers/
      bytes.png                         fetched and evicted GiB per step
      lane_utilization.png              share of lane-seconds
      by_geometry.png                   lane utilization per geometry
      by_geometry_bytes.png             the same, in GiB per step
      by_graph_pair_selection/
        <micro>x<accum>.png             lane bytes per selection
        <micro>x<accum>_shares.png      lane utilization per selection
    vs_unconstrained/
      by_geometry.png                   each geometry against its floor
    orderings/
      <micro>x<accum>.png               one geometry's step time under every
                                        ordering the search tried
  real/
    throughput.png                      measured against simulated
    sim_fidelity.png                    the signed error, and which part of
                                        the step it came from
  raw_data/
    search.json                         the report itself, lossless
    points.csv                          one row per geometry, ordering, and
                                        budget
    graph_pair_selections.csv           one row per geometry, budget, and
                                        graph-pair selection
    run_budgets.csv                     one row per executed budget
    steps.csv                           one row per budget and step
```

## How the tree is organised

Three axes decide where a figure lands, and every directory is one answer.

**Plan or measurement.** `sim/` is what the planner promised and is available
from a search that executed nothing. `real/` is what the hardware delivered.
Keeping them apart means a figure never silently mixes a prediction with a
measurement; `real/sim_fidelity.png` is the one place they are deliberately
compared.

**What is being spent.** Under `sim/`, `throughput/` is the headline rate,
`overheads/` is where the step's time went, `transfers/` is what moved over the
lanes, and `vs_unconstrained/` is the distance from the compute floor.

**Waste, or the whole step.** `overheads/by_geometry.png` is the waste alone:
each bar is the recomputation a plan chose plus the stall it could not avoid,
which is the comparison between geometries at its own scale, with the makespan
written above the bar because the bar no longer carries it. The
`_with_compute` pair puts the same two segments on top of the compute floor,
so a bar is the whole step and its three parts partition it exactly; there the
number above the bar is the makespan itself.

**What varies within a figure.** `winners.png` shows only each budget's winning
geometry. `by_geometry.png` shows every geometry at every budget, each at its
best ordering there. A `by_graph_pair_selection/` directory goes one level
deeper still: one figure per geometry, in which each budget's bars are the
individual graph-pair selections the search evaluated, not only the one it
answered with. `orderings/` is the other ladder behind a geometry's line: its
step time under every microbatch ordering the search tried, labelled
`<depth>x<breadth>` with `r` for the reversed backward walk and `p` for the
paired loss, the best at each budget circled. So the levels are *the winner*,
*every geometry*, *every ordering within one geometry*, and *every selection
within one geometry*.

## Quantities

| Term | Meaning |
|---|---|
| Recompute | Compute the selection spends above the cheapest option of every group: `recomputation_overhead_seconds`. |
| Stall | Everything in the step that is not compute — waiting between tasks and the terminal writeback together. |
| Unconstrained | The compute floor: every group at its cheapest option, no waiting at all. The ceiling a budget is traded against. |
| Makespan | The simulated step time of a selection's best plan. |
| Lane utilization | A lane's busy seconds as a share of the step, per direction. |

A share figure and its seconds counterpart show the same data; the share
version answers "what fraction of the step" and the absolute one answers "how
long", and neither is derivable from the other without the step time.

## Reading conventions

These are consistent across every figure in the tree.

**Centered halves.** Fetch is drawn above the axis and evict below it, with the
axis labelled `Fetch` above and `Evict` below and the y axis showing absolute
values on both sides. This halves the width of what would otherwise be a
side-by-side pair, and puts the two directions of the same budget in one
column.

**Colour is the series, position is the budget.** Within a group, bars run left
to right in ascending microbatch size, or up the recompute ladder. One geometry
keeps its colour across every by-geometry figure, so a split followed in one is
the same split in the next; a by-selection figure colours by recompute level
instead, because within one geometry that is what varies. What the lane figures
share across both families is the direction convention: fetch solid, evict
faded.

**Red text is makespan.** Where selections are compared, the simulated step
time of each appears above its bar in red, with a `Makespan` key in the legend.
It is the number the bars are explaining.

**Recompute levels are named by share.** A selection's legend entry is the
share of groups it recomputes, to the nearest eighth — `0%`, `12.5%`, …,
`87.5%`, `100% of Groups Recomputing` — rather than a raw count, so the ladder
reads the same across geometries with different group counts.

**A detail panel.** `sim/throughput/by_geometry.png` is two panels over one x
axis: every geometry above, and below it only the band within 1.25x of the best
throughput. A slow split compresses the fast ones into a band, and those are
exactly the geometries a reader is choosing between.

**A gap is data.** A geometry-budget point that never planned is still a row in
the CSVs and still a gap in the line. Nothing is dropped for being infeasible.

**Simulator error is measured minus simulated.** Positive means the step ran
*slower* than the plan predicted, negative that it ran faster. The direction is
what carries the meaning: an optimistic prediction is time the step spends
somewhere the simulator does not model, and that is a thing to go and find.
This is the convention the performance gate reports, so a number here and a
number there mean the same thing. `real/sim_fidelity.png` draws it in two
panels: the signed error per budget against the 5% and 10% bands above, and
below it the two task windows side by side, each stacked as effective compute,
recompute and stall, with the simulated bar the faded one of the pair. So an
error can be attributed to one of those three. Recompute is a counterfactual
against the save-only floor rather than a measurement, so the same figure
stands on both bars and the comparison is left to compute and stall. Nothing
outside the task window is drawn.

## Redrawing from `raw_data/`

`raw_data/` holds everything the figures were drawn from, so they can be drawn
again in another style or another tool without replanning.

`run_budgets.csv` keeps one row per budget, whose `measured_step_seconds` is a
median; `steps.csv` keeps the steps behind it, one row each, because a median
cannot say whether a budget was steady or erratic. `plot_step_run()` writes both
as its last step, and `write_run_tables()` writes them alone, so a run that
measures one budget at a time keeps the record current and leaves what it
measured even if it stops early.

`search.json` is the report itself and is lossless: it is the same value
`plot_step_search()` was handed, and `StepSearchReport.load()` reads it back to
an equal report, so the whole `sim/` tree can be rebuilt from it alone. The CSVs
are its tidy view. There are two rather than one per figure because all but the
ladder are projections of the same per-point row, and writing that row twenty
times under different names would be twenty copies to disagree with each
other.

### `benchmarking/replot.py`

Redraws a run's figures from its `raw_data/`, optionally over a subset of it,
into a directory of your choosing. The source tree is only read.

```bash
python -m benchmarking.replot RAW_DATA OUTPUT
python -m benchmarking.replot RAW_DATA OUTPUT --budget-gib 8,16,29
python -m benchmarking.replot RAW_DATA OUTPUT --geometry 8x8,4x16 --resolution 3/4,1
```

`RAW_DATA` is a run's `raw_data/`, its `figures/` directory, or the run root --
whichever is at hand. `OUTPUT` is created and filled exactly as a run fills one,
`raw_data/` included, so a redraw can itself be redrawn and narrowed again.

| filter | keeps | narrows |
|---|---|---|
| `--budget-gib 8,16,29` | those execution budgets, named as the axis names them, so `29` selects a 29.0137 GiB point | both trees |
| `--geometry 8x8,4x16` | those geometries, as `microbatch x accumulation` | `sim/` |
| `--resolution 0,1/4,1/2,3/4,1` | those recompute shares, as the search names them | the `sim/` figures drawn per graph-pair selection |

Narrowing is subtraction only: nothing is recomputed, so every figure still
draws numbers the search actually produced. Dropping a budget's winning
geometry drops that budget rather than promoting a runner-up. A column an older
run did not write reads as zero, so its figures still draw with the parts it did
not record simply absent.

## Related

- [Quickstart](../../benchmarking/quickstart.md) runs the search that produces
  these figures and shows the tree in context.
- [Plan report](plan-report.md) and [its field
  reference](plan-report-fields.md) describe the values behind every `sim/`
  figure.
- [Step diagnostics](step-diagnostics.md) describes the measured values behind
  `real/`.
