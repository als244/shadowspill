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

`plot_step_search()` writes everything under `sim/`, which needs only a plan.
`plot_step_run()` writes `real/`, which needs a step to have executed. Both
write into the directory they are given and key nothing themselves, so what
distinguishes one run from another is the caller's to choose: point a second
run at a second directory.

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
      by_geometry.png                   one bar per geometry per budget
      by_geometry_shares.png            the same, as shares
      by_graph_pair_selection/
        <micro>x<accum>.png             one figure per geometry, seconds
        <micro>x<accum>_shares.png      the same, as shares
    transfers/
      bytes.png                         fetched and evicted GiB per step
      lane_utilization.png              share of lane-seconds
      by_geometry.png                   lane utilization per geometry
      by_geometry_bytes.png             the same in raw bytes
      by_graph_pair_selection/
        <micro>x<accum>.png             lane bytes per selection
        <micro>x<accum>_shares.png      lane utilization per selection
    vs_unconstrained/
      by_geometry.png                   each geometry against its floor
  real/
    throughput.png                      measured against simulated
    sim_fidelity.png                    where the prediction fell short
  raw_data/
    search.json                         the report itself, lossless
    points.csv                          one row per geometry and budget
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

**What varies within a figure.** `winners.png` shows only each budget's winning
geometry. `by_geometry.png` shows every geometry at every budget. A
`by_graph_pair_selection/` directory goes one level deeper still: one figure per
geometry, in which each budget's bars are the individual graph-pair selections
the search evaluated, not only the one it answered with. So the three levels
are *the winner*, *every geometry*, and *every selection within one geometry*.

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
to right in ascending microbatch size, or up the recompute ladder. The colour
scheme is shared between the by-geometry and by-selection lane figures, so a
colour means the same thing across both.

**Red text is makespan.** Where selections are compared, the simulated step
time of each appears above its bar in red, with a `Makespan` key in the legend.
It is the number the bars are explaining.

**Recompute levels are named by share.** A selection's legend entry is the
share of groups it recomputes, to the nearest eighth — `0%`, `12.5%`, …,
`87.5%`, `100% of Groups Recomputing` — rather than a raw count, so the ladder
reads the same across geometries with different group counts.

**Detail insets.** The throughput figure carries an inset covering the points
within 1.25x of the best, because the interesting budgets crowd together at the
top and the full range hides them.

**A gap is data.** A geometry-budget point that never planned is still a row in
the CSVs and still a gap in the line. Nothing is dropped for being infeasible.

## Redrawing from `raw_data/`

`raw_data/` holds everything the figures were drawn from, so they can be drawn
again in another style or another tool without replanning.

`run_budgets.csv` keeps one row per budget, whose `measured_step_seconds` is a
median; `steps.csv` keeps the steps behind it, one row each, because a median
cannot say whether a budget was steady or erratic.

`search.json` is the report itself and is lossless: it is the same value
`plot_step_search()` was handed, so the whole `sim/` tree can be rebuilt from it
alone. The CSVs are its tidy view. There are three rather than one per figure
because all but the ladder are projections of the same per-point row, and
writing that row twenty times under different names would be twenty copies to
disagree with each other.

## Related

- [Quickstart](../../benchmarking/quickstart.md) runs the search that produces
  these figures and shows the tree in context.
- [Plan report](plan-report.md) and [its field
  reference](plan-report-fields.md) describe the values behind every `sim/`
  figure.
- [Step diagnostics](step-diagnostics.md) describes the measured values behind
  `real/`.
