# Quickstart

`benchmarking/quickstart.py` is the one-command tour of ShadowSpill. Give
it a model, a sequence length, and how many sequences one optimizer step
must consume; it searches every way of splitting that total into
microbatches and accumulation rounds, across every execution budget you
name, then optionally renders figures over the results and runs the
winning plan — reporting what the plan promised, what each step
delivered, and how a traced step reconciles with the simulator.

## Three ways to use it

**The default demo.** No budget flags: the model's retained qualification
budget is searched and run.

```bash
python -m benchmarking.quickstart mlops_olmoe
```

**Planning only.** Search budgets but no run budgets: every geometry
plans under every budget, figures render if asked, and nothing executes.
This mode needs the device only for profiling fresh geometries; warm
artifact stores keep it cheap.

```bash
python -m benchmarking.quickstart mlops_olmoe \
  --sequences-per-step 64 --search-budget-gib 8,10,12,14,16 \
  --plots --plot-dir benchmarking/quickstart_plots/mlops_olmoe
```

**Search, then run.** Run budgets must appear among the search budgets;
each one gets the full treatment — plan promise, steps, and the
traced-step reconciliation — at that budget's winning geometry.

```bash
python -m benchmarking.quickstart mlops_llama3 \
  --sequence-length 1024 --sequences-per-step 64 \
  --search-budget-gib 10,12,16,20,24,28 \
  --run-budget-gib 10,12,16,20,24,28 \
  --spill-gib 112 --steps 5 \
  --plots --plot-dir benchmarking/quickstart_plots/mlops_llama3
```

## Command line

Geometry:

| Argument | Meaning | Default |
|---|---|---|
| `model` | One of `mlops_llama3`, `mlops_qwen35`, `mlops_olmoe`, `pytorch_llama3`, `pytorch_qwen35` | required |
| `--sequence-length` | Tokens per sequence | the model's retained value |
| `--sequences-per-step` | Sequences one optimizer step consumes; the search splits this into microbatches times accumulation | retained value |
| `--sequences-per-microbatch` | Choose the geometry yourself instead of searching; must divide the sequences per step, and requires a run budget | search decides |
| `--min-tokens-per-microbatch` | Skip splits whose microbatch would hold fewer tokens | none |
| `--max-tokens-per-microbatch` | Skip splits whose microbatch would hold more tokens | none |

Budgets:

| Argument | Meaning | Default |
|---|---|---|
| `--search-budget-gib` | Comma-separated execution budgets to search and plot across, for example `10,12,16` | the run budgets, or the retained value |
| `--run-budget-gib` | Comma-separated execution budgets to actually run; every one must appear among the search budgets | retained value when no budget flag is given; otherwise none |
| `--spill-gib` | Pinned-host spill budget, shared by every point | retained value |

Output and caching:

| Argument | Meaning | Default |
|---|---|---|
| `--plots` | Render the figures below | off |
| `--report-path` | Where the search report JSON is saved — every point's plan summary, build phase times, and statuses, for post-hoc analysis | `benchmarking/quickstart_reports/<model>.json` |
| `--search-log` | Where the plan-style progress log goes: planner phase lines and search progress, wall-clock stamped | `benchmarking/quickstart_reports/<model>.search.log` |
| `--plot-dir` | Where figures are written | `benchmarking/quickstart_plots/<model>` |
| `--steps` | Optimizer steps per run budget; the last is traced | 5 |
| `--seed` | Model and data seed | 0 |
| `--artifact-store` | Compile/profile/plan cache; reuse skips most planning cost | `benchmarking/quickstart_store/<model>` |
| `--force-fresh` | Ignore the artifact store and replan from scratch | off |

## What the output shows, in order

1. **Configuration.** The effective geometry, the search and run budget
   lists, and the spill budget.
2. **Geometry search** — `plan_step_search` from the
   [frontend API](../docs/python/api/frontend.md). Every admitted split
   plans through capture, profiling, lowering, and the PressureFit search
   ([planning orchestration](../docs/architecture/planning.md),
   [PressureFit](../docs/architecture/pressurefit.md)); each distinct
   microbatch shape compiles and profiles once, deduplicated by the
   artifact store. The table lists every split under every budget with
   its simulated step and marks each budget's winner; skipped splits show
   their reasons, and build/search wall totals close the section. The
   full report — every point's `PlanSummary`, per-geometry build phase
   times, statuses, and skips — is saved as JSON to `--report-path`.
   `--sequences-per-microbatch` replaces this phase with your choice.
3. **Figures**, with `--plots`: the chosen-geometry table and six line
   charts over the execution budgets — throughput, simulated step time,
   recomputation/waiting/wasted-compute overheads (raw seconds and as
   shares of the step), transfer traffic, and simulated lane utilization.
   After the runs, a seventh chart overlays measured throughput on the
   simulated line for the budgets that executed.
4. **Per run budget**: the chosen geometry, then
   **the chosen plan's promise** — from
   [`PlanReport.summary`](../docs/python/plan-report.md): the simulated
   step beside the unconstrained floor, the three-way split of the
   difference, the recomputation selection fraction, transfer traffic,
   planning capacities, and the calibrated bandwidths planning assumed —
   then **steps** (wall time, throughput, loss; the first step may use
   the dedicated first-step plan that initializes lazy optimizer state),
   then **the traced step versus simulation**, using the fields defined
   in the [StepResult diagnostics guide](../docs/python/step-diagnostics.md).
   The boundary behavior it reports — the opening restore and the
   terminal writeback — is defined in
   [step boundaries](../docs/architecture/step-boundaries.md).
5. **Where the time went.** The command's own wall time by category —
   runtime calibration, model construction and import, the geometry
   builds split by frontend phase, the PressureFit searches, per-budget
   run planning, step execution, figures, and the unattributed rest — so
   the cost of what you just watched is never a mystery.

## Terms the output uses

| Term | Meaning |
|---|---|
| geometry | One split of the step's sequence total: sequences per microbatch times accumulation rounds. |
| unconstrained | The compute floor: every recomputation group priced at its cheapest option, with no waiting of any kind. Real plans exceed it on purpose — see [recomputation selection](../docs/architecture/recomputation-selection.md). |
| extra recomputation | Compute the selection added over that floor by choosing to recompute rather than hold memory. |
| waiting between tasks | Time the simulated step spends with tasks waiting on data or capacity rather than computing. |
| wasted compute | The sum of the two rows above: everything the step spends beyond the floor, before the terminal writeback. |
| terminal writeback | Transfers that return spill-final objects to the spill pool after the last task; the simulated step includes them. |
| task window | From the first task's compute start through the last task's end. It excludes the step's boundary regions by construction. |
| opening restore | The unmodeled fetch of the schedule's initial device objects at each invocation's start. |
| lane utilization | Simulated transfer bytes over the assumed lane bandwidth over the simulated step: the share of the step each transfer lane spends busy. |
| infeasible / search_exhausted | A geometry the planner proved cannot fit the budget, or whose bounded candidate search ended without a feasible schedule. Reported in the table, never raised. |
| artifact store | The on-disk cache of compilation, profiling, and plan artifacts, keyed by content digests — see [reusable planning](../docs/examples/reusable-planning.md). |

The traced-step deltas are real minus simulated: positive start deltas
mean the real timeline ran behind the prediction, and positive duration
deltas mean the work took longer than profiled.
