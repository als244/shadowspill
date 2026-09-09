# Quickstart

`benchmarking/quickstart.py` is the one-command tour of ShadowSpill. Give
it a model, a sequence length, and how many sequences one optimizer step
must consume; it searches every way of splitting that total into
microbatches and accumulation rounds, across every execution budget you
name, then optionally renders figures over the results and runs the
winning plan — reporting how the plan breaks down, what each step
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
  --plots
```

**Search, then run.** Run budgets must appear among the search budgets;
each one gets the full treatment — the plan's breakdown, steps, and the
traced-step reconciliation — at that budget's winning geometry.

```bash
python -m benchmarking.quickstart mlops_llama3 \
  --sequence-length 1024 --sequences-per-step 64 \
  --search-budget-gib 6,7,8,9,10,12,16,20,24,28,30 \
  --run-budget-gib 6,7,8,9,10,12,16,20,24,28,30 \
  --spill-gib 112 --steps 5 \
  --min-tokens-per-microbatch 4096 \
  --plots
```

The floor is what makes that command finish in reasonable time. Splitting 64
sequences of 1024 tokens gives seven geometries, and the two narrowest --
one sequence per microbatch across 64 rounds, and two across 32 -- run far
longer than the rest for the same work, because a step's fixed per-task and
per-boundary costs are paid once per round. A floor of 4096 tokens drops
exactly those two and keeps the five that are worth comparing. Raise or
remove it when the narrow end is the thing being studied.

The low end is there on purpose. A geometry cannot plan below the largest
amount one task must hold at once: its inputs, outputs and mutations counted
once per alias group, plus its workspace. For this model at sequence length
1024 that floor is 15.8 GiB at 64 sequences per microbatch, 8.4 GiB at 32,
and about 5.1 to 5.5 GiB from 16 down, where it stops falling because a
parameter-sized activation and its gradient do not shrink with the
microbatch. Plans need roughly 20 to 30 percent above the floor to leave room
for the resident slice and the dynamic reserve, so budgets from 6 GiB up
straddle the real limit rather than sitting above it. A budget where no
geometry plans is reported and skipped, which is the answer, not a failure.

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
| `--output-dir` | Where this run writes: its search report, log, traced steps, figures, and — unless `--artifact-store` or `--plan-store` point elsewhere — its two stores | `benchmarking/quickstart_reports/<model>_<revision>/seq<length>/seqsperstep<n>` |

Each run budget also writes its traced step beside those, as
`<model>_seq<length>_seqsperstep<n>.step-<budget>gib.json`: the complete `StepDiagnostics`
for that budget's final step, which is the only step run with
`runtime_trace=True`. The figures keep nine aggregate numbers per budget, and
those answer *how far* the prediction was from the measurement; the trace is
what answers *which* transfers drifted and *which* tasks ran long. It is the
same `shadowspill.step_diagnostics` schema the performance matrix writes, so
`python -m tools.qualification.gap_report` reads a quickstart run the same way
it reads a matrix.

Everything a run writes lands in one directory, keyed by model and then by
each parameter of the run's shape, so another shape is a sibling rather than an
overwrite:

```text
benchmarking/quickstart/
  mlops_llama3/
    seq1024/
      seqsperstep64/
        search.json             the search report, lossless
        progress.log            planner phases and search progress, wall-clock
                                stamped and tailable while it runs
        steps/
          12gib.json            one traced step per run budget
          16gib.json
        figures/
          sim/  real/  raw_data/
        artifact_store/         this run's captures, graph pairs, profiles
                                and lowered programs, reusable by other runs
        plan_store/             this run's plans: every request, selection
                                and plan manifest
      seqsperstep32/            another shape, beside the first
    seq2048/
      ...
```

Each directory level is exactly one parameter, so the shapes at one sequence
length sit together, which is the comparison worth making most often.

A run owns both stores by default, so everything it measured is in one place
and nothing it reused is ambiguous. That means a fresh run pays capture,
compilation and profiling in full. To skip work already done, point
`--artifact-store` at another run's store: the store is content-addressed, so
whatever matches by structural digest is reused and the rest is built. The
plans stay this run's own, in its `plan_store`, so a shared artifact store
never answers a point with a plan another run searched; that is what makes a
planning-time comparison between two runs on one store honest.

`--output-dir` moves the whole tree somewhere else; `--artifact-store` and
`--plan-store` point the two stores at existing ones. Those are the only path
flags, because everything else a run writes has a fixed name inside the run
directory.
| `--steps` | Optimizer steps per run budget; the last is traced | 5 |
| `--seed` | Model and data seed | 0 |
| `--artifact-store` | The captures, graph pairs, profiles and lowered programs to read and write. Point it at another run's store to skip work already paid for there | `<output-dir>/artifact_store` |
| `--plan-store` | Where this run's plans go: every selection request, selection and plan manifest, kept apart from the artifact store so a shared store never hands a run another run's plans | `<output-dir>/plan_store` |
| `--orderings` | Which microbatch orderings the search tries per geometry: `factors` (the default) lowers every `depth x breadth` factor pair of the accumulation count into its own program and plans each under every budget, so the winner at a budget may be any walk of any geometry; `depth-first` tries only the walk every step used before there was a choice. The loss stays paired and the backward walk reversed either way. The run phase plans the winner's ordering | `factors` |
| `--resolution-options` | Which resolutions the search and the runs plan: the shares of flexible groups to recompute, as `quarters` (the library default), `eighths`, `halves`, or a comma-separated list of exact fractions such as `0,1/2,7/8,1`. More shares plan more programs per point: on the llama3 frontier `eighths` cost 1.75x the search wall and beat the quarter rungs by a median of 0.00 % (mean 0.77 %). The options are part of every plan's identity in the store, and the runs plan the same options the search did | `quarters` |
| `--transfer-bandwidths` | Plan the search against this calibration instead of the one the runtime measures at start: `FETCH,EVICT` in GB/s, optionally followed by the fetch and evict latencies in microseconds (`26,26,8,4`), or the path of another run's `search.json` to pin to what that run planned against, latencies included. Two runs are comparable only when they plan against the same lanes, and a fresh calibration differs run to run (22 against 26 GB/s on one machine, one hour apart). The run phase keeps the live calibration | calibrated |
| `--deterministic` / `--no-deterministic` | Make the **search** reproduce exactly at any worker count: a candidate's placement gate consults only its own placed plans rather than the shared best-placed record, so every graph-pair selection reports the plan it actually found rather than showing up only if it was measured before a better plan existed. Costs wall time, because the shared bound is what lets a candidate skip measuring a plan that cannot win. It does not reach the per-budget replan a run does before executing, which has no such option | on |
| `--incumbents` / `--no-incumbents` | Hand each budget the best plan found at a smaller budget of the same program as the plan to beat, so no program plans worse with more memory: the search plans budgets ascending, and a point that did not beat the plan it was handed answers with it and says which budget it came from (`plan from 6 GiB` in the table, `incumbent_budget_bytes` in `search.json`). The run phase is handed the search's winning plan as its plan to beat, so it executes that plan or better even when its live calibration differs from what the search planned against. `--no-incumbents` searches every point alone, for comparing the two | on |

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
   times and the transfer calibration each geometry planned against,
   statuses, and skips — is saved as `search.json` in the run directory.
   `--sequences-per-microbatch` replaces this phase with your choice.
3. **Figures**, with `--plots`, written into `figures/` in the run directory.
   Everything under `sim/` reads a plan, so it is available from a search
   with nothing executed; `real/` needs a step to have run.

   ```text
   sequence-1024/
     sim/
       geometry_table.png          winning geometry per budget
       throughput/
         winners.png               tokens per second of each budget's winner
         winners_step_time.png     the same in seconds
         by_geometry.png           every geometry, over two panels: the whole
                                   range, and the band within 1.25x of the
                                   fastest step, where the choice is made
       overheads/
         winners.png               recomputation, waiting and their total
         winners_shares.png        the same as a share of the step
         by_geometry.png           one bar per geometry per budget, ordered by
                                   ascending microbatch, recomputation below
                                   and waiting above at a lighter opacity,
                                   both labelled in seconds with the total on
                                   top; log-spaced because one geometry wastes
                                   a thousand times another
         by_geometry_shares.png    the same, linear, where a segment's drawn
                                   thickness is its value
         by_graph_pair_selection/
           <micro>x<accum>.png     one figure per geometry: at each budget,
           <micro>x<accum>_shares.png  every graph-pair selection the search
                                   evaluated, not only the one it answered
                                   with. A selection with no plan leaves a
                                   gap, which is the useful negative result
       transfers/
         bytes.png                 fetched and evicted GiB per step
         lane_utilization.png      share of lane-seconds
       vs_unconstrained/
         by_geometry.png           each geometry's unconstrained compute floor
                                   over its simulated step, with that floor in
                                   seconds in the legend
     raw_data/
       search.json                 the report itself, lossless: every figure
                                   above can be rebuilt from it exactly
       points.csv                  one row per geometry and budget, including
                                   the ones that never planned
       graph_pair_selections.csv   one row per geometry, budget and
                                   graph-pair selection
       run_budgets.csv             one row per executed budget
       steps.csv                   one row per budget and step
     real/
       throughput.png              measured against simulated, per run budget
       sim_fidelity.png            how far the prediction fell at each budget,
                                   against the bounds the performance gate
                                   holds the simulator to, and which part of
                                   the step it missed: compute, waiting, or
                                   the opening restore the simulator does not
                                   model at all
   ```

   The sequence length leads, so repeating the search at another length
   writes its own tree beside the first rather than over it. `raw_data/`
   holds what the figures were drawn from, so they can be drawn again in
   another style or another tool. `search.json` is the report itself and is
   lossless. The CSVs are its tidy view, two rather than one per figure
   because all but the ladder are projections of the same per-point row, and
   writing that row twenty times under different names would be twenty
   copies to disagree with each other. A point that never planned is still a
   row, because a gap in a line is data too.

   The winning geometry at each budget is circled in the line figures and
   outlined in the bars, and a geometry keeps one colour throughout.

4. **Per run budget**: the chosen geometry, then
   **the chosen plan's breakdown** — from
   [`PlanReport.summary`](../docs/python/plan-report.md): the simulated
   step beside the unconstrained floor, the three-way split of the
   difference, the graph-pair selection fraction, transfer traffic,
   planning capacities, and the calibrated bandwidths planning assumed —
   then **steps** (each step's cycle on the device clock, its throughput,
   its head wait and its loss, each line appearing once the next step has
   begun. The loss is the step's mean over its microbatches, and every
   budget runs from the same initial weights and a fresh optimizer on the
   same seeded tokens per step, so the losses of one budget agree with
   every other budget's bar reduction order: a run that disagrees is a
   correctness signal, not a measurement. The first step may use the
   dedicated first-step plan that
   initializes lazy optimizer state, and the measured step is the median
   of the steps after it),
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
6. **Where the host memory went.** The pinned spill arena, the peak and
   exit resident bytes, and the cgroup ceiling in force with how much of
   it went unused at the peak. The arena is one page-locked mapping and
   counts in full from the moment the runtime registers it; everything
   else the frontend holds on the host counts on top of it, and the
   largest of those is one optimizer state per plan. The progress log
   carries the same reading stamped at each boundary — pools registered,
   model imported, search finished, each budget planned and closed —
   because a batch scheduler enforces its reservation as a cgroup limit
   and the kernel answers an overrun with `SIGKILL`, leaving no Python
   traceback behind. The high-water mark is then the only evidence of
   what the run was holding.

## Terms the output uses

| Term | Meaning |
|---|---|
| geometry | One split of the step's sequence total: sequences per microbatch times accumulation rounds. |
| unconstrained | The compute floor: every graph-pair group priced at its cheapest option, with no waiting of any kind. Real plans exceed it on purpose — see [graph-pair selection](../docs/architecture/graph-pair-selection.md). |
| extra recomputation | Compute the selection added over that floor by choosing to recompute rather than hold memory. |
| stalled | Time the simulated step spends with tasks waiting on data or capacity rather than computing. |
| wasted compute | The sum of the two rows above: everything the step spends beyond the floor, before the terminal writeback. |
| terminal writeback | Transfers that return spill-final objects to the spill pool after the last task; the simulated step includes them. |
| task window | From the first task's compute start through the last task's end. It excludes the step's boundary regions by construction. |
| opening restore | The unmodeled fetch of the schedule's initial device objects at each invocation's start. |
| lane utilization | Simulated transfer bytes over the assumed lane bandwidth over the simulated step: the share of the step each transfer lane spends busy. |
| infeasible / search_exhausted | A geometry the planner proved cannot fit the budget, or whose bounded candidate search ended without a feasible schedule. A geometry whose build exhausts the device reports every one of its budgets infeasible too, since profiling runs real kernels and the largest microbatch can run out of memory before any plan exists. Reported in the table, never raised. |
| rejected | A point the planner refused, before or during its search; `error` carries its reason. The sweep goes on with the next point. |
| artifact store | The on-disk cache of compilation, profiling, and plan artifacts, keyed by content digests — see [reusable planning](../docs/examples/reusable-planning.md). |

The traced-step deltas are real minus simulated: positive start deltas
mean the real timeline ran behind the prediction, and positive duration
deltas mean the work took longer than profiled.
