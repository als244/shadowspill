# Quickstart

`benchmarking.quickstart` searches named microbatch candidates, runs the winner
at each budget, saves artifacts and reports traced execution against simulation.
The runner accepts ordinary models and objectives. Text presets are optional
factories under `workloads/recipes/text`; the runner does not inspect text shapes
or model families.

To reuse a result in a presentation, see [Timeline plots for slides](quickstart_timeline.md).
`python -m benchmarking.quickstart_timeline` exports aligned memory,
compute and transfer plots from saved simulated or traced HTML pages, including
an optional PowerPoint slide with the report's summary statistics.
`python -m benchmarking.quickstart_tradeoff` exports the budget/throughput and
overhead-share comparison; both exporters' options are in the same guide.

## Any model or objective

Supply a Python factory receiving the selected local `device` and returning a
plain mapping. It is called after runtime installation, so it can safely create
model resources then. No base class or registration is required.

```bash
python -u -m benchmarking.quickstart \
  --factory workloads.recipes.regression:experiment \
  --factory-args '{"rows": 64, "width": 16, "outputs": 8}' \
  --search-budget-gib 2,3 --run-budget-gib 2,3 --spill-gib 2 \
  --steps 5 --plots --resolution-plans \
  --output-dir benchmarking/quickstart_reports/regression
```

The [regression factory](../workloads/recipes/regression.py) is a complete
non-text example. Its mapping contains:

| Key | Meaning |
| --- | --- |
| `model_factory` or `model` | Construct fresh model state for each budget, or copy a supplied initialized CPU model |
| `objective` | Ordinary `objective(model, *microbatch_inputs)` function |
| `optimizer` | Constructor receiving the model parameters |
| `candidates` | Nonempty mapping of candidate names to positional microbatch-input sequences |
| `initialize` | Optional explicit model initializer |
| `hyperparams` | Runtime hyperparameter values, such as learning rate |
| `plan_options` | Additional shared planning settings, such as partition, masters or gradient dtype |
| `units_per_step`, `unit_label` | Optional throughput units; default `1`, `updates` |
| `metadata` | Optional report metadata |
| `metric_reducer` | Optional host reducer of completed-step observations |
| `context` | Optional context factory enclosing search and execution |
| `cleanup_model` | Optional `cleanup_model(model)` called after imported state is released, including budget rebuilds; releases caller-owned communication resources |

Every candidate must represent the same normalized update. The objective owns
normalization, including any unit count; the runner only sums its returned
contributions. A model factory must recreate the same initial values at each
budget, for example with a local seeded RNG context. The provided factory does
this. Prepared candidate data should stay on CPU until the runner uses it.

`--search-workers` limits planner CPU threads **per process** (0 means automatic).
For a distributed sweep, choose a per-rank limit that fits the node's CPU count.
`--external-headroom-gib` reserves device memory outside the execution slab;
its default remains 0.5 GiB. Workloads with communication allocations can request
more. This reserve comes out of each requested physical execution budget, and
the console reports the resulting slab budgets.

The Python `run(...)` API accepts `search_workers`, `external_headroom_gib`, and
`plan_store` as well. A stable `artifact_store` and `plan_store` let later
invocations reuse completed builds/searches while keeping separate output
directories for their logs and measurements.

Python callers may use the same workflow:

```python
from benchmarking.quickstart import run
from workloads.recipes.regression import experiment

run(
    experiment, search_budget_gib=[2, 3], run_budget_gib=[2, 3],
    spill_gib=2, steps=5, output_dir="benchmarking/quickstart_reports/regression",
    plots=True, resolution_plans=True,
)
```

`--device` selects a local device. Automatic selection uses the only visible
device or `LOCAL_RANK` when several are visible. Device selection alone does
not enable distributed collectives; coordinated distributed preparation is a
separate phase of the implementation.

## Three ways to use the supplied text presets

**The default demo.** No budget flags: the model's retained qualification
budget is searched and run.

```bash
python -m benchmarking.quickstart mlops_olmoe
```

**Planning only.** Search budgets but no run budgets: every geometry
plans under every budget, figures render if asked, and nothing executes.
This mode needs the device only for profiling fresh geometries; a warm
build store keeps it cheap.

```bash
python -m benchmarking.quickstart mlops_olmoe \
  --sequences-per-step 64 --search-budget-gib 8,10,12,14,16 \
  --plots --resolution-plans
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
  --plots --resolution-plans
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
once per alias group, plus its workspace. That floor falls as the microbatch
narrows, then stops falling, because a parameter-sized activation and its
gradient do not shrink with the microbatch. A plan needs headroom above the
floor for the resident slice and the dynamic reserve, so the budgets worth
naming straddle the limit rather than sitting safely above it. A budget where
no geometry plans is reported and skipped, which is the answer, not a failure.

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
| `--spill-gib` | Spill budget, shared by every point. Pinned host memory unless `--remote-spill` names a peer, and the same size either way | retained value |
| `--remote-spill` | Spill to a memory daemon on another machine, `HOST:PORT`, instead of to pinned host memory. Everything else about the tour is unchanged -- `--spill-gib` still sets the size -- so the only thing that differs is where the pool lives, which is what makes a remote tour comparable with a local one. **The arena is then the peer's memory and is not counted in the host figures the closing report prints**, where a pinned one is; the report says which it is | pinned host |
| `--steps` | Optimizer steps per run budget; the last is traced | 5 |
| `--seed` | Model and data seed | 0 |

What the search plans:

| Argument | Meaning | Default |
|---|---|---|
| `--orderings` | Which microbatch orderings the search tries per geometry: `factors` lowers every `depth x breadth` factor pair of the accumulation count into its own program and plans each under every budget, so the winner at a budget may be any walk of any geometry; `depth-first` tries only the plain walk, one microbatch start to finish before the next. The loss stays paired and the backward walk reversed either way. The run phase plans the winner's ordering | `factors` |
| `--resolution-options` | The shares of flexible groups to recompute, as `quarters`, `eighths`, `halves`, or a comma-separated list of exact fractions such as `0,1/2,7/8,1`. More shares plan more programs per point, so the search wall grows with the count and the finer rungs may or may not be worth it for a given model. The options are part of every plan's identity in the store, and the runs plan the same options the search did | `quarters` |
| `--transfer-bandwidths` | Plan the search against this calibration instead of the one the runtime measures at start: `FETCH_SOLO,FETCH_CONCURRENT,EVICT_SOLO,EVICT_CONCURRENT` in GB/s, optionally followed by fetch/evict latency in microseconds (`40,26,56,26,8,4`); a two-number form gives fixed fetch/evict rates, or the path of another run's `search.json` to pin to what that run planned against, latencies included. Two runs are comparable only when they plan against the same lanes, and a fresh calibration differs run to run on one machine. The run phase plans against the same lanes as the search either way, pinned or calibrated once at start, so each budget asks the store the search's question | calibrated |
| `--deterministic` / `--no-deterministic` | Make the **search** reproduce exactly at any worker count: a candidate's placement gate consults only its own placed plans rather than the shared best-placed record, so every graph-pair selection reports the plan it actually found rather than showing up only if it was measured before a better plan existed. Costs wall time, because the shared bound is what lets a candidate skip measuring a plan that cannot win. It does not reach the per-budget replan a run does before executing, which has no such option | on |
| `--incumbents` / `--no-incumbents` | Hand each budget the best plan found at a smaller budget of the same program as the plan to beat, so no program plans worse with more memory: the search plans budgets ascending, and a point that did not beat the plan it was handed answers with it and says which budget it came from (`plan from 6 GiB` in the table, `incumbent_budget_bytes` in `search.json`). The run phase is handed the search's winning plan as its plan to beat, so it executes that plan or better even when its facts differ from the search's. `--no-incumbents` searches every point alone, for comparing the two | on |

Precision, named as the [training harness](../training/README.md#run-a-supplied-text-recipe)
names it, so a tour and a training run at one configuration are the same
arithmetic:

| Argument | Meaning | Default |
|---|---|---|
| `--model-dtype` | Model weights and activations: `bfloat16`, `float16`, or `float32`. This default stays BF16 on every GPU; select FP16 explicitly on older GPUs | `bfloat16` |
| `--master-dtype` | A dtype -- `float32` -- to keep a master copy of every weight trained at another dtype at; the optimizer steps the masters in the weights' place and each step writes the weights from them. `none` steps the weights themselves | `none` |
| `--grad-dtype` | The dtype gradients are created and summed at over a step's microbatches. Naming one also asks the mlops kernels for weight gradients at it and has the optimizer read gradients at it, the two settings the harness lists beside it, because a step that names one and sets neither silently rounds the sum back to bf16 | the weights' dtype |
| `--opt-state-dtype` | The dtype the optimizer keeps its state at, AdamW's moments: `bfloat16`, `float16`, `float32`, or `parameter` for the dtype of what it steps | the optimizer's own default |
| `--parameter-rounding` | How the optimizer rounds the weights it steps: `nearest`, or `stochastic`, which keeps small updates in expectation | the optimizer's own default, nearest |
| `--opt-state-rounding` | How it rounds the state it stores: `nearest` or `stochastic` | ShadowSpill: stochastic for BF16 AdamW moments, nearest otherwise |
| `--round-accumulation-once` | `plan_step`'s: a matrix multiply adds its product into running gradients kept narrower than it sums at -- bf16 -- as it writes them, avoiding a separate product buffer and addition. BLAS controls rounding; a single rounding is not guaranteed. Off, the narrow-dtype addition stays separate | off |

Every one of these is part of the plan's identity in the store and of the
request a run records, so `--reproduce` replays them, and the banner names
them even when every one is the default: two runs at different precisions are
not the same arithmetic. Two microbatch geometries of one step train on the
same batch (the step's tokens and targets are drawn once and split), and their
losses agree to about a thousandth per step whichever dtype the gradients are
summed at: what separates them is bf16 rounding inside the kernels, whose
tiling differs with the rows a microbatch holds, and a gradient at
initialization is a cancelling sum, so a rounding difference of one bf16 ulp
in its terms is a difference of the same relative size in the gradient.

### GPUs without BF16 support

Quickstart keeps its BF16 defaults on every machine. Choose FP16 weights and
FP32 optimizer moments explicitly on a GPU below SM80, such as an RTX 2080 Ti:

```bash
python -u -m benchmarking.quickstart mlops_llama3 \
  --model-dtype float16 --opt-state-dtype float32 \
  --master-dtype none --grad-dtype float16 \
  --sequence-length 1024 --sequences-per-step 8 \
  --search-budget-gib 8 --run-budget-gib 8 --spill-gib 112 \
  --min-tokens-per-microbatch 1024 --steps 5 \
  --plots --resolution-plans
```

The four precision settings are independent. Omit `--grad-dtype` to accumulate
at the model weights' dtype, or use `--grad-dtype float32` for FP32 accumulation.
Use `--master-dtype float32` to train through FP32 master weights; those extra
weights require more host storage (for this full Llama 8B example, use
`--spill-gib 160`). FP16 weights and FP32 optimizer moments do not require
master weights when using the mlops optimizer.

The selected model dtype reaches model construction before storage is
allocated. It is printed in the banner and saved in `request.json`, so
`--reproduce` preserves it. The existing manifest and optimizer dtype defaults
remain BF16 when no precision flags are given.

Task profiling:

All profiling defaults can be changed. These options apply to both search and
execution builds and are recorded in `request.json`; each stored task profile
also records its effective policy and observed conditioning/measurement windows.

| Argument | Meaning | Default |
|---|---|---|
| `--profile-warmup-iterations` | Exact-task initialization warmups | `3` |
| `--profile-stabilization-iterations` | Additional allocation stabilization limit | `16` |
| `--profile-conditioning-seconds` | Device time to condition each task before timing | `1.0` |
| `--profile-conditioning-wall-seconds` | Conditioning wall-time cap | `3.0` |
| `--profile-minimum-samples` | Minimum timed invocations | `15` |
| `--profile-measurement-seconds` | Minimum device time accumulated by timing samples | `0.3` |
| `--profile-measurement-wall-seconds` | Wall-time cap after the minimum sample count | `2.0` |
| `--profile-relative-mad-threshold` | Allowed relative median absolute deviation | `0.03` |
| `--profile-half-drift-threshold` | Allowed relative half-window median drift | `0.03` |

For example, append `--profile-conditioning-seconds 0.5
--profile-minimum-samples 20` to a quickstart command. Zero duration targets
disable their floors. A wall cap cannot interrupt a task or skip the minimum
sample count. Profiles that miss a duration target or stability threshold are
marked unstable. See [ProfilingOptions](../docs/python/api/frontend.md#profilingoptions).

Output and stores:

| Argument | Meaning | Default |
|---|---|---|
| `--plots` | Render the figures below | off |
| `--timelines` | Write every plan's pages under `timelines/` as the run closes (see below); `--no-timelines` skips them | on |
| `--resolution-plans` | Keep every resolution's best plan in the plan store beside the answer, certified, so the timelines carry a page per resolution; several times the plan store. Every resolution reports its plan under `--deterministic` (the default); without it one bounded away before it placed has none | off |
| `--output-dir` | Where this run writes: its console and progress logs, search report, traced steps, figures, timelines, and — unless a store flag points elsewhere — its two stores | `benchmarking/quickstart_reports/<model>_<revision>_<MMDD_HHMM>/seq<length>/seqsperstep<n>` |
| `--force-overwrite` | Replace an existing run at that directory. Its stores are kept, being content-addressed | off |
| `--reproduce RUN` | Repeat the run at `RUN` (its `seq<length>/seqsperstep<n>` directory) exactly: every setting comes from its `request.json`, the search is pinned to the calibration its `search.json` records, and plan-store mode is `require`, so a plan the store lacks refuses instead of being searched again. Only `--output-dir`, `--plots`, `--timelines` and `--resolution-plans` may be given with it | none |
| `--export-bypass-key` | The caller's name for the code this run builds from. With it, a build reads each ordering's step program back from the build store and captures only what is not there; without it every build captures | none |
| `--artifact-store` | Roots both store trees | `<output-dir>/artifact_store` |
| `--build-store` | The captures, graph pairs, profiles and compiled artifacts to read and write; overrides `--artifact-store` for the build tree. Point it at another run's store to skip work already paid for there | the artifact store |
| `--plan-store` | Where this run's plans go: every selection request, result and plan manifest; overrides `--artifact-store` for the planning tree | `<output-dir>/plan_store` |
| `--build-store-mode`, `--plan-store-mode` | What this run does with each tree; the four modes are defined in [the artifact store guide](../docs/python/artifact-store.md#store-modes) | `contribute` |

A run owns both trees by default, so everything it measured is in one place
and nothing it reused is ambiguous. Since every run gets its own directory,
that also means every run pays capture, compilation and profiling in full
unless told otherwise. To skip work already done, point `--build-store` at
another run's build tree: it is content-addressed, so whatever matches by
structural digest is reused and the rest is built. Pair it with
`--build-store-mode reuse` to read that tree without writing into it, which
is how a later run borrows an earlier one's builds while leaving the earlier
run's directory exactly as it was measured. The
plans stay this run's own, so a shared build store never answers a point with
a plan another run searched; that is what makes a planning-time comparison
between two runs on one store honest.

## What a run writes

Everything lands in one directory, keyed by what the run measured — the model,
the revision, and the minute it started — and then by each parameter of the
run's shape, so another run and another shape are both siblings rather than
overwrites:

```text
benchmarking/quickstart_reports/
  mlops_llama3_<revision>_<MMDD_HHMM>/
    seq1024/
      seqsperstep64/
        request.json            the request in full, as --reproduce reads it
        search.json             the search report, lossless
        console.log             everything the run printed, as it was
                                printed: the geometry table, the chosen
                                plans and their breakdowns, and the
                                per-step numbers
        progress.log            planner phases and search progress, wall-clock
                                stamped and tailable while it runs
        steps/
          12gib.json            one traced step per run budget
          16gib.json
        figures/
          sim/  real/  raw_data/
        timelines/              every plan's pools and lanes over the step,
          index.html  summary.csv   as pages, budget by budget; see below
          <budget>/  all_save/
        artifact_store/         this run's captures, graph pairs, profiles
                                and lowered programs, reusable by other runs
        plan_store/             this run's plans: every request, selection
                                and plan manifest
      seqsperstep32/            another shape, beside the first
    seq2048/
      ...
```

The top level names what was measured. `<revision>` is the short commit the
run built from, with `_dirty` appended when the tree carried uncommitted
changes, because such a run cannot be reproduced from the hash alone;
`nogit` stands in outside a checkout. `<MMDD_HHMM>` is the minute the run
started, which is what keeps two runs of one revision apart — the same commit
is worth measuring more than once, on a quiet machine or against another
run's store. Below that, each directory level is exactly one parameter, so the
shapes at one sequence length sit together, which is the comparison worth
making most often.

A directory that already holds a run is refused rather than replaced, because
a measurement costs real time and overwriting one loses it. The default path
carries the start minute, so this catches an explicit `--output-dir` and two
runs that begin in the same minute; `--force-overwrite` or `--output-dir` is
the way past it.

Each traced step under `steps/` is the complete `StepDiagnostics` for that
budget's final step, which is the only step run with `runtime_trace=True`.
The figures keep a handful of aggregate numbers per budget, and those answer
*how far* the prediction was from the measurement; the trace is what answers
*which* transfers drifted and *which* tasks ran long. It is the same
`shadowspill.step_diagnostics` schema the performance matrix writes, so
`python -m qualification.gap_report` reads a quickstart run the same way
it reads a matrix.

Missing transfer timestamps do not fail a completed run. The console and
fidelity figure label the traced invocation as **unknown**; throughput and
step timings remain available, and subsequent budgets still run. Missing
measurements are `null` in step JSON and empty cells in the run CSV, including
the traced invocation and its simulator error. Replotting preserves these gaps
instead of substituting zero or treating an incomplete trace as a full one.

`timelines/` is written as the run closes, unless `--no-timelines`, by
[the occupancy tool](../docs/python/occupancy.md); a failure to write it is
reported and does not fail the run, whose data is complete by then. Budget
first: for every plan the search made, one page on the simulated clock
under `<budget>/<geometry>_<walk>/recompute_<share>/` -- the folder named
by the share of the flexible groups the plan recomputes -- with the step's
summary, what occupies the spill pool and the execution pool at each
moment by what the objects are for, and the fetch, compute and evict
lanes, on one zoom, and beside it the plan's own unconstrained page, its
floor at the alternatives it fixed; for every budget that ran, the traced
page on the device's clock in that folder and a copy at `<budget>/traced.html`;
for every geometry, its all-save page under `all_save/<geometry>_<walk>/`,
the compute floor with every alternative at its cheapest, every object
resident and nothing spilled, to read the budgeted pages against; with
`--resolution-plans`, for every other resolution the search kept, its
simulated page and its own unconstrained page under its share beside the
choice, which every index marks; and a table of contents at every level --
the root `index.html` for everything, one per budget, per geometry within
it and per plan -- with `summary.csv` beside the root, one row per page
carrying the summary its cards show.
`python -m shadowspill.diagnostics.occupancy --run <run directory>` writes the same
for a run made before the pages existed.

## What the output shows, in order

1. **Configuration.** The effective geometry, the search and run budget
   lists, the spill budget, the orderings and resolutions the search will
   try, the precision the step trains under -- master dtype, gradient
   dtype, optimizer state dtype and roundings -- and the transfer lanes both
   ways round: the rate and latency the
   simulator will be built with, beside the effective, concurrent and solo
   rates the runtime measured. The planned figure is the effective one
   [coarsened by magnitude](../docs/python/plan-report.md), so it is
   deliberately not the measurement, and a plan is priced against it.
2. **Geometry search** — `plan_step_search` from the
   [frontend API](../docs/python/api/frontend.md). Every admitted split
   plans through capture, profiling, lowering, and the search
   ([the planning pipeline](../docs/architecture/planning-pipeline.md),
   [PressureFit](../docs/architecture/pressurefit.md)); each distinct
   microbatch shape compiles and profiles once, deduplicated by the
   build store. The table lists every split under every budget with
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
   figures/
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
         by_geometry_bytes.png     what each geometry moves, not only the
         by_geometry.png           winner: bytes, and as a share of the step
         by_graph_pair_selection/
           <micro>x<accum>.png     one geometry's lane cost by selection:
           <micro>x<accum>_shares.png  recomputing less means keeping more,
                                   and keeping more is traffic
       orderings/
         <micro>x<accum>.png       the ladder behind one geometry's line:
                                   its step time under every ordering the
                                   search tried, so the walk's own worth at
                                   each budget is visible
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
       sim_fidelity.png            how far the measurement fell from the
                                   prediction at each budget -- positive means
                                   the step ran slower than predicted --
                                   against the bounds the performance gate
                                   holds the simulator to, and which part of
                                   the step it missed: entry delay, compute,
                                   waiting, or terminal transfers
   ```

   `raw_data/` holds what the figures were drawn from, so they can be drawn
   again in another style or another tool. `search.json` is the report itself
   and is lossless. The CSVs are its tidy view, two rather than one per
   figure because all but the ladder are projections of the same per-point
   row, and writing that row twenty times under different names would be
   twenty copies to disagree with each other. A point that never planned is
   still a row, because a gap in a line is data too.

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
   begun. Each microbatch's objective is its share of the step's mean loss
   over trained tokens, using the requested sequences per step as the
   denominator. The displayed loss sums the head-loss metrics when the
   objective reports them separately; otherwise it sums the objectives.
   The MoE balancing term remains in the differentiated objective. Every
   budget runs from the same initial weights and a fresh optimizer on the
   same seeded tokens per step, so the losses of one budget agree with
   every other budget's bar reduction order: a run that disagrees is a
   correctness signal, not a measurement. Optimizer state is allocated and
   filled before the first step rather than created by it, so every step
   runs the same plan, and the reported step is the median of every step
   after the first),
   then **the traced step versus simulation**, using the fields defined
   in the [StepResult diagnostics guide](../docs/python/step-diagnostics.md).
   The boundary behavior it reports — scheduled entry fetches and the
   terminal writeback — is defined in
   [step boundaries](../docs/architecture/step-boundaries.md).
5. **Where the time went.** The command's own wall time by category —
   runtime calibration, model construction and import, the geometry
   builds split by frontend phase, the searches, per-budget
   run planning, step execution, figures, and the unattributed rest — so
   the cost of what you just watched is never a mystery.
6. **Where the host memory went.** The spill arena, the peak and exit
   resident bytes, and the cgroup ceiling in force with how much of it
   went unused at the peak. A **pinned** arena is one page-locked mapping
   and counts in full from the moment the runtime registers it;
   everything else the frontend holds on the host counts on top of it,
   and the largest of those is one optimizer state per plan. A
   **remote** arena counts for nothing here, because it is the peer's
   memory — so the resident figures of a local tour and a remote one are
   not the same measurement, and the line names which it is rather than
   leaving them to be read side by side. The progress log
   carries the same reading stamped at each boundary — pools registered,
   model imported, search finished, each budget planned and closed —
   because a batch scheduler enforces its reservation as a cgroup limit
   and the kernel answers an overrun with `SIGKILL`, leaving no Python
   traceback behind. The high-water mark is then the only evidence of
   what the run was holding.

## Terms the output uses

| Term | Meaning |
|---|---|
| candidate / geometry | A named representative update and its microbatch count. Text recipes name candidates by sequences per microbatch. |
| unconstrained | The compute floor: every graph-pair group priced at its cheapest option, with no waiting of any kind. Real plans exceed it on purpose — see [graph-pair selection](../docs/architecture/graph-pair-selection.md). |
| extra recomputation | Compute the selection added over that floor by choosing to recompute rather than hold memory. |
| stalled | Time the simulated step spends with tasks waiting on data or capacity rather than computing. |
| wasted compute | The sum of the two rows above: everything the step spends beyond the floor, before the terminal writeback. |
| terminal writeback | Transfers that return spill-final objects to the spill pool after the last task; the simulated step includes them. |
| task window | From the first task's compute start through the last task's end. It excludes the step's boundary regions by construction. |
| entry delay | Invocation origin to the first computation, including its scheduled fetches and measured frontend preparation. |
| lane utilization | Simulated lane-busy time divided by step time, including startup latency. Fetch and evict are independent percentages, each bounded by 100%. |
| infeasible / search_exhausted | A geometry the planner proved cannot fit the budget, or whose bounded candidate search ended without a feasible schedule. A geometry whose build exhausts the device reports every one of its budgets infeasible too, since profiling runs real kernels and the largest microbatch can run out of memory before any plan exists. Reported in the table, never raised. |
| rejected | A point the planner refused, before or during its search; `error` carries its reason. The sweep goes on with the next point. |
| artifact store | The on-disk store of build and planning artifacts, keyed by content digests — see [reusable planning](../docs/examples/reusable-planning.md). |

The traced-step deltas are real minus simulated: positive start deltas
mean the real timeline ran behind the prediction, and positive duration
deltas mean the work took longer than profiled. The simulator error the
figures and the gate report follows the same convention: positive means the
step ran slower than predicted.

## Distributed runs

`--symmetric-planning` verifies matching planning memory requirements and shares
CPU search work across ranks. All ranks still profile; shared timing estimates
are conservative, and every selected plan is admitted locally. A mismatch falls
back to independent searches. `--no-symmetric-planning` disables the optimization;
omitting both flags preserves the factory's `Distributed` setting. The Python
runner accepts the same override as `run(..., symmetric_planning=True)`.
See [verified symmetric planning](../docs/python/api/distributed.md#verified-symmetric-planning)
for the checks, work assignment, and artifact records.

The distributed path uses the same model-independent planner as `Trainer`. Its
separate-device qualification is in progress. Launch one process per GPU:

```bash
torchrun --standalone --nproc-per-node=2 -m benchmarking.quickstart mlops_llama3 \
  --distributed --sequence-length 1024 --sequences-per-step 16 \
  --search-budget-gib 12,16 --run-budget-gib 12,16 --spill-gib 40 \
  --steps 5 --plots --resolution-plans --output-dir runs/llama_dp
```

Budgets and `sequences-per-step` are per rank. The text recipe produces distinct
rank inputs and scales summed token losses by the total token count across the
whole update. Generic factories define their own objective normalization.

Gloo is initialized first for combined host-memory admission. The text recipe
then creates its NCCL group after Runtime installation and declares replicated
parameters. `--host-headroom-gib` (default 2 per rank) and
`--preparation-timeout` (default 1800 seconds) are configurable. Data-parallel
optimizer state and optional masters are sharded by default.

Reports, figures, traces and stdout logs live in `<output-dir>/rank-00000/`, etc.
Stores use `<output-dir>/artifact_store/rank-00000/` and
`<output-dir>/plan_store/rank-00000/`. Explicit artifact/build/plan store paths also
receive a rank suffix. Timing, loss observations, and throughput in each report
are local to that rank. Do not describe the sum of rank throughputs as measured
global throughput when ranks have different durations.

For a custom factory, return `distributed=Distributed(...)` alongside the
ordinary experiment fields. It may instead be a function of each fresh model
returning its ownership specification. Supply `model_factory` so every budget can
rebuild the model and its registered parameter ownership. The existing `context`
field can manage factory-created groups/resources. The factory is called after
Runtime installation. `--distributed` initializes only the CPU control group for
a custom factory; that factory owns its accelerator communication groups and mathematical scaling.

The Python interface accepts an already-created Gloo group:

```python
run(
    experiment, control_group=control_group,
    search_budget_gib=[12,16], spill_gib=40, output_dir="runs/custom_dp",
)
```

Run all participants with the same sweep. `--reproduce runs/llama_dp` selects the
current launcher's rank report and reuses its calibration and rank-specific
stores. No global task barriers are added to the measured runtime sequence.

### Overriding transfer calibration

Calibration normally supplies separate solo and concurrent rates automatically.
To override them, use `--transfer-bandwidths 40,26,56,26` (GB/s, in fetch solo,
fetch concurrent, evict solo, evict concurrent order). Two optional trailing
numbers set fetch/evict startup latency in microseconds: `40,26,56,26,4,4`.
The short form `26,26` gives fixed fetch and evict rates. A saved `search.json`
also supplies all four rates and both latencies. These values participate in
plan identity; this remains the development v1 schema.

Winner and geometry lane utilization uses simulated transfer intervals. Resolution
plots use each retained resolution's simulation (`--resolution-plans`); missing
lane timing stays unknown rather than being estimated from a fixed bandwidth.
Blended bandwidth is total bytes divided by lane-busy time, including startup
latency. It depends on each schedule's actual solo/concurrent overlap, not an
arithmetic average of calibration rates. `transfers/blended_bandwidth.png` and
`raw_data/points.csv` expose those rates alongside busy times and utilization.
Timeline HTML, timeline `summary.csv`, and slide exports show the same planned
blend. A measured page compares its achieved rate against that simulated blend;
its trace never replaces the assumed rate. Solo/concurrent calibration remains
visible separately. A lane with no transfers has no defined blended rate.
