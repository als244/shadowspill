# PressureFit tracing and videos

Capture real planner decisions, analyze their scores, and render the algorithm
and its resulting plan. All three video implementations live in this directory.
These source-tree tools contain no workload names or hard-coded run paths.
Generated data belongs in a separate, ignored run directory such as `runs/`.

| Video | Input | Rendering file | Usage guide |
| --- | --- | --- | --- |
| Residency cuts | Raw PressureFit trace directory | [`render.py`](render.py) | This guide |
| Emission + placement | Saved `AnnotatedProgramPlan` JSON | [`plan/render_emission_placement.py`](plan/render_emission_placement.py) | [Plan videos](plan/README.md) |
| Execution timelines | Saved `AnnotatedProgramPlan` JSON | [`plan/render_execution.py`](plan/render_execution.py) | [Plan videos](plan/README.md) |

Capture once with `dump`, search with `plan`, then record decisions with `trace`.
The resulting raw trace feeds the cut video; the saved plan feeds the other two.
The [trace API guide](../../../docs/python/pressurefit-trace.md) covers recording
from Python. The [algorithm guide](../../../docs/architecture/pressurefit.md)
defines the policies and scores. The [occupancy guide](../../../docs/python/occupancy.md)
describes the plan's memory/timing evidence and the existing HTML view.

Run from a development checkout with the `shadowspill` environment active:

```sh
python -m tools.pressurefit_trace --help
python -m tools.pressurefit_trace render --help
python -m tools.pressurefit_trace.plan --help
python -m tools.pressurefit_trace.plan render --help
```

Python, NumPy, and Matplotlib are already project dependencies. MP4 rendering
also needs `ffmpeg` on `PATH`. Analysis and PNG rendering run on the CPU and
do not need an accelerator. `dump` uses the quickstart frontend and its normal
accelerator/workload dependencies. This package lives under `src/tools/` and
is not included in the product wheel.

If necessary, set `PYTHONPATH=src` when running from the repository root.
For emitted actions, physical placement, and simulated compute/fetch/evict
execution, use the [plan tools in this directory](plan/README.md). They read
the annotated plan saved by `plan`, alongside optional semantic labels; a cut
trace alone does not contain physical offsets or transfer timings.

Titles and labels use Matplotlib's bundled STIX serif font, chosen for readable
small labels alongside Computer Modern math. A TeX compiler or extra font
installation is not required.

## Start with an existing trace

`TRACE` below is a **raw trace directory** containing `manifest.json`,
`program.json`, and candidate `.pftrace` files. It is not an exported `.npz`
directory or a collection's parent directory.

```sh
TRACE=runs/example/traces/program_0000/pressurefit-EXAMPLE

# All cuts, score comparisons, alias sizes, semantic labels, and task timings.
python -m tools.pressurefit_trace analyze "$TRACE" --output runs/example/analysis

# Teaching movie: spaced explanations, increasingly fast passages, and cut zooms.
python -m tools.pressurefit_trace render "$TRACE" --reduction all \
    --output runs/example/trajectory.mp4

# Only the reduction that produced the winning plan, with the same teaching pace.
python -m tools.pressurefit_trace render "$TRACE" --reduction answer \
    --output runs/example/winning_reduction.mp4

# Equal-width tasks instead of widths proportional to profiled duration.
python -m tools.pressurefit_trace render "$TRACE" --axis task \
    --output runs/example/trajectory_tasks.mp4

# Choose your own 1-based cut examples in the selected reduction.
# Replace these cut numbers with decisions present in your trace.
python -m tools.pressurefit_trace render "$TRACE" --reduction answer \
    --detail-at 1,20,800 --explain-seconds 15 --slow-seconds 8 \
    --step-seconds 1 --bridge-seconds 24 --accelerated-seconds 45 \
    --output runs/example/teaching.mp4

# Keep the boundary-local focus instead of showing the chosen object's full gap.
python -m tools.pressurefit_trace render "$TRACE" --focus-mode boundary \
    --output runs/example/local.mp4

# A quick layout check, or a presentation export.
python -m tools.pressurefit_trace render "$TRACE" --resolution 720p --quality preview \
    --output runs/example/preview.mp4
python -m tools.pressurefit_trace render "$TRACE" --resolution 4k --quality best \
    --output runs/example/presentation.mp4

# Inspect the first cut's selection formulas and alternatives, before applying it.
python -m tools.pressurefit_trace frame "$TRACE" --reduction 0 --cut 0 --explain \
    --output runs/example/choice.png

# Inspect a later state around a chosen boundary, with a wider local window.
python -m tools.pressurefit_trace frame "$TRACE" --reduction 0 --cut 100 \
    --boundary 65 --context-tasks 5 --focus-rows 10 --axis task \
    --output runs/example/state.png
```

`--candidate winner` is the default; alternatively pass a zero-based index
into `manifest.json`'s `candidates` list. `render --reduction all` shows every
seed-to-fit attempt as a separate chapter. Use `--reduction answer` for the
attempt that supplies the returned plan, or an integer for a specific attempt.
The answer can come from an earlier attempt. `frame` defaults to `answer` and
`--cut final`. Cut counts are **numbers of applied cuts**: 0 is the seed, 1 is
after the first cut. `frame --explain` explains the *next* cut from that state.

### Pacing and layout controls

Common input and state selectors:

| Argument | Commands | Meaning |
| --- | --- | --- |
| `input` | `analyze`, `frame`, `render` | Raw trace directory containing `manifest.json`. |
| `--output` | All | Required output directory for analysis, PNG path for a frame, or MP4 path for a movie. |
| `--candidate` | `analyze`, `frame`, `render` | `winner` (default), or a zero-based index in the trace manifest. |
| `--reduction` | `frame`, `render` | Zero-based pass index, `answer`, or `all` (movies only). Defaults: `answer` for frames, `all` for movies. |
| `--cut` | `frame` | Number of already applied cuts; `0` is the seed, `final` (default) is the last state. |
| `--boundary` | `frame` | Override the focus with a zero-based boundary position. |
| `--explain` | `frame` | Explain the next cut from the selected state. |
| `--object-timeline` | `frame` | Show the selected object's prior use, removed residency, and next use. |

Layout options below apply to `frame` and `render`. Timing, pacing, encoding,
and automatic example-selection options apply only to `render`.

| Argument | Default | Meaning |
| --- | --- | --- |
| `--title` | Recorded workload | Override the workload label, for example to distinguish measured bandwidth from a simulated bandwidth assumption. |
| `--axis compute` / `--axis task` | `compute` | Cumulative profiled seconds, or equal-width tasks with task indices. Changes drawing coordinates only. |
| `--pressure-scale` | `adaptive` | Shrink the pressure y-axis as cuts approach the budget, keeping zero and the flat budget line visible. `fixed` retains one scale. |
| `--pressure-view` | `total` | Include the fixed resident pool by role. `reducer` omits it from both usage and budget. Both views include task workspace as a separate layer beneath a flat budget; boundary excess and decisions are identical. |
| `--explain-cuts` | 3 | Automatically spaced detailed examples in the first rendered reduction; 0 disables. |
| `--detail-at` | automatic | Comma-separated **1-based** cut numbers, e.g. `1,20,800`; overrides `--explain-cuts`. Applies to the first rendered reduction. |
| `--explain-seconds` | 15 | Hold for each formula and alternatives card. |
| `--slow-seconds` | 8 | Selection and application time for each detailed cut, outside the explanation and extra camera holds. |
| `--opening-cuts` | 6 | Minimum initial cuts shown individually; also the minimum spacing for automatic examples. With explanations disabled, these receive slow selection/application views. Later reductions show one slow initial cut. |
| `--step-seconds` | 1 | Hold per individual cut in short passages between early examples. |
| `--bridge-seconds` | 24 | Duration of each longer passage between detailed examples. |
| `--bridge-hold-seconds` | 0.6 | Minimum hold for each displayed batch between examples. |
| `--accelerated-seconds` | 45 | Duration of the final accelerated passage in each reduction. |
| `--accelerated-hold-seconds` | 0.1 | Minimum hold per displayed batch in the fastest passage. Video frame rate remains unchanged. |
| `--seed-seconds` / `--final-seconds` | 5 / 4 | Holds at the seed and final state of each reduction. |
| `--focus-mode` | `cut` | During a cut, span the chosen object's prior and next uses. `boundary` retains the local multi-object view. |
| `--cut-zoom` / `--no-cut-zoom` | enabled | Animate detailed cuts from the local boundary to the full gap, then return after applying the cut. |
| `--zoom-seconds` | 1.2 | Duration of **each** outward and returning zoom. |
| `--zoom-hold-seconds` | 3 | Extra viewing time, split between the uncut full gap and the local result. |
| `--pacing overview` | `overview` | Batches later cuts into exact replay states, with explicit batch counts. |
| `--pacing every-cut` | | Shows every later decision for one frame; ignores `--accelerated-seconds`. Large traces make long movies. |
| `--fps` | 30 | Video frames per second. |
| `--context-tasks` | 3 | Tasks before and after the focused boundary (plus the task starting at it). |
| `--focus-rows` | 7 | Maximum readable local rows; selected alias stays visible. Full local membership is in the storyboard. |
| `--resolution` | `1080p` | `720p`, `1080p`, `1440p`, or `4k`; 16:9 PNG/video dimensions. |
| `--quality` | `high` | H.264 CRF: `preview` = 26, `high` = 18, `best` = 14. Lower CRF retains more detail and produces larger files. |
| `--width` | | Custom width instead of `--resolution`; height is 9/16 of width and MP4 dimensions must be even. |

`--quality` changes MP4 compression only. All three settings use the same
`libx264` encoder and `fast` encoding preset; resolution, frame rate, layout,
and pacing are independent. `best` retains more fine text and line detail than
`high`, but is still lossy H.264. It does not affect PNG exports.

The compressed movie replays **every cut**, but a displayed batch shows only
its endpoint. Its detail panel describes the last decision in the batch.
Use `every-cut`, `frame`, or the analysis files to inspect intervening decisions.
The short movie never labels a batch as one cut. Pressure changes discretely
when the shown decision or batch is applied; the fading highlight is decoration.

Automatic examples start with the first cut, then seek the strongest early
visual contrast in object size, removable span, role, and release/eviction.
The early window is the first 10% of cuts, after a six-cut gap. A different
comparison criterion breaks ties, so a sibling weight is not selected just
because its tie-breaker changes. The third example seeks a change between
release and eviction. If those changes do not exist, the renderer chooses
spaced fallback positions. No workload, stage name, or cut number is embedded
in this selection. A short trace gets only as many unique examples as it has
cuts. Use `--detail-at` for an exact script, and `--reduction` to choose which
reduction receives it. All chosen cut numbers and timing settings are saved
in the storyboard.

The third automatic example is at least one-third of the way through the
selected reduction, so an early eviction cannot place two teaching cards
almost next to each other. `--detail-at` can override this spacing.

Short passages (at most `max(24, opening-cuts)` intervening cuts) show each cut
for `--step-seconds`. Longer passages use `--bridge-seconds`, holding each batch
for at least `--bridge-hold-seconds`. The final passage uses the accelerated
settings. Within a batched passage, cut counts ramp quadratically while the
requested duration is preserved exactly to the nearest video frame. Repeated
states are held rather than dropped. `--pacing every-cut` visits every
intervening decision for one frame instead of batching those passages.

## What the drawing means

- The overview orders aliases by their **first input, mutation, or output
  reference** in the selected task sequence. Ties use the stable alias index.
  Colors encode object role, not row order. Row heights are proportional to
  alias bytes; subpixel rows contribute their actual area to the overview.
- Aliases below the run's recorded `minimum_object_bytes_evict_eligible` are
  hidden. Rows whose seed residency is entirely required use are also hidden,
  except optimizer-state rows.
  An alias with a recorded boundary split remains visible even when it has no
  positive-duration idle interval. Filters are fixed across the movie and both
  horizontal-axis modes.
- At the seed, final state, and return from a detailed cut, the local focus
  pins the chosen alias first. Parameters, activations, gradients, and optimizer
  state take priority over all other roles, regardless of size. Within those
  roles, current-task aliases come first by descending size, with
  parameter/activation/gradient/optimizer-state ties, then nearby task references.
  Other-role neighbors only fill unused slots (six neighbors by default).
  During a cut, the default view isolates that object and spans
  its most recent use through its next use, keeping both required-use endpoints
  visible around the removed interval. Every intervening task still has a bar;
  long gaps label the prior/next tasks and triggering boundary to stay readable.
  Short windows keep individual task and boundary labels. The camera only
  changes drawing coordinates; residency and pressure remain exact replay states.
  `--focus-mode boundary` keeps the local multi-object view throughout.
- The chosen row says **Release** (a spill copy is available) or **Evict**
  (writeback is needed), becoming **Released** or **Evicted** after the cut.
  It names the release/eviction boundary and the next task's semantic name and
  starting boundary. The duration is profiled compute from that release/eviction
  boundary until the next task begins; the selected pressure boundary may lie
  inside this gap. The release/eviction label marks the start of removed
  residency, not a simulated transfer completion timestamp.
  This duration stays in seconds, milliseconds, or microseconds in either axis
  mode; it does not include future transfer stalls. A red outline marks the exact
  span selected for removal; its residency fades out after applying the cut, leaving
  a dashed outline and pointer on the gap. A cut with no positive-duration gap
  receives a boundary marker. Formula pauses are followed by an unobscured
  selection view, an outward camera move, the cut, and a return to the trigger's
  local neighborhood. `--no-cut-zoom` shows the full span directly.
- Solid bars show residency. Hatched bands cover whole tasks that access or
  produce an alias. Input anchors extend through the task's end; output anchors
  extend back through the producing task. An alias can contain several objects
  sharing storage. Its primary role determines its color; labels retain member
  identity, and activation labels adapt to the focused task context. Shared
  member counts and identities remain in the analysis exports, without adding
  annotations to the focus rows.
- Boundary position `p` is before task `p` and after task `p-1`; the final
  position is after all tasks. This differs by one from the signed cut endpoint
  convention in the binary format. A split cut may change headroom without
  clearing any boundary cells. Both axis modes label boundaries above the focus
  task strip. Every boundary keeps its own label and exact tick; leader lines
  separate labels when tasks are very short. Tasks too narrow for their names
  retain a boxed label pointing to their actual position in the strip.
  Optimizer tasks without a stage index use the compact label `Opt`; boundary
  indices distinguish their positions and the trace retains their full task IDs.
- **Drawing filters never remove pressure contributions**, including hidden
  rows. Pressure is replayed from the recorded operations and includes the reducer's
  reservations and repair pressure. It is not recomputed from the visible bars
  or presented as total process/device memory. The dashed line is a **flat
  planning budget**. Per-task workspace is added back as its own pink layer,
  alongside the alias roles. This is planning memory, not measured GPU allocation.
- Optimizer-state rows remain visible even if their only residency is required
  by one optimizer task. The size threshold still applies. Their total storage
  across the model is different from their simultaneous device residency: the
  pressure chart includes only the planner's boundary charges and fixed
  reservations. Short optimizer tasks can produce narrow violet intervals on
  the compute-time axis. The sidebar reports both visible aliases and total tasks.
- Non-evictable aliases occupy a fixed resident pool, already subtracted from
  the reducer's capacity. The default chart includes these reservations by role
  (including optimizer state in violet), **even when their object rows are hidden**.
  Each reservation includes the alias's generations and alignment padding, following
  the planner's static homes and checked against its returned pool size. These
  constant bands show reserved capacity, not changing tensor liveness. Add the
  same pool size to pressure and capacity, so every boundary's excess and every
  decision stays identical. `--pressure-view reducer` omits both additions.
- Pressure is stacked by the alias's primary role, counting shared storage
  once. Persistent state (parameters, optimizer state, gradients, and buffers)
  takes precedence over saved activation views of that same storage. Its stable
  name labels the row; member labels and task references remain available in
  `aliases.json`. This also corrects the display of older trace recordings.
  Output reservations and fetch headroom are assigned to that same role;
  repair pressure and any additional capacity reservation have a separate gray
  layer. The flat budget is the maximum recorded `capacity + workspace` across
  reductions, plus the resident pool in the `total` view. Any difference between
  that budget and a boundary's `capacity + workspace + resident pool` is shown
  as reserved bytes. Every stack is checked to preserve exactly
  `displayed usage - flat budget = recorded pressure - recorded capacity`.
  The red outline marks over-budget pressure. Adaptive scaling keeps zero at
  the bottom and 8% margin above the current usage/budget maximum.
- The header shows `Cut X/N`, with the seed at cut zero. The decision panel
  identifies `Cut X (Boundary Y)` and colors the selected object's name by its
  role, matching the timeline and pressure legend.
- The decision panel breaks down the task immediately after the selected
  boundary into inputs, outputs, mutated objects, and profiled workspace.
  Parentheses count distinct declared objects; bytes count their unique full
  storage aliases, matching the planner's allocation unit. Categories may
  overlap. The total unions all three object/alias sets before adding workspace,
  so shared views and in-place mutations are counted once. These are required
  task bytes, not the boundary's total pressure, which can include other resident
  aliases and reservations. `tasks.json` and each storyboard's `current_task`
  retain the counts, bytes, and overlap for analysis.
- The lower-left note shows `Reduction X/Y` and that pass's object capacity.
  Capacity refinements lower this budget; pressure repairs can restart the
  reducer at the same object capacity. The value comes from recorded search
  steps and is carried across repair-only passes. An unlinked refinement or
  pass with no recorded reason shows `not recorded` when its exact
  capacity is unavailable. Pool/allocation causes remain in analysis and
  storyboard metadata.
- Role attribution and the total view require `pressure_accounting` in the
  manifest context. Current recordings include it automatically. Older traces
  remain readable with `--pressure-view reducer`; missing composition is shown
  as **Unattributed**, preserving exact totals. Record the candidate again to
  obtain its role breakdown and resident reservations.
- The compute axis uses summed profiled task durations, not the emitted plan's
  wall-clock start times. A task-index view uses the same state and pressure.
  Neither view invents fetch/eviction timestamps from a cut's score.
- A selected recompute backward task starts with a **lime R segment**, labeled
  with the same stage and microbatch as its backward segment, for example
  `R16 · mb14` and `B16 · mb14`. Its
  length is the profiled recompute-backward duration minus the save-backward
  duration from the same task-alternative group. The remaining task stays in
  the backward color. On the task-index axis the lime segment occupies the
  same fraction of the equal-width task. This is a duration decomposition,
  not a separately traced kernel. The signed difference and both source task
  profiles remain in the analysis/storyboard; a negative profiling difference
  draws no lime segment. Ambiguous or missing save/recompute pairs stay
  unshaded instead of borrowing a duration from another stage.
- Release/eviction labels describe the reducer's proposed residency cut.
  Execution timing requires the emitted schedule for that candidate step;
  neither the cut score nor a profiled compute gap is a transfer timestamp.
- Every reduction resets to the seed. Capacity and extra-pressure refinements
  carry over as recorded. A reduction ending analytically infeasible is labeled
  as such; it is not described as a fit.

The explanation card shows `argmax(used - capacity)` with the earlier-boundary
tie break, then `E, writeback, -slack, -size`, alias, start, and cut ID. It
highlights each alternative's first losing field. Transfer strategies omit
`E`; both maximize signed slack without clamping it. `E` is the sum of
separately clamped eviction/fetch exposure estimates, without queue contention.
Final simulation measures contention. See the [algorithm description](../../../docs/architecture/pressurefit.md)
for the formulas and where residency strategies and fetch rules act.

## Outputs

`analyze` writes:

- `analysis.json`: size bins/quantiles, display filters, per-reduction totals,
  final pressure, and explicit answer references.
- `cuts.csv`: one row per applied cut, selected alias and semantic name, trigger,
  pressure before/after, peak, excess, score, and decisive comparison field.
- `decisions.jsonl`: selected cut and up to three alternatives, complete named
  scores, their first distinguishing field, and top-three boundary measurements.
- `aliases.json`: every alias, all shared members, semantic labels, reference
  positions, sizes, and hide reasons. `tasks.json`: selected task order and
  recorded profiled/plan durations, task workspace, and semantic labels.
- `reduction_NNN_metrics.npy`: columns `cut_count`, `peak_pressure_bytes`,
  `max_excess_bytes`, with excess clipped at zero.
- `pressure_roles.npz`: role names and role-by-boundary byte arrays for each
  reduction's seed and final reducer state, plus `resident_reservations` by role
  to add for the total view, a boundary `workspace` array, and
  `budget_excluding_resident_bytes`. These seed/final arrays retain raw reducer
  values. `analysis.json` reports the total planning budget and whether roles
  could be attributed; all intermediate compositions are verified as well.

`render` writes an MP4, FFmpeg log, seed/first-cut/final PNGs for each chapter,
and `<stem>.storyboard.json`. Each storyboard entry names the exact reduction,
cut count, axis, video frame range, boundary, visible/full local alias lists,
pressure summary, flat budget, y-axis limits, and role bytes at the selected
boundary. Cut highlights include the release/eviction action and
`departure_position`, the next-use task position, and `compute_gap_ns` between
those boundaries. `compute_until_next_use_ns` separately retains the duration
from the pressure trigger to the next use.
Camera states additionally record `focus_task_range`, `focus_axis_limits`, and
`object_journey` (prior/next use, departure, and zoom progress), so a rendered
wide view can be related to its original tasks in either axis mode.
`recompute_prefixes` records each lime interval's task position, duration,
save-backward baseline, and exact drawing coordinates. The header's
`recompute_tasks` table keeps all matched backward timing differences.
The storyboard header records the source directory and candidate file,
`cut_score`, `cut_ranking`, exact `ranking_criteria`, `answer_reduction`, and
`rendered_reductions`. Check these fields when matching a movie to a run.
`frame` writes a PNG and corresponding state metadata.
Named analysis/render outputs are replaced when rerun; raw inputs are read-only.

All analysis/render commands verify every recorded boundary ranking, cut
comparison, and final C residency/pressure digest before producing output.
Analysis additionally checks that required-task intervals remain covered at
each reduction's seed and final state. The renderer streams through the trace;
it does not retain a cuts-by-alias-by-time array.

## Capture new traces

For an arbitrary program, the simplest route is the normal planning API:

```python
from shadowspill.planner import GenericPlanningOptions, SearchOptions, plan_program

plan = plan_program(
    problem,
    search_options=SearchOptions(
        generic=GenericPlanningOptions(deterministic=True),
        trace_directory="runs/example/traces",
    ),
)
```

The API finds the winner, reruns it with tracing, and verifies the result.
Explicit `trace_candidates` pairs record other resolutions/policies. See the
[trace API guide](../../../docs/python/pressurefit-trace.md) for scope, metadata,
and normal frontend semantic-label export. The optional `semantic_labels.json`
sidecar must have the same program digest; the renderer works with the generic
object/task labels when that sidecar is absent.

For a quickstart workload, the collection commands separate accelerator
capture from CPU planning and recording:

```sh
# Pass the same preset/factory and profiling arguments used by quickstart.
python -m tools.pressurefit_trace dump runs/example/dump -- <quickstart arguments>

python -m tools.pressurefit_trace plan runs/example/dump --output runs/example/plans
python -m tools.pressurefit_trace trace runs/example/plans --output runs/example/traces

# Optional full experiment: every ordering/geometry program, every resolution/policy.
python -m tools.pressurefit_trace trace runs/example/plans \
    --programs all --all-candidates --output runs/example/all_traces

# Optional compressed, verified export of one trace or a whole collection.
python -m tools.pressurefit_trace export runs/example/all_traces \
    --output runs/example/export
```

Capture/plan/trace/export require fresh output directories. `plan` defaults
to the captured budgets; `--execution-gib` and `--spill-gib` override them.
`--execution-gib` is usable planner capacity. To specify a physical device cap
like quickstart's budget, use `--physical-gib` instead: it subtracts the captured
runtime/headroom carve-out. The two execution-budget options are mutually
exclusive. Capture with a pool at least as large as the largest intended device
budget: the saved problem rejects execution budgets above the pool used for
profiling. Increasing `--physical-gib` does not enlarge that profiling contract.
Within that ceiling, one capture can feed several independent budget searches:

```sh
python -m tools.pressurefit_trace plan runs/example/dump \
    --physical-gib 7 --spill-gib 128 --output runs/example/budget_7/plans
python -m tools.pressurefit_trace plan runs/example/dump \
    --physical-gib 16 --spill-gib 128 --output runs/example/budget_16/plans

# Reuse the compute profiles, but search against hypothetical transfer rates.
# Each direction retains 450 decimal GB/s when both lanes copy simultaneously.
python -m tools.pressurefit_trace plan runs/example/dump \
    --physical-gib 16 --spill-gib 128 --transfer-bandwidths 450,450 \
    --output runs/example/budget_16_450gbps/plans
python -m tools.pressurefit_trace trace runs/example/budget_16_450gbps/plans \
    --output runs/example/budget_16_450gbps/traces
```

`--transfer-bandwidths` changes planning and simulation; it does not run a GPU
copy at that speed or alter the captured compute profiles. The two-number form
sets fetch and eviction rates, including concurrent operation. Four numbers
set fetch solo/concurrent and eviction solo/concurrent separately; for example,
`450,225,450,225` models an aggregate 450 GB/s link shared equally while both
lanes copy. Transfer latencies retain the captured values. The saved plan records
the effective calibration, and `trace` reuses it when recording the winner, so
the cut trajectory corresponds to the hypothetical plan rather than the capture's
original transfer rates. This records planner decisions, not measured execution.

If capture exhausts device capacity while profiling a geometry, it uses the
normal search's typed out-of-memory detection and rollback validation,
records that geometry and every associated ordering as `profiling_infeasible`,
and continues with the remaining geometries. It does not enlarge the pool.
These failures are preserved in `infeasible_NNNN.json`, `dump.json`, and each
budget's `plans.json`. Non-capacity failures and incomplete cleanup still stop
capture. The physical/usable budget mapping and workload metadata also remain
in `dump.json`.

`--workers` controls CPU planning. `trace` defaults to the winning program and
winning candidate. `--candidates-json pairs.json` supplies a JSON list of
`[selection_id, candidate_id]` pairs and is mutually exclusive with
`--all-candidates`. Full collections can be much larger than winner-only traces.

Collection arguments:

| Command | Argument | Meaning/default |
| --- | --- | --- |
| `dump` | `output` | Fresh output directory for captured programs and profiling metadata. |
| `dump` | `-- <quickstart arguments>` | Forward workload, geometry, and profiling options to quickstart. Use `python -m benchmarking.quickstart --help` for that frontend's arguments. |
| `plan` | `input`, `--output` | Capture directory and required fresh planning output directory. |
| `plan` | `--physical-gib` | Physical execution cap; subtract the capture's runtime/headroom carve-out. |
| `plan` | `--execution-gib` | Usable planner execution capacity; mutually exclusive with `--physical-gib`. Without either, retain captured budgets. |
| `plan` | `--spill-gib` | Override the captured spill capacity. |
| `plan` | `--transfer-bandwidths` | Optional decimal GB/s override: `FETCH,EVICT` or `FETCH_SOLO,FETCH_CONCURRENT,EVICT_SOLO,EVICT_CONCURRENT`. Compute profiles and transfer latencies stay as captured. |
| `plan` | `--workers` | CPU planning workers; `0` (default) delegates worker selection to the planner. |
| `trace` | `input`, `--output` | Planning directory and required fresh trace collection directory. |
| `trace` | `--programs winner\|all` | Record the winning program (default) or every feasible program. |
| `trace` | `--all-candidates` | Record every resolution/policy pair instead of only each program's winner. |
| `trace` | `--candidates-json` | JSON list of `[selection_id, candidate_id]` pairs; mutually exclusive with `--all-candidates`. |
| `export` | `input`, `--output` | Raw trace or collection, and required fresh directory for compressed, verified exports. |

The collection's `dump.json`, `plans.json`, and `collection.json` link stages
and retain their source paths; keep the collection tree in place while using
those links. A raw per-search trace directory is self-contained and can be
moved for analysis. The exporter stores large context tables once per resolution.
Collected raw traces also include `collection_context.json`, checked against
the program digest. It supplies the movie's workload, budget, ordering and
microbatch labels. Each movie also names its recompute share and current
reduction, independently of whether it displays all reductions or only the answer.

## Every ordering and recompute resolution

The matrix workflow keeps each resolved program's best policy from a saved
collection. It reproduces the source search, retains the individual schedules,
certifies each layout, and records a separate winning cut trace for each case.
It preserves infeasible combinations and their candidate failure reasons in
`matrix.json`; those cases have no successful execution movie.
The cut trace retains the search simulation. Plan videos use the certified
simulation with address-reuse dependencies; if timings change during
certification, the matrix preserves both `search_makespan_ns` and `makespan_ns`.

```sh
python -m tools.pressurefit_trace.matrix prepare runs/example/plans \
    --dump runs/example/dump --output runs/example/matrix --workers 4

python -m tools.pressurefit_trace.matrix render runs/example/matrix \
    --output runs/example/matrix_videos --jobs 6 \
    --title 'Example model · Measured bandwidth' \
    --resolution 4k --quality best --fps 30 --motion-fps 30 \
    --publish-to "$HOME/Desktop/Example videos"
```

Both commands resume completed work. `prepare` checks the saved schedules and
simulated makespans against the source diagnostics. `render` records the renderer
source hashes and options and refuses to mix changed configurations in one output
directory. Use a fresh output directory for a different render configuration.
Jobs run in separate processes; on Linux they divide the available CPU affinity.
A failed movie does not stop the other jobs, and a later invocation retries it.

The default output includes cut trajectories, emission/placement, and execution,
arranged as `seqs_N/order_ORDER/recompute_FRACTION/`. Each MP4 is checked against
its report and fully decoded before publication. Publication copies are verified
by SHA-256. `README.md` is a live linked index; `progress.json` records running,
failed, and complete movies. Encoded review frames are extracted beside each
movie for human inspection; successful decoding alone is not a visual review.

| Command | Argument | Meaning/default |
| --- | --- | --- |
| `prepare` | `planning`, `--output` | Source collection with `plans.json`, and resumable matrix directory. |
| `prepare` | `--dump` | Capture directory; use this when the original capture moved to another machine. |
| `prepare` | `--workers` | CPU planning workers, default `4`. Source searches must be deterministic. |
| `render` | `matrix`, `--output` | Prepared matrix directory and resumable video directory. |
| `render` | `--publish-to` | Optional library directory receiving each verified video and a linked index. |
| `render` | `--videos` | One or more of `cuts`, `emission-placement`, `execution`; default all three. |
| `render` | `--jobs` | Concurrent video processes, default `4`. |
| `render` | `--title` | Workload title; each video's ordering, geometry and resolution remain visible. |
| `render` | `--resolution`, `--quality` | Default `4k`, `best`; same choices as the individual renderers. |
| `render` | `--fps`, `--motion-fps` | Output rate and distinct plan-animation states per second; default `30` each. |
| `render` | `--emission-timing` | Explain, motion, build, sweep, hold seconds; default `15,12,30,50,8`. |
| `render` | `--execution-timing` | The same five durations; default `18,15,30,60,8`. |
| `render` | `--cut-options-json` | Extra cut-render CLI settings as a JSON object, for example `'{"explain-seconds":20,"detail-at":"1,50,200"}'`. Defaults to the cut renderer's teaching pace and automatic examples. |

## A browsable video library

`library.py` builds one offline catalog for published movies from multiple runs.
It produces a searchable HTML index, README results tables, CSV/JSON exports,
and a video page plus README for every configuration. It also indexes a running
matrix, including queued movies and infeasible cases. It does not render or move
movies and does not change the saved plans.

```sh
python -m tools.pressurefit_trace.library runs/library_sources.json \
    --output "$HOME/Desktop/Example videos"

# Refresh every 30 seconds until all matrix movies reach a final status.
python -m tools.pressurefit_trace.library runs/library_sources.json \
    --output "$HOME/Desktop/Example videos" --watch 30
```

Open the output's `index.html` directly in a browser; no server, external fonts,
network connection, or JavaScript framework is required. Search and filter by
model/budget, choose winners or the full matrix, and sort by simulated throughput.
Refresh the browser page to see newly published movies. The generator leaves the
batch renderer's matrix `README.md` alone; intermediate folders get `CONTENTS.md`.

| Argument | Meaning |
| --- | --- |
| `config` | JSON source configuration with `schema: 1`; example below. |
| `--output` | Library root, containing the already published MP4 files. |
| `--watch SECONDS` | Optional polling interval, at least 1 second. Rebuilds indexes and exits once all requested matrix movies are complete or failed. Without this flag, builds once. |

Input paths (`winner`, `plan`, `sources[].path`, `matrix`, `progress`) are absolute
or relative to the configuration JSON. Output paths (`directory`, `videos[].path`)
are relative to the library root. Each entry needs a unique ID and page directory.
Supply workload dimensions explicitly; the catalog never infers tokens per step
from a filename or video duration.

```json
{
  "schema": 1,
  "title": "Example videos",
  "notes": ["Saved simulation using measured task profiles."],
  "entries": [{
    "id": "example-16-measured",
    "model": "Example model",
    "training": "Full-model training",
    "sequence_length": 4096,
    "tokens_per_step": 65536,
    "budget_gib": 16,
    "spill_gib": 128,
    "transfer": "Measured bandwidth",
    "category": "Winner",
    "featured": true,
    "directory": "Example model/16 GiB/Measured bandwidth",
    "winner": "example/winner.json",
    "plan": "example/plans/program_0000.plan.json",
    "videos": [{
      "kind": "execution",
      "path": "Example model/16 GiB/Measured bandwidth/execution_4k.mp4",
      "seconds": 124,
      "width": 3840,
      "height": 2160,
      "decode": "passed"
    }],
    "sources": [{"label": "Run", "path": "example"}]
  }],
  "matrices": [{
    "id": "example-16-matrix",
    "matrix": "example/matrix",
    "progress": "example/matrix_videos/progress.json",
    "directory": "Example model/16 GiB/All orderings and resolutions",
    "defaults": {
      "model": "Example model",
      "training": "Full-model training",
      "sequence_length": 4096,
      "tokens_per_step": 65536,
      "budget_gib": 16,
      "spill_gib": 128,
      "transfer": "Measured bandwidth"
    }
  }]
}
```

Entries accept `facts` for explicit metadata overrides (geometry, ordering,
resolution, policy, reduction counts). The saved plan's **certified simulation**
always supplies execution statistics when `plan` is provided. For cut-only
archives, supply the recorded trace makespan in `facts.makespan_ns` and identify
`facts.timing_basis`; missing metrics remain blank. `category` is `Winner`,
`Earlier cut`, or `Archive`; generated matrix entries use `Matrix`. Keep different
planner versions in separate entries, even if their movies share a folder.

`videos[].kind` is `cuts`, `cuts-task`, `emission-placement`, or `execution`;
optional `label` overrides the displayed title. A video becomes clickable only
when its status is `complete` and the published file exists. Static entries may
provide duration, dimensions, SHA-256, and decode evidence from their validation
reports. Matrix entries obtain those values from `progress.json`. The catalog
does not substitute file existence for decoding or visual review. Matrices can
override the default three 4K filenames with a `video_files` mapping from video
kind to filename, matching the batch's requested video types.

Throughput is `tokens_per_step / (makespan_ns / 1e9)`, labeled simulated tokens/s.
An optional `measurement` object adds actual training throughput alongside it:

```json
"measurement": {
  "step_seconds": 20.0,
  "aggregation": "Median of steps 2-6, using the slowest rank in each step across 8 ranks",
  "useful_tflops_per_gpu": 300.0
}
```

These example values must be replaced with validated measurements of this exact
plan. Include the raw measurement directory in `sources`. The catalog computes
measured tokens/s from `tokens_per_step` and measured seconds, and simulation
error as `simulated_seconds / measured_seconds - 1`. The timing aggregation is
required; missing measurements stay blank. Measurements never replace the
simulation used for the video. Useful TFLOP/s/GPU is optional and must use the
workload's documented FLOP count.

The CSV retains both `search_makespan_ns` and certified `makespan_ns` when they
differ. Task waits sum `start_ns - ready_ns`; final transfer drain is excluded.
Compute time sums compute task durations, and execution peak includes workspace.
Infeasible configurations retain failure details and have no throughput value.
Reduction indices in JSON/CSV are zero-based; displayed reduction numbers are
one-based. Plan summaries are cached by file path, size, and modification time.

Each configuration's `Data/` contains named symlinks to its saved inputs. These
are convenient entry points, not additional data copies. Preserve their targets
when relocating the library, or copy with symlinks dereferenced. Keep source
configuration, videos, cache, indexes, and generated results outside Git (for
example under `runs/` or a Desktop library). Only the generic generator belongs
in the repository.

## Executed winners and hypothetical host bandwidth

For distributed quickstart runs, opt in to the observers before starting the
search. Each process uses its own directory; install the search-input observer
only on the reporting rank:

```python
from tools.pressurefit_trace.winner_capture import install as capture_winners
from tools.pressurefit_trace.search_capture import install as capture_search

capture_winners(rank_output / "winners")
if rank == 0:
    capture_search(rank_output / "search-inputs")
```

The winner observer saves the exact admitted program, schedule, physical layout,
semantic names and the native search input. The search observer saves the
verified distributed problems, including conservative cross-rank profiles,
collective timing and allocation requirements. Both write outside measured
training steps. Export errors are recorded and must be resolved before using
the export; they do not change benchmark feasibility.

The launcher also records actually executed plans in `executed/<budget>bytes.json`
(candidate, ordering, capture identity, physical/logical budgets and completed
step times), archives `report.winner_plans` in `search-winners/`, and writes
`search-inputs/metadata.json` after search with `complete: true`, an optional
`physical_to_execution_bytes` mapping, and `global_tokens_per_step`.
The rank directory is the source argument for offline replanning below.

Prepare only the plans that actually completed the benchmark, then render all
three views. The cuts show the answering reduction; all timings and example
selection remain configurable through the batch renderer:

```sh
python -m tools.pressurefit_trace.winners RUN/quickstart \
  --output RUN/winner-videos-inputs --rank 0 --workload 'Example model'
python -m tools.pressurefit_trace.matrix render RUN/winner-videos-inputs \
  --output RUN/videos --publish-to '/path/to/Video Library/Example model' \
  --resolution 4k --quality best --jobs 2
```

On the original GPU environment, capture separate Nsight Systems reports of the
already benchmarked winners. The replay requires matching program, schedule,
layout and capture identities. It reuses the search report and stored incumbent;
profiler timings are not throughput measurements:

Replay restores the winner's recorded timing model instead of replanning with
new collective/profiling measurements. Before doing so, it checks that the fresh
capture has the same executable structure, tensor/storage facts, residency,
transfer operations and resource order. Workspace and scratch must still fit
the saved admission. A change to measured durations alone is allowed; a change
to operations, bytes or memory requirements is rejected. Each rank retains a
`replay-contract.json` record, and the final program/schedule/layout/capture
identity checks still apply before capture begins.

```sh
python -m tools.pressurefit_trace.profile_winners RUN/quickstart \
  --output RUN/nsys --world-size 8 --warmup-steps 3 --capture-steps 3 \
  --gpu-metrics-frequency 10000 --nsys /path/to/nsys
```

`--budget-gib` limits capture to one executed winner. Each attempt retains its
command, logs, `.nsys-rep`, SQLite export and validation. Validation requires CUDA
kernels on every GPU, device metrics, semantic task/transfer NVTX and step ranges
from every rank. Resume checks identities and hashes before skipping a capture.
GPU metrics require permission on the allocated node.
Capture boundaries coordinate over a CPU/Gloo group. The command disables
Nsight's synchronous flush inside `cudaProfilerStop`; Nsight still flushes at
collection end. This avoids a documented multi-context profiler-stop hang
without changing the captured schedule or excluding GPU metrics. Validation
also requires every requested captured step from every rank.

Repeat the entire saved planning search without importing PyTorch or executing
training:

```sh
python -m tools.pressurefit_trace.replan RUN/quickstart/rank-00000 \
  --output RUN/simulated-300GBps --host-gbps 300 --jobs 3 --workers 4 --plots
```

`--host-gbps` is decimal GB/s **per direction**, both alone and during simultaneous
fetch/evict. Only copy rates change. Shard sizes, GPU memory sizes, calibrated
collective durations/resources, task profiles, overlap costs and launch latencies
stay unchanged. All saved microbatches, orderings, budgets, resolutions and
policies are searched again, carrying smaller-budget incumbents as in quickstart.
Uncaptured geometries retain their capture failure. Missing or unverified
distributed inputs fail explicitly. Separate plan/artifact stores, `search.json`,
`winners.json`, self-contained winner plans and `RESULTS.md` make the experiment
reproducible. A different bandwidth requires a different output directory.
`--plots` also saves the normal search figures and their raw plotting data.
`--jobs` runs independent geometry/ordering searches in separate CPU processes
(default 1). Each process uses `--workers` planner workers and walks budgets in
order, preserving smaller-budget incumbents. Parallel runs keep resumable stores
under `geometry-search/`; their results merge in the original search order, so
completion order does not change winner tie-breaking.

## Code map and validation

```text
pressurefit_trace/
  __main__.py                 capture, trace analysis, and cut-video CLI
  collect.py                  dump / plan / trace / export
  matrix.py                   retain and trace each resolution's best policy
  render_batch.py             resumable matrix rendering and verified copies
  winner_capture.py           opt-in exact admitted-plan export
  search_capture.py           opt-in verified distributed search input export
  winners.py                  prepare executed winners for the three video types
  profile_winners.py          exact-plan Nsight replay with metrics/NVTX checks
  replan.py                   CPU-only search with hypothetical host-copy rates
  library.py                  indexed results, configuration pages, live progress
  library_html.py             offline HTML views
  library_assets/             local stylesheet and search/player controls
  data.py, pressure.py         cut replay geometry and memory accounting
  explain.py                  cut selection formulas and alternatives
  render.py                   cut-trajectory renderer and encoding
  style.py                    shared palette and duration labels
  plan/
    __main__.py               plan analysis and video CLI
    data.py, waits.py         actions, transfers, occupancy, and task blockers
    placement.py             verified replay of physical placement
    storyboard.py            examples, timing, and camera movement
    render_common.py         shared canvas, lanes, and placement raster
    render_emission_placement.py
    render_execution.py
    render.py                renderer selection and plan-video encoding
```

The binary reader/replay implementation stays in
`shadowspill.diagnostics.pressurefit_trace`. The old `tools.plan_video` command
is a thin CLI alias to `tools.pressurefit_trace.plan`; all implementation is
here. Both video guides document their complete CLI arguments and outputs.

```sh
python -m pytest tests/tools/test_pressurefit_trace.py tests/tools/test_plan_video.py
ruff check src/tools/pressurefit_trace src/tools/plan_video tests/tools
```

Tests use a fresh real planner trace, verify filters and ordering preserve
pressure, check required-use coverage, compare incremental raster updates with
an independent exact state, and exercise analysis and explanation rendering.
