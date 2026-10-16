# PressureFit trace tools

Capture real planner decisions, analyze their scores, and render residency-cut
trajectories. These source-tree tools use the installed trace API; they contain
no workload names or hard-coded run paths. Generated data belongs in a separate
run directory.

Run from a development checkout with the `shadowspill` environment active:

```sh
python -m tools.pressurefit_trace --help
python -m tools.pressurefit_trace render --help
```

Python, NumPy, and Matplotlib are already project dependencies. MP4 rendering
also needs `ffmpeg` on `PATH`. Analysis and PNG rendering run on the CPU and
do not need an accelerator. `dump` uses the quickstart frontend and its normal
accelerator/workload dependencies. This package lives under `src/tools/` and
is not included in the product wheel.

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

| Argument | Default | Meaning |
| --- | --- | --- |
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
exclusive. One capture can feed several independent physical-budget searches:

```sh
python -m tools.pressurefit_trace plan runs/example/dump \
    --physical-gib 7 --spill-gib 128 --output runs/example/budget_7/plans
python -m tools.pressurefit_trace plan runs/example/dump \
    --physical-gib 16 --spill-gib 128 --output runs/example/budget_16/plans
```

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

The collection's `dump.json`, `plans.json`, and `collection.json` link stages
and retain their source paths; keep the collection tree in place while using
those links. A raw per-search trace directory is self-contained and can be
moved for analysis. The exporter stores large context tables once per resolution.
Collected raw traces also include `collection_context.json`, checked against
the program digest. It supplies the movie's workload, budget, ordering and
microbatch labels. Each movie also names its recompute share and current
reduction, independently of whether it displays all reductions or only the answer.

## Code map and validation

`collect.py` owns capture/plan/trace/export. `data.py` joins metadata, builds
task-use geometry, and exports analysis. `explain.py` decodes comparator fields
and draws the explanation card. `render.py` draws states and encodes movies.
`__main__.py` owns the CLI. The binary reader/replay implementation stays in
`shadowspill.diagnostics.pressurefit_trace`.

```sh
python -m pytest tests/tools/test_pressurefit_trace.py
ruff check src/tools/pressurefit_trace tests/tools/test_pressurefit_trace.py
```

Tests use a fresh real planner trace, verify filters and ordering preserve
pressure, check required-use coverage, compare incremental raster updates with
an independent exact state, and exercise analysis and explanation rendering.
