# Residency-cut videos

Render exact replay states from a raw PressureFit trace directory. Frames and MP4s
run on CPU with NumPy and Matplotlib; movies additionally require `ffmpeg` on PATH.
The [capture guide](../../README.md) explains how to obtain the input trace.

By default the movie advances through the cuts without detailed examples,
explanation cards, camera excursions or seed/final holds. Add `--details` for the
optional teaching sequence. Explicit `--detail-at` or `--explain-cuts` also selects
examples without enabling the other preset holds.

## Start with an existing trace

`TRACE` below is a **raw trace directory** containing `manifest.json`,
`program.json`, and candidate `.pftrace` files. It is not an exported `.npz`
directory or a collection's parent directory.

```sh
TRACE=runs/example/traces/program_0000/pressurefit-EXAMPLE

# All cuts, score comparisons, alias sizes, semantic labels, and task timings.
python -m tools.pressurefit_trace analyze "$TRACE" --output runs/example/analysis

# Continuous cut playback; no explanation cards or slow examples by default.
python -m tools.pressurefit_trace.video render "$TRACE" --reduction all \
    --output runs/example/trajectory.mp4

# Only the reduction that produced the winning plan.
python -m tools.pressurefit_trace.video render "$TRACE" --reduction answer \
    --output runs/example/winning_reduction.mp4

# Equal-width tasks instead of widths proportional to profiled duration.
python -m tools.pressurefit_trace.video render "$TRACE" --axis task \
    --output runs/example/trajectory_tasks.mp4

# Optionally choose 1-based cut examples in the selected reduction.
# Replace these cut numbers with decisions present in your trace.
python -m tools.pressurefit_trace.video render "$TRACE" --reduction answer \
    --detail-at 1,20,800 --explain-seconds 15 --slow-seconds 8 \
    --step-seconds 1 --bridge-seconds 24 --accelerated-seconds 45 \
    --output runs/example/teaching.mp4

# Keep the boundary-local focus instead of showing the chosen object's full gap.
python -m tools.pressurefit_trace.video render "$TRACE" --focus-mode boundary \
    --output runs/example/local.mp4

# A quick layout check, or a presentation export.
python -m tools.pressurefit_trace.video render "$TRACE" --resolution 720p --quality preview \
    --output runs/example/preview.mp4
python -m tools.pressurefit_trace.video render "$TRACE" --resolution 4k --quality best \
    --output runs/example/presentation.mp4

# Inspect the first cut's selection formulas and alternatives, before applying it.
python -m tools.pressurefit_trace.video frame "$TRACE" --reduction 0 --cut 0 --explain \
    --output runs/example/choice.png

# Inspect a later state around a chosen boundary, with a wider local window.
python -m tools.pressurefit_trace.video frame "$TRACE" --reduction 0 --cut 100 \
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
| `--details` | off | Add the teaching sequence with explanation cards and slow examples. |
| `--explain-cuts` | 0; 3 with `--details` | Automatically spaced detailed examples in the first rendered reduction; 0 disables. |
| `--detail-at` | none; automatic with `--details` | Comma-separated **1-based** cut numbers, e.g. `1,20,800`; overrides `--explain-cuts`. Applies to the first rendered reduction. |
| `--explain-seconds` | 15 | Hold for each formula and alternatives card. |
| `--slow-seconds` | 8 | Selection and application time for each detailed cut, outside the explanation and extra camera holds. |
| `--opening-cuts` | 0; 6 with `--details` | Minimum initial cuts shown individually; also the minimum spacing for automatic examples. With explanations disabled, these receive slow selection/application views. When enabled, later reductions show one slow initial cut. |
| `--step-seconds` | 1 | Hold per individual cut in short passages between early examples. |
| `--bridge-seconds` | 24 | Duration of each longer passage between detailed examples. |
| `--bridge-hold-seconds` | 0.6 | Minimum hold for each displayed batch between examples. |
| `--accelerated-seconds` | 45 | Duration of the final accelerated passage in each reduction. |
| `--accelerated-hold-seconds` | 0.1 | Minimum hold per displayed batch in the fastest passage. Video frame rate remains unchanged. |
| `--seed-seconds` / `--final-seconds` | 0 / 0; 5 / 4 with `--details` | Holds at the seed and final state of each reduction. |
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

When enabled, automatic examples start with the first cut, then seek the strongest early
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
Final simulation measures contention. See the [algorithm description](../../../../../docs/architecture/pressurefit.md)
for the formulas and where residency strategies and fetch rules act.

## Render outputs

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

## Implementation

`canvas.py` composes the `panels/` drawing functions. `view.py` resolves the viewport;
`geometry.py` and `overview.py` handle interval geometry and incremental rasterization.
`playback.py` advances exact replay states; `recording.py` streams frames and their
metadata. `movie.py` orchestrates chapters using the shared `video/encoding.py`.

For a change, render a review frame and short movie, inspect the image, and compare
its storyboard to the original trace. Video tests are not part of the repository suite.
