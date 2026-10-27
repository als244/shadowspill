# Plan videos

Part of the shared [PressureFit tracing and video tools](../README.md).
The cut renderer lives in the parent directory; emission/placement and execution
have separate rendering modules here, with common fonts, colors, and drawing
primitives.

Render a saved plan as two related explanations:

- **Emission + placement:** follow release, eviction, and fetch actions; then
  construct the physical address layout and sweep through its lifetimes.
- **Execution:** explain why selected tasks cannot start, with their required
  input transfers or storage-reuse dependencies highlighted on compute/fetch/
  evict lanes. Show byte backlog and execution/spill occupancy on the same clock.

The tools accept any single-device `AnnotatedProgramPlan` JSON. They do not
contain workload names, model-specific rules, or experiment paths. Everything
runs on the CPU; no profiling or training run is needed. NumPy and Matplotlib
are project dependencies. MP4 rendering additionally requires `ffmpeg`.

Run from a development checkout with the `shadowspill` environment active:

```sh
python -m tools.pressurefit_trace.plan --help
python -m tools.pressurefit_trace.plan render --help
```

If the source tree is not already importable, set `PYTHONPATH=src`. The tools
are source-tree utilities, not part of the product wheel. Keep generated data
under `runs/` or another ignored experiment directory. Movies, frame PNGs, and
logs are ignored by git; the source and this guide are tracked.

## Inputs and first render

Save `plan_program(...).to_json()` to obtain an annotated plan, or use the
`program_XXXX.plan.json` produced by
[`tools.pressurefit_trace plan`](../README.md). A cut trace
alone does not contain physical allocations or transfer intervals.

Semantic annotations are optional, but recommended for real programs. Supply
the capture's `program_XXXX.semantics.json` or a trace's
`semantic_labels.json`. Its program digest must match. Without annotations,
task/object IDs remain usable; the renderer never guesses a layer or microbatch
from an opaque ID. Recompute prefixes use the paired backward profiles from
the program itself, including for models other than OLMoE.

Optional `--metadata` takes a JSON object with `ordering`, `geometry`,
`microbatches`, `resolution`, and `physical_budget_gib` for the header. A
collection's `winner.json` is suitable. When a schedule digest is supplied,
it must match the plan. `--title` supplies the workload name independently.

```sh
PLAN=runs/example/program_0000.plan.json
SEMANTICS=runs/example/semantic_labels.json

# Verify every placement offset and export evidence for all actions.
python -m tools.pressurefit_trace.plan analyze "$PLAN" --semantics "$SEMANTICS" \
    --output runs/example/analysis

# Review scene compositions before spending time encoding a movie.
python -m tools.pressurefit_trace.plan gallery "$PLAN" --semantics "$SEMANTICS" \
    --video emission-placement --resolution 1080p \
    --output runs/example/review

python -m tools.pressurefit_trace.plan render "$PLAN" --semantics "$SEMANTICS" \
    --title 'Example model' --video emission-placement \
    --output runs/example/emission-placement.mp4

python -m tools.pressurefit_trace.plan render "$PLAN" --semantics "$SEMANTICS" \
    --title 'Example model' --video execution \
    --output runs/example/execution.mp4
```

Both movies default to **3840 x 2160**, H.264 **CRF 14**, and 30 output FPS.
`--quality preview|high|best` sets CRF 26/18/14, respectively; it changes
compression, not dimensions or evidence. `--resolution 720p|1080p|1440p|4k`
controls dimensions. Motion is sampled at 15 states/second by default and
repeated into the output stream; `--motion-fps 30 --fps 30` gives 30 distinct
states/second. Explanations hold for reading rather than moving continuously.

## Examples, timing, and camera

Automatic emission examples show a clean release with a later fetch, an eviction,
and a fetch trigger. Execution examples select **exposed compute stalls**:
an input delayed by the fetch queue, evictions holding addresses needed by a task,
or an eviction preventing a required prefetch. A queue wait that is entirely
hidden behind useful computation is not a detailed example. Examples prefer
transfers at least 1 MiB; every byte still counts in backlog, occupancy, and
placement. A program without these cases has fewer detailed examples.

All commands accept the following input/output arguments:

| Argument | Meaning | Default |
| --- | --- | --- |
| `plan` | Saved `AnnotatedProgramPlan` JSON; positional argument | Required |
| `--output` | Directory for `analyze`/`gallery`, PNG path for `frame`, MP4 path for `render` | Required |
| `--semantics` | Semantic-label JSON with the same program digest | Generic object/task labels |
| `--metadata` | Winner/context JSON for budget, ordering, geometry, and resolution | No extra header metadata |

Rendering controls apply to `frame`, `gallery`, and `render` unless a command
is named explicitly. `analyze` exports all evidence without selecting examples.

| Argument | Meaning | Default |
| --- | --- | --- |
| `--video emission-placement\|execution` | Choose the video composition | `emission-placement` |
| `--title` | Workload name in the header | `PressureFit` |
| `--resolution 720p\|1080p\|1440p\|4k` | 16:9 PNG/movie dimensions | `4k` (3840 × 2160) |
| `--scene` | `frame`: 1-based storyboard scene, after any chapter filtering | 1 |
| `--at` | `frame`: fraction through that scene, between 0 and 1 | 0.5 |
| `--fps` | `render`: output frames per second | 30 |
| `--motion-fps` | `render`: distinct animation states per second; must not exceed `--fps` | 15 |
| `--quality preview\|high\|best` | `render`: H.264 CRF 26/18/14; independent of resolution | `best` |

Example selection, timing, and camera arguments:

| Argument | Meaning | Default |
|---|---|---|
| `--actions 48,51,200` | Emission examples: schedule actions, 1-based, in presentation order | Automatic |
| `--tasks 10,58,103` | Execution examples: selected task positions with recorded stalls, 1-based | Automatic |
| `--placements 1,2,100` | Detailed insertion ranks in the verified placement order, 1-based | Automatic |
| `--explain-seconds` | Hold each explanation | 12 |
| `--motion-seconds` | Follow an emission example; execution examples use twice this | 10 |
| `--build-seconds` | Final accelerated placement construction | 30 |
| `--sweep-seconds` | Whole-program sliding sweep | 40 |
| `--hold-seconds` | Introduction, transition, and final holds | 6 |
| `--window-tasks` | Approximate tasks in a sliding focus window | 8 |
| `--chapter emission\|placement\|all` | Render a chapter of the first video on its own | `all` |

Longer placement gaps before detailed examples get additional brief accelerated
passes. Actual duration therefore depends on the examples. The JSON storyboard
records every scene and its duration. Horizontal coordinates are simulated
elapsed seconds, including waits; they are not the cut video's summed-compute
axis. Sliding sweeps keep a fixed time scale, sized from `--window-tasks` and
the average task duration including waits. Detailed examples use a fitted
window spanning the relevant triggers and dependencies. Placement uses the
certificate's predicted allocation lifetimes.

```sh
# One exact frame, halfway through storyboard scene 3.
python -m tools.pressurefit_trace.plan frame "$PLAN" --semantics "$SEMANTICS" \
    --video execution --scene 3 --at 0.5 --output runs/example/detail.png

# Short preview using the identical plan and example selection.
python -m tools.pressurefit_trace.plan render "$PLAN" --semantics "$SEMANTICS" \
    --video execution --resolution 720p --quality preview \
    --explain-seconds 2 --motion-seconds 2 --sweep-seconds 5 --hold-seconds 1 \
    --output runs/example/preview.mp4
```

## What the drawings mean

**Actions.** The final emitted schedule supplies the action and trigger task.
Boundary position `p` is immediately before selected task `p`; an action
triggered after task `i` is therefore shown at `B(i+1)`, matching the cut
renderer. Required-use bands come from task declarations. Solid allocation
bands come from the physical certificate, so they include destination holding
and source retention through eviction completion. Clean release, eviction,
and a subsequent fetch remain distinct.

The legal fetch window is reconstructed from the preceding departure and next
read. The selected trigger is the **final** trigger, including any repair.
The video does not claim to recover the sequence of earlier fetch-rule
proposals or repair moves; the annotated plan does not save that history.

**Placement.** Every rectangle is a physical allocation, not necessarily a
whole object lifetime. Its height and offset are exact bytes; its width is
the saved predicted lifetime. Several rectangles can name the same object.
The construction is replayed with the shipped C placer, respecting resident
slice exclusions, alignment, input-order tie breaks, and half-open lifetimes.
Every offset and the final span must equal the saved certificate before a
video is allowed. Focused explanations independently recover the union of
blocking address ranges and the lowest aligned fit. Numbers are insertion
ranks, not execution order. Small allocations keep their true addresses.

The final live-allocation chart counts the certificate's fixed allocations
throughout the step, including workspace and optimizer state. It is the same
attribution as the HTML occupancy view. Caller-owned dynamic/scratch reserves
are recorded separately in the certificate's required bytes; they do not have
invented allocation addresses. Peak live bytes and layout span are distinct;
the renderer does not assert the greedy packing is optimal.

**Execution and backlog.** Detailed scenes start from a waiting task. Required
fetches are matched by input alias; storage dependencies come from the saved
simulation admission edges. An eviction is labeled as holding output or task
workspace addresses only when the physical allocation ranges support that
attribution. The last blocking completion is connected to the task's start.
Other transfers are dimmed, even when they have the same role/color as an input.
Same-lane predecessors are distinguished from the task's actual dependencies.
Where applicable, analysis also identifies copies needed only by later tasks.

Transfer bars use the saved simulator's start/end timestamps. For sharded
transfers, each bar covers the complete recipe, including collective or zeroing
phases. Saved simulation intervals do not expose individual phase timestamps,
so the renderer does not invent copy/collective boundaries. A dashed connector
and boundary label show the last blocker's trigger-to-start wait. **Displayed
backlog is byte-based**: GiB histories and byte-proportional role stacks, with
a shared scale for both lanes. Counts remain available only in analysis.
Pending bytes count the CPU-copy payload of transfers whose triggers have fired
but whose recipes have not started; active recipes are excluded. A sharded
transfer therefore contributes its owned CPU shard, while GPU occupancy,
placement, and task input sizes still count the full GPU object. Action details
show both sizes when they differ. This is an
outstanding-transfer backlog, including admission/dependency waits where
present, not a measurement of the runtime worker's queue length. Saved stall
reasons distinguish `lane-busy` from other causes. No separate successful
enqueue timestamp is invented. The waiting charts retain every event, even
very short peaks. At simultaneous starts/ends, half-open accounting prevents
double-counting.

Saved stall flags are aggregate reasons, not a timed partition of the wait.
The renderer does not invent a per-reason duration or an eviction dependency
from a nearby transfer. A capacity wait without a saved causal edge is labeled
as capacity waiting rather than attributed to an arbitrary object.

Role colors match the cut-trajectory renderer, including violet optimizer
state and pink **Task workspace**. Lime marks recomputation within backward.

Compute hatching shows recorded ready-to-start waiting. Recompute prefixes
use `max(0, recompute_backward - save_backward)` and are a profile-derived
duration decomposition, not a separately timed kernel. All tasks and transfers
use one simulated clock. Measured GPU execution is a future input adapter;
these movies are explicitly labeled simulated.

## Outputs and organization

`analyze` writes `analysis.json`, `actions.jsonl`, `task_waits.jsonl`, and
`placement_order.jsonl`. JSONL task/action indices are zero-based; placement numbers
and CLI selectors are one-based. Analysis retains full semantic labels.

`render` writes the MP4, a JSON sidecar containing source digests, accounting
definitions, scene timestamps, frame counts and encoding settings, an ffmpeg
log, and `<movie>_frames/` with a review still for every scene. `gallery`
writes midpoint stills and `storyboard.json`. `frame` writes a PNG and the
exact scene state/complete storyboard beside it.

- `data.py`: validation, labels, action/transfer matching, exact interval sums.
- `waits.py`: task input blockers, admission reuse edges, and queue predecessors.
- `placement.py`: verified placement order, conflicts, alignment, and reuse.
- `storyboard.py`: example selection, timing, and sliding camera.
- `render_emission_placement.py`: release/evict/fetch scenes and placement construction.
- `render_execution.py`: task stalls, compute/fetch/evict timelines, and byte backlog.
- `render_common.py`: shared canvas, task strips, lanes, text spacing, and placement raster.
- `render.py`: renderer selection and MP4 encoding; `../render.py` renders residency cuts.
- `../style.py`: shared palette and duration labels for the three video types.
- `__main__.py`: the public source-tree CLI.

Validation:

```sh
python -m pytest tests/tools/test_plan_video.py
ruff check src/tools/pressurefit_trace/plan tests/tools/test_plan_video.py
```
