# Batch rendering

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
python -m tools.pressurefit_trace.video.matrix prepare runs/example/plans \
    --dump runs/example/dump --output runs/example/matrix --workers 4

python -m tools.pressurefit_trace.video.batch runs/example/matrix \
    --output runs/example/matrix_videos --jobs 6 \
    --title 'Example model · Measured bandwidth' \
    --resolution 4k --quality best --fps 30 --motion-fps 30 \
    --publish-to "$HOME/Desktop/Example videos"
```

All three omit story frames and detailed examples by default. Add `--details`
to the batch command to include them.

Both commands resume completed work. `prepare` checks the saved schedules and
simulated makespans against the source diagnostics. `render` records the renderer
source hashes and options and refuses to mix changed configurations in one output
directory. Use a fresh output directory for a different render configuration.
Jobs run in separate processes; on Linux they divide the available CPU affinity.
A failed movie does not stop the other jobs, and a later invocation retries it.

The default output includes continuous cut trajectories, emission/placement, and execution,
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
| `render` | `--details` | Add story frames and detailed examples to all three videos; default off. |
| `render` | `--videos` | One or more of `cuts`, `emission-placement`, `execution`; default all three. |
| `render` | `--jobs` | Concurrent video processes, default `4`. |
| `render` | `--title` | Workload title; each video's ordering, geometry and resolution remain visible. |
| `render` | `--resolution`, `--quality` | Default `4k`, `best`; same choices as the individual renderers. |
| `render` | `--fps`, `--motion-fps` | Output rate and distinct plan-animation states per second; default `30` each. |
| `render` | `--emission-timing` | Explain, motion, build, sweep, hold seconds; default `15,12,30,50,8`. |
| `render` | `--execution-timing` | The same five durations; default `18,15,30,60,8`. |
| `render` | `--cut-options-json` | Extra cut-render CLI settings as a JSON object, for example `'{"explain-seconds":20,"detail-at":"1,50,200"}'`. Defaults to continuous cut playback with no detailed examples. |

## Completed quickstart winners

Read the saved Plans and completed results directly, then render all three views.
The collector checks that every logical rank completed the same Plan. Execution
and placement videos use its exact schedule and physical certificate; cut videos
replay its selected search candidate on CPU. No export hooks or model imports are
needed. The cut replay must reproduce the saved schedule and timing; a mismatch
stops the export. `--rank` labels a representative rank of the symmetric program.

```sh
python -m tools.pressurefit_trace.video.winners RUN/quickstart \
  --output RUN/winner-videos-inputs --rank 0 --workload 'Example model'
python -m tools.pressurefit_trace.video.batch RUN/winner-videos-inputs \
  --output RUN/videos --publish-to '/path/to/Video Library/Example model' \
  --resolution 4k --quality best --jobs 2
```
