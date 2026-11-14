# Video library

`library.py` builds one offline catalog for published movies from multiple runs.
It produces a searchable HTML index, README results tables, CSV/JSON exports,
and a video page plus README for every configuration. It also indexes a running
matrix, including queued movies and infeasible cases. It does not render or move
movies and does not change the saved plans.

```sh
python -m tools.pressurefit_trace.video.library runs/library_sources.json \
    --output "$HOME/Desktop/Example videos"

# Refresh every 30 seconds until all matrix movies reach a final status.
python -m tools.pressurefit_trace.video.library runs/library_sources.json \
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
