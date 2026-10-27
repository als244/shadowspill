# Planner videos

All video code lives here. Capture and analysis remain in the [parent tool](../README.md).
The default is continuous playback without story cards or detailed examples.
Add `--details` to any renderer or the batch command to include them.

Run from a source checkout with the project environment; rendering uses CPU, NumPy,
Matplotlib and `ffmpeg`. Outputs belong in an ignored run directory.

| View | Input | Implementation | Guide |
| --- | --- | --- | --- |
| Residency cuts | Raw PressureFit trace | `cuts/` | [Cut videos](cuts/README.md) |
| Emission and placement | Saved annotated plan | `plan/render_emission_placement.py` | [Plan videos](plan/README.md) |
| Execution timeline | Saved annotated plan | `plan/render_execution.py` | [Plan videos](plan/README.md) |

```sh
python -m tools.pressurefit_trace.video render TRACE --output runs/cuts.mp4
python -m tools.pressurefit_trace.video.plan render PLAN --video execution --output runs/execution.mp4
python -m tools.pressurefit_trace.video.batch MATRIX --output runs/videos
```

- `encoding.py`: shared encoder process and H.264 settings.
- `style.py`: shared palette and duration labels.
- `matrix.py`: prepare cut traces and certified plans for every resolution.
- `winners/`: prepare video inputs from completed quickstart Plans.
- `batch.py`: resumable rendering, decode checks and verified publication.
- `library.py`, `library_html.py`, `library_assets/`: offline catalog and player pages.

See [batch rendering](batch.md) and the [video library](library.md) for larger collections.
Video rendering is verified manually when changed; it is excluded from repository tests.
