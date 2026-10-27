"""Render a selected exact replay state and its metadata."""

import json

import matplotlib.pyplot as plt

from .canvas import Renderer
from .geometry import subtract_intervals


def render_frame(
    data,
    output,
    *,
    title=None,
    reduction,
    cut,
    width=1920,
    context=3,
    focus_rows=7,
    boundary=None,
    explain=False,
    journey=None,
    pressure_scale="adaptive",
    pressure_view="total",
    focus_mode="cut",
):
    state = data.trace.state(reduction, cut)
    event = min(max(cut - 1, 0), len(data.trace.reductions[reduction].cuts) - 1)
    if explain:
        if cut >= len(data.trace.reductions[reduction].cuts):
            raise ValueError("there is no next cut to explain in this final state")
        event = cut
    removed = ()
    if cut > 0 and not explain:
        before = data.trace.state(reduction, event)
        alias = int(data.trace.reductions[reduction].cuts[event, 4])
        removed = subtract_intervals(
            data.intervals(alias, before.spans[alias]),
            data.intervals(alias, state.spans[alias]),
        )
    renderer = Renderer(
        data,
        title=title,
        width=width,
        context=context,
        focus_rows=focus_rows,
        pressure_scale=pressure_scale,
        pressure_view=pressure_view,
        focus_mode=focus_mode,
    )
    renderer.draw(
        reduction,
        cut,
        state.spans,
        state.pressure,
        state.capacity,
        event=event if event >= 0 else None,
        phase="selected" if explain else "seed" if cut == 0 else "state",
        boundary=boundary,
        removed=removed,
        explain=explain,
        journey=journey,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    renderer.save(output)
    output.with_suffix(".json").write_text(
        json.dumps(renderer.last_record, indent=2) + "\n"
    )
    plt.close(renderer.fig)
