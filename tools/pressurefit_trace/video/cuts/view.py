"""Resolve the selected cut and its local or object-journey viewport."""

from dataclasses import dataclass

import numpy as np

from .geometry import cut_intervals


@dataclass(frozen=True)
class FocusView:
    row: object
    selected: int | None
    next_use: int | None
    departure: int | None
    highlight_cut: bool
    journey: bool
    applied: bool
    removed: tuple
    position: int
    lo: int
    hi: int
    xmin: float
    xmax: float
    prior_use: int | None
    focus: list
    local: list


def focus_view(
    canvas,
    rid,
    count,
    spans,
    pressure,
    capacity,
    *,
    event,
    phase,
    boundary,
    removed,
    journey,
    zoom_progress,
):
    d = canvas.data
    row = None if event is None else d.trace.reductions[rid].cuts[event]
    selected = None if row is None else int(row[4])
    next_use = None if row is None else d.next_use(row)
    departure = None if row is None else d.departure_position(row)
    highlight_cut = row is not None and phase not in ("seed", "final")
    if journey is None:
        journey = highlight_cut and canvas.focus_mode == "cut"
    applied = row is not None and count > event
    if highlight_cut and not applied:
        removed = cut_intervals(d, spans, row)
    position = (
        boundary
        if boundary is not None
        else (
            int(row[0])
            if row is not None
            else int(np.argmax(pressure.astype(np.int64) - capacity.astype(np.int64)))
        )
    )
    lo, hi = (
        max(0, position - canvas.context),
        min(len(d.tasks), position + canvas.context + 1),
    )
    xmin, xmax = d.edges[lo], d.edges[hi]
    prior_use = None
    journey = journey and selected is not None and next_use is not None
    if journey:
        prior_use = max((i for i in d.uses[selected] if i < departure), default=None)
        wide_lo = max(0, (prior_use if prior_use is not None else departure) - 1)
        wide_hi = min(len(d.tasks), next_use + 2)
        xmin = (1 - zoom_progress) * xmin + zoom_progress * d.edges[wide_lo]
        xmax = (1 - zoom_progress) * xmax + zoom_progress * d.edges[wide_hi]
        lo, hi = wide_lo, wide_hi
    if xmin == xmax:
        xmin, xmax = max(0, xmin - 0.01), xmin + 0.01
    focus, local = d.focus_aliases(
        position, selected, canvas.context, canvas.focus_rows
    )
    if journey:
        focus = [selected]

    return FocusView(
        row=row,
        selected=selected,
        next_use=next_use,
        departure=departure,
        highlight_cut=highlight_cut,
        journey=journey,
        applied=applied,
        removed=removed,
        position=position,
        lo=lo,
        hi=hi,
        xmin=xmin,
        xmax=xmax,
        prior_use=prior_use,
        focus=focus,
        local=local,
    )
