"""Exact intervals and collision-free task-label placement."""

from __future__ import annotations

import numpy as np

from shadowspill.diagnostics.pressurefit_trace import _cut


def recompute_endpoint(data, position):
    """The lime prefix occupies the same duration fraction on either axis."""
    task = data.tasks[position]
    duration = task["profiled_duration_ns"]
    extra = task.get("recompute_prefix_ns", 0)
    if task.get("phase") != "backward" or duration <= 0 or extra <= 0:
        return float(data.edges[position])
    fraction = extra / duration
    if fraction > 1:
        raise ValueError("recompute prefix exceeds the backward task")
    return float(
        data.edges[position]
        + fraction * (data.edges[position + 1] - data.edges[position])
    )


def spaced_labels(anchors, widths, gap):
    """Keep labels inside [0, 1], separated, and ordered like their anchors."""
    centers = np.clip(np.asarray(anchors, dtype=float), widths / 2, 1 - widths / 2)
    for i in range(1, len(centers)):
        centers[i] = max(
            centers[i], centers[i - 1] + (widths[i - 1] + widths[i]) / 2 + gap
        )
    centers[-1] = min(centers[-1], 1 - widths[-1] / 2)
    for i in range(len(centers) - 2, -1, -1):
        centers[i] = min(
            centers[i], centers[i + 1] - (widths[i] + widths[i + 1]) / 2 - gap
        )
    centers[0] = max(centers[0], widths[0] / 2)
    for i in range(1, len(centers)):
        centers[i] = max(
            centers[i], centers[i - 1] + (widths[i - 1] + widths[i]) / 2 + gap
        )
    return centers


def free_label_intervals(occupied):
    """Unoccupied horizontal space in the focus task strip."""
    intervals, end = [], 0.0
    for left, right in sorted(occupied):
        if left > end:
            intervals.append((end, min(left, 1.0)))
        end = max(end, right)
    if end < 1:
        intervals.append((end, 1.0))
    return intervals


def subtract_intervals(before, after):
    """The displayed idle-time intervals removed by a cut."""
    removed = []
    for left, right in before:
        pieces = [(left, right)]
        for start, end in after:
            pieces = [
                (a, b)
                for lo, hi in pieces
                for a, b in [(lo, min(hi, start)), (max(lo, end), hi)]
                if b > a
            ]
        removed.extend(pieces)
    return removed


def cut_intervals(data, spans, row):
    """Preview the recorded cut using replay semantics, without changing state."""
    alias = int(row[4])
    projected = list(spans)
    _cut(projected, row[3:15])
    return subtract_intervals(
        data.intervals(alias, spans[alias]),
        data.intervals(alias, projected[alias]),
    )
