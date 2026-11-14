"""Draw the tasks panel without changing replay state."""

from __future__ import annotations

import numpy as np

from ....data import short_task
from ...style import BG, MUTED, PANEL, RECOMPUTE, WHITE
from ..geometry import free_label_intervals, recompute_endpoint, spaced_labels


def draw_tasks(self, view):
    d = self.data
    next_use = view.next_use
    departure = view.departure
    journey = view.journey
    position = view.position
    lo = view.lo
    hi = view.hi
    xmin = view.xmin
    xmax = view.xmax
    prior_use = view.prior_use
    focus = view.focus
    local = view.local
    self.style(self.tasks, ticks=False)
    self.tasks.set_xlim(xmin, xmax)
    self.tasks.set_ylim(0, 1)
    strip_points = self.tasks.get_position().width * self.fig.get_figwidth() * 72
    occupied, narrow = [], []
    task_labels = set(range(lo, hi))
    wide_journey = journey and hi - lo > 12
    if wide_journey:
        task_labels = {i for i in (prior_use, next_use, position) if i is not None}
    for i in range(lo, hi):
        color = "#97512b" if d.tasks[i].get("phase") == "forward" else "#553c53"
        self.tasks.axvspan(
            d.edges[i],
            d.edges[i + 1],
            facecolor=color,
            edgecolor=BG,
            lw=0.35 if journey else 1,
        )
        recompute_end = recompute_endpoint(d, i)
        if recompute_end > d.edges[i]:
            # Floating labels for nearby tiny tasks must not cover the
            # recompute interval or its R marker.
            if i in task_labels and recompute_end > xmin and d.edges[i] < xmax:
                occupied.append(
                    (
                        max(0, (d.edges[i] - xmin) / (xmax - xmin)),
                        min(1, (recompute_end - xmin) / (xmax - xmin)),
                    )
                )
            self.tasks.axvspan(
                d.edges[i],
                recompute_end,
                facecolor=RECOMPUTE,
                edgecolor=BG,
                lw=0.35 if journey else 1,
            )
            if i in task_labels and recompute_end > xmin and d.edges[i] < xmax:
                name = short_task(d.tasks[i], phase="recompute")
                label = name.replace(" · ", "\n")
                midpoint = (d.edges[i] + recompute_end) / 2
                center = (midpoint - xmin) / (xmax - xmin)
                label_width = (
                    3.5 * max(map(len, label.splitlines())) + 4
                ) / strip_points
                if (
                    not wide_journey
                    and (recompute_end - d.edges[i]) / (xmax - xmin) >= label_width
                ):
                    self.tasks.text(
                        midpoint,
                        0.5,
                        label,
                        ha="center",
                        va="center",
                        fontsize=7,
                        color=BG,
                        clip_on=True,
                    )
                else:
                    narrow.append((center, label, label_width))
        if i not in task_labels or d.edges[i + 1] < xmin or d.edges[i] > xmax:
            continue
        name = short_task(d.tasks[i])
        label = f"{i}\n{name}" if d.axis == "task" else name
        label_midpoint = (recompute_end + d.edges[i + 1]) / 2
        center = (label_midpoint - xmin) / (xmax - xmin)
        label_width = (4 * max(map(len, label.splitlines())) + 6) / strip_points
        if (
            not wide_journey
            and (d.edges[i + 1] - recompute_end) / (xmax - xmin) >= label_width
        ):
            self.tasks.text(
                label_midpoint,
                0.5,
                label,
                ha="center",
                va="center",
                fontsize=8,
                clip_on=True,
            )
            occupied.append((center - label_width / 2, center + label_width / 2))
        else:
            narrow.append((center, label, label_width))
            occupied.append((center - 8 / strip_points, center + 8 / strip_points))
    ordered_centers, label_scale = None, 1.0
    if wide_journey and narrow:
        # Only the prior use, focused task, and next use are labeled here.
        # Keep their R/B labels in time order even at the viewport edges.
        widths = np.asarray([item[2] for item in narrow])
        gap = 5 / strip_points
        label_scale = min(1, 1 / (widths.sum() + gap * (len(widths) - 1)))
        ordered_centers = spaced_labels(
            [item[0] for item in narrow], widths * label_scale, gap * label_scale
        )
    for label_index, (center, label, label_width) in enumerate(narrow):
        gaps = free_label_intervals(occupied)
        fits = [(a, b) for a, b in gaps if b - a >= label_width]
        if ordered_centers is not None:
            label_center = ordered_centers[label_index]
            label_width *= label_scale
            size = 8 * label_scale
        elif fits:
            label_center = min(
                (
                    np.clip(center, a + label_width / 2, b - label_width / 2)
                    for a, b in fits
                ),
                key=lambda x: abs(x - center),
            )
            size = 8
        else:
            a, b = max(gaps or [(0, 1)], key=lambda ab: ab[1] - ab[0])
            label_center = (a + b) / 2
            size = 8 * min(1, (b - a) / label_width)
            label_width = min(label_width, b - a)
        occupied.append(
            (label_center - label_width / 2, label_center + label_width / 2)
        )
        target = xmin + center * (xmax - xmin)
        self.tasks.plot([target, target], [0.25, 0.75], color=WHITE, lw=0.8)
        label_x = xmin + label_center * (xmax - xmin)
        # Route long leaders below the text, so they cannot strike through
        # another task name when several short tasks share a boundary.
        self.tasks.plot(
            [label_x, label_x, target, target],
            [0.27, 0.12, 0.12, 0.30],
            color=WHITE,
            lw=0.7,
        )
        self.tasks.text(
            label_x,
            0.5,
            label,
            ha="center",
            va="center",
            fontsize=size,
            color=WHITE,
            bbox={
                "boxstyle": "round,pad=0.18",
                "fc": PANEL,
                "ec": MUTED,
                "lw": 0.5,
            },
        )
    # Labels can move slightly to avoid collisions; ticks and their leaders
    # retain the exact boundary positions, including zero-duration tasks.
    indices = list(range(lo, hi + 1))
    if journey and hi - lo > 12:
        indices = sorted({lo, hi, departure, next_use, position} | task_labels)
    if journey:
        indices = [i for i in indices if xmin <= d.edges[i] <= xmax]
    widths = np.asarray([(4 * len(f"b{i}") + 4) / strip_points for i in indices])
    gap = 5 / strip_points
    scale = min(1, 1 / (widths.sum() + gap * (len(widths) - 1)))
    anchors = (d.edges[indices] - xmin) / (xmax - xmin)
    centers = spaced_labels(anchors, widths * scale, gap * scale)
    for i, center in zip(indices, centers, strict=True):
        x = xmin + center * (xmax - xmin)
        self.tasks.plot(
            [d.edges[i], d.edges[i], x],
            [1, 1.045, 1.13],
            color=MUTED,
            lw=0.65,
            clip_on=False,
        )
        self.tasks.text(
            x,
            1.13,
            f"b{i}",
            ha="center",
            va="bottom",
            fontsize=8 * scale,
            color=MUTED,
            clip_on=False,
        )
    self.label(
        0.255,
        0.623,
        (
            "OBJECT TIMELINE  ·  prior use to next use"
            if journey
            else f"BOUNDARY {position}  ·  local task window  ·  "
            f"{len(focus)} of {len(local)} relevant aliases"
        ),
        fontsize=10,
        weight="bold",
    )
    self.label(0.035, 0.566, "FOCUS", fontsize=9, color=MUTED)
    self.label(0.255, 0.580, "", fontsize=9)
