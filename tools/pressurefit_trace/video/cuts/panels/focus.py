"""Draw the focus panel without changing replay state."""

from __future__ import annotations

import textwrap

import numpy as np
from matplotlib.patches import Rectangle

from ....data import ROLE_COLORS, byte_label, short_task
from ...style import MUTED, PANEL, RED, WHITE, duration_label


def draw_focus(self, view, spans, event):
    d = self.data
    row = view.row
    selected = view.selected
    next_use = view.next_use
    departure = view.departure
    highlight_cut = view.highlight_cut
    journey = view.journey
    applied = view.applied
    removed = view.removed
    position = view.position
    lo = view.lo
    hi = view.hi
    xmin = view.xmin
    xmax = view.xmax
    prior_use = view.prior_use
    focus = view.focus
    self.style(self.focus, ticks=False)
    self.focus.set_xlim(xmin, xmax)
    weights = np.asarray(
        [1.8 if a == selected and highlight_cut else 1.0 for a in focus]
    )
    centers = np.cumsum(weights) - weights / 2
    self.focus.set_ylim(max(float(weights.sum()), 1), 0)
    for y, weight, alias in zip(centers, weights, focus, strict=True):
        meta = d.aliases[alias]
        color = ROLE_COLORS.get(meta["role"], MUTED)
        # The overview preserves byte proportions. Expand the focus rows
        # uniformly so required-use patterns remain legible beside a much
        # larger alias; explicit byte labels retain the size information.
        height = 0.85 if weight > 1 else 0.62
        if alias == selected:
            self.focus.axhspan(
                y - weight / 2 + 0.03,
                y + weight / 2 - 0.03,
                color=RED if highlight_cut else WHITE,
                alpha=0.07,
            )
        for a, b in d.intervals(alias, spans[alias]):
            self.focus.add_patch(
                Rectangle(
                    (a, y - height / 2),
                    b - a,
                    height,
                    facecolor=color,
                    alpha=0.70,
                    linewidth=0,
                )
            )
        for task in d.uses[alias]:
            if lo <= task < hi:
                a, b = d.edges[task : task + 2]
                self.focus.add_patch(
                    Rectangle(
                        (a, y - height / 2),
                        b - a,
                        height,
                        facecolor=color,
                        edgecolor=WHITE,
                        hatch="////",
                        linewidth=0.25,
                        alpha=0.9,
                    )
                )
        name = d.focus_label(alias, position).removeprefix(meta["role"] + " ")
        name = textwrap.shorten(name, width=44, placeholder="…")
        chosen_cut = alias == selected and highlight_cut
        figure_y = 0.31 + 0.24 * (1 - y / max(weights.sum(), 1))
        self.label(
            0.244,
            figure_y + (0.018 if chosen_cut else 0.007),
            name,
            fontsize=10,
            ha="right",
            va="center",
            color=WHITE if alias == selected else color,
            weight="bold" if alias == selected else "normal",
        )
        action = ("EVICT" if int(row[8]) else "RELEASE") if chosen_cut else ""
        if chosen_cut and applied:
            action += "ED" if int(row[8]) else "D"
        self.label(
            0.244,
            figure_y if chosen_cut else figure_y - 0.009,
            (
                f"{action} at b{departure}  ·  cut {event + 1:,}  ·  "
                if chosen_cut
                else ""
            )
            + byte_label(int(d.trace.alias_sizes[alias])),
            fontsize=9 if chosen_cut else 7.8,
            ha="right",
            va="center",
            color=RED if chosen_cut else MUTED,
        )
        if chosen_cut:
            next_label = "No later task use"
            if next_use is not None:
                wait = duration_label(d.gap_duration_ns(row, next_use))
                next_label = (
                    f"Next: b{next_use} · {short_task(d.tasks[next_use])} · "
                    f"+{wait} compute"
                )
            self.label(
                0.244,
                figure_y - 0.018,
                next_label,
                fontsize=8.5,
                ha="right",
                va="center",
                color=MUTED,
            )
            for a, b in removed:
                if b <= xmin or a >= xmax:
                    continue
                # Keep the exact gap outlined after the temporary red
                # residency ghost fades. The marker also survives on PNGs
                # and on the last decision of an accelerated batch.
                self.focus.add_patch(
                    Rectangle(
                        (a, y - height / 2),
                        b - a,
                        height,
                        facecolor="none" if applied else RED,
                        edgecolor=RED,
                        alpha=0.95 if applied else 0.45,
                        linewidth=1.5,
                        linestyle=(0, (3, 2)) if applied else "solid",
                        zorder=6,
                    )
                )
                self.focus.scatter(
                    [(max(a, xmin) + min(b, xmax)) / 2],
                    [y - height / 2 - 0.04],
                    s=28,
                    marker="v",
                    color=RED,
                    zorder=8,
                )
                if not applied:
                    continue
                patch = Rectangle(
                    (a, y - height / 2),
                    b - a,
                    height,
                    facecolor=RED,
                    edgecolor=RED,
                    alpha=0.55,
                    linewidth=0,
                    hatch="xx",
                    visible=False,
                    zorder=5,
                )
                self.focus.add_patch(patch)
                self.flashes.append(patch)
            if not removed:
                # A logical split can change headroom without removing a
                # positive-duration task interval. Mark its boundary; do
                # not draw a fabricated gap through required use.
                x = d.edges[departure]
                self.focus.plot(
                    [x, x], [y - 0.38, y + 0.38], color=RED, lw=2.5, zorder=8
                )
                self.focus.scatter(
                    [x], [y - 0.38], s=28, marker="v", color=RED, zorder=8
                )
    self.focus.axvline(d.edges[position], color=WHITE, lw=1, alpha=0.8)
    if journey:
        self.focus.axvline(d.edges[departure], color=RED, lw=1.2, alpha=0.9)
        for task, title, x, align in (
            (prior_use, "Prior use", 0.01, "left"),
            (next_use, "Next use", 0.99, "right"),
        ):
            if task is None:
                continue
            self.focus.annotate(
                f"{title}: b{task} · {short_task(d.tasks[task])}",
                xy=(float(np.mean(d.edges[task : task + 2])), 0.9),
                xytext=(x, 0.89),
                textcoords="axes fraction",
                ha=align,
                fontsize=10,
                color=WHITE,
                arrowprops={"arrowstyle": "->", "color": MUTED, "lw": 0.8},
                bbox={"facecolor": PANEL, "edgecolor": "none", "pad": 2},
                annotation_clip=True,
            )
        self.focus.text(
            0.5,
            0.13,
            f"{'Eviction' if int(row[8]) else 'Release'} b{departure} → "
            f"next use b{next_use}   ·   "
            f"{next_use - departure:,} intervening tasks   ·   "
            f"{duration_label(d.gap_duration_ns(row, next_use))} compute",
            transform=self.focus.transAxes,
            ha="center",
            fontsize=10,
            color=WHITE,
            bbox={"facecolor": PANEL, "edgecolor": "none", "pad": 3},
            zorder=10,
        )
    ticks = np.linspace(xmin, xmax, 4)
    if d.axis == "task":
        ticks = np.unique(ticks.astype(int))
    self.focus.set_xticks(ticks)
    self.focus.set_xticklabels(
        [str(int(x)) if d.axis == "task" else f"{x:.3f} s" for x in ticks]
    )
    self.label(
        0.255,
        0.278,
        "Solid = resident    //// = required during task    "
        "Focus rows expanded for readability",
        fontsize=8,
        color=MUTED,
    )
