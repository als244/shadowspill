"""Draw the memory panel without changing replay state."""

from __future__ import annotations

import numpy as np
from matplotlib.ticker import MaxNLocator

from ....data import GIB
from ...style import GREEN, MUTED, RED, WHITE


def draw_memory(self, view, rid, count, pressure, capacity):
    d = self.data
    position = view.position
    self.style(self.pressure)
    composition = self.composition.display(rid, count, pressure, capacity)
    displayed_pressure = composition.sum(axis=0)
    p = displayed_pressure / GIB
    c = np.full_like(p, self.composition.flat_budget / GIB)
    bottom = np.zeros(len(d.edges))
    for index, _label, color in self.composition.legend():
        top = bottom + composition[index] / GIB
        self.pressure.fill_between(
            d.edges, bottom, top, step="post", color=color, linewidth=0
        )
        bottom = top
    self.pressure.plot(d.edges, p, color=WHITE, lw=0.75, drawstyle="steps-post")
    self.pressure.plot(
        d.edges,
        np.where(p > c, p, np.nan),
        color=RED,
        lw=1,
        drawstyle="steps-post",
    )
    self.pressure.plot(
        d.edges, c, color=WHITE, lw=1, ls=(0, (4, 3)), drawstyle="steps-post"
    )
    self.pressure.axvline(d.edges[position], color=WHITE, lw=0.8, alpha=0.65)
    self.pressure.set_xlim(0, d.edges[-1])
    ceiling = (
        max(float(p.max()), float(c.max()), 1e-9) * 1.08
        if self.pressure_scale == "adaptive"
        else self.pressure_max
    )
    self.pressure.set_ylim(0, ceiling)
    self.pressure.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=3))
    legend = self.composition.legend()
    for index, (_, label, color) in enumerate(legend):
        self.label(
            0.255 + index * min(0.105, 0.525 / max(len(legend), 1)),
            0.237,
            "● " + label,
            fontsize=7.8,
            color=color,
        )
    self.pressure.set_ylabel("GiB", fontsize=9, labelpad=4)
    self.pressure.set_xlabel(
        "Task index" if d.axis == "task" else "Profiled compute time (s)",
        fontsize=9,
    )
    self.pressure.grid(axis="y", alpha=0.07)
    self.label(0.035, 0.217, "CURRENT PEAK", fontsize=9, color=MUTED)
    self.label(0.035, 0.182, f"{p.max():.2f} GiB", fontsize=21, color=WHITE)
    excess = max(0, float(np.max(p - c)))
    self.label(0.035, 0.142, "WORST EXCESS", fontsize=9, color=MUTED)
    self.label(
        0.035,
        0.108,
        f"{excess:.2f} GiB",
        fontsize=21,
        color=RED if excess else GREEN,
    )
    return composition, displayed_pressure, ceiling
