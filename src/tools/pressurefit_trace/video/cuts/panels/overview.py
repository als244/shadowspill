"""Draw the overview panel without changing replay state."""

from __future__ import annotations

import numpy as np

from ...style import BG, WHITE


def draw_overview(self, view):
    d = self.data
    selected = view.selected
    position = view.position
    xmin = view.xmin
    xmax = view.xmax
    self.style(self.over, ticks=False)
    self.over.imshow(
        self.overview.image(),
        aspect="auto",
        origin="upper",
        interpolation="nearest",
        extent=(0, d.edges[-1], d.y_edges[-1] or 1, 0),
    )
    self.over.axvspan(
        xmin, xmax, facecolor="white", alpha=0.08, edgecolor=WHITE, lw=0.9
    )
    self.over.axvline(d.edges[position], color=WHITE, lw=0.8, alpha=0.9)
    if selected is not None and d.y_rank[selected] >= 0:
        y = np.mean(d.y_edges[d.y_rank[selected] : d.y_rank[selected] + 2])
        self.over.scatter(
            [d.edges[position]], [y], s=35, facecolor=WHITE, edgecolor=BG, zorder=5
        )
