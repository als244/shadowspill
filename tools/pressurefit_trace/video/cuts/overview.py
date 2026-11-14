"""Incremental byte-weighted overview raster."""

from __future__ import annotations

import math

import numpy as np
from matplotlib.colors import to_rgb

from ...data import ROLE_COLORS, Trajectory
from ..style import MUTED, PANEL


class Overview:
    """Byte-weighted vertical raster; subpixel rows contribute their true area."""

    def __init__(self, data: Trajectory, spans, height=180, width=1100):
        self.data, self.height, self.width = data, height, width
        self.pixels = np.broadcast_to(to_rgb(PANEL), (height, width, 3)).copy()
        self.previous = {}
        self.weights = {}
        total = data.y_edges[-1] or 1
        for rank, alias in enumerate(data.order):
            start, end = data.y_edges[rank : rank + 2] * height / total
            rows = np.arange(int(start), min(height, math.ceil(end)))
            weights = np.minimum(rows + 1, end) - np.maximum(rows, start)
            self.weights[alias] = (rows, weights)
            required = [(data.edges[i], data.edges[i + 1]) for i in data.uses[alias]]
            self.paint(alias, required, 0.18)
        self.update(spans, data.order)

    def paint(self, alias, intervals, strength):
        rows, weights = self.weights[alias]
        color = np.asarray(
            to_rgb(ROLE_COLORS.get(self.data.aliases[alias]["role"], MUTED))
        )
        for start, end in intervals:
            left, right = (
                start / self.data.edges[-1] * self.width,
                end / self.data.edges[-1] * self.width,
            )
            cols = np.arange(max(0, int(left)), min(self.width, math.ceil(right)))
            if not len(cols):
                continue
            coverage = np.minimum(cols + 1, right) - np.maximum(cols, left)
            for y, weight in zip(rows, weights, strict=True):
                self.pixels[y, cols] += strength * weight * coverage[:, None] * color

    def update(self, spans, aliases):
        for alias in aliases:
            if self.data.y_rank[alias] < 0:
                continue
            if alias in self.previous:
                self.paint(alias, self.previous[alias], -0.64)
            intervals = self.data.intervals(alias, spans[alias])
            self.paint(alias, intervals, 0.64)
            self.previous[alias] = intervals

    def image(self):
        return np.clip(self.pixels, 0, 1)
