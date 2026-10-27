"""Canvas setup and composition of independent residency-cut panels."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from ...data import GIB, ROLE_COLORS, Trajectory
from ...pressure import PressureComposition
from ..style import BG, MUTED, PANEL, RECOMPUTE, WHITE
from .overview import Overview
from .panels.details import draw_details
from .panels.focus import draw_focus
from .panels.labels import draw_labels
from .panels.memory import draw_memory
from .panels.overview import draw_overview
from .panels.tasks import draw_tasks
from .record import record_frame
from .view import focus_view


class Renderer:
    def __init__(
        self,
        data: Trajectory,
        *,
        title=None,
        width=1920,
        context=3,
        focus_rows=7,
        pressure_scale="adaptive",
        pressure_view="total",
        focus_mode="cut",
    ):
        self.data, self.context, self.focus_rows = data, context, focus_rows
        self.title = title
        self.pressure_scale = pressure_scale
        self.focus_mode = focus_mode
        self.composition = PressureComposition(
            data, include_resident=pressure_view == "total"
        )
        self.pressure_offset = self.composition.offset
        # STIX keeps small semantic labels readable and ships with Matplotlib;
        # Computer Modern gives the formulas their TeX appearance.
        plt.rcParams.update(
            {
                "font.family": "STIXGeneral",
                "mathtext.fontset": "cm",
                "text.color": WHITE,
                "axes.labelcolor": MUTED,
                "xtick.color": MUTED,
                "ytick.color": MUTED,
                "font.size": 10,
                "hatch.linewidth": 0.6,
            }
        )
        self.fig = plt.figure(figsize=(16, 9), dpi=width / 16, facecolor=BG)
        self.over = self.fig.add_axes((0.255, 0.655, 0.525, 0.145))
        self.focus = self.fig.add_axes((0.255, 0.31, 0.525, 0.24))
        self.tasks = self.fig.add_axes((0.255, 0.554, 0.525, 0.04))
        self.pressure = self.fig.add_axes((0.255, 0.08, 0.525, 0.15))
        self.side = self.fig.add_axes((0.813, 0.10, 0.165, 0.73))
        self.dynamic_text = []
        self.flashes = []
        self.overview = None
        self.rid = None
        self.last_record = None
        self.explanation = None
        self.pressure_max = (
            (
                self.composition.flat_budget
                + max(
                    0,
                    max(
                        int(
                            np.max(
                                state.pressure.astype(np.int64)
                                - state.capacity.astype(np.int64)
                            )
                        )
                        for state in (
                            data.trace.state(r, 0)
                            for r in range(len(data.trace.reductions))
                        )
                    ),
                )
            )
            / GIB
            * 1.08
        )
        self.fig.text(0.035, 0.955, "PressureFit", fontsize=25, weight="bold")
        self.fig.text(
            0.255, 0.958, "A trajectory of residency cuts", fontsize=16, color=MUTED
        )
        self.fig.add_artist(
            Line2D(
                [0.035, 0.975],
                [0.857, 0.857],
                transform=self.fig.transFigure,
                color=MUTED,
                lw=0.6,
                alpha=0.3,
            )
        )
        self.fig.text(0.255, 0.837, "FULL PROGRAM", fontsize=10, weight="bold")
        self.fig.text(
            0.035, 0.831, "One row per storage alias", fontsize=10, weight="bold"
        )
        self.fig.text(
            0.035,
            0.792,
            f"{len(data.order):,} visible aliases\n{len(data.tasks):,} total tasks",
            fontsize=11,
            color=MUTED,
            linespacing=1.45,
            va="top",
        )
        row_legend = [
            ("● Parameters", ROLE_COLORS["parameter"]),
            ("● Activations", ROLE_COLORS["activation"]),
            ("● Gradients", ROLE_COLORS["gradient"]),
        ]
        if any(a["role"] == "optimizer_state" for a in data.aliases):
            row_legend.append(("● Optimizer state", ROLE_COLORS["optimizer_state"]))
        row_legend.append(("● Other roles", MUTED))
        if any(task.get("recompute_prefix_ns", 0) > 0 for task in data.tasks):
            row_legend.append(("R  Recompute", RECOMPUTE))
        for i, (name, color) in enumerate(row_legend):
            self.fig.text(
                0.035 + i % 2 * 0.107,
                0.706 - i // 2 * 0.032,
                name,
                color=color,
                fontsize=10,
            )
        self.fig.text(
            0.255,
            0.252,
            "MEMORY PRESSURE",
            fontsize=10,
            weight="bold",
        )

    def label(self, x, y, content, **kwargs):
        artist = self.fig.text(x, y, content, **kwargs)
        self.dynamic_text.append(artist)
        return artist

    @staticmethod
    def style(ax, *, ticks=True):
        ax.clear()
        ax.set_facecolor(PANEL)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0, labelsize=8)
        if not ticks:
            ax.set_xticks([])
            ax.set_yticks([])

    def draw(
        self,
        rid,
        count,
        spans,
        pressure,
        capacity,
        *,
        event=None,
        phase="state",
        batch_start=None,
        boundary=None,
        removed=(),
        touched=(),
        explain=False,
        journey=None,
        zoom_progress=1.0,
    ):
        d = self.data
        if self.explanation is not None:
            self.explanation.remove()
            self.explanation = None
        for text in self.dynamic_text:
            text.remove()
        self.dynamic_text.clear()
        self.flashes = []
        self.focus.set_visible(not explain)
        self.tasks.set_visible(not explain)
        if self.rid != rid:
            self.rid = rid
            self.overview = Overview(d, spans)
        else:
            self.overview.update(spans, touched)
        view = focus_view(
            self,
            rid,
            count,
            spans,
            pressure,
            capacity,
            event=event,
            phase=phase,
            boundary=boundary,
            removed=removed,
            journey=journey,
            zoom_progress=zoom_progress,
        )
        reduction_context = draw_labels(
            self, view, rid, count, event, phase, batch_start
        )
        draw_overview(self, view)
        draw_tasks(self, view)
        draw_focus(self, view, spans, event)
        composition, displayed_pressure, ceiling = draw_memory(
            self, view, rid, count, pressure, capacity
        )
        current_task, current_memory = draw_details(
            self, view, rid, count, event, pressure, capacity
        )
        record_frame(
            self,
            view,
            rid,
            count,
            phase,
            event,
            batch_start,
            pressure,
            capacity,
            reduction_context,
            composition,
            displayed_pressure,
            ceiling,
            current_task,
            current_memory,
            zoom_progress,
        )
        if explain and event is not None:
            from .explanation import draw_explanation

            self.explanation = draw_explanation(
                self.fig, d, rid, event, pressure_budget=self.composition.flat_budget
            )
            self.last_record["explanation"] = True

    def rgba(self, flash=None):
        if flash is not None:
            for patch in self.flashes:
                patch.set_visible(flash > 1e-3)
                patch.set_alpha(0.55 * flash)
        self.fig.canvas.draw()
        return memoryview(self.fig.canvas.buffer_rgba()).cast("B")

    def save(self, path):
        self.fig.savefig(path, dpi=self.fig.dpi, facecolor=BG)
