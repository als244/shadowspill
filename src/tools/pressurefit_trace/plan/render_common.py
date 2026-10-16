"""Shared canvas, task/transfer lanes, typography, and placement raster."""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_rgba
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator

from shadowspill.diagnostics.occupancy import WORKSPACE

from ..data import GIB, ROLE_COLORS, byte_label, short_task
from ..pressure import ROLE_GROUPS
from ..style import BG, MUTED, PANEL, RECOMPUTE, RED, WHITE, duration_label

ROLE_COLOR = {
    **ROLE_COLORS,
    WORKSPACE: next(color for key, _, color in ROLE_GROUPS if key == "workspace"),
}
PHASE_COLOR = {
    "forward": "#dc642b",
    "backward": "#84533e",
    "optimizer": "#9655ff",
    "control": "#78889e",
}
ROLE_NAMES = {
    "parameter": "Parameters",
    "activation": "Activations",
    "gradient": "Gradients",
    "optimizer_state": "Optimizer state",
    "input": "Inputs",
    "unattributed": "Other roles",
    WORKSPACE: "Task workspace",
}
LANE_COLOR = {"fetch": "#6aa9ff", "evict": "#f0af60"}


def truncate(value, length=45):
    return value if len(value) <= length else value[: length - 1] + "…"


def boxes(
    ax, rectangles, colors, *, edgecolor="none", linewidth=0, alpha=None, zorder=2
):
    if not rectangles:
        return None
    vertices = [
        [(x, y), (x + w, y), (x + w, y + h), (x, y + h)] for x, y, w, h in rectangles
    ]
    collection = PolyCollection(
        vertices,
        facecolors=colors,
        edgecolors=edgecolor,
        linewidths=linewidth,
        alpha=alpha,
        zorder=zorder,
    )
    ax.add_collection(collection)
    return collection


class LayoutRaster:
    """Paint each insertion once at its exact time/address coordinates."""

    def __init__(self, data, replay, width):
        self.data, self.replay = data, replay
        self.fig = plt.figure(figsize=(10.08, 3.06), dpi=width / 16, facecolor=PANEL)
        self.ax = self.fig.add_axes((0, 0, 1, 1), facecolor=PANEL)
        self.ax.set_xlim(0, data.end_ns / 1e9)
        self.ax.set_ylim(0, data.pool_bytes / GIB)
        self.ax.set_axis_off()
        self.fig.canvas.draw()
        self.placed = 0

    def image(self, placed):
        if placed < self.placed:
            self.fig.canvas.draw()
            self.placed = 0
        ids = self.replay.order[self.placed : placed]
        r = self.replay
        collection = boxes(
            self.ax,
            [
                (
                    r.starts[i] / 1e9,
                    r.offsets[i] / GIB,
                    (r.ends[i] - r.starts[i]) / 1e9,
                    r.sizes[i] / GIB,
                )
                for i in ids
            ],
            [ROLE_COLOR.get(self.data.leases[i]["role"], MUTED) for i in ids],
            edgecolor=BG,
            linewidth=0.2,
        )
        if collection is not None:
            self.ax.draw_artist(collection)
            collection.remove()
        self.placed = placed
        return np.asarray(self.fig.canvas.buffer_rgba())


class PlanCanvas:
    """Shared drawing primitives; subclasses supply one video composition."""

    def __init__(self, data, replay, *, width=3840, title=None):
        self.data, self.replay = data, replay
        self.width = width
        self.title = title or "PressureFit"
        plt.rcParams.update(
            {
                "font.family": "STIXGeneral",
                "mathtext.fontset": "cm",
                "text.color": WHITE,
                "axes.labelcolor": MUTED,
                "xtick.color": MUTED,
                "ytick.color": MUTED,
                "font.size": 11,
                "hatch.linewidth": 0.7,
            }
        )
        self.fig = plt.figure(figsize=(16, 9), dpi=width / 16, facecolor=BG)
        self.detail_cache = {}
        self._lease_colors = [ROLE_COLOR.get(p["role"], MUTED) for p in data.leases]
        self.task_starts = np.array([t["start_ns"] for t in data.tasks])
        self.task_ends = np.array([t["end_ns"] for t in data.tasks])
        self.transfer_starts = np.array(
            [a["transfer"]["start_ns"] for a in data.transfers]
        )
        self.transfer_ends = np.array([a["transfer"]["end_ns"] for a in data.transfers])
        self.axes = []
        self.layout_raster = LayoutRaster(data, replay, width)

    def close(self):
        plt.close(self.fig)
        plt.close(self.layout_raster.fig)

    def axis(self, rect, *, xlim=None, ylim=None, ylabel=None, xlabel=None, ticks=True):
        ax = self.fig.add_axes(rect, facecolor=PANEL)
        self.axes.append(ax)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0, pad=7, labelsize=10)
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.xaxis.set_major_locator(MaxNLocator(6))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=11, labelpad=9)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=11, labelpad=8)
        if not ticks:
            ax.set_xticks([])
            ax.set_yticks([])
        return ax

    def text(self, x, y, value, *, size=13, color=WHITE, **kw):
        return self.fig.text(x, y, value, fontsize=size, color=color, **kw)

    def header(self, state):
        mode = (
            "Execution"
            if state["kind"].startswith("execution")
            else "Emission & placement"
        )
        self.text(0.045, 0.95, self.title, size=24)
        self.text(0.955, 0.95, mode, size=15, color=MUTED, ha="right")
        meta = self.data.metadata
        details = []
        if "physical_budget_gib" in meta:
            details.append(f"{meta['physical_budget_gib']:g} GiB device budget")
        if "ordering" in meta:
            details.append(f"ordering {meta['ordering']}")
        if "geometry" in meta:
            details.append(
                f"{meta['geometry']} seq/mb × {meta.get('microbatches', '?')} mb"  # noqa: RUF001
            )
        if "resolution" in meta:
            details.append(f"resolution {meta['resolution']}")
        details += [f"{len(self.data.tasks):,} tasks", self.data.policy]
        self.text(0.045, 0.91, "   ·   ".join(details), size=11, color=MUTED)
        self.text(0.07, 0.857, state["title"], size=19)
        self.text(0.955, 0.857, "Simulated", size=11, color=MUTED, ha="right")
        x = 0.07
        roles = [
            "parameter",
            "activation",
            "gradient",
            "optimizer_state",
        ]
        observed = {p["role"] for p in self.data.leases}
        if "input" in observed:
            roles.append("input")
        if observed - set(roles) - {WORKSPACE, "output"}:
            roles.append("unattributed")
        roles.append(WORKSPACE)
        for role in roles:
            self.fig.patches.append(
                Rectangle(
                    (x, 0.028),
                    0.008,
                    0.012,
                    transform=self.fig.transFigure,
                    facecolor=ROLE_COLOR[role],
                    edgecolor="none",
                )
            )
            self.text(x + 0.012, 0.028, ROLE_NAMES[role], size=10, color=MUTED)
            x += {
                "parameter": 0.114,
                "activation": 0.11,
                "gradient": 0.101,
                "optimizer_state": 0.143,
                "input": 0.08,
                "unattributed": 0.105,
                WORKSPACE: 0.13,
            }[role]

    def card(self, top=0.755):
        ax = self.axis(
            (0.75, 0.095, 0.225, top - 0.095), ylim=(0, 1), xlim=(0, 1), ticks=False
        )
        return ax

    @staticmethod
    def card_text(ax, y, value, *, size=13, color=WHITE, wrap=34, weight=None):
        wrap = min(wrap, max(18, int(33 * 13 / size)))
        renderer = ax.figure.canvas.get_renderer()
        font = FontProperties(family="STIXGeneral", size=size, weight=weight)

        def measured_width(text):
            return renderer.get_text_width_height_descent(text, font, False)[0]

        paragraphs = str(value).splitlines()
        while wrap > 8 and any(
            measured_width(line) > ax.bbox.width * 0.86
            for paragraph in paragraphs
            for line in textwrap.wrap(paragraph, wrap, break_on_hyphens=False)
        ):
            wrap -= 1
        lines = "\n".join(
            line
            for paragraph in paragraphs
            for line in textwrap.wrap(
                paragraph, wrap, break_long_words=True, break_on_hyphens=False
            )
        )
        if "$" in str(value):
            lines = str(value)
        label = ax.text(
            0.065,
            y,
            lines,
            transform=ax.transAxes,
            va="top",
            fontsize=size,
            color=color,
            linespacing=1.3,
            fontweight=weight,
            clip_on=True,
        )
        return y - label.get_window_extent(renderer).height / ax.bbox.height - 0.025

    def task_strip(self, window, *, rect=(0.07, 0.729, 0.63, 0.05), boundaries=True):
        lo, hi = window
        ax = self.axis(rect, xlim=(lo / 1e9, hi / 1e9), ylim=(0, 1), ticks=False)
        indices = np.flatnonzero((self.task_ends >= lo) & (self.task_starts <= hi))
        widths = max(1, hi - lo)
        task_boxes, task_colors, prefixes = [], [], []
        for i in indices:
            t = self.data.tasks[i]
            start, end = t["start_ns"], t["end_ns"]
            color = PHASE_COLOR.get(t["phase"], MUTED)
            task_boxes.append((start / 1e9, 0.08, (end - start) / 1e9, 0.82))
            task_colors.append(color)
            extra = min(t.get("recompute_prefix_ns", 0), end - start)
            if extra:
                prefixes.append((start / 1e9, 0.08, extra / 1e9, 0.82))
            segments = (
                [(start, start + extra, "recompute"), (start + extra, end, None)]
                if extra
                else [(start, end, None)]
            )
            for a, b, phase in segments:
                if (min(b, hi) - max(a, lo)) / widths > 0.065:
                    ax.text(
                        (max(a, lo) + min(b, hi)) / 2e9,
                        0.49,
                        short_task(t, phase=phase),
                        ha="center",
                        va="center",
                        fontsize=10,
                        color=BG if phase == "recompute" else WHITE,
                        clip_on=True,
                    )
        # Full-program placement builds redraw this strip many times. Two
        # collections keep its geometry exact without thousands of artists.
        boxes(ax, task_boxes, task_colors, edgecolor=BG, linewidth=0.4)
        boxes(ax, prefixes, [RECOMPUTE] * len(prefixes))
        if boundaries and len(indices) <= 18:
            # Exact task-end positions; zero-duration tasks may share a tick.
            previous = -1e100
            for i in indices:
                x = self.data.tasks[i]["end_ns"]
                if lo <= x <= hi:
                    ax.axvline(
                        x / 1e9, ymax=0.12, color=WHITE, alpha=0.65, linewidth=0.65
                    )
                    if (x - previous) / widths > 0.035:
                        ax.text(
                            x / 1e9,
                            1.12,
                            f"B{i + 1}",
                            fontsize=8,
                            color=MUTED,
                            ha="center",
                            va="bottom",
                            clip_on=False,
                        )
                        previous = x
        return ax

    def minimap(self, state, *, placement=False):
        if placement:
            ax = self.axis(
                (0.07, 0.795, 0.63, 0.038),
                xlim=(0, self.data.end_ns / 1e9),
                ylim=(0, self.data.pool_bytes / GIB),
                ticks=False,
            )
            ax.imshow(
                self.layout_raster.image(
                    int(state.get("placed", len(self.data.leases)))
                ),
                extent=(0, self.data.end_ns / 1e9, 0, self.data.pool_bytes / GIB),
                origin="upper",
                aspect="auto",
                interpolation="antialiased",
            )
        else:
            ax = self.axis(
                (0.07, 0.795, 0.63, 0.038),
                xlim=(0, self.data.end_ns / 1e9),
                ylim=(-0.1, 3),
                ticks=False,
            )
            self.lanes(ax, [0, self.data.end_ns], overview=True)
        lo, hi = state["window_ns"]
        ax.axvspan(
            lo / 1e9, hi / 1e9, facecolor="none", edgecolor=WHITE, linewidth=1, zorder=5
        )
        ax.axvline(state["cursor_ns"] / 1e9, color=WHITE, linewidth=1, zorder=6)

    def pressure(self, state, *, rect=(0.07, 0.105, 0.63, 0.17), spill=False):
        data = self.data
        window = state["window_ns"]
        series = data.spill if spill else data.execution
        capacity = data.spill_bytes if spill else data.pool_bytes
        lo, hi = window
        left = max(0, int(np.searchsorted(series.times, lo, side="right")) - 1)
        right = min(
            len(series.times), int(np.searchsorted(series.times, hi, side="right")) + 1
        )
        times = series.times[left:right] / 1e9
        values = series.values[left:right].T / GIB
        ceiling = max(capacity / GIB, float(values.sum(axis=0).max(initial=0))) * 1.12
        ax = self.axis(
            rect,
            xlim=(lo / 1e9, hi / 1e9),
            ylim=(0, ceiling),
            ylabel=(
                "Spill GiB"
                if spill
                else "Final live GiB"
                if state["kind"].startswith("placement")
                else "Pool GiB"
            ),
            xlabel="Simulated time (s)",
        )
        order = sorted(
            range(len(series.names)),
            key=lambda i: (
                [
                    "parameter",
                    "optimizer_state",
                    "gradient",
                    "activation",
                    WORKSPACE,
                ].index(series.names[i])
                if series.names[i]
                in ["parameter", "optimizer_state", "gradient", "activation", WORKSPACE]
                else 4
            ),
        )
        if len(times) and order:
            ax.stackplot(
                times,
                *values[order],
                colors=[ROLE_COLOR.get(series.names[i], MUTED) for i in order],
                step="post",
                linewidth=0,
            )
        ax.axhline(capacity / GIB, color=RED, linestyle=(0, (4, 4)), linewidth=1)
        ax.axvline(state["cursor_ns"] / 1e9, color=WHITE, linewidth=0.9)
        ax.text(
            0.99,
            0.97,
            f"Capacity {byte_label(capacity)}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            color=RED,
        )
        return ax

    def lanes(
        self, ax, window, *, overview=False, selected=None, task_index=None, cursor=None
    ):
        lo, hi = window
        wait = self.data.task_detail(task_index) if task_index is not None else None
        related = {}
        if wait is not None:
            selected = wait["critical_action"]
            related = {b["action"]: b["relation"] for b in wait["blocking_transfers"]}
            related.update(
                {
                    b["action"]: "fetch_storage_reuse"
                    for b in wait["upstream_storage_dependencies"]
                }
            )
        rectangles, colors = [], []
        for i in np.flatnonzero((self.task_ends > lo) & (self.task_starts < hi)):
            task = self.data.tasks[i]
            s, e = task["start_ns"], task["end_ns"]
            extra = min(task.get("recompute_prefix_ns", 0), e - s)
            rectangles.append((s / 1e9, 2.12, (e - s) / 1e9, 0.66))
            colors.append(PHASE_COLOR.get(task["phase"], MUTED))
            if extra:
                rectangles.append((s / 1e9, 2.12, extra / 1e9, 0.66))
                colors.append(RECOMPUTE)
            if not overview:
                if task["start_ns"] > task["ready_ns"]:
                    ax.add_patch(
                        Rectangle(
                            (task["ready_ns"] / 1e9, 2.12),
                            (s - task["ready_ns"]) / 1e9,
                            0.66,
                            facecolor=RED,
                            alpha=0.23,
                            hatch="///",
                            edgecolor=RED,
                            linewidth=0.5,
                        )
                    )
                segments = (
                    [(s, s + extra, "recompute"), (s + extra, e, None)]
                    if extra
                    else [(s, e, None)]
                )
                for start, end, phase in segments:
                    label = short_task(task, phase=phase)
                    width = (
                        self.fig.canvas.get_renderer().get_text_width_height_descent(
                            label, FontProperties(family="STIXGeneral", size=10), False
                        )[0]
                    )
                    available = (
                        (min(end, hi) - max(start, lo))
                        / max(1, hi - lo)
                        * ax.bbox.width
                    )
                    if available > width + 8:
                        ax.text(
                            (max(start, lo) + min(end, hi)) / 2e9,
                            2.45,
                            label,
                            ha="center",
                            va="center",
                            fontsize=10,
                            color=BG if phase == "recompute" else WHITE,
                            zorder=5,
                            clip_on=True,
                        )
        for i in np.flatnonzero(
            (self.transfer_ends > lo) & (self.transfer_starts < hi)
        ):
            a = self.data.transfers[i]
            t = a["transfer"]
            y = 1.12 if t["direction"] == "fetch" else 0.12
            rectangles.append(
                (t["start_ns"] / 1e9, y, (t["end_ns"] - t["start_ns"]) / 1e9, 0.66)
            )
            colors.append(
                to_rgba(
                    ROLE_COLOR.get(a["role"], MUTED),
                    0.24 if wait and a["index"] not in related else 1,
                )
            )
        boxes(
            ax,
            rectangles,
            colors,
            edgecolor=BG if not overview else "none",
            linewidth=0.3,
        )
        if wait is not None:
            task = self.data.tasks[task_index]
            for action_index, relation in related.items():
                a = self.data.actions[action_index]
                t = a["transfer"]
                if t["end_ns"] < lo or t["start_ns"] > hi:
                    continue
                y = 1.12 if t["direction"] == "fetch" else 0.12
                ax.add_patch(
                    Rectangle(
                        (t["start_ns"] / 1e9, y),
                        (t["end_ns"] - t["start_ns"]) / 1e9,
                        0.66,
                        facecolor="none",
                        edgecolor=WHITE
                        if relation == "required_input"
                        else LANE_COLOR["evict"],
                        linewidth=1,
                        zorder=5,
                    )
                )
            ax.add_patch(
                Rectangle(
                    (task["start_ns"] / 1e9, 2.12),
                    (task["end_ns"] - task["start_ns"]) / 1e9,
                    0.66,
                    facecolor="none",
                    edgecolor=WHITE,
                    linewidth=1.5,
                    zorder=6,
                )
            )
            if wait["stall_ns"]:
                center = (task["ready_ns"] + task["start_ns"]) / 2e9
                ax.text(
                    center,
                    3.12,
                    f"{wait['label']} waits {duration_label(wait['stall_ns'])}",
                    ha="center",
                    va="center",
                    fontsize=12,
                    color=WHITE,
                    clip_on=True,
                )
                ax.plot(
                    [task["ready_ns"] / 1e9, task["start_ns"] / 1e9],
                    [2.92, 2.92],
                    color=RED,
                    linewidth=1.5,
                    marker="|",
                    markersize=7,
                )
            # Name the preceding task even when it is too narrow for a label
            # inside its bar. Keep it below the selected wait's top caption.
            if task_index:
                prior = self.data.tasks[task_index - 1]
                if (prior["end_ns"] - prior["start_ns"]) / max(1, hi - lo) <= 0.085:
                    ax.text(
                        (prior["start_ns"] + prior["end_ns"]) / 2e9,
                        2.84,
                        short_task(prior),
                        ha="center",
                        va="bottom",
                        fontsize=10,
                        clip_on=True,
                    )
        if selected is not None:
            a = self.data.actions[selected]
            t = a["transfer"]
            y = 1.45 if t["direction"] == "fetch" else 0.45
            ax.plot(
                [a["trigger_ns"] / 1e9, t["start_ns"] / 1e9],
                [y, y],
                color=WHITE,
                linestyle=(0, (3, 3)),
                linewidth=2,
                zorder=6,
            )
            if wait is not None:
                ax.text(
                    max(lo, a["trigger_ns"]) / 1e9,
                    y + 0.48,
                    f"Trigger B{a['boundary']} · {a['trigger_ns'] / 1e9:.4f} s",
                    fontsize=10,
                    va="bottom",
                    color=WHITE,
                    clip_on=True,
                    bbox={"facecolor": PANEL, "edgecolor": "none", "pad": 1.5},
                )
            ax.plot(
                a["trigger_ns"] / 1e9,
                y,
                marker="o",
                markersize=6,
                color=WHITE,
                zorder=6,
            )
            ax.add_patch(
                Rectangle(
                    (t["start_ns"] / 1e9, y - 0.33),
                    (t["end_ns"] - t["start_ns"]) / 1e9,
                    0.66,
                    facecolor="none",
                    edgecolor=WHITE,
                    linewidth=2,
                    zorder=6,
                )
            )
            next_task = (
                task_index
                if wait is not None
                else self.data.action_detail(selected)["next_task"]
            )
            if next_task is not None and (
                wait is not None or t["direction"] == "fetch"
            ):
                start = self.data.tasks[next_task]["start_ns"]
                if lo <= start <= hi:
                    ax.annotate(
                        "",
                        xy=(start / 1e9, 2.1),
                        xytext=(t["end_ns"] / 1e9, y + 0.34),
                        arrowprops={"arrowstyle": "->", "color": WHITE, "linewidth": 1},
                        zorder=7,
                    )
        if cursor is not None:
            ax.axvline(cursor / 1e9, color=WHITE, linewidth=1.2, zorder=8)
        if not overview:
            ax.set_yticks([2.45, 1.45, 0.45], ["Compute", "Fetch", "Evict"])

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(path, dpi=self.fig.dpi, facecolor=BG)

    def draw_scene(self, state):
        raise NotImplementedError

    def draw(self, state):
        self.fig.clear()
        self.axes = []
        self.header(state)
        self.draw_scene(state)
        self.fig.canvas.draw()
        return memoryview(self.fig.canvas.buffer_rgba()).cast("B")
