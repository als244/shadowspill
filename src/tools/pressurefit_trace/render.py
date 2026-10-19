"""Draw exact replay states, with a dense overview and a boundary-local view.

The short movie batches decisions after its opening cuts. Every batch endpoint
is an exact replay state; no synthetic intermediate residency is introduced.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator

from shadowspill.diagnostics.pressurefit_trace import _cut

from .data import GIB, ROLE_COLORS, Trajectory, byte_label, short_task
from .pressure import ROLE_GROUPS, PressureComposition
from .style import BG, GREEN, MUTED, PANEL, RECOMPUTE, RED, WHITE, duration_label


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


def compressed_targets(first: int, total: int, frames: int) -> list[int]:
    """Accelerate smoothly, without repeats, and always reach the final cut."""
    if total <= first:
        return []
    count = min(total - first, frames)
    return sorted(
        {
            first + max(1, round((total - first) * (i / count) ** 1.5))
            for i in range(1, count + 1)
        }
        | {total}
    )


def paced_steps(first, total, frames, *, hold_frames=1):
    """Exact frame budget, with a quadratic ramp and repeated states held."""
    if total <= first:
        return []
    slots = max(1, frames // hold_frames)
    steps = []
    for i in range(1, slots + 1):
        target = first + max(1, math.ceil((total - first) * (i / slots) ** 2))
        duration = round(i * frames / slots) - round((i - 1) * frames / slots)
        if steps and steps[-1][0] == target:
            steps[-1] = (target, steps[-1][1] + duration)
        else:
            steps.append((target, duration))
    return steps


def explanation_events(data, rid, amount, *, minimum_gap=6):
    """Choose visually distinct early cuts, then a release/eviction contrast."""
    cuts = data.trace.reductions[rid].cuts
    total = len(cuts)
    if amount <= 0 or not total:
        return []

    def reason(index):
        comparisons = data.trace.comparisons(rid, index)
        return comparisons[0]["criterion"] if comparisons else None

    def span_length(row):
        first, last = map(int, row[5:7].view("<i8"))
        return max(1, last - first + 1)

    events = [0]
    if amount > 1 and total > 1:
        start = min(max(1, minimum_gap), total - 1)
        end = min(total, max(start + 1, total // 10))
        first = cuts[0]
        first_alias = int(first[4])
        first_size = max(1, int(data.trace.alias_sizes[first_alias]))
        first_span = span_length(first)
        first_reason = reason(0)

        def contrast(index):
            row = cuts[index]
            alias = int(row[4])
            size = max(1, int(data.trace.alias_sizes[alias]))
            span = span_length(row)
            # A changed tie-breaker alone can select a visually identical
            # sibling weight. Prefer differences the viewer can actually see.
            visual = abs(math.log2(size / first_size)) + abs(
                math.log2(span / first_span)
            )
            visual += 3 * (
                data.aliases[alias]["role"] != data.aliases[first_alias]["role"]
            )
            visual += 3 * (int(row[8]) != int(first[8]))
            substantive = reason(index) not in (
                None,
                first_reason,
                "alias",
                "start",
                "cut_id",
            )
            return visual, substantive, -index

        early = max(range(start, end), key=contrast)
        events.append(early)
    if amount > 2 and events[-1] < total - 1:
        start = min(max(events[-1] + max(1, minimum_gap), total // 3), total - 1)
        later = next(
            (i for i in range(start, total) if int(cuts[i, 8]) != int(cuts[0, 8])),
            max(start, total // 3),
        )
        events.append(later)
    if amount > len(events):
        events.extend(
            i
            for i in np.linspace(0, total - 1, min(amount, total), dtype=int)
            if i not in events
        )
    return sorted(int(event) for event in events[:amount])


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


def explain_comparison(comparison):
    field = comparison["criterion"]
    a, b = comparison["chosen_value"], comparison["alternative_value"]
    if field == "exposed_ns":
        return f"Less exposed transfer time\n{a:,} vs {b:,} ns"
    if field == "writeback":
        return "No writeback needed\nAlternative requires writeback"
    if field == "negative_slack_ns":
        return f"More signed transfer slack\n{-a / 1e6:,.3f} vs {-b / 1e6:,.3f} ms"
    if field == "negative_size_bytes":
        return f"Larger object\n{byte_label(-a)} vs {byte_label(-b)}"
    return f"Tie-break: {field.replace('_', ' ')}\n{a} precedes {b}"


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
        total = len(d.trace.reductions[rid].cuts)
        row = None if event is None else d.trace.reductions[rid].cuts[event]
        selected = None if row is None else int(row[4])
        next_use = None if row is None else d.next_use(row)
        departure = None if row is None else d.departure_position(row)
        highlight_cut = row is not None and phase not in ("seed", "final")
        if journey is None:
            journey = highlight_cut and self.focus_mode == "cut"
        applied = row is not None and count > event
        if highlight_cut and not applied:
            removed = cut_intervals(d, spans, row)
        position = (
            boundary
            if boundary is not None
            else (
                int(row[0])
                if row is not None
                else int(
                    np.argmax(pressure.astype(np.int64) - capacity.astype(np.int64))
                )
            )
        )
        lo, hi = (
            max(0, position - self.context),
            min(len(d.tasks), position + self.context + 1),
        )
        xmin, xmax = d.edges[lo], d.edges[hi]
        prior_use = None
        journey = journey and selected is not None and next_use is not None
        if journey:
            prior_use = max(
                (i for i in d.uses[selected] if i < departure), default=None
            )
            wide_lo = max(0, (prior_use if prior_use is not None else departure) - 1)
            wide_hi = min(len(d.tasks), next_use + 2)
            xmin = (1 - zoom_progress) * xmin + zoom_progress * d.edges[wide_lo]
            xmax = (1 - zoom_progress) * xmax + zoom_progress * d.edges[wide_hi]
            lo, hi = wide_lo, wide_hi
        if xmin == xmax:
            xmin, xmax = max(0, xmin - 0.01), xmin + 0.01
        focus, local = d.focus_aliases(
            position, selected, self.context, self.focus_rows
        )
        if journey:
            focus = [selected]

        context = d.collection_context or {}
        if context or self.title:
            workload = self.title or context.get("workload")
            parts = [workload] if workload else []
            metadata = context.get("metadata", {})
            if metadata.get("trainable", {}).get("mode") == "full":
                parts.append("full training")
            sequence = metadata.get("sequence_length")
            if sequence:
                parts.append(f"{sequence:,} tokens / sequence")
            if sequence and metadata.get("sequences_per_step"):
                parts.append(
                    f"{sequence * metadata['sequences_per_step']:,} tokens / step"
                )
            budget = context.get("physical_budget_bytes")
            if budget is not None:
                parts.append(f"{byte_label(budget)} budget")
            if context.get("spill_budget_bytes") is not None:
                parts.append(f"{byte_label(context['spill_budget_bytes'])} spill")
            self.label(0.035, 0.914, "  ·  ".join(parts), fontsize=11, color=WHITE)
        resolution = d.candidate.get("resolution_recompute_fraction")
        resolution_label = (
            f"recompute resolution {resolution}"
            if resolution is not None
            else f"recompute {d.candidate.get('recompute_share', 0):.0%} of groups"
        )
        policy = [d.candidate["candidate_id"], resolution_label]
        if context.get("ordering"):
            policy.append(f"ordering {context['ordering']}")
        if context.get("geometry"):
            policy.append(f"{context['geometry']} seq / microbatch")
        if context.get("microbatches"):
            policy.append(f"{context['microbatches']} microbatches")
        policy.extend(
            [
                f"{d.compute_seconds:.3f} s of profiled compute",
            ]
        )
        self.label(
            0.035,
            0.879 if context or self.title else 0.914,
            "  ·  ".join(policy),
            fontsize=10.5,
            color=MUTED,
        )
        reduction_context = d.reduction_context(rid)
        object_capacity = reduction_context["object_capacity_bytes"]
        capacity_label = (
            f"{object_capacity / GIB:,.3f} GiB"
            if object_capacity is not None
            else "not recorded"
        )
        self.label(
            0.035,
            0.024,
            f"Reduction {rid + 1}/{len(d.trace.reductions)}  ·  "
            f"object capacity: {capacity_label}",
            fontsize=8.5,
            color=MUTED,
        )
        current_cut = event + 1 if row is not None else count
        counter = f"Cut {current_cut:,}/{total:,}"
        self.label(
            0.975, 0.956, counter, fontsize=17, ha="right", weight="bold", color=WHITE
        )
        if phase == "seed":
            caption = (
                "Start from seed"
                if rid == 0
                else "Restart from seed with revised pressure / capacity"
            )
        elif phase == "final":
            caption = (
                "All boundaries fit"
                if d.trace.reductions[rid].status == 0
                else "Reduction stopped: no legal cut for an over-budget boundary"
            )
        elif batch_start is not None and count - batch_start > 1:
            caption = (
                f"Accelerated  ·  cuts {batch_start + 1:,}-{count:,}  ·  "
                f"{count - batch_start:,} decisions this frame"
            )
        elif phase == "selected":
            caption = f"Choose cut {event + 1:,}  ·  state before applying it"
        else:
            caption = f"Apply cut {count:,}  ·  update residency and pressure"
        self.label(
            0.255,
            0.817,
            caption,
            fontsize=9,
            color=GREEN if phase == "final" else MUTED,
        )

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
        self.style(self.side, ticks=False)
        self.side.set_xlim(0, 1)
        self.side.set_ylim(0, 1)

        def side(y, content, **kw):
            self.side.text(0.06, y, content, va="top", **kw)

        side(0.975, "DECISION DETAIL", fontsize=10.5, weight="bold")
        if row is not None:
            side(
                0.935,
                f"Cut {event + 1:,} (Boundary {position})",
                fontsize=13,
                color=GREEN,
            )
            name = d.aliases[selected]["label"]
            name_lines = textwrap.wrap(name, 28, max_lines=3, placeholder="…")
            side(
                0.900,
                "\n".join(name_lines),
                fontsize=10.5,
                linespacing=1.2,
                color=ROLE_COLORS.get(d.aliases[selected]["role"], MUTED),
            )
            size_y = 0.894 - 0.028 * len(name_lines)
            side(
                size_y,
                byte_label(int(d.trace.alias_sizes[selected]))
                + f"  ·  {d.aliases[selected]['role']}",
                fontsize=10,
                color=MUTED,
            )
            side(
                size_y - 0.033,
                "Evict modified data"
                if int(row[8])
                else "Release · spill copy available",
                fontsize=10,
                color=MUTED,
            )
            side(
                size_y - 0.066,
                f"Exposed transfer score  {int(row[7]) / 1e6:.3f} ms",
                fontsize=10,
            )
            self.side.plot(
                [0.06, 0.94], [0.714, 0.714], color=MUTED, alpha=0.25, lw=0.7
            )
            comparisons = d.trace.comparisons(rid, event)
            side(
                0.695,
                "WHY IT BEATS RUNNER-UP",
                fontsize=9.7,
                weight="bold",
                color=MUTED,
            )
            side(
                0.662,
                explain_comparison(comparisons[0])
                if comparisons
                else "Only eligible cut",
                fontsize=10.5,
                linespacing=1.2,
            )
            side(
                0.591,
                f"TOP BOUNDARIES BEFORE CUT {event + 1:,}",
                fontsize=9.4,
                weight="bold",
                color=MUTED,
            )
            for i, (pos, _used, _cap, over) in enumerate(
                row[52:64].reshape(3, 4)[: int(row[51])]
            ):
                pos = int(pos)
                task = short_task(d.tasks[pos]) if pos < len(d.tasks) else "end"
                side(
                    0.558 - i * 0.058,
                    f"{i + 1}.  b{pos}  ·  +{int(over) / GIB:.2f} GiB\n"
                    f"     before {task}",
                    fontsize=10.5,
                    linespacing=1.2,
                    color=WHITE if i == 0 else MUTED,
                )
            boundary_offset = self.composition.flat_budget - int(capacity[int(row[0])])
            used = (int(row[1]) + boundary_offset) / GIB
            self.side.plot(
                [0.06, 0.94], [0.388, 0.388], color=MUTED, alpha=0.25, lw=0.7
            )
            side(
                0.373,
                f"Used at b{int(row[0])}",
                fontsize=10.5,
                color=MUTED,
            )
            if count > event:
                after = (int(pressure[int(row[0])]) + boundary_offset) / GIB
                side(
                    0.341,
                    f"{used:.3f} → {after:.3f} GiB",
                    fontsize=13,
                    weight="bold",
                    color=GREEN,
                )
            else:
                side(
                    0.341,
                    f"{used:.3f} GiB",
                    fontsize=13,
                    weight="bold",
                    color=WHITE,
                )
        else:
            side(0.90, "No cut selected", fontsize=11)
        current_task = d.tasks[position] if position < len(d.tasks) else None
        current_memory = current_task["required_memory"] if current_task else None
        self.side.plot([0.06, 0.94], [0.305, 0.305], color=MUTED, alpha=0.25, lw=0.7)
        side(
            0.289,
            f"CURRENT TASK · b{position}",
            fontsize=9.7,
            weight="bold",
            color=MUTED,
        )
        side(
            0.257,
            short_task(current_task) if current_task else "End of program",
            fontsize=11,
        )
        if current_memory is not None:
            for y, key, label in (
                (0.213, "inputs", "Inputs"),
                (0.181, "outputs", "Outputs"),
                (0.149, "mutated", "Mutated"),
            ):
                entry = current_memory[key]
                side(y, f"{label} ({entry['object_count']:,})", fontsize=10.5)
                self.side.text(
                    0.94,
                    y,
                    byte_label(entry["bytes"]),
                    ha="right",
                    va="top",
                    fontsize=10.5,
                )
            side(0.117, "Workspace", fontsize=10.5)
            self.side.text(
                0.94,
                0.117,
                byte_label(current_memory["workspace_bytes"]),
                ha="right",
                va="top",
                fontsize=10.5,
            )
            entry = current_memory["total"]
            side(
                0.073, f"Total ({entry['object_count']:,})", fontsize=11, weight="bold"
            )
            self.side.text(
                0.94,
                0.073,
                byte_label(entry["bytes"]),
                ha="right",
                va="top",
                fontsize=11,
                weight="bold",
            )
        self.last_record = {
            "axis": d.axis,
            "reduction": rid,
            "reduction_context": reduction_context,
            "cut_count": count,
            "phase": phase,
            "event_index": event,
            "batch_start": batch_start,
            "boundary_position": position,
            "current_task": (
                {
                    "task_position": position,
                    "task_id": current_task["task_id"],
                    "label": short_task(current_task),
                    "required_memory": current_memory,
                }
                if current_task
                else None
            ),
            "focus_aliases": [int(a) for a in focus],
            "all_local_aliases": local,
            "focus_task_range": [lo, hi],
            "focus_axis_limits": [float(xmin), float(xmax)],
            "recompute_prefixes": [
                {
                    "task_position": i,
                    "task_id": d.tasks[i]["task_id"],
                    "label": short_task(d.tasks[i], phase="recompute"),
                    "duration_ns": d.tasks[i]["recompute_prefix_ns"],
                    "save_backward_duration_ns": d.tasks[i][
                        "save_backward_duration_ns"
                    ],
                    "axis_start": float(d.edges[i]),
                    "axis_end": recompute_endpoint(d, i),
                }
                for i in range(lo, hi)
                if d.tasks[i].get("recompute_prefix_ns", 0) > 0
            ],
            "peak_pressure_bytes": int(pressure.max()),
            "display_peak_pressure_bytes": int(displayed_pressure.max()),
            "display_budget_bytes": self.composition.flat_budget,
            "resident_pool_bytes": self.pressure_offset,
            "pressure_ylim_gib": [0, ceiling],
            "pressure_roles_at_boundary": {
                name: int(composition[i, position])
                for i, (name, _, _) in enumerate(ROLE_GROUPS)
            },
            "max_excess_bytes": max(
                0, int(np.max(pressure.astype(np.int64) - capacity.astype(np.int64)))
            ),
        }
        if journey:
            self.last_record["object_journey"] = {
                "alias_index": selected,
                "prior_use_position": prior_use,
                "departure_position": departure,
                "next_use_position": next_use,
                "intervening_tasks": next_use - departure,
                "zoom_progress": zoom_progress,
            }
        if highlight_cut:
            self.last_record["cut_highlight"] = {
                "alias_index": selected,
                "status": "applied" if applied else "selected",
                "removed_intervals": removed,
                "boundary_only": not removed,
                "action": "evict" if int(row[8]) else "release",
                "departure_position": departure,
                "next_use_position": next_use,
                "compute_gap_ns": (
                    d.gap_duration_ns(row, next_use) if next_use is not None else None
                ),
                "compute_until_next_use_ns": (
                    d.time_to_use_ns(row, next_use) if next_use is not None else None
                ),
            }
        if explain and event is not None:
            from .explain import draw_explanation

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


def render_movie(
    data,
    output,
    *,
    title=None,
    reductions,
    width=1920,
    fps=30,
    context=3,
    focus_rows=7,
    opening_cuts=6,
    slow_seconds=8,
    accelerated_seconds=45,
    pacing="overview",
    explain_cuts=3,
    explain_seconds=15,
    detail_at=None,
    step_seconds=1,
    bridge_seconds=24,
    bridge_hold_seconds=0.6,
    accelerated_hold_seconds=0.1,
    seed_seconds=5,
    final_seconds=4,
    cut_zoom=True,
    zoom_seconds=1.2,
    zoom_hold_seconds=3,
    quality="high",
    pressure_scale="adaptive",
    pressure_view="total",
    focus_mode="cut",
):
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "MP4 output requires ffmpeg on PATH; PNG frames work without it"
        )
    details = (
        sorted(set(cut - 1 for cut in detail_at))
        if detail_at is not None
        else explanation_events(
            data, reductions[0], explain_cuts, minimum_gap=opening_cuts
        )
    )
    if any(
        i < 0 or i >= len(data.trace.reductions[reductions[0]].cuts) for i in details
    ):
        raise ValueError(
            "--detail-at cut numbers must belong to the first rendered reduction"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
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
    height = int(width * 9 / 16)
    if width % 2 or height % 2:
        raise ValueError(
            "--width must give even 16:9 video dimensions (e.g. 1280, 1600, 1920)"
        )
    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "warning",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgba",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        str({"preview": 26, "high": 18, "best": 14}[quality]),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    storyboard = []
    frame_count = 0
    with output.with_suffix(".ffmpeg.log").open("w") as log:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
        try:

            def emit(seconds=0, *, flash=False, frames=None):
                nonlocal frame_count
                frames = max(1, round(seconds * fps)) if frames is None else frames
                storyboard.append(
                    {
                        **renderer.last_record,
                        "first_video_frame": frame_count,
                        "video_frame_count": frames,
                    }
                )
                if flash:
                    fade_frames = min(frames, max(2, round(0.8 * fps)))
                    for i in range(fade_frames):
                        process.stdin.write(
                            renderer.rgba(flash=(1 - i / max(1, fade_frames - 1)) ** 2)
                        )
                    rgba = renderer.rgba(flash=0)
                    for _ in range(frames - fade_frames):
                        process.stdin.write(rgba)
                else:
                    rgba = renderer.rgba()
                    for _ in range(frames):
                        process.stdin.write(rgba)
                frame_count += frames

            for chapter, rid in enumerate(reductions):
                reduction = data.trace.reductions[rid]
                walk = data.trace._walk(rid)
                count, spans, pressure, capacity = next(walk)
                total = len(reduction.cuts)
                renderer.draw(
                    rid,
                    count,
                    spans,
                    pressure,
                    capacity,
                    event=0 if total else None,
                    phase="seed",
                )
                renderer.save(output.with_name(f"{output.stem}_r{rid:02d}_seed.png"))
                emit(seed_seconds)

                def advance(
                    target,
                    frames,
                    *,
                    flash=False,
                    reduction=reduction,
                    walk=walk,
                    rid=rid,
                ):
                    nonlocal count, spans, pressure, capacity
                    start = count
                    touched = set()
                    while count < target:
                        alias = int(reduction.cuts[count, 4])
                        touched.add(alias)
                        if count == target - 1:
                            before = data.intervals(alias, spans[alias])
                        count, spans, pressure, capacity = next(walk)
                    removed = subtract_intervals(
                        before, data.intervals(alias, spans[alias])
                    )
                    renderer.draw(
                        rid,
                        count,
                        spans,
                        pressure,
                        capacity,
                        event=count - 1,
                        batch_start=start if count - start > 1 else None,
                        removed=removed,
                        touched=touched,
                    )
                    emit(frames=frames, flash=flash)

                def flow(
                    first,
                    target,
                    seconds,
                    hold_seconds,
                    *,
                    rid=rid,
                    total=total,
                    advance=advance,
                ):
                    steps = (
                        [(i, 1) for i in range(first + 1, target + 1)]
                        if pacing == "every-cut"
                        else paced_steps(
                            first,
                            target,
                            max(1, round(seconds * fps)),
                            hold_frames=max(1, round(hold_seconds * fps)),
                        )
                    )
                    for index, (target_count, frames) in enumerate(steps):
                        advance(target_count, frames)
                        if index % 100 == 0:
                            print(
                                f"RENDER reduction={rid} cut={target_count}/{total} "
                                f"frames={frame_count}",
                                flush=True,
                            )

                examples = details if chapter == 0 else []
                slow_events = examples or list(
                    range(min(total, opening_cuts if chapter == 0 else 1))
                )
                for event in slow_events:
                    gap = event - count
                    if gap <= max(opening_cuts, 24):
                        for target in range(count + 1, event + 1):
                            advance(
                                target, max(1, round(step_seconds * fps)), flash=True
                            )
                    else:
                        flow(count, event, bridge_seconds, bridge_hold_seconds)
                    alias = int(reduction.cuts[event, 4])
                    before = data.intervals(alias, spans[alias])
                    explanation = event in examples
                    journey = bool(
                        explanation
                        and cut_zoom
                        and focus_mode == "cut"
                        and data.next_use(reduction.cuts[event]) is not None
                    )
                    renderer.draw(
                        rid,
                        count,
                        spans,
                        pressure,
                        capacity,
                        event=event,
                        phase="selected",
                        explain=explanation,
                        journey=False if journey else None,
                    )
                    if explanation:
                        emit(explain_seconds)
                        renderer.draw(
                            rid,
                            count,
                            spans,
                            pressure,
                            capacity,
                            event=event,
                            phase="selected",
                            journey=False if journey else None,
                        )
                    emit(slow_seconds * 0.4)
                    if journey:
                        zoom_frames = max(1, round(zoom_seconds * fps))
                        for frame in range(1, zoom_frames + 1):
                            fraction = frame / zoom_frames
                            renderer.draw(
                                rid,
                                count,
                                spans,
                                pressure,
                                capacity,
                                event=event,
                                phase="selected",
                                journey=True,
                                zoom_progress=fraction * fraction * (3 - 2 * fraction),
                            )
                            emit(frames=1)
                        emit(zoom_hold_seconds / 2)
                    count, spans, pressure, capacity = next(walk)
                    removed = subtract_intervals(
                        before, data.intervals(alias, spans[alias])
                    )
                    renderer.draw(
                        rid,
                        count,
                        spans,
                        pressure,
                        capacity,
                        event=event,
                        removed=removed,
                        touched=[alias],
                        journey=True if journey else None,
                    )
                    emit(
                        slow_seconds * 0.6,
                        flash=True,
                    )
                    if journey:
                        for frame in range(1, zoom_frames + 1):
                            fraction = 1 - frame / zoom_frames
                            renderer.draw(
                                rid,
                                count,
                                spans,
                                pressure,
                                capacity,
                                event=event,
                                removed=removed,
                                journey=True,
                                zoom_progress=fraction * fraction * (3 - 2 * fraction),
                            )
                            emit(frames=1)
                        renderer.draw(
                            rid,
                            count,
                            spans,
                            pressure,
                            capacity,
                            event=event,
                            removed=removed,
                            journey=False,
                        )
                        emit(zoom_hold_seconds / 2)
                    if event == 0:
                        renderer.save(
                            output.with_name(f"{output.stem}_r{rid:02d}_cut00001.png")
                        )
                if chapter == 0:
                    for target in range(count + 1, min(total, opening_cuts) + 1):
                        advance(target, max(1, round(step_seconds * fps)), flash=True)
                flow(count, total, accelerated_seconds, accelerated_hold_seconds)
                data.assert_required_covered(spans)
                renderer.draw(
                    rid,
                    count,
                    spans,
                    pressure,
                    capacity,
                    event=count - 1 if count else None,
                    phase="final",
                )
                renderer.save(output.with_name(f"{output.stem}_r{rid:02d}_final.png"))
                emit(final_seconds)
                print(
                    f"RENDER reduction={rid} complete cuts={count} "
                    f"frames={frame_count}",
                    flush=True,
                )
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError(
                    f"ffmpeg failed; see {output.with_suffix('.ffmpeg.log')}"
                )
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            plt.close(renderer.fig)
    report = {
        "title": title,
        "width": width,
        "height": height,
        "quality": quality,
        "axis": data.axis,
        "source": str(data.directory),
        "candidate_index": data.candidate_index,
        "candidate_file": data.candidate["file"],
        "collection_context": data.collection_context,
        "recompute_share": data.candidate.get("recompute_share"),
        "resolution_recompute_fraction": data.candidate.get(
            "resolution_recompute_fraction"
        ),
        "recompute_tasks": [
            {
                key: task[key]
                for key in (
                    "task_id",
                    "profiled_duration_ns",
                    "save_backward_task_id",
                    "save_backward_duration_ns",
                    "recompute_extra_ns",
                    "recompute_prefix_ns",
                )
            }
            for task in data.tasks
            if "recompute_prefix_ns" in task
        ],
        "cut_score": data.trace.cut_score,
        "cut_ranking": data.trace.cut_ranking,
        "ranking_criteria": list(data.trace.ranking_criteria),
        "answer_reduction": data.trace.answer_reduction,
        "rendered_reductions": list(reductions),
        "fps": fps,
        "frames": frame_count,
        "seconds": frame_count / fps,
        "pacing": pacing,
        "detailed_cuts": [i + 1 for i in details],
        "timing": {
            "opening_cuts": opening_cuts,
            "slow_seconds": slow_seconds,
            "explain_seconds": explain_seconds,
            "step_seconds": step_seconds,
            "bridge_seconds": bridge_seconds,
            "bridge_hold_seconds": bridge_hold_seconds,
            "accelerated_seconds": accelerated_seconds,
            "accelerated_hold_seconds": accelerated_hold_seconds,
            "seed_seconds": seed_seconds,
            "final_seconds": final_seconds,
            "cut_zoom": cut_zoom,
            "zoom_seconds": zoom_seconds,
            "zoom_hold_seconds": zoom_hold_seconds,
        },
        "pressure_scale": pressure_scale,
        "pressure_view": pressure_view,
        "focus_mode": focus_mode,
        "pressure_attributed": renderer.composition.attributed,
        "row_filter": data.row_filter_summary(),
        "shots": storyboard,
    }
    output.with_suffix(".storyboard.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report
