#!/usr/bin/env python3
"""Turn a quickstart timeline HTML/JSON or report directory into slide assets.

No training run, browser, GPU, or Google credentials are needed. Run with
python -m benchmarking.quickstart_timeline; see quickstart_timeline.md.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MaxNLocator

COLORS = dict(
    params="#3268DE",
    gradients="#008D80",
    optimizer_state="#8860C9",
    activations="#E1AD4F",
    forward="#D65A22",
    backward="#75452E",
    optimizer="#151F2B",
    recompute="#84CC16",
    idle="#FCE9E7",
    ink="#15263D",
    muted="#62738A",
    line="#D6DFEB",
    red="#CF5555",
)
MEMORY_CLASSES = {
    "params": "Parameters",
    "gradients": "Gradients",
    "optimizer_state": "Optimizer state",
    "activations": "Activations",
}


def category(name):
    """Presentation convention: temporary workspace is included in activations."""
    return {
        "weights": "params",
        "model gradients": "gradients",
        "optimizer state": "optimizer_state",
    }.get(name, "activations")


def select_source(args):
    if args.source.is_file():
        return args.source.resolve()
    summaries = list(args.source.rglob("timelines/summary.csv"))
    direct = args.source / "summary.csv"
    if direct.is_file():
        summaries.append(direct)
    summaries = sorted(set(p.resolve() for p in summaries))
    if len(summaries) != 1:
        raise ValueError(
            "Choose a single report's seq*/seqsperstep*/ directory; "
            f"found {len(summaries)} timeline summaries: {summaries}"
        )
    with summaries[0].open() as f:
        rows = list(csv.DictReader(f))
    if args.list:
        for row in rows:
            print(
                " | ".join(
                    row.get(k, "")
                    for k in (
                        "budget",
                        "geometry",
                        "resolution",
                        "view",
                        "step_seconds",
                        "page",
                    )
                )
            )
        return None
    view = (args.view or "traced").replace("-", "_")
    # Choose the simulated winner, then locate its requested timeline. A
    # 'selected' row alone is only the best resolution within ONE geometry.
    selection_view = (
        "simulated" if view in ("simulated", "traced", "unconstrained") else view
    )
    candidates = [r for r in rows if r["view"] == selection_view]
    if selection_view == "simulated" and not args.resolution:
        candidates = [
            r for r in candidates if r.get("selected", "true").lower() in ("true", "1")
        ]
    if args.budget_gib is not None:
        candidates = [
            r
            for r in candidates
            if r["budget"].endswith("gib")
            and math.isclose(float(r["budget"][:-3]), args.budget_gib, abs_tol=1e-6)
        ]
    elif selection_view == "simulated":
        raise ValueError(
            "A report directory needs --budget-gib; or pass the exact timeline HTML."
        )
    if args.geometry:
        candidates = [r for r in candidates if r["geometry"] == args.geometry]
    if args.resolution:
        candidates = [r for r in candidates if r["resolution"] == args.resolution]
    if not candidates:
        raise ValueError("No matching timeline. Use --list to inspect the report.")
    if selection_view == "all_save" and len(candidates) > 1:
        raise ValueError(
            "Multiple all-save geometries: specify --geometry or the exact HTML."
        )
    winner = min(candidates, key=lambda r: float(r["step_seconds"]))
    page = summaries[0].parent / winner["page"]
    if view != selection_view:
        page = page.with_name(view + ".html")
    if not page.is_file():
        raise ValueError(f"The selected plan has no {view} timeline: {page}")
    return page


def load_timeline(path):
    content = path.read_text()
    if path.suffix.lower() == ".html":
        match = re.search(
            r"<script\b[^>]*\bid=[\"\']data[\"\'][^>]*>(.*?)</script>", content, re.S
        )
        if not match:
            raise ValueError(f"No embedded quickstart timeline JSON in {path}")
        content = match[1]
    data = json.loads(content)
    required = {"pools", "compute", "fetch", "evict", "end_seconds", "summary", "view"}
    if missing := required - data.keys():
        raise ValueError(f"Missing timeline fields: {sorted(missing)}")
    return data


def lane_runs(items, duration, *, compute=False, columns=1800):
    """Same dominant-duration overview binning as the report's HTML timeline."""
    tally = [{} for _ in range(columns)]

    def paint(start, end, kind):
        a, b = (
            max(0, start) / duration * columns,
            min(duration, end) / duration * columns,
        )
        for i in range(max(0, math.floor(a)), min(columns - 1, math.floor(b)) + 1):
            part = min(b, i + 1) - max(a, i)
            if part > 0:
                tally[i][kind] = tally[i].get(kind, 0) + part

    items = sorted(items, key=lambda item: item[0])
    for item in items:
        a, b = item[:2]
        if compute:
            split = min(b, a + item[4])
            paint(a, split, "recompute")
            paint(split, b, item[2])
        else:
            paint(a, b, category(item[3]))
    if compute and items:
        latest = items[0][1]
        for item in items[1:]:
            if item[0] > latest:
                paint(latest, item[0], "idle")
            latest = max(latest, item[1])
    kinds = [max(t, key=t.get) if t else None for t in tally]
    runs, start = [], 0
    for i in range(1, columns + 1):
        if i == columns or kinds[i] != kinds[start]:
            if kinds[start] is not None:
                runs.append(
                    (start * duration / columns, i * duration / columns, kinds[start])
                )
            start = i
    return runs


def render(data, source, args):
    pool = next(p for p in data["pools"] if p["name"] == "execution")
    duration = float(data["end_seconds"])
    xmax = args.time_max or duration
    if xmax < duration:
        raise ValueError("--time-max must include the full timeline.")
    budget = args.budget_gib
    if budget is None and data["view"] in ("simulated", "traced"):
        match = re.search(r"execution budget ([\d.]+)gib", data.get("plan", ""))
        budget = float(match[1]) if match else None
    if args.persistent_gradients_gib is not None and data["view"] not in (
        "all_save",
        "unconstrained",
    ):
        raise ValueError(
            "Persistent-gradient illustration is only allowed "
            "for unconstrained timelines."
        )
    show_transfers = args.transfers == "show" or (
        args.transfers == "auto" and bool(data["fetch"] or data["evict"])
    )
    if args.transfers == "hide" and (data["fetch"] or data["evict"]):
        raise ValueError("Cannot hide transfers that exist in the source timeline.")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "text.color": COLORS["ink"],
            "axes.labelcolor": COLORS["muted"],
            "axes.edgecolor": COLORS["line"],
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["muted"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(12, 3.65), facecolor="white")
    left, width = 0.063, 0.926
    ax_mem = fig.add_axes(
        [
            left,
            0.50 if show_transfers else 0.34,
            width,
            0.405 if show_transfers else 0.565,
        ]
    )
    lanes = [("compute", "GPU", 0.355 if show_transfers else 0.155)]
    if show_transfers:
        lanes.extend([("fetch", "Fetch", 0.245), ("evict", "Evict", 0.135)])
    times = np.asarray(pool["times"])
    rows = {k: np.zeros(len(times)) for k in MEMORY_CLASSES}
    for name, values in pool["rows"].items():
        values = np.asarray(values)
        if len(values) != len(times) or not np.isfinite(values).all():
            raise ValueError(f"Invalid memory samples for {name}")
        rows[category(name)] += values
    original_sample_peak = float(np.max(sum(rows.values())))
    if args.persistent_gradients_gib is not None:
        rows["gradients"][:] = args.persistent_gradients_gib
    peak = (
        float(np.max(sum(rows.values())))
        if args.persistent_gradients_gib is not None
        else float(pool["peak_gib"])
    )
    ax_mem.stackplot(
        times,
        *rows.values(),
        colors=[COLORS[k] for k in rows],
        step="post",
        linewidth=0,
    )
    ymax = max(peak, budget or 0) * 1.15
    if budget:
        ax_mem.axhline(budget, color=COLORS["red"], lw=0.9, dashes=(4, 3))
        ax_mem.text(
            xmax,
            budget + ymax * 0.012,
            f"{budget:g} GiB budget",
            color=COLORS["red"],
            fontsize=9,
            ha="right",
            va="bottom",
            bbox=dict(facecolor="white", edgecolor="none", pad=0.3),
        )
        ax_mem.set_yticks([0, budget / 2, budget])
    else:
        ax_mem.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
    ax_mem.set_ylim(0, ymax)
    ax_mem.tick_params(axis="y", length=0, pad=5)
    ax_mem.tick_params(axis="x", bottom=False, labelbottom=False)
    fig.text(left, 0.965, "GPU memory (GiB)", fontsize=11, weight="bold", va="top")
    peak_label = (
        f"Illustrative peak ≈{peak:.0f} GiB"
        if args.persistent_gradients_gib is not None
        else f"Peak {peak:.2f} GiB"
    )
    fig.text(0.989, 0.965, peak_label, fontsize=11, weight="bold", va="top", ha="right")
    axes, counts, used = [ax_mem], {}, set()
    for key, label, bottom in lanes:
        ax = fig.add_axes([left, bottom, width, 0.077])
        axes.append(ax)
        runs = lane_runs(
            data[key], duration, compute=key == "compute", columns=args.columns
        )
        counts[key] = len(runs)
        for a, b, kind in runs:
            ax.broken_barh(
                [(a, b - a)], (0, 1), facecolors=COLORS[kind], edgecolors="none"
            )
            if key == "compute":
                used.add(kind)
        if key == "compute":
            for a, b, *_ in data[key]:
                if (b - a) / xmax * args.columns > 3:
                    ax.add_patch(
                        Rectangle(
                            (a, 0),
                            b - a,
                            1,
                            facecolor="none",
                            edgecolor=(0, 0, 0, 0.20),
                            linewidth=0.25,
                        )
                    )
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.text(
            -0.018,
            0.5,
            label,
            transform=ax.transAxes,
            color=COLORS["muted"],
            fontsize=10,
            ha="right",
            va="center",
        )
        ax.tick_params(axis="x", length=0, pad=4, labelbottom=key == lanes[-1][0])
    ticks = MaxNLocator(nbins=10, steps=[1, 2, 2.5, 5, 10]).tick_values(0, xmax)
    ticks = ticks[(ticks >= 0) & (ticks <= xmax)]
    for ax in axes:
        ax.set_xlim(0, xmax)
        ax.set_xticks(ticks)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_linewidth(0.6)
    axes[-1].text(
        1,
        -0.9,
        "Time (s)",
        transform=axes[-1].transAxes,
        fontsize=9,
        ha="right",
        color=COLORS["muted"],
    )
    handles = [
        Patch(facecolor=COLORS[k], edgecolor="none", label=name)
        for k, name in (
            ("forward", "Forward"),
            ("backward", "Backward"),
            ("recompute", "Recompute"),
            ("optimizer", "Optimizer Update"),
            ("idle", "Stalled"),
        )
        if k in used
    ]
    fig.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(left - 0.004, -0.007),
        ncol=5,
        frameon=False,
        fontsize=9,
        handlelength=1.1,
        columnspacing=1.8,
        borderaxespad=0,
    )
    args.outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(
            args.outdir / f"timeline.{ext}",
            dpi=args.dpi,
            transparent=ext != "pdf",
            facecolor="white" if ext == "pdf" else "none",
        )
    plt.close(fig)
    meta = dict(
        source=str(source),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        view=data["view"],
        clock=data.get("clock"),
        plan=data.get("plan"),
        step_seconds=data["summary"]["seconds"],
        duration_seconds=duration,
        tokens_per_second=data["summary"].get("tokens_per_second"),
        time_max=xmax,
        peak_gib=peak,
        original_peak_gib=pool["peak_gib"],
        original_sample_peak_gib=original_sample_peak,
        budget_gib=budget,
        pool_capacity_gib=pool.get("capacity_gib"),
        persistent_gradients_gib=args.persistent_gradients_gib,
        spill_peak_gib=data["summary"].get("spill_peak_gib"),
        events={k: len(data[k]) for k in ("compute", "fetch", "evict")},
        summary=data["summary"],
        rendered_runs=counts,
        overview_columns=args.columns,
        interpretation={
            "memory": (
                "Report tensor leases plus task workspace envelope; "
                "not all physical CUDA allocations."
            ),
            "categories": (
                "Workspace and control temporaries are included in Activations; "
                "no bytes dropped."
            ),
            "recompute": (
                "Profile-cost split within backward tasks, as in source HTML; "
                "not separately traced kernels."
            ),
            "idle": "Only between compute tasks; initial/final waiting is not shaded.",
            "transfers": (
                "Colors identify object class. Overlapping/narrow events "
                "use dominant-duration overview bins."
            ),
            "illustration": (
                "Persistent gradients replace the original series "
                "only when explicitly requested."
            ),
        },
    )
    (args.outdir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    if args.pptx:
        make_slide(args, meta)
    return meta


def make_slide(args, meta):
    """Portable slide using the existing deck's result-slide design."""
    from pptx import Presentation
    from pptx.dml.color import RGBColor
    from pptx.enum.shapes import MSO_SHAPE
    from pptx.util import Inches, Pt

    deck = Presentation()
    deck.slide_width, deck.slide_height = Inches(13.333333), Inches(7.5)
    slide = deck.slides.add_slide(deck.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = RGBColor.from_string("F7F9FC")

    def text(value, x, y, width, height, size, color="ink", bold=False):
        box = slide.shapes.add_textbox(
            Inches(x), Inches(y), Inches(width), Inches(height)
        )
        tf = box.text_frame
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        p = tf.paragraphs[0]
        p.text = value
        p.font.name, p.font.size, p.font.bold = "Arial", Pt(size), bold
        p.font.color.rgb = RGBColor.from_string(COLORS[color][1:])

    def rect(x, y, width, height, color, rounded=False):
        shape = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE,
            Inches(x),
            Inches(y),
            Inches(width),
            Inches(height),
        )
        shape.fill.solid()
        shape.fill.fore_color.rgb = RGBColor.from_string(color.lstrip("#"))
        shape.line.fill.background()
        if rounded:
            shape.adjustments[0] = 0.08

    label = {
        "traced": "MEASURED",
        "simulated": "SIMULATED",
        "all_save": "BEFORE PLANNING",
    }.get(meta["view"], "TIMELINE")
    text(f"SHADOWSPILL / {label}", 0.62, 0.40, 12.1, 0.3, 11, "muted", True)
    text(
        args.title or f"{label.title()} training timeline",
        0.62,
        0.86,
        12.1,
        0.60,
        30,
        bold=True,
    )
    subtitle = args.subtitle or (
        f"{meta['view']} · {meta['step_seconds']:.2f} s · "
        f"peak {meta['peak_gib']:.2f} GiB"
    )
    show_summary = args.summary and meta["view"] in ("simulated", "traced")
    text(subtitle, 0.64, 1.50 if show_summary else 1.63, 12.05, 0.38, 15, "muted")
    for x, (key, label) in zip(
        (0.76, 2.32, 3.88, 5.86), MEMORY_CLASSES.items(), strict=True
    ):
        rect(x, 1.975 if show_summary else 2.185, 0.13, 0.13, COLORS[key])
        text(label, x + 0.2, 1.945 if show_summary else 2.155, 1.75, 0.3, 11, "muted")
    rect(0.62, 2.15 if show_summary else 2.46, 12.11, 3.75, "FFFFFF", True)
    slide.shapes.add_picture(
        str(args.outdir / "timeline.png"),
        Inches(0.80),
        Inches(2.21 if show_summary else 2.53),
        width=Inches(11.73),
    )
    if show_summary:
        rect(0.62, 5.95, 12.11, 1.12, "E9EFF8", True)
        for i, (label, value) in enumerate(summary_cards(meta["summary"])):
            x, y = 0.78 + (i % 6) * 1.995, 6.02 + (i // 6) * 0.52
            text(label, x, y, 1.89, 0.21, 12, "muted")
            text(value, x, y + 0.20, 1.89, 0.25, 14, bold=True)
    else:
        rect(0.62, 6.40, 12.11, 0.50, "E9EFF8", True)
        text(
            args.caption
            or f"{meta['step_seconds']:.2f} s per step · {meta['view']} timeline",
            0.82,
            6.49,
            11.7,
            0.30,
            15,
            bold=True,
        )
    slide.notes_slide.notes_text_frame.text = json.dumps(meta, indent=2)
    deck.save(args.outdir / "timeline-slide.pptx")


def summary_cards(summary):
    """The twelve summary statistics from the quickstart HTML, in slide order."""

    def number(key, spec, unit=""):
        value = summary.get(key)
        return "n/a" if value is None else f"{value:{spec}}{unit}"

    def rate(direction):
        value = number(direction + "_gbps", ".1f")
        assumed = number("assumed_" + direction + "_gbps", ".1f")
        return f"{value} / {assumed} GB/s"

    return [
        ("Step time", number("seconds", ".2f", " s")),
        ("Tokens/s", number("tokens_per_second", ",.0f")),
        ("Peak GPU memory", number("execution_peak_gib", ".2f", " GiB")),
        ("Peak spill", number("spill_peak_gib", ".2f", " GiB")),
        ("Stalled", number("idle_percent", ".1f", "%")),
        ("Recompute", number("recompute_percent", ".1f", "%")),
        ("Fetch lane busy", number("fetch_utilization_percent", ".1f", "%")),
        ("Evict lane busy", number("evict_utilization_percent", ".1f", "%")),
        ("Fetched", number("fetch_gib", ".1f", " GiB")),
        ("Evicted", number("evict_gib", ".1f", " GiB")),
        ("Fetch rate / planned blend", rate("fetch")),
        ("Evict rate / planned blend", rate("evict")),
    ]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "source",
        type=Path,
        help=(
            "Timeline HTML/JSON, timelines directory, "
            "or one quickstart result directory"
        ),
    )
    parser.add_argument(
        "--view", choices=("traced", "simulated", "all-save", "unconstrained")
    )
    parser.add_argument(
        "--budget-gib",
        type=float,
        help="Select a budget's fastest simulated plan; also label the budget line",
    )
    parser.add_argument(
        "--geometry", help="Optional exact report geometry, e.g. 8x8_1x8rp"
    )
    parser.add_argument("--resolution", help="Optional resolution, e.g. 3/4")
    parser.add_argument(
        "--list",
        action="store_true",
        help="List a report directory's available timeline choices",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("timeline-slide"),
        help="Directory for generated figures and slide",
    )
    parser.add_argument(
        "--time-max",
        type=float,
        help="Common x-axis endpoint for comparable plots; must cover all events",
    )
    parser.add_argument(
        "--transfers",
        choices=("auto", "show", "hide"),
        default="auto",
        help="Show Fetch/Evict lanes; auto omits empty lanes",
    )
    parser.add_argument(
        "--persistent-gradients-gib",
        type=float,
        help="Explicit illustration only: replace gradients with this constant bank",
    )
    parser.add_argument("--columns", type=int, default=1800, help="Overview time bins")
    parser.add_argument("--dpi", type=int, default=300, help="PNG resolution")
    parser.add_argument(
        "--pptx",
        action="store_true",
        help="Also create a one-slide PowerPoint for import into Slides",
    )
    parser.add_argument(
        "--summary",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Include the HTML's summary statistics "
            "in simulated/traced PowerPoint slides"
        ),
    )
    parser.add_argument("--title", help="PowerPoint title")
    parser.add_argument("--subtitle", help="PowerPoint subtitle")
    parser.add_argument(
        "--caption", help="PowerPoint takeaway when summary cards are not shown"
    )
    args = parser.parse_args()
    try:
        if args.list and args.source.is_file():
            raise ValueError(
                "--list needs a report directory, not a single timeline file."
            )
        if args.columns <= 0 or args.dpi <= 0:
            raise ValueError("--columns and --dpi must be positive.")
        for name in ("budget_gib", "time_max", "persistent_gradients_gib"):
            value = getattr(args, name)
            if value is not None and (
                not math.isfinite(value)
                or value < 0
                or (name != "persistent_gradients_gib" and value == 0)
            ):
                raise ValueError(
                    f"--{name.replace('_', '-')} must be finite and positive."
                )
        source = select_source(args)
        if source is None:
            return
        data = load_timeline(source)
        if args.view and data["view"] != args.view.replace("-", "_"):
            raise ValueError(
                f"Source contains {data['view']!r}, not requested {args.view!r}."
            )
        meta = render(data, source, args)
    except (ValueError, OSError, KeyError, StopIteration) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            dict(
                outdir=str(args.outdir.resolve()),
                view=meta["view"],
                seconds=meta["step_seconds"],
                peak_gib=meta["peak_gib"],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
