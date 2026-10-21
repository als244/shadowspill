"""Draw the throughput-against-memory frontier of every model and hardware
pair from result rows, and index the figures.

    python -m baselines.report ROWS_ROOT [ROWS_ROOT ...] --output DIR \\
        [--memory allocated|reserved] [--dpi N]

Rows are every ``row.json`` under the roots, a case's attempts counting once
by the last that wrote a row. They are grouped by model, training mode,
device and world size, and each group is one figure of two panels: global
tokens per second against the peak device memory of the binding rank, and
against the host memory the ranks' processes peaked at in sum. Every
completed throughput row is a point, colored by harness; the Pareto frontier
of each harness, the rows no other row of that harness beats on both axes,
is drawn as a step and its points are labeled with the settings that vary
among that harness's rows of the group. Rows that ran out of device memory
are the boundary and are listed with the group. Each figure is written as
PNG, SVG and PDF with a table of its points, and ``index.html`` links them
all.
"""

from __future__ import annotations

import argparse
import csv
import html
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from baselines.common.results import read_rows, varied

#: The device-memory peak a point's abscissa is, by the ``--memory`` choice.
MEMORY_KEYS = {
    "allocated": "peak_allocated_bytes",
    "reserved": "peak_reserved_bytes",
}

#: Geometry fields derived from the others, left out of a point's settings;
#: a system's replicas follow from the parallelism its configuration names.
DERIVED_GEOMETRY = (
    "accumulation",
    "replicas",
    "sequences_per_microbatch",
    "tokens_per_rank_per_step",
)

#: One color per harness, in the order harnesses usually appear.
COLORS = (
    "#3268db",
    "#008b83",
    "#d9480f",
    "#7048e8",
    "#c2255c",
    "#2b8a3e",
    "#e67700",
)
MARKERS = ("o", "D", "s", "^", "v", "P", "X")

INDEX_STYLE = """
body { max-width:1200px; margin:40px auto; padding:0 24px;
       font-family:Georgia,serif; color:#14263f; background:#f7f9fc }
h1, h2 { font-weight:normal }
img { width:100%; background:white }
table { border-collapse:collapse; width:100%; background:white; margin:16px 0 }
th,td { text-align:left; padding:8px; border-bottom:1px solid #e3e8ef;
        font-size:14px; vertical-align:top }
td.number { text-align:right; font-variant-numeric:tabular-nums }
a { color:#3268db }
p { line-height:1.6 }
"""


def group_key(row: dict[str, Any]) -> tuple[str, str, str, int]:
    """What a figure is of: the model, its training mode, the device and
    the world size."""

    device = (row.get("hardware") or {}).get("device") or {}
    return (
        str(row["model"]["model"]),
        str(row["model"].get("mode", "full")),
        str(device.get("name") or "unknown device"),
        int(row["geometry"].get("world_size") or 0),
    )


def settings_of(row: dict[str, Any]) -> dict[str, Any]:
    """The geometry a harness was asked for and its configuration, as one
    mapping; what distinguishes rows of one harness within a group."""

    geometry = {
        key: value
        for key, value in row["geometry"].items()
        if key not in DERIVED_GEOMETRY
    }
    return {**geometry, **row["configuration"]}


def point(row: dict[str, Any], memory: str = "allocated") -> dict[str, Any] | None:
    """A completed throughput row as a point; ``None`` for any other row."""

    if row["status"] != "completed":
        return None
    measurement = row["measurement"] or {}
    if not measurement.get("performance_result", True):
        return None
    peaks = row["memory"] or {}
    return {
        "harness": row["harness"],
        "key": group_key(row),
        "tokens_per_second": float(measurement["global_tokens_per_second"]),
        "step_seconds": float(measurement["step_seconds"]),
        "useful_tflops_per_gpu": measurement.get("useful_tflops_per_gpu"),
        "device_gib": float(peaks[MEMORY_KEYS[memory]]) / 2**30,
        "host_gib": float(peaks["host_peak_rss_bytes"]) / 2**30,
        "settings": settings_of(row),
        "precision": row["precision"],
        "path": row.get("_path", ""),
    }


def frontier(points: list[dict[str, Any]], x: str, y: str) -> list[dict[str, Any]]:
    """The points no other point beats on both axes: less of ``x`` and no
    less of ``y``, or no more of ``x`` and more of ``y``; in ascending
    ``x``."""

    result: list[dict[str, Any]] = []
    best = float("-inf")
    for item in sorted(points, key=lambda item: (item[x], -item[y])):
        if item[y] > best:
            result.append(item)
            best = item[y]
    return result


def label(item: dict[str, Any], keys: list[str]) -> str:
    """``key=value`` for the settings that vary within the group."""

    parts = []
    for key in keys:
        value = item["settings"].get(key)
        if isinstance(value, float):
            value = f"{value:g}"
        parts.append(f"{key}={value}")
    return ", ".join(parts)


def slug(key: tuple[str, str, str, int]) -> str:
    text = " ".join(str(part) for part in key).lower()
    return re.sub(r"[^a-z0-9]+", "-", text).strip("-")


def title(key: tuple[str, str, str, int]) -> str:
    model, mode, device, world_size = key
    return f"{model} ({mode}) on {world_size} x {device}"


def groups(
    rows: list[dict[str, Any]], memory: str
) -> dict[tuple[str, str, str, int], dict[str, Any]]:
    """Per group: its points, the setting keys that vary within each
    harness, the frontier of each harness on both axes, and the rows that
    ran out of device memory."""

    collected: dict[tuple[str, str, str, int], dict[str, Any]] = defaultdict(
        lambda: {"points": [], "boundary": []}
    )
    for row in rows:
        item = point(row, memory)
        if item is not None:
            collected[item["key"]]["points"].append(item)
        elif row["status"] == "cuda_out_of_memory":
            collected[group_key(row)]["boundary"].append(
                {
                    "harness": row["harness"],
                    "settings": settings_of(row),
                    "path": row.get("_path", ""),
                }
            )
    result = {}
    for key in sorted(collected):
        group = collected[key]
        everything = group["points"] + group["boundary"]
        harnesses = sorted({item["harness"] for item in everything})
        keys = {
            harness: varied(
                {
                    index: item["settings"]
                    for index, item in enumerate(everything)
                    if item["harness"] == harness
                }
            )
            for harness in harnesses
        }
        frontiers = {
            harness: {
                axis: frontier(
                    [item for item in group["points"] if item["harness"] == harness],
                    axis,
                    "tokens_per_second",
                )
                for axis in ("device_gib", "host_gib")
            }
            for harness in harnesses
        }
        result[key] = {
            "points": group["points"],
            "boundary": group["boundary"],
            "keys": keys,
            "harnesses": harnesses,
            "frontiers": frontiers,
        }
    return result


def draw(key, group, output: Path, *, memory: str, dpi: int) -> list[Path]:
    """The group's figure as PNG, SVG and PDF under ``output/figures``."""

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot

    figure, axes = pyplot.subplots(1, 2, figsize=(15, 5.6))
    styles = {
        harness: (COLORS[index % len(COLORS)], MARKERS[index % len(MARKERS)])
        for index, harness in enumerate(group["harnesses"])
    }
    for axis, column, caption in (
        (axes[0], "device_gib", f"peak device memory, {memory} (GiB)"),
        (axes[1], "host_gib", "peak host memory, sum over ranks (GiB)"),
    ):
        widest = max((item[column] for item in group["points"]), default=0.0)
        for harness in group["harnesses"]:
            color, marker = styles[harness]
            items = [item for item in group["points"] if item["harness"] == harness]
            if not items:
                continue
            axis.scatter(
                [item[column] for item in items],
                [item["tokens_per_second"] for item in items],
                color=color,
                marker=marker,
                s=28,
                alpha=0.6,
                label=harness,
            )
            best = group["frontiers"][harness][column]
            if best:
                axis.step(
                    [item[column] for item in best],
                    [item["tokens_per_second"] for item in best],
                    where="post",
                    color=color,
                    linewidth=1.6,
                )
                for item in best:
                    # Labels of points in the right third read leftwards, so
                    # the axis edge does not cut them.
                    leftwards = item[column] > 0.67 * widest
                    text = axis.annotate(
                        label(item, group["keys"][item["harness"]]),
                        (item[column], item["tokens_per_second"]),
                        fontsize=6,
                        xytext=(-4 if leftwards else 4, 2),
                        textcoords="offset points",
                        horizontalalignment="right" if leftwards else "left",
                        color=color,
                        clip_on=True,
                    )
                    text.set_in_layout(False)
        axis.set_xlabel(caption)
        axis.set_ylabel("global tokens per second")
        axis.set_ylim(bottom=0)
        axis.grid(color="#e3e8ef", linewidth=0.8)
        axis.set_axisbelow(True)
        if group["harnesses"]:
            axis.legend(frameon=False, fontsize=8, loc="lower right")
    figure.suptitle(title(key))
    figure.tight_layout()
    paths = []
    (output / "figures").mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg", "pdf"):
        path = output / "figures" / f"{slug(key)}.{suffix}"
        figure.savefig(path, dpi=dpi)
        paths.append(path)
    pyplot.close(figure)
    return paths


def table_rows(group) -> list[dict[str, Any]]:
    """One record per point and per boundary row, frontier membership
    marked, for the tables and the CSV."""

    on_frontier = {
        axis: {
            id(item)
            for harness in group["harnesses"]
            for item in group["frontiers"][harness][axis]
        }
        for axis in ("device_gib", "host_gib")
    }
    records = []
    for item in sorted(
        group["points"], key=lambda item: (item["harness"], item["device_gib"])
    ):
        records.append(
            {
                "harness": item["harness"],
                "settings": label(item, group["keys"][item["harness"]]),
                "tokens_per_second": item["tokens_per_second"],
                "step_seconds": item["step_seconds"],
                "useful_tflops_per_gpu": item["useful_tflops_per_gpu"],
                "device_gib": item["device_gib"],
                "host_gib": item["host_gib"],
                "frontier": ", ".join(
                    name
                    for name, axis in (("device", "device_gib"), ("host", "host_gib"))
                    if id(item) in on_frontier[axis]
                ),
                "status": "completed",
                "path": item["path"],
            }
        )
    for item in sorted(group["boundary"], key=lambda item: item["harness"]):
        records.append(
            {
                "harness": item["harness"],
                "settings": label(item, group["keys"][item["harness"]]),
                "tokens_per_second": None,
                "step_seconds": None,
                "useful_tflops_per_gpu": None,
                "device_gib": None,
                "host_gib": None,
                "frontier": "",
                "status": "cuda_out_of_memory",
                "path": item["path"],
            }
        )
    return records


TABLE_COLUMNS = (
    ("harness", "Harness", False),
    ("settings", "Settings", False),
    ("status", "Status", False),
    ("tokens_per_second", "tok/s", True),
    ("step_seconds", "s/update", True),
    ("useful_tflops_per_gpu", "TFLOP/s/GPU", True),
    ("device_gib", "Device GiB", True),
    ("host_gib", "Host GiB", True),
    ("frontier", "Frontier", False),
    ("path", "Row", False),
)


def _format(key: str, value: Any) -> str:
    if value is None:
        return "—"
    if key == "tokens_per_second":
        return f"{value:,.0f}"
    if key == "step_seconds":
        return f"{value:.4f}"
    if key in ("useful_tflops_per_gpu", "device_gib", "host_gib"):
        return f"{value:.2f}"
    return str(value)


def markdown_table(records: list[dict[str, Any]]) -> str:
    lines = [
        "| " + " | ".join(heading for _, heading, _ in TABLE_COLUMNS) + " |",
        "|"
        + "|".join("---:" if number else "---" for _, _, number in TABLE_COLUMNS)
        + "|",
    ]
    for record in records:
        lines.append(
            "| "
            + " | ".join(_format(key, record[key]) for key, _, _ in TABLE_COLUMNS)
            + " |"
        )
    return "\n".join(lines) + "\n"


def html_table(records: list[dict[str, Any]]) -> str:
    head = "".join(
        f"<th>{html.escape(heading)}</th>" for _, heading, _ in TABLE_COLUMNS
    )
    body = []
    for record in records:
        cells = "".join(
            f'<td class="{"number" if number else "text"}">'
            f"{html.escape(_format(key, record[key]))}</td>"
            for key, _, number in TABLE_COLUMNS
        )
        body.append(f"<tr>{cells}</tr>")
    return f"<table><tr>{head}</tr>{''.join(body)}</table>"


def write_index(output: Path, sections: list[dict[str, Any]], memory: str) -> None:
    parts = [
        "<!doctype html>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>Baselines frontier</title>",
        f"<style>{INDEX_STYLE}</style>",
        "<h1>Throughput against memory</h1>",
        "<p>One figure per model, training mode, device and world size: global "
        "tokens per second against the binding rank's peak device memory "
        f"({html.escape(memory)}) and against the ranks' summed peak host memory. "
        "A step is one harness's Pareto frontier, labeled with the settings that "
        "vary within the group; out-of-memory rows are the boundary.</p>",
        '<p><a href="rows.csv">CSV of every point</a></p>',
    ]
    for section in sections:
        name = slug(section["key"])
        parts.append(f"<h2>{html.escape(title(section['key']))}</h2>")
        parts.append(
            f'<img src="figures/{name}.png" alt="{html.escape(title(section["key"]))}">'
        )
        parts.append(
            f'<p><a href="figures/{name}.png">PNG</a> · '
            f'<a href="figures/{name}.svg">SVG</a> · '
            f'<a href="figures/{name}.pdf">PDF</a> · '
            f'<a href="tables/{name}.md">Table</a></p>'
        )
        parts.append(html_table(section["records"]))
    (output / "index.html").write_text("\n".join(parts) + "\n")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("roots", nargs="+", type=Path, help="where rows are read")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--memory", choices=tuple(MEMORY_KEYS), default="allocated")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args(argv)
    rows = []
    for root in args.roots:
        for row in read_rows(root):
            relative = Path(row["_path"]).relative_to(root.resolve())
            row["_path"] = str(Path(root.name) / relative)
            rows.append(row)
    if not rows:
        parser.error("no row.json under the given roots")
    output = args.output.resolve()
    (output / "tables").mkdir(parents=True, exist_ok=True)
    sections = []
    with (output / "rows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["group", *(key for key, _, _ in TABLE_COLUMNS)]
        )
        writer.writeheader()
        for key, group in groups(rows, args.memory).items():
            draw(key, group, output, memory=args.memory, dpi=args.dpi)
            records = table_rows(group)
            (output / "tables" / f"{slug(key)}.md").write_text(
                f"# {title(key)}\n\n" + markdown_table(records)
            )
            for record in records:
                writer.writerow({"group": title(key), **record})
            sections.append({"key": key, "records": records})
    write_index(output, sections, args.memory)
    print(
        f"{len(rows)} rows, {len(sections)} figures; index: {output / 'index.html'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
