"""Draw the throughput-against-memory frontier of every model and hardware
pair from result rows, and index the figures.

    python -m baselines.report ROWS_ROOT [ROWS_ROOT ...] --output DIR \\
        [--memory allocated|reserved] [--dpi N]

Rows are every ``row.json`` under the roots, a case's attempts counting once
by the last that wrote a row. They are grouped by what an experiment's cases
share, the model, training mode, sequence length, sequences per step, device
and world size, which title the group's figures, and each group has two
figures: global tokens per
second against the peak device memory of the binding rank (``-device``),
and against the host memory the ranks' processes peaked at in sum
(``-host``). Every completed throughput row is a point, colored and marked
by engine. Each engine's Pareto frontier against device memory, the rows no
other row of that engine beats on both throughput and device memory, is
drawn solid and outlined on both figures, and the rest faint. The group's
table lists every point with the settings that vary among its engine's rows
of the group, and marks the frontier. Rows that ran out of device memory are
the boundary and are listed with the group. Each figure is written as PNG,
SVG and PDF, and ``index.html`` links them all.
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

#: A group's figures: each one's name, the column it draws throughput
#: against, and its axis caption.
FIGURES = (
    ("device", "device_gib", "peak device memory, {memory} (GiB)"),
    ("host", "host_gib", "peak host memory, sum over ranks (GiB)"),
)

#: One color per engine, in the order engines usually appear.
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


def group_key(row: dict[str, Any]) -> tuple[str, str, int, int, str, int]:
    """What a figure is of, the settings its experiment shares: the model,
    its training mode, the sequence length, the sequences per update, the
    device and the world size."""

    device = (row.get("hardware") or {}).get("device") or {}
    geometry = row["geometry"]
    length = int(geometry.get("sequence_length") or 0)
    tokens = int(geometry.get("tokens_per_step") or 0)
    return (
        str(row["model"]["model"]),
        str(row["model"].get("mode", "full")),
        length,
        tokens // length if length else 0,
        str(device.get("name") or "unknown device"),
        int(geometry.get("world_size") or 0),
    )


def settings_of(row: dict[str, Any]) -> dict[str, Any]:
    """The geometry an engine was asked for and its configuration, as one
    mapping; what distinguishes rows of one engine within a group."""

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
        "engine": row["engine"],
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


def slug(key: tuple[str, str, int, int, str, int]) -> str:
    text = " ".join(str(part) for part in key).lower()
    return re.sub(r"[^a-z0-9]+", "-", text).strip("-")


def title(key: tuple[str, str, int, int, str, int]) -> str:
    """The settings a group's cases share, as a figure's title."""

    model, mode, length, sequences, device, world_size = key
    parts = [model, {"full": "full training", "lora": "LoRA"}.get(mode, mode)]
    if length:
        parts.append(f"{length:,}-token sequences")
    if sequences:
        parts.append(f"{sequences} sequences per step")
    parts.extend([f"world size {world_size}", device])
    return ", ".join(parts)


def wrapped(text: str, width: int) -> str:
    """``text`` on as few lines of at most ``width`` characters as its
    comma-separated parts allow, a part never split."""

    lines: list[str] = []
    for part in text.split(", "):
        if lines and len(lines[-1]) + 2 + len(part) <= width:
            lines[-1] += ", " + part
        else:
            lines.append(part)
    return ",\n".join(lines)


def groups(
    rows: list[dict[str, Any]], memory: str
) -> dict[tuple[str, str, int, int, str, int], dict[str, Any]]:
    """Per group: its points, the setting keys that vary within each
    engine, the frontier of each engine on both axes, and the rows that
    ran out of device memory. A row that names no device, which a case
    refused before its engine probed the device writes, joins the one group
    that shares everything else of its key."""

    collected: dict[tuple[str, str, int, int, str, int], dict[str, Any]] = defaultdict(
        lambda: {"points": [], "boundary": []}
    )
    for row in rows:
        item = point(row, memory)
        if item is not None:
            collected[item["key"]]["points"].append(item)
        elif row["status"] == "cuda_out_of_memory":
            collected[group_key(row)]["boundary"].append(
                {
                    "engine": row["engine"],
                    "settings": settings_of(row),
                    "path": row.get("_path", ""),
                }
            )
    unknown = "unknown device"
    for key in [key for key in collected if key[4] == unknown]:
        shared = key[:4] + key[5:]
        known = [
            other
            for other in collected
            if other[4] != unknown and other[:4] + other[5:] == shared
        ]
        if len(known) == 1 and not collected[key]["points"]:
            collected[known[0]]["boundary"].extend(collected.pop(key)["boundary"])
    result = {}
    for key in sorted(collected):
        group = collected[key]
        everything = group["points"] + group["boundary"]
        engines = sorted({item["engine"] for item in everything})
        keys = {
            engine: varied(
                {
                    index: item["settings"]
                    for index, item in enumerate(everything)
                    if item["engine"] == engine
                }
            )
            for engine in engines
        }
        frontiers = {
            engine: frontier(
                [item for item in group["points"] if item["engine"] == engine],
                "device_gib",
                "tokens_per_second",
            )
            for engine in engines
        }
        result[key] = {
            "points": group["points"],
            "boundary": group["boundary"],
            "keys": keys,
            "engines": engines,
            "frontiers": frontiers,
        }
    return result


def draw(key, group, output: Path, *, memory: str, dpi: int) -> list[Path]:
    """The group's two figures, throughput against device memory and against
    host memory, as PNG, SVG and PDF under ``output/figures``."""

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot

    styles = {
        engine: (COLORS[index % len(COLORS)], MARKERS[index % len(MARKERS)])
        for index, engine in enumerate(group["engines"])
    }
    # Each engine's frontier against the device memory it held stands out on
    # both figures; the points it beats are drawn faint behind it.
    best = {
        id(item) for engine in group["engines"] for item in group["frontiers"][engine]
    }
    paths = []
    (output / "figures").mkdir(parents=True, exist_ok=True)
    for name, column, caption in FIGURES:
        figure, axis = pyplot.subplots(figsize=(7.5, 5.6))
        for engine in group["engines"]:
            color, marker = styles[engine]
            items = [item for item in group["points"] if item["engine"] == engine]
            for chosen, alpha, size, edge in (
                (False, 0.22, 22, "none"),
                (True, 1.0, 40, "black"),
            ):
                drawn = [item for item in items if (id(item) in best) == chosen]
                if not drawn:
                    continue
                axis.scatter(
                    [item[column] for item in drawn],
                    [item["tokens_per_second"] for item in drawn],
                    color=color,
                    marker=marker,
                    s=size,
                    alpha=alpha,
                    edgecolors=edge,
                    linewidths=0.6,
                    label=engine if chosen else None,
                    zorder=3 if chosen else 2,
                )
        axis.set_xlabel(caption.format(memory=memory))
        axis.set_ylabel("global tokens per second")
        axis.set_ylim(bottom=0)
        axis.grid(color="#e3e8ef", linewidth=0.8)
        axis.set_axisbelow(True)
        if group["engines"]:
            axis.legend(frameon=False, fontsize=8, loc="lower right")
        axis.set_title(wrapped(title(key), 64), fontsize=10)
        figure.tight_layout()
        for suffix in ("png", "svg", "pdf"):
            path = output / "figures" / f"{slug(key)}-{name}.{suffix}"
            figure.savefig(path, dpi=dpi)
            paths.append(path)
        pyplot.close(figure)
    return paths


def table_rows(group) -> list[dict[str, Any]]:
    """One record per point and per boundary row, frontier membership
    marked, for the tables and the CSV."""

    on_frontier = {
        id(item) for engine in group["engines"] for item in group["frontiers"][engine]
    }
    records = []
    for item in sorted(
        group["points"], key=lambda item: (item["engine"], item["device_gib"])
    ):
        records.append(
            {
                "engine": item["engine"],
                "settings": label(item, group["keys"][item["engine"]]),
                "tokens_per_second": item["tokens_per_second"],
                "step_seconds": item["step_seconds"],
                "useful_tflops_per_gpu": item["useful_tflops_per_gpu"],
                "device_gib": item["device_gib"],
                "host_gib": item["host_gib"],
                "frontier": "yes" if id(item) in on_frontier else "",
                "status": "completed",
                "path": item["path"],
            }
        )
    for item in sorted(group["boundary"], key=lambda item: item["engine"]):
        records.append(
            {
                "engine": item["engine"],
                "settings": label(item, group["keys"][item["engine"]]),
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
    ("engine", "Engine", False),
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
        "<p>Two figures per model, training mode, sequence length, sequences "
        "per step, device and world size, which title them: global "
        "tokens per second against the binding rank's peak device memory "
        f"({html.escape(memory)}), and against the ranks' summed peak host memory. "
        "Each point is a completed case, colored and marked by engine; each "
        "engine's Pareto frontier against device memory is solid and outlined "
        "on both figures, the rest faint. Each "
        "table lists the points with their settings, and out-of-memory rows "
        "are the boundary.</p>",
        '<p><a href="rows.csv">CSV of every point</a></p>',
    ]
    for section in sections:
        name = slug(section["key"])
        parts.append(f"<h2>{html.escape(title(section['key']))}</h2>")
        for figure, _, _ in FIGURES:
            parts.append(
                f'<img src="figures/{name}-{figure}.png" '
                f'alt="{html.escape(title(section["key"]))}, {figure} memory">'
            )
            parts.append(
                f'<p>{figure} memory: <a href="figures/{name}-{figure}.png">PNG</a> · '
                f'<a href="figures/{name}-{figure}.svg">SVG</a> · '
                f'<a href="figures/{name}-{figure}.pdf">PDF</a></p>'
            )
        parts.append(f'<p><a href="tables/{name}.md">Table</a></p>')
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
    for root in [root.resolve() for root in args.roots]:
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
        f"{len(rows)} rows, {len(sections)} groups of {len(FIGURES)} figures;"
        f" index: {output / 'index.html'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
