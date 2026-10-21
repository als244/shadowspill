"""Summarize an experiment: each engine's cases by outcome, the fastest case
of each engine within each GPU-memory budget, and each engine's frontier.

    python -m baselines.summary DIR [--budgets GIB [GIB ...]] [--publish DIR]

DIR is the output directory of ``baselines.experiment``. The summary reads
every row its sweeps hold, whichever run of the experiment wrote them, so an
experiment collected one engine at a time is summarized like one collected
in a single run; a case counts by its latest attempt. GPU memory is what an
engine held at its peak, a row's ``peak_reserved_bytes``: the execution
pool's capacity for ShadowSpill, which holds the pool whole, and the peak
its allocator reserved for any other engine; what was in use at that peak,
``peak_allocated_bytes``, is shown beside it. The budgets are ``--budgets``,
else the spec's ``summary_budgets_gib``; with neither, the summary gives the
frontiers alone. It names the commits the rows ran from and the span of the
experiment's runs, and embeds the report's two figures of each group of
shared settings (model, mode, sequence length, sequences per step, device
and world size) when ``DIR/report`` holds them. The summary is printed
and written to ``DIR/summary.md``, and as data to ``DIR/summary.json``.
``--publish`` also writes it, with the figures beside it, into a folder of
its own, the experiment's ``results`` in the repository, with no path of the
machine the experiment ran on and no link into its results directory.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from baselines.common import write_json
from baselines.common.results import read_rows
from baselines.report import FIGURES, groups, label, slug, title

#: How a case's latest outcome reads in a count, in the order counts list.
OUTCOMES = {
    "completed": "completed",
    "cuda_out_of_memory": "out of device memory",
    "failed": "failed",
    "timeout": "timed out",
    "interrupted": "interrupted",
    "pending": "not yet run",
}

#: A budget admits a peak this far above it, so that a pool sized exactly
#: at the budget is not lost to rounding.
SLACK_GIB = 1 / 1024


def sweeps_held(directory: Path, spec: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Every sweep of the experiment, in the spec's order and then any other
    the directory holds, with its engine, its case count and its cases'
    latest outcomes; a sweep that has not run has no count."""

    named = {sweep["name"]: sweep["engine"] for sweep in (spec or {}).get("sweeps", [])}
    held = sorted(
        path.parent.name for path in (directory / "sweeps").glob("*/manifest.json")
    )
    result = []
    for name in [*named, *(name for name in held if name not in named)]:
        path = directory / "sweeps" / name
        if not (path / "manifest.json").is_file():
            result.append(
                {"sweep": name, "engine": named[name], "cases": None, "outcomes": {}}
            )
            continue
        manifest = json.loads((path / "manifest.json").read_text())
        latest = {}
        if (path / "results.json").is_file():
            latest = {
                case["case"]: case["status"]
                for case in json.loads((path / "results.json").read_text())["cases"]
            }
        counts = Counter(
            latest.get(case["name"], "pending") for case in manifest["cases"]
        )
        result.append(
            {
                "sweep": name,
                "engine": manifest["engine"],
                "cases": len(manifest["cases"]),
                "outcomes": {
                    status: counts[status]
                    for status in sorted(counts, key=_outcome_order)
                },
            }
        )
    return result


def _outcome_order(status: str) -> int:
    order = list(OUTCOMES)
    return order.index(status) if status in order else len(order)


def _entry(
    item: dict[str, Any], keys: list[str], row: dict[str, Any], directory: Path
) -> dict[str, Any]:
    """A point as the summary lists it."""

    return {
        "tokens_per_second": item["tokens_per_second"],
        "step_seconds": item["step_seconds"],
        "gpu_gib": item["device_gib"],
        "gpu_in_use_gib": row["memory"]["peak_allocated_bytes"] / 2**30,
        "host_gib": item["host_gib"],
        "settings": label(item, keys) or "—",
        "row": str(Path(item["path"]).relative_to(directory)),
    }


def summarize(directory: Path, budgets: list[float] | None = None) -> dict[str, Any]:
    """The summary of everything the experiment in ``directory`` holds, as
    data: its engines, and per group of shared settings the
    fastest case of each engine within each budget and each engine's
    frontier, fastest first."""

    directory = Path(directory).resolve()
    record_path = directory / "experiment.json"
    record = json.loads(record_path.read_text()) if record_path.is_file() else {}
    spec = record.get("spec")
    if budgets is None:
        budgets = list((spec or {}).get("summary_budgets_gib") or [])
    listed = sweeps_held(directory, spec)
    order = list(dict.fromkeys(engine["engine"] for engine in listed))
    rows = []
    for path in sorted((directory / "sweeps").glob("*")):
        if path.is_dir():
            rows.extend(read_rows(path))
    by_path = {row["_path"]: row for row in rows}
    sections = []
    for key, group in groups(rows, "reserved").items():
        engines = sorted(
            group["engines"],
            key=lambda engine: (
                order.index(engine) if engine in order else len(order),
                engine,
            ),
        )

        def entry(item: dict[str, Any], keys=group["keys"]) -> dict[str, Any]:
            return _entry(item, keys[item["engine"]], by_path[item["path"]], directory)

        best = []
        for budget in sorted(budgets):
            fastest = {}
            for engine in engines:
                fitting = [
                    item
                    for item in group["points"]
                    if item["engine"] == engine
                    and item["device_gib"] <= budget + SLACK_GIB
                ]
                fastest[engine] = (
                    entry(max(fitting, key=lambda item: item["tokens_per_second"]))
                    if fitting
                    else None
                )
            best.append({"budget_gib": budget, "engines": fastest})
        sections.append(
            {
                "title": title(key),
                "figures": [
                    f"report/figures/{slug(key)}-{name}.png"
                    for name, _, _ in FIGURES
                    if (
                        directory / "report" / "figures" / f"{slug(key)}-{name}.png"
                    ).is_file()
                ],
                "engines": engines,
                "best": best,
                "frontiers": {
                    engine: [
                        entry(item) for item in reversed(group["frontiers"][engine])
                    ]
                    for engine in engines
                },
            }
        )
    commits = Counter(
        (
            ((row.get("provenance") or {}).get("source") or {}).get("git_head"),
            bool(((row.get("provenance") or {}).get("source") or {}).get("git_dirty")),
        )
        for row in rows
    )
    runs = [run for run in record.get("runs", []) if not run.get("dry_run")]
    spec_path = record.get("spec_path")
    return {
        "schema_version": 1,
        "name": Path(spec_path).parent.name if spec_path else directory.name,
        "experiment": str(directory),
        "commits": [
            {"commit": head, "dirty": dirty, "rows": count}
            for (head, dirty), count in sorted(
                commits.items(), key=lambda item: -item[1]
            )
        ],
        "runs": {
            "count": len(runs),
            "first_started_at": min((run["started_at"] for run in runs), default=None),
            "last_finished_at": max(
                (run["finished_at"] for run in runs if run.get("finished_at")),
                default=None,
            ),
        },
        "generated_at": datetime.datetime.now()
        .astimezone()
        .isoformat(timespec="seconds"),
        "budgets_gib": sorted(budgets),
        "sweeps": listed,
        "groups": sections,
    }


def counts(engine: dict[str, Any]) -> str:
    """``25 cases: 20 completed, 5 out of device memory``, or ``not run``."""

    if engine["cases"] is None:
        return "not run"
    outcomes = ", ".join(
        f"{count} {OUTCOMES.get(status, status)}"
        for status, count in engine["outcomes"].items()
    )
    return f"{engine['cases']} cases: {outcomes}"


def _measured(entry: dict[str, Any]) -> str:
    return (
        f"{entry['tokens_per_second']:,.1f} tok/s  {entry['step_seconds']:.2f} s/update"
        f"  GPU {entry['gpu_gib']:.2f} GiB ({entry['gpu_in_use_gib']:.2f} in use)"
        f"  host {entry['host_gib']:.2f} GiB  {entry['settings']}"
    )


def provenance(summary: dict[str, Any]) -> str:
    """``rows from fadc671c (306); 2 runs, 2026-10-21 17:40 EDT to ...``."""

    commits = ", ".join(
        f"{(item['commit'] or 'unknown')[:8]}{' dirty' if item['dirty'] else ''}"
        f" ({item['rows']})"
        for item in summary["commits"]
    )
    runs = summary["runs"]
    text = f"rows from {commits or 'no commit yet'}"
    if runs["count"]:
        first = _moment(runs["first_started_at"])
        last = _moment(runs["last_finished_at"]) if runs["last_finished_at"] else "now"
        text += f"; {runs['count']} runs, {first} to {last}"
    return text


def _moment(value: str) -> str:
    return (
        datetime.datetime.fromisoformat(value)
        .astimezone()
        .strftime("%Y-%m-%d %H:%M %Z")
    )


def render(summary: dict[str, Any]) -> str:
    """The summary as console text."""

    generated = datetime.datetime.fromisoformat(summary["generated_at"]).astimezone()
    lines = [
        f"SUMMARY {summary['experiment']}",
        f"  generated {generated.strftime('%Y-%m-%d %H:%M:%S %Z')}; GPU memory is"
        " what an engine held at its peak, what was in use beside it",
        f"  {provenance(summary)}",
        "  engines:",
    ]
    names = [
        engine["sweep"]
        + ("" if engine["engine"] == engine["sweep"] else f" ({engine['engine']})")
        for engine in summary["sweeps"]
    ]
    width = max((len(name) for name in names), default=0)
    for name, engine in zip(names, summary["sweeps"], strict=True):
        lines.append(f"    {name:<{width}}  {counts(engine)}")
    if not summary["groups"]:
        lines.append("  no completed or out-of-memory rows yet")
    for group in summary["groups"]:
        width = max(len(engine) for engine in group["engines"])
        lines.extend(["", f"  {group['title']}"])
        if group["best"]:
            lines.append("    fastest within each GPU budget:")
        for budget in group["best"]:
            lines.append(f"      {budget['budget_gib']:g} GiB")
            for engine, entry in budget["engines"].items():
                text = "nothing fits" if entry is None else _measured(entry)
                lines.append(f"        {engine:<{width}}  {text}")
        lines.append("    frontier of each engine, fastest first:")
        for engine, entries in group["frontiers"].items():
            lines.append(f"      {engine}")
            if not entries:
                lines.append("        no completed case")
            for entry in entries:
                lines.append(f"        {_measured(entry)}")
    return "\n".join(lines) + "\n"


MARKDOWN_COLUMNS = (
    "tok/s",
    "s/update",
    "GPU GiB",
    "In use GiB",
    "Host GiB",
    "Settings",
)


def _cells(entry: dict[str, Any] | None, links: bool = True) -> list[str]:
    if entry is None:
        return ["—"] * (len(MARKDOWN_COLUMNS) - 1) + ["nothing fits"]
    return [
        f"{entry['tokens_per_second']:,.1f}",
        f"{entry['step_seconds']:.2f}",
        f"{entry['gpu_gib']:.2f}",
        f"{entry['gpu_in_use_gib']:.2f}",
        f"{entry['host_gib']:.2f}",
        f"[{entry['settings']}]({entry['row']})" if links else entry["settings"],
    ]


def markdown(summary: dict[str, Any], published: bool = False) -> str:
    """The summary as ``summary.md``: in the experiment's directory, its
    figures in the report and its points linked to their rows; published,
    its figures beside it and nothing that names the results directory."""

    where = (
        f"`{summary['name']}`"
        if published
        else f"`{summary['name']}` in `{summary['experiment']}`"
    )
    lines = [
        f"# {summary['name']}: results",
        "",
        f"{where}, generated {summary['generated_at']}; {provenance(summary)}. GPU"
        " memory is what an engine held at its peak, with what was in use at that"
        " peak beside it.",
        "",
        "## Engines",
        "",
        "| Sweep | Engine | Outcomes |",
        "|---|---|---|",
    ]
    for engine in summary["sweeps"]:
        lines.append(f"| {engine['sweep']} | {engine['engine']} | {counts(engine)} |")
    for group in summary["groups"]:
        lines.extend(["", f"## {group['title']}"])
        for figure in group["figures"]:
            shown = f"figures/{Path(figure).name}" if published else figure
            lines.extend(["", f"![{group['title']}]({shown})"])
        if group["best"]:
            lines.extend(
                [
                    "",
                    "### Fastest within each GPU budget",
                    "",
                    "| Budget | Engine | " + " | ".join(MARKDOWN_COLUMNS) + " |",
                    "|---:|---|" + "---:|" * (len(MARKDOWN_COLUMNS) - 1) + "---|",
                ]
            )
            for budget in group["best"]:
                for engine, entry in budget["engines"].items():
                    cells = [
                        f"{budget['budget_gib']:g} GiB",
                        engine,
                        *_cells(entry, not published),
                    ]
                    lines.append("| " + " | ".join(cells) + " |")
        lines.extend(
            [
                "",
                "### Frontier of each engine, fastest first",
                "",
                "| Engine | " + " | ".join(MARKDOWN_COLUMNS) + " |",
                "|---|" + "---:|" * (len(MARKDOWN_COLUMNS) - 1) + "---|",
            ]
        )
        for engine, entries in group["frontiers"].items():
            for entry in entries:
                cells = [engine, *_cells(entry, not published)]
                lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write(directory: Path, summary: dict[str, Any]) -> dict[str, Any]:
    """Write ``summary.md`` and ``summary.json`` into the experiment's
    directory; the summary is returned."""

    directory = Path(directory)
    (directory / "summary.md").write_text(markdown(summary))
    write_json(directory / "summary.json", summary)
    return summary


def publish(directory: Path, summary: dict[str, Any], target: Path) -> Path:
    """Write the summary into ``target`` with its figures beside it, as the
    experiment's folder in the repository keeps it: no path of the machine
    it ran on and no link into its results directory."""

    directory, target = Path(directory), Path(target)
    target.mkdir(parents=True, exist_ok=True)
    for group in summary["groups"]:
        for figure in group["figures"]:
            source = directory / figure
            for path in (source, source.with_suffix(".svg")):
                if path.is_file():
                    (target / "figures").mkdir(exist_ok=True)
                    shutil.copy2(path, target / "figures" / path.name)
    (target / "summary.md").write_text(markdown(summary, published=True))
    data = copy.deepcopy(summary)
    data.pop("experiment")
    for group in data["groups"]:
        group["figures"] = [
            f"figures/{Path(figure).name}" for figure in group["figures"]
        ]
        entries = [*group["frontiers"].values()]
        entries += [list(budget["engines"].values()) for budget in group["best"]]
        for listed in entries:
            for entry in listed:
                if entry is not None:
                    entry.pop("row", None)
    write_json(target / "summary.json", data)
    return target / "summary.md"


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("directory", type=Path, help="an experiment's output")
    parser.add_argument(
        "--budgets",
        type=float,
        nargs="+",
        metavar="GIB",
        help="the GPU-memory budgets of the table (default: the spec's)",
    )
    parser.add_argument(
        "--publish",
        type=Path,
        metavar="DIR",
        help="also write the summary and its figures into DIR, the experiment's"
        " results folder",
    )
    args = parser.parse_args(argv)
    if not (args.directory / "sweeps").is_dir():
        parser.error(f"{args.directory} holds no sweeps")
    if args.budgets is not None and min(args.budgets) <= 0:
        parser.error("budgets are positive")
    try:
        summary = summarize(args.directory, args.budgets)
    except ValueError as error:
        parser.error(str(error))
    print(render(write(args.directory, summary)), end="", flush=True)
    if args.publish is not None:
        print(f"published: {publish(args.directory, summary, args.publish)}")


if __name__ == "__main__":
    main()
