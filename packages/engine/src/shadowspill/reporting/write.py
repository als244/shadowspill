"""CPU reports from existing artifacts; saving a Plan does not write a report."""

from __future__ import annotations

import csv
import json
from html import escape
from pathlib import Path
from typing import Any

from shadowspill.artifacts.plan import Plan
from shadowspill.artifacts.sweep import SweepResult
from shadowspill.diagnostics.occupancy.page import html_parts, script_json

from .occupancy import occupancy_data
from .plot import plot_plan
from .rows import task_rows, transfer_rows


def _table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "<p>None</p>"
    keys = list(rows[0])
    header = "".join(f"<th>{escape(key)}</th>" for key in keys)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(str(row[key]))}</td>" for key in keys) + "</tr>"
        for row in rows
    )
    return (
        f"<div class='scroll'><table><thead>{header}</thead>"
        f"<tbody>{body}</tbody></table></div>"
    )


def _page(title: str, content: str) -> str:
    return f"""<!doctype html><html lang="en"><meta charset="utf-8">
<title>{escape(title)}</title><style>
body{{font:15px system-ui;margin:2rem auto;max-width:1250px;
padding:0 1rem;color:#233044}}
table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}
th,td{{padding:.45rem;text-align:left;border-bottom:1px solid #ddd;white-space:nowrap}}
th{{background:#edf1f6}}.scroll{{overflow:auto}}img{{width:100%}}summary{{cursor:pointer}}
</style><h1>{escape(title)}</h1>{content}</html>"""


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def write_report(
    result: Plan | SweepResult, path: str | Path, *, plots: bool = True
) -> Path:
    """Write an HTML report of a Plan or a planning sweep into a directory.

    The report is made from the result alone and needs no device. The
    directory is created if it is missing, and report files already in it
    are overwritten.

    For a `Plan`, the directory receives ``index.html``, ``tasks.csv``,
    ``transfers.csv``, ``summary.json``, ``timeline.json`` (the tasks, the
    transfers, and why the Plan is simulation only, if it is),
    ``occupancy.html`` (an interactive page of pool occupancy over the step)
    and, with ``plots``, ``timeline.svg`` and ``timeline.png``. For a
    `SweepResult`, it receives ``cases.csv``, one row per case with its
    budgets, duration and status; an ``index.html`` linking every case that
    planned; and each planned case's report in a subdirectory named by its
    position, ``case-0000``, ``case-0001`` and so on.

    Parameters
    ----------
    result : Plan or SweepResult
        The Plan or sweep to report.
    path : str or path-like
        Directory to write into.
    plots : bool, default True
        Whether to draw the simulated timeline of compute, transfers and
        memory.

    Returns
    -------
    Path
        The report's ``index.html``.

    Raises
    ------
    TypeError
        If ``result`` is neither a `Plan` nor a `SweepResult`.
    """
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True)
    if isinstance(result, SweepResult):
        content = []
        rows = []
        for index, case in enumerate(result.cases):
            label = f"case-{index:04d}"
            if case.plan is not None:
                write_report(case.plan, destination / label, plots=plots)
                content.append(
                    f'<li><a href="{label}/index.html">{escape(case.candidate)}'
                    f" — {case.memory_budgets.execution_bytes} bytes</a></li>"
                )
            rows.append(
                {
                    "candidate": case.candidate,
                    "execution_bytes": case.memory_budgets.execution_bytes,
                    "spill_bytes": case.memory_budgets.spill_bytes,
                    "duration_seconds": None
                    if case.plan is None
                    else case.plan.summary["duration_seconds"],
                    "status": "planned"
                    if case.plan is not None
                    else case.failure["type"],
                    "failure": case.failure.get("message", ""),
                }
            )
        _csv(destination / "cases.csv", rows)
        html = _page(
            "Planning sweep", _table(rows) + "<ul>" + "".join(content) + "</ul>"
        )
    elif isinstance(result, Plan):
        tasks, transfers = task_rows(result), transfer_rows(result)
        _csv(destination / "tasks.csv", tasks)
        _csv(destination / "transfers.csv", transfers)
        (destination / "summary.json").write_text(
            json.dumps(result.summary, indent=2) + "\n"
        )
        (destination / "timeline.json").write_text(
            json.dumps(
                {
                    "tasks": tasks,
                    "transfers": transfers,
                    "simulation_only_reasons": result.simulation_only_reasons,
                },
                indent=2,
            )
            + "\n"
        )
        if plots:
            plot_plan(result, destination)
        prefix, suffix = html_parts(result.program.name or "Plan report")
        (destination / "occupancy.html").write_text(
            prefix + script_json(occupancy_data(result)) + suffix
        )
        note = (
            (
                "<p>Simulation only: "
                + escape("; ".join(result.simulation_only_reasons))
                + "</p>"
            )
            if result.simulation_only_reasons
            else "<p>Plan includes physical allocation evidence.</p>"
        )
        figure = (
            '<img src="timeline.svg" alt="Simulated compute, transfers and memory">'
            if plots
            else ""
        )
        html = _page(
            result.program.name or "Plan report",
            note + '<p><a href="occupancy.html">Interactive report: summary, lanes, '
            "execution and spill memory breakdowns</a></p>"
            + _table([result.summary])
            + figure
            + "<details open><summary>Tasks</summary>"
            + _table(tasks)
            + "</details>"
            + "<details><summary>Transfers — host copies and collective "
            "traffic</summary>"
            + "<p>Copy throughput includes latency. Composite transfers have no "
            "isolated copy-rate value; their individual phase timestamps are "
            "not retained by the simulator. Collective bytes are per-rank "
            "ring-equivalent traffic.</p>" + _table(transfers) + "</details>",
        )
    else:
        raise TypeError("write_report requires a Plan or SweepResult")
    page = destination / "index.html"
    page.write_text(html)
    return page
