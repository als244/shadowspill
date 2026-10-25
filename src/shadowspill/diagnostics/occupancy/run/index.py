"""Run index for pool-occupancy diagnostics."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from html import escape
from pathlib import Path

from ..html import _page, _table
from .records import (
    SUMMARY_COLUMNS,
    _Entry,
    _gib_of,
    _links,
    _seconds,
    _share,
    _summary_rows,
)


def _pages_of(entry: _Entry, base: Path) -> str:
    return _links(
        entry.directory.relative_to(base), [view.view for view in entry.views]
    )


def _others_of(entries: Sequence[_Entry], entry: _Entry, base: Path) -> str:
    kept = [
        item
        for item in entries
        if item.kind == "resolution"
        and item.budget == entry.budget
        and item.geometry == entry.geometry
    ]
    return "<br>".join(f"{item.label()}: {_pages_of(item, base)}" for item in kept)


def _geometry_rows(entries: Sequence[_Entry], budget: str, base: Path) -> str:
    """One row per geometry planned at ``budget``: the choice and what was kept."""

    rows = []
    for entry in entries:
        if entry.kind != "chosen" or entry.budget != budget:
            continue
        rows.append(
            [
                escape(entry.geometry),
                escape(entry.resolution or "?"),
                _seconds(entry.simulated),
                _seconds(entry.traced),
                _gib_of(entry.simulated.spill),
                _gib_of(entry.simulated.execution),
                _pages_of(entry, base),
                _others_of(entries, entry, base),
            ]
        )
    return _table(
        [
            "geometry",
            "recompute share chosen",
            "simulated step",
            "traced step",
            "spill peak",
            "execution peak",
            "the choice's pages",
            "other resolutions the search kept",
        ],
        rows,
    )


def _write_plan_index(out: Path, directory: Path, entries: Sequence[_Entry]) -> None:
    """``<budget>/<geometry>/index.html``: the choice and the others kept."""

    here = [entry for entry in entries if entry.directory.parent == directory]
    answer = next(entry for entry in here if entry.kind == "chosen")
    kept = [entry for entry in here if entry.kind == "resolution"]
    rows = [
        [
            answer.label(),
            _seconds(answer.simulated),
            _seconds(answer.traced),
            _gib_of(answer.simulated.spill),
            _gib_of(answer.simulated.execution),
            _pages_of(answer, directory),
        ]
    ] + [
        [
            item.label(),
            _seconds(item.simulated),
            "",
            _gib_of(item.simulated.spill),
            _gib_of(item.simulated.execution),
            _pages_of(item, directory),
        ]
        for item in kept
    ]
    (directory / "index.html").write_text(
        _page(
            f"Timelines: {answer.geometry} at {answer.budget}",
            f"<h1>{escape(answer.geometry)} at {escape(answer.budget)}</h1>"
            f'<p><a href="../index.html">all of {escape(answer.budget)}</a> · '
            f'<a href="../../all_save/{escape(answer.geometry)}/index.html">'
            f"{escape(answer.geometry)} at its cheapest, all save</a> · "
            '<a href="../../index.html">everything</a></p>'
            + _table(
                [
                    "recompute share",
                    "simulated step",
                    "traced step",
                    "spill peak",
                    "execution peak",
                    "pages",
                ],
                rows,
            ),
        )
    )


def _write_run_indexes(
    out: Path,
    run_root: Path,
    entries: Sequence[_Entry],
    missing: Sequence[tuple[str, str]],
    tokens: int | None,
) -> None:
    """The root table of contents, one per budget, ``all_save/``, the summary."""

    out.mkdir(parents=True, exist_ok=True)
    with (out / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for entry in entries:
            for row in _summary_rows(entry, out, tokens):
                writer.writerow(row)
    budgets = sorted(
        {entry.budget for entry in entries if entry.kind == "chosen"},
        key=lambda label: (
            _share(label.removesuffix("gib")) if label.endswith("gib") else 1e9,
            label,
        ),
    )
    floors = [entry for entry in entries if entry.kind == "all_save"]

    def ran_rows(base: Path) -> str:
        rows = [
            [
                escape(entry.budget),
                escape(entry.geometry),
                escape(entry.resolution or "?"),
                _seconds(entry.simulated),
                _seconds(entry.traced),
                _gib_of(entry.traced.spill if entry.traced else None),
                _gib_of(entry.simulated.execution),
                f'<a href="{escape(entry.budget)}/traced.html">traced.html</a>',
                _pages_of(entry, base),
            ]
            for entry in entries
            if entry.kind == "chosen" and entry.traced is not None
        ] + [
            [escape(stem), escape(why), "", "", "", "", "", "", ""]
            for stem, why in missing
        ]
        return _table(
            [
                "budget",
                "geometry that ran",
                "recompute share chosen",
                "simulated step",
                "traced step",
                "spill peak, traced",
                "execution peak",
                "traced page, at the budget",
                "the choice's pages",
            ],
            rows,
        )

    def floor_rows(base: Path) -> str:
        return _table(
            ["geometry", "compute floor, all save", "resident peak", "pages"],
            [
                [
                    escape(entry.geometry),
                    _seconds(entry.simulated),
                    _gib_of(entry.simulated.execution),
                    _pages_of(entry, base),
                ]
                for entry in floors
            ],
        )

    intro = (
        "<p>One page per plan and view: the step's summary, the spill and"
        " execution pools over the step by category, and the fetch, compute and"
        " evict lanes, on one zoom. Under each budget, each geometry's folder"
        " holds the plan the search chose there, under the share of the"
        " flexible groups it recomputes: its simulated page, its own"
        " unconstrained page -- its alternatives with nothing to plan around:"
        " tasks back to back at their profiled floor, every object resident,"
        " nothing spilled -- and, for the geometry that ran at that budget, the"
        " traced page on the device's clock. A search asked to keep its"
        " resolutions has the others beside it under their own shares, each with"
        " its simulated page and its own unconstrained page. Each budget that"
        " ran also carries a copy of its traced page at its own level. Each"
        " geometry also has its all-save page, every alternative at its"
        " cheapest, the floor the planner reports as unconstrained.</p>"
    )
    body = (
        f"<h1>Timelines: {escape(str(run_root))}</h1>"
        + intro
        + '<nav><a href="summary.csv">summary.csv, one row per page</a>'
        '<a href="all_save/index.html">all-save floors</a>'
        + "".join(
            f'<a href="{escape(budget)}/index.html">{escape(budget)}</a>'
            for budget in budgets
        )
        + "</nav><h2>Budgets that ran</h2>"
        + ran_rows(out)
        + "".join(
            f'<h2 id="{escape(budget)}"><a href="{escape(budget)}/index.html">'
            f"{escape(budget)}</a></h2>" + _geometry_rows(entries, budget, out)
            for budget in budgets
        )
        + "<h2>Each geometry at its cheapest alternatives, all save</h2>"
        + floor_rows(out)
    )
    (out / "index.html").write_text(_page(f"Timelines {run_root.name}", body))
    for budget in budgets:
        base = out / budget
        ran = next(
            (
                entry
                for entry in entries
                if entry.kind == "chosen"
                and entry.budget == budget
                and entry.traced is not None
            ),
            None,
        )
        (base / "index.html").write_text(
            _page(
                f"Timelines: {budget}",
                f"<h1>{escape(budget)}</h1>"
                '<p><a href="../index.html">everything</a> · '
                '<a href="../all_save/index.html">all-save floors</a></p>'
                + (
                    f"<p>This budget ran with {escape(ran.geometry)}, recompute"
                    f" {escape(ran.resolution or '?')}: "
                    f'<a href="traced.html">traced.html</a>, its step on the'
                    " device's clock, a copy of the one in that plan's folder.</p>"
                    if ran is not None
                    else ""
                )
                + _geometry_rows(entries, budget, base),
            )
        )
    floor_dir = out / "all_save"
    floor_dir.mkdir(parents=True, exist_ok=True)
    (floor_dir / "index.html").write_text(
        _page(
            "Timelines: all-save floors",
            "<h1>Every geometry at its cheapest alternatives, all save</h1>"
            '<p><a href="../index.html">everything</a></p>' + floor_rows(floor_dir),
        )
    )
