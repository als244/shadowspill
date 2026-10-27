"""Cli for pool-occupancy diagnostics."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from shadowspill.planner.plan_store import read_stored_selection

from .page import describe_plan, table, write_pages
from .plan import all_save, attribute, unconstrained
from .stored import program_path_for


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m shadowspill.diagnostics.occupancy",
        description="What occupies each pool over a step, by object category or role,"
        " and the fetch, compute and evict lanes.",
    )
    parser.add_argument(
        "selection",
        type=Path,
        nargs="?",
        help="a plan store's planning/results/<..>/selection.json",
    )
    parser.add_argument(
        "--program",
        type=Path,
        help="its program.json; found beside it in the store by default",
    )
    parser.add_argument(
        "--step",
        type=Path,
        help="the traced step's diagnostics JSON: adds the view on the device's"
        " clock, both pools and the lanes",
    )
    parser.add_argument(
        "--unconstrained",
        action="store_true",
        help="also the plan's own floor: its alternatives as it fixed them, with"
        " nothing else to plan around -- tasks back to back at their profiled"
        " floor, every object resident, nothing spilled",
    )
    parser.add_argument(
        "--all-save",
        action="store_true",
        help="also the program at its cheapest alternatives (all save for"
        " save-or-recompute choices), the floor the planner reports as"
        " unconstrained; with --program and no selection, only that view",
    )
    parser.add_argument("--by", choices=("category", "role"), default="category")
    parser.add_argument(
        "--tokens-per-step",
        type=int,
        help="the step's tokens, for the tokens-per-second figure in the summary",
    )
    parser.add_argument(
        "--at",
        default="",
        help="comma-separated seconds into the step to add columns for",
    )
    parser.add_argument(
        "--html",
        type=Path,
        metavar="DIRECTORY",
        help="write the pages there: one per view, with the summary, both pools"
        " and the lanes, and an index",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the snapshots as JSON instead of tables",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.selection is None and not (arguments.program and arguments.all_save):
        raise SystemExit("give a selection.json or --program with --all-save")
    selection: dict[str, Any] = (
        {}
        if arguments.selection is None
        else read_stored_selection(arguments.selection)
    )
    program_path = arguments.program or program_path_for(arguments.selection, selection)
    program = json.loads(program_path.read_text())
    try:
        views = [] if arguments.selection is None else [attribute(selection, program)]
        if arguments.step:
            diagnostics = json.loads(arguments.step.read_text())
            views.append(attribute(selection, program, diagnostics=diagnostics))
        if arguments.unconstrained and arguments.selection is not None:
            views.append(unconstrained(program, selection.get("selections", ())))
        if arguments.all_save:
            views.append(all_save(program))
    except ValueError as error:
        raise SystemExit(str(error)) from error
    at_ns = [
        int(float(value) * 1e9) for value in arguments.at.split(",") if value.strip()
    ]
    if arguments.json:
        payload = {
            result.view: {
                occupancy.pool: {
                    "peak_bytes": occupancy.peak()[0],
                    "peak_ns": occupancy.peak()[1],
                    "reported_peak_bytes": occupancy.reported_peak_bytes,
                    "capacity_bytes": occupancy.capacity_bytes,
                    "at_peak": occupancy.at(occupancy.peak()[1], by=arguments.by),
                    "at": {str(t): occupancy.at(t, by=arguments.by) for t in at_ns},
                    "own_peaks": {
                        name: list(value)
                        for name, value in occupancy.peaks_by(arguments.by).items()
                    },
                }
                for occupancy in result.pools
                if occupancy.intervals
            }
            for result in views
        }
        print(json.dumps(payload, indent=2))
    else:
        first = views[0]
        print(
            f"{arguments.selection or program_path}: {len(first.facts.objects)}"
            f" objects, {len(first.facts.task_order)} executing tasks"
        )
        for result in views:
            print(f"\n== {result.clock_label} ==")
            for occupancy in result.pools:
                if occupancy.intervals:
                    print()
                    print(table(occupancy, arguments.by, at_ns))
    if arguments.html:
        source = arguments.selection or program_path
        pages = write_pages(
            views,
            arguments.html,
            title=source.parent.name[:12],
            by=arguments.by,
            tokens_per_step=arguments.tokens_per_step,
            plan=describe_plan(selection),
            plan_by_view={
                "unconstrained": describe_plan(
                    {"selections": selection.get("selections", ())}, unconstrained=True
                ),
                "all_save": describe_plan({}, unconstrained=True),
            },
        )
        print(f"\npages: {pages[0]}")
    return 0
