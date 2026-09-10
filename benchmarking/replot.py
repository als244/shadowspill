"""Redraw a run's figures from its raw data, over a subset of what it measured.

A quickstart writes `figures/raw_data` beside the figures themselves, and that
directory is the record: the run-side tables carry every executed budget and
every step it ran, and `search.json` carries every point the search evaluated.
This redraws from that record alone, so narrowing a figure costs a second, not a
rerun, and the original tree is never written to.

    python -m benchmarking.replot RAW_DATA OUTPUT
    python -m benchmarking.replot RAW_DATA OUTPUT --budget-gib 8,16,24
    python -m benchmarking.replot RAW_DATA OUTPUT --geometry 64x1,16x4

`RAW_DATA` is a run's `figures/raw_data` directory, or the `figures` directory
above it, or the run directory above that -- whichever is convenient. `OUTPUT` is
created, and the figure tree is written inside it exactly as a run writes one, so
a redraw and an original are read the same way.

Filters name what to keep. A budget is named in gibibytes as it appears on the
axis, so `16` selects the 16 GiB point and `29` selects a 29.0137 GiB one; a
geometry is named as `microbatch x accumulation`, the way the search reports it.
Naming nothing keeps everything.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Sequence
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

from shadowspill.plots import RunBudgetOutcome, plot_step_run, plot_step_search
from shadowspill.pytorch.step_search import StepSearchReport

_GIB = 1 << 30


def _raw_data_directory(value: Path) -> Path:
    """Accept the raw-data directory, the figures directory, or the run root.

    A run root carries its own `search.json` beside the one in `raw_data`, so a
    directory holding only that is not evidence of having found the raw data.
    The run tables settle it: whichever candidate holds `run_budgets.csv` is the
    raw data, and a lone `search.json` is the fallback for a planning-only run
    that executed nothing and so wrote no tables.
    """

    candidates = (value, value / "raw_data", value / "figures" / "raw_data")
    for candidate in candidates:
        if (candidate / "run_budgets.csv").is_file():
            return candidate
    for candidate in reversed(candidates):
        if (candidate / "search.json").is_file():
            return candidate
    raise SystemExit(
        f"{value}: no run_budgets.csv or search.json here, nor under raw_data/"
        " or figures/raw_data/. Point this at a quickstart's raw data."
    )


def _budget_filter(value: str) -> tuple[float, ...]:
    try:
        return tuple(float(item) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{value}: {error}") from error


def _geometry_filter(value: str) -> tuple[tuple[int, int], ...]:
    pairs: list[tuple[int, int]] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        head, _, tail = item.lower().partition("x")
        try:
            pairs.append((int(head), int(tail)))
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                f"{item}: expected a geometry as MICROBATCH x ACCUMULATION,"
                " for example 16x4"
            ) from error
    return tuple(pairs)


def _resolution_filter(value: str) -> tuple[float, ...]:
    """Recompute shares, as the search names them: `0,1/4,1/2,3/4,1`."""

    shares: list[float] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            shares.append(float(Fraction(item)))
        except (ValueError, ZeroDivisionError) as error:
            raise argparse.ArgumentTypeError(
                f"{item}: expected a share as a fraction or decimal, for"
                " example 1/2 or 0.5"
            ) from error
    return tuple(shares)


def _keeps_budget(gibibytes: float, wanted: Sequence[float]) -> bool:
    """Whether a budget is one the caller named.

    A budget resolves to an exact byte count, so the 30 GiB a caller asked for
    reaches the table as 29.0137. Matching to a tenth lets a filter name a budget
    the way the axis does rather than the way the pool resolved it.
    """

    return not wanted or any(abs(gibibytes - item) < 0.5 for item in wanted)


def _float(row: dict[str, str], name: str, default: float = 0.0) -> float:
    """Read a column that older runs may not have written."""

    value = row.get(name)
    return default if value in (None, "") else float(value)


def _run_entries(
    directory: Path, budgets: Sequence[float]
) -> tuple[RunBudgetOutcome, ...]:
    table = directory / "run_budgets.csv"
    if not table.is_file():
        return ()
    steps: dict[float, list[tuple[int, float]]] = {}
    step_table = directory / "steps.csv"
    if step_table.is_file():
        with step_table.open(newline="") as handle:
            for row in csv.DictReader(handle):
                gib = float(row["execution_budget_gib"])
                steps.setdefault(gib, []).append(
                    (int(row["step"]), float(row["seconds"]))
                )
    entries: list[RunBudgetOutcome] = []
    with table.open(newline="") as handle:
        for row in csv.DictReader(handle):
            gib = float(row["execution_budget_gib"])
            if not _keeps_budget(gib, budgets):
                continue
            ordered = tuple(seconds for _step, seconds in sorted(steps.get(gib, [])))
            entries.append(
                RunBudgetOutcome(
                    execution_budget_bytes=round(gib * _GIB),
                    simulated_step_seconds=_float(row, "simulated_step_seconds"),
                    measured_step_seconds=_float(row, "measured_step_seconds"),
                    profiled_task_seconds=_float(row, "profiled_task_seconds"),
                    real_task_seconds=_float(row, "real_task_seconds"),
                    simulated_idle_seconds=_float(row, "simulated_idle_seconds"),
                    real_idle_seconds=_float(row, "real_idle_seconds"),
                    prologue_seconds=_float(row, "prologue_seconds"),
                    terminal_tail_seconds=_float(row, "terminal_tail_seconds"),
                    real_terminal_tail_seconds=_float(
                        row, "real_terminal_tail_seconds"
                    ),
                    recomputation_seconds=_float(row, "recomputation_seconds"),
                    step_seconds=ordered,
                )
            )
    return tuple(entries)


def _filtered_report(
    report: StepSearchReport,
    budgets: Sequence[float],
    geometries: Sequence[tuple[int, int]],
    resolutions: Sequence[float],
) -> StepSearchReport:
    """The same search, narrowed to the points a caller asked to see.

    Narrowing is subtraction only: nothing is recomputed, so every figure still
    draws the numbers the search actually produced. `winners` is a property over
    the points that remain, so dropping a budget's winner drops that budget from
    the figures rather than promoting a runner-up.
    """

    def keeps_geometry(microbatch: int, accumulation: int) -> bool:
        return not geometries or (microbatch, accumulation) in tuple(geometries)

    def keeps_selection(outcome: object) -> bool:
        if not resolutions:
            return True
        groups = getattr(outcome, "group_count", 0)
        if not groups:
            return True
        share = getattr(outcome, "recompute_groups", 0) / groups
        return any(abs(share - item) < 0.125 for item in resolutions)

    points = tuple(
        replace(
            point,
            graph_pair_selections=tuple(
                item for item in point.graph_pair_selections if keeps_selection(item)
            ),
        )
        for point in report.points
        if _keeps_budget(point.execution_budget_bytes / _GIB, budgets)
        and keeps_geometry(point.sequences_per_microbatch, point.accumulation_count)
    )
    kept = {point.execution_budget_bytes for point in points}
    return replace(
        report,
        budgets=tuple(item for item in report.budgets if item[0] in kept),
        geometries=tuple(
            item
            for item in report.geometries
            if keeps_geometry(item.sequences_per_microbatch, item.accumulation_count)
        ),
        points=points,
        skipped=(
            report.skipped
            if not geometries
            else tuple(
                item for item in report.skipped if keeps_geometry(item[0], item[1])
            )
        ),
    )


def _tokens_per_step(directory: Path) -> int:
    """Tokens one step consumes, which turns a step time into throughput."""

    report = directory / "search.json"
    if report.is_file():
        payload = json.loads(report.read_text())
        sequences = payload.get("total_sequences_per_step")
        length = payload.get("sequence_length")
        if isinstance(sequences, int) and isinstance(length, int):
            return sequences * length
    raise SystemExit(
        f"{directory}: cannot tell how many tokens a step consumes."
        " search.json is missing or does not record the geometry."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "raw_data",
        type=Path,
        help="a run's figures/raw_data, its figures directory, or the run root",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="where the redrawn figure tree is written; created if absent",
    )
    parser.add_argument(
        "--budget-gib",
        type=_budget_filter,
        default=(),
        help="comma-separated execution budgets to keep, in gibibytes as the"
        " axis names them, for example 8,16,24. Every budget by default",
    )
    parser.add_argument(
        "--geometry",
        type=_geometry_filter,
        default=(),
        help="comma-separated geometries to keep, as MICROBATCH x ACCUMULATION,"
        " for example 64x1,16x4. Every geometry by default. Narrows the"
        " plan-side figures; the measured ones carry one geometry per budget"
        " already, the winner's",
    )
    parser.add_argument(
        "--resolution",
        type=_resolution_filter,
        default=(),
        help="comma-separated recompute shares to keep, as the search names"
        " them: 0,1/4,1/2,3/4,1. Every share by default. Narrows the plan-side"
        " figures that draw one line per graph-pair selection",
    )
    arguments = parser.parse_args()

    directory = _raw_data_directory(arguments.raw_data)
    tokens = _tokens_per_step(directory)
    output = arguments.output
    output.mkdir(parents=True, exist_ok=True)

    entries = _run_entries(directory, arguments.budget_gib)
    if entries:
        written = plot_step_run(entries, output, tokens_per_step=tokens)
        print(f"  measured: {len(entries)} budgets")
        for path in written:
            print(f"    {path}")
    else:
        print("  measured: nothing to draw (no run_budgets.csv, or none kept)")

    report_path = directory / "search.json"
    if report_path.is_file():
        report = _filtered_report(
            StepSearchReport.load(report_path),
            arguments.budget_gib,
            arguments.geometry,
            arguments.resolution,
        )
        if report.points:
            plot_step_search(report, output)
            print(
                f"  planned: {len(report.points)} points,"
                f" {len(report.geometries)} geometries,"
                f" {len(report.budgets)} budgets -> {output / 'sim'}"
            )
        else:
            print("  planned: nothing kept by the filters")
    else:
        print("  planned: no search.json beside the tables")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
