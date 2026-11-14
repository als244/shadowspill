"""Page for pool-occupancy diagnostics."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from html import escape
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from .html import _page
from .intervals import GIB, Occupancy
from .plan import PlanOccupancy
from .summary import summarize


def table(occupancy: Occupancy, by: str, at_ns: Sequence[int] = ()) -> str:
    """One pool at its peak and at the times asked for, by category or role."""

    peak_bytes, peak_ns = occupancy.peak()
    columns = [("peak", peak_ns), *((f"{t / 1e9:.2f} s", t) for t in at_ns)]
    snapshots = [occupancy.at(t, by=by) for _, t in columns]
    names = sorted(
        {name for snapshot in snapshots for name in snapshot},
        key=lambda n: -snapshots[0].get(n, 0),
    )
    own_peaks = occupancy.peaks_by(by)
    width = max([len(n) for n in names] + [12])
    reported = (
        f" (planner reported {occupancy.reported_peak_bytes / GIB:.2f} GiB)"
        if occupancy.reported_peak_bytes is not None
        else ""
    )
    capacity = (
        f", capacity {occupancy.capacity_bytes / GIB:.2f} GiB"
        if occupancy.capacity_bytes is not None
        else ""
    )
    required = (
        f", layout requires {occupancy.required_bytes / GIB:.2f} GiB"
        if occupancy.required_bytes is not None
        else ""
    )
    lines = [
        f"{occupancy.pool} pool: peak {peak_bytes / GIB:.2f} GiB"
        f" at {peak_ns / 1e9:.2f} s{reported}{capacity}{required}",
        "  "
        + f"{by:<{width}}"
        + "".join(f"{label:>16}" for label, _ in columns)
        + f"{'own peak':>16}",
    ]
    for name in names:
        row = f"  {name:<{width}}"
        for snapshot in snapshots:
            value = snapshot.get(name, 0)
            share = f" {100 * value / peak_bytes:3.0f}%" if peak_bytes else "    "
            row += f"{value / GIB:>10.2f} GiB" + share
        own, own_at = own_peaks[name]
        row += f"{own / GIB:>10.2f} GiB {own_at / 1e9:5.1f}s"
        lines.append(row)
    totals = [sum(snapshot.values()) for snapshot in snapshots]
    lines.append(
        "  "
        + f"{'total':<{width}}"
        + "".join(f"{value / GIB:>10.2f} GiB     " for value in totals)
    )
    return "\n".join(lines)


def page_data(
    result: PlanOccupancy,
    *,
    by: str = "category",
    points: int = 1200,
    tokens_per_step: int | None = None,
    plan: str = "",
) -> dict[str, Any]:
    """What a page draws: the step's summary, the pools sampled over the
    step, execution then spill, and the lanes, all on this view's clock;
    ``plan`` is the line under the title that says whose step this is."""

    end_ns = max(
        [result.clock.makespan_ns]
        + [span.end_ns for span in result.tasks]
        + [span.end_ns for span in result.transfers]
    )
    data: dict[str, Any] = {
        "view": result.view,
        "clock": result.clock_label,
        "plan": plan,
        "untimed_transfers": result.untimed_transfers,
        "end_seconds": end_ns / 1e9,
        "summary": summarize(result, tokens_per_step=tokens_per_step),
    }
    pools = []
    for occupancy in result.pools:
        if not occupancy.intervals:
            continue  # the unconstrained view spills nothing
        times, rows = occupancy.series(by=by, points=points, end_ns=end_ns)
        peak_bytes, peak_ns = occupancy.peak()
        note = ""
        if occupancy is result.execution and result.view == "traced":
            note = "leases at the device times of the events that open and close them"
            if result.interpolated_leases:
                note += (
                    f", {result.interpolated_leases} lease instants interpolated"
                    " between task boundaries"
                )
        pools.append(
            {
                "name": occupancy.pool,
                "note": note,
                "times": [round(t, 4) for t in times],
                "rows": {
                    name: [round(v, 4) for v in values] for name, values in rows.items()
                },
                "peak_gib": peak_bytes / GIB,
                "peak_seconds": peak_ns / 1e9,
                "reported_peak_gib": (
                    None
                    if occupancy.reported_peak_bytes is None
                    else occupancy.reported_peak_bytes / GIB
                ),
                "capacity_gib": (
                    None
                    if occupancy.capacity_bytes is None
                    else occupancy.capacity_bytes / GIB
                ),
            }
        )
    data["pools"] = pools
    # In start order on this clock, which on a traced step is not the
    # schedule's order.
    data["compute"] = [
        [
            span.start_ns / 1e9,
            span.end_ns / 1e9,
            span.phase,
            span.name,
            None if span.overhead_ns is None else span.overhead_ns / 1e9,
        ]
        for span in sorted(result.tasks, key=lambda span: span.start_ns)
    ]
    for direction in ("fetch", "evict"):
        data[direction] = [
            [
                span.start_ns / 1e9,
                span.end_ns / 1e9,
                span.execution_bytes,
                span.category,
                span.alias_group_id,
                span.bytes,
            ]
            for span in sorted(result.transfers, key=lambda span: span.start_ns)
            if span.direction == direction
        ]
    if result.collectives is not None:
        data["collectives"] = result.collectives
    if result.transfer_phases is not None:
        data["transfer_phases"] = result.transfer_phases
    return data


def _gib(value: float) -> str:
    return f"{value / GIB:.2f}".rstrip("0").rstrip(".")


def _describe_geometry(label: str) -> str:
    """Describe a named candidate, accumulation and ordering."""
    head, _, ordering = label.rpartition("_")
    candidate, separator, count = head.rpartition("x")
    if not (separator and count.isdigit() and ordering):
        return label
    return f"candidate {unquote(candidate)}, {count} microbatches, ordering {ordering}"


def describe_plan(
    selection: Mapping[str, Any],
    *,
    model: str = "",
    geometry: str = "",
    budget: str = "",
    sequence_length: int | None = None,
    sequences_per_step: int | None = None,
    unconstrained: bool = False,
    resolution: str = "",
    selected: bool = False,
) -> str:
    """The line under a page's title: whose step this is and what it was
    planned within -- the model, the geometry, the tokens a step, the
    execution budget and the pool it resolved to, the spill capacity, the
    resolution when the page is one of several the search kept -- or,
    ``unconstrained``, that nothing constrained it."""

    parts: list[str] = []
    if model:
        parts.append(model)
    if geometry:
        parts.append(_describe_geometry(geometry))
    if sequence_length and sequences_per_step:
        parts.append(
            f"{sequence_length} tokens by {sequences_per_step} sequences"
            f" = {sequence_length * sequences_per_step} tokens a step"
        )
    layout = selection.get("admission_certificate", {}).get("layout", {})
    if "pool_capacity_bytes" in layout:
        pool = f"pool {_gib(int(layout['pool_capacity_bytes']))} GiB"
        parts.append(
            f"execution budget {budget} ({pool})" if budget else f"execution {pool}"
        )
    spill = (selection.get("simulation") or {}).get("spill_capacity_bytes")
    if spill is not None:
        parts.append(f"spill pool {_gib(int(spill))} GiB")
    if resolution:
        parts.append(
            f"resolution: {resolution} of the flexible groups recompute"
            + (", the search's choice" if selected else "")
        )
    if unconstrained:
        # whose alternatives the floor is taken at: a resolution's, a plan's
        # (a selection given), or the program's cheapest
        parts.append(
            "unconstrained: every object resident, nothing spilled, "
            + (
                "alternatives as this resolution fixes them"
                if resolution
                else "alternatives as this plan fixes them"
                if selection.get("selections")
                else "every alternative at its cheapest"
            )
        )
    return " · ".join(parts)


def write_pages(
    views: Sequence[PlanOccupancy],
    directory: Path,
    *,
    title: str,
    by: str = "category",
    tokens_per_step: int | None = None,
    plan: str = "",
    plan_by_view: Mapping[str, str] | None = None,
    related: Sequence[tuple[str, str]] = (),
) -> list[Path]:
    """Write one page per view -- ``simulated.html``, ``traced.html`` -- each
    with the summary, both pools and the lanes on one zoom, and an index
    that links them and carries the tables. ``plan`` names the step under
    every title, and ``plan_by_view`` names it differently for a view;
    ``related`` is ``(label, href)`` links the index adds, to the folders
    around this one. Returns the pages written, the index first."""

    directory.mkdir(parents=True, exist_ok=True)
    template = Path(__file__).with_name("occupancy_page.html").read_text()
    written: list[Path] = []
    links: list[str] = []
    tables: list[str] = []
    for result in views:
        name = f"{result.view}.html"
        named = (plan_by_view or {}).get(result.view, plan)
        payload = json.dumps(
            page_data(result, by=by, tokens_per_step=tokens_per_step, plan=named),
            separators=(",", ":"),
        )
        page = template.replace(
            "__TITLE__", escape(f"{title} · {result.view}")
        ).replace("__OCCUPANCY_DATA__", payload)
        (directory / name).write_text(page)
        written.append(directory / name)
        links.append(f'<li><a href="{name}">{result.clock_label}</a></li>')
        tables.append(
            f"<h2>{escape(result.clock_label)}</h2>"
            + "".join(
                f"<pre>{escape(table(occupancy, by))}</pre>"
                for occupancy in result.pools
                if occupancy.intervals
            )
        )
    index = directory / "index.html"
    around = "".join(
        f'<li><a href="{escape(href)}">{escape(label)}</a></li>'
        for label, href in related
    )
    index.write_text(
        _page(
            f"Occupancy {title}",
            f"<h1>Occupancy over the step: {escape(title)}</h1>"
            f"<p>{escape(plan)}</p><ul>{''.join(links)}</ul>"
            + (f"<p>Around this plan:</p><ul>{around}</ul>" if around else "")
            + "".join(tables),
        )
    )
    return [index, *written]
