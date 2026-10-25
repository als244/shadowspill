"""Run records for pool-occupancy diagnostics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from html import escape
from pathlib import Path

from ..intervals import GIB, Occupancy
from ..plan import PlanOccupancy
from ..summary import summarize

SUMMARY_COLUMNS = (
    "kind",  # chosen, resolution, all_save
    "geometry",
    "budget",
    "resolution",  # the share of the flexible groups recomputed
    "selected",
    "view",
    "page",
    "step_seconds",
    "tokens_per_second",
    "spill_peak_gib",
    "execution_peak_gib",
    "idle_percent",
    "recompute_percent",
    "fetch_utilization_percent",
    "evict_utilization_percent",
    "fetch_gib",
    "evict_gib",
    "fetch_host_gib",
    "evict_host_gib",
    "fetch_gbps",
    "evict_gbps",
    "assumed_fetch_gbps",
    "assumed_evict_gbps",
    "assumed_fetch_solo_gbps",
    "assumed_fetch_concurrent_gbps",
    "assumed_evict_solo_gbps",
    "assumed_evict_concurrent_gbps",
    "untimed_transfers",
    "interpolated_leases",
    "pool_capacity_gib",
    "spill_capacity_gib",
)


@dataclass(slots=True)
class _Entry:
    """One folder of pages in a run's tree: what it is, and its views."""

    kind: str  # chosen, resolution, all_save
    geometry: str
    budget: str
    resolution: str
    selected: bool
    directory: Path
    views: list[PlanOccupancy]
    pool_capacity_bytes: int | None = None
    spill_capacity_bytes: int | None = None

    @property
    def simulated(self) -> PlanOccupancy:
        return self.views[0]

    @property
    def traced(self) -> PlanOccupancy | None:
        return next((view for view in self.views if view.view == "traced"), None)

    def label(self) -> str:
        """The resolution as the indexes name it."""

        return escape(self.resolution) + (
            " (the search's choice)" if self.selected else ""
        )


def _summary_rows(
    entry: _Entry, out: Path, tokens: int | None
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for view in entry.views:
        summary = summarize(view, tokens_per_step=tokens)
        rows.append(
            {
                "kind": entry.kind,
                "geometry": entry.geometry,
                "budget": entry.budget,
                "resolution": entry.resolution,
                "selected": entry.selected,
                "view": view.view,
                "page": str((entry.directory / f"{view.view}.html").relative_to(out)),
                "step_seconds": summary["seconds"],
                "tokens_per_second": summary["tokens_per_second"],
                "spill_peak_gib": summary["spill_peak_gib"],
                "execution_peak_gib": summary["execution_peak_gib"],
                "idle_percent": summary["idle_percent"],
                "recompute_percent": summary["recompute_percent"],
                "fetch_utilization_percent": summary["fetch_utilization_percent"],
                "evict_utilization_percent": summary["evict_utilization_percent"],
                "fetch_gib": summary["fetch_gib"],
                "evict_gib": summary["evict_gib"],
                "fetch_host_gib": summary["fetch_host_gib"],
                "evict_host_gib": summary["evict_host_gib"],
                "fetch_gbps": summary["fetch_gbps"],
                "evict_gbps": summary["evict_gbps"],
                "assumed_fetch_gbps": summary["assumed_fetch_gbps"],
                "assumed_evict_gbps": summary["assumed_evict_gbps"],
                "assumed_fetch_solo_gbps": summary["assumed_fetch_solo_gbps"],
                "assumed_fetch_concurrent_gbps": summary[
                    "assumed_fetch_concurrent_gbps"
                ],
                "assumed_evict_solo_gbps": summary["assumed_evict_solo_gbps"],
                "assumed_evict_concurrent_gbps": summary[
                    "assumed_evict_concurrent_gbps"
                ],
                "untimed_transfers": view.untimed_transfers,
                "interpolated_leases": view.interpolated_leases,
                "pool_capacity_gib": (
                    None
                    if entry.pool_capacity_bytes is None
                    else entry.pool_capacity_bytes / GIB
                ),
                "spill_capacity_gib": (
                    None
                    if entry.spill_capacity_bytes is None
                    else entry.spill_capacity_bytes / GIB
                ),
            }
        )
    return rows


def _share(text: str) -> float:
    """A recompute share as a number, for ordering; an unreadable one last."""

    try:
        return float(Fraction(text))
    except (ValueError, ZeroDivisionError):
        return 2.0


def _links(relative: Path, views: Sequence[str]) -> str:
    parts = [f'<a href="{relative}/{view}.html">{view}</a>' for view in views]
    parts.append(f'<a href="{relative}/index.html">tables</a>')
    return " · ".join(parts)


def _gib_of(occupancy: Occupancy | None) -> str:
    return "" if occupancy is None else f"{occupancy.peak()[0] / GIB:.2f} GiB"


def _seconds(result: PlanOccupancy | None) -> str:
    return "" if result is None else f"{summarize(result)['seconds']:.3f} s"
