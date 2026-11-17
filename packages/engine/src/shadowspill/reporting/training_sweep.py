"""Offline comparisons linking the ordinary timeline/lane/memory reports."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING

from shadowspill.artifacts._values import plain

from .write import _page, _table
from .write import write_report as plan_report

if TYPE_CHECKING:
    from shadowspill.training.preparation.local_sweep import PlanningSweep


def write_report(result: PlanningSweep, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    assets = destination.with_name(destination.stem + "_reports")
    assets.mkdir(exist_ok=True)
    (assets / "rows.json").write_text(
        json.dumps([row.to_dict() for row in result.rows], indent=2) + "\n"
    )
    (assets / "measurements.json").write_text(
        json.dumps(plain(result.measurements), indent=2) + "\n"
    )
    tables = []
    for comparison in dict.fromkeys(row.comparison for row in result.rows):
        members = [row for row in result.rows if row.comparison == comparison]
        rows, links = [], []
        for index, row in enumerate(members):
            label = f"point-{row.point}-variant-{index}"
            values = row.to_dict()
            rows.append(
                {
                    "Point / variant": label,
                    "GPU GiB": row.device_gib,
                    "Spill GiB": row.spill_gib,
                    "Transfer scale": row.transfer_scale,
                    "Microbatches": row.microbatch_count,
                    "Build": json.dumps(values["build"]),
                    "Ordering": "—"
                    if row.ordering is None
                    else json.dumps(values["ordering"]),
                    "Predicted seconds": "—"
                    if row.predicted_seconds is None
                    else f"{row.predicted_seconds:.6g}",
                    "GPU GiB required": "—"
                    if row.device_peak_bytes is None
                    else f"{row.device_peak_bytes / 2**30:.3f}",
                    "Spill GiB required": "—"
                    if row.spill_peak_bytes is None
                    else f"{row.spill_peak_bytes / 2**30:.3f}",
                    "Status": row.failure
                    or ("Best for this workload" if row.best else "Planned"),
                }
            )
            for rank, plans in result._plans.get(
                (row.point, row.candidate), {}
            ).items():
                for member, plan in plans.items():
                    relative = (
                        Path(comparison[:12]) / label / f"rank-{rank}-{member[:12]}"
                    )
                    plan_report(plan, assets / relative, plots=False)
                    href = (Path(assets.name) / relative / "occupancy.html").as_posix()
                    links.append(
                        f'<li><a href="{escape(href, quote=True)}">'
                        f"{escape(label)} · Rank {rank} · "
                        f"{escape(plan.program.bindings['kind'])}</a></li>"
                    )
        tables.append(
            f"<h2>{escape(members[0].operation)}</h2><p>Scheduled capacities: "
            + escape(json.dumps(dict(members[0].capacities)))
            + "</p>"
            + _table(rows)
            + "<details><summary>Detailed Plan reports</summary><ul>"
            + "".join(links)
            + "</ul></details>"
        )
    destination.write_text(
        _page(
            "Offline planning sweep",
            "<p>Predicted server execution, replanned at every point. Transfer scale "
            "multiplies fetch/evict solo and concurrent bandwidths. Compute, latency, "
            "collectives and measured memory evidence are unchanged. Capacities "
            "include scheduled padding; they are not observed token counts. Best "
            "rows compare only equal declared workloads. GPU/spill requirements "
            "include the selected worker Plans' shared state. Service admission "
            "still checks the complete prepared family before training.</p>"
            f'<p><a href="{escape(assets.name, quote=True)}/measurements.json">'
            "Original settings, measurement identities and transfer rates</a></p>"
            + "".join(tables),
        )
    )
    return destination
