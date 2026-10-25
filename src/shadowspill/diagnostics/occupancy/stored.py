"""Stored for pool-occupancy diagnostics."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import quote

from shadowspill.schema import ARTIFACT_VERSION

from .intervals import GIB


def _stored_selection(path: Path, *, layout: bool = True) -> dict[str, Any]:
    """A plan record with its certificate, which a plan store keeps in a file
    of its own beside it; `layout=False` reads only the certificate's byte
    counts. A record with no such file reads as it is."""

    from shadowspill.planner.plan_store import read_stored_selection

    return read_stored_selection(path, layout=layout)


def program_path_for(selection_path: Path, selection: Mapping[str, Any]) -> Path:
    """Where the store keeps the program a selection was made for."""

    digest = str(selection["program_digest"])
    # <store>/vN/planning/results/<xx>/<key>/selection.json -> <store>/vN/planning
    planning = selection_path.resolve().parents[3]
    return planning / "programs" / digest[:2] / digest / "program.json"


def _requested_budget(pool_capacity_bytes: int, budgets: Sequence[int]) -> int | None:
    """The requested execution budget a plan's pool capacity resolved from:
    the smallest budget at or above the capacity, since a slab is a budget
    less the leeway the pool keeps."""

    candidates = [budget for budget in budgets if budget >= pool_capacity_bytes]
    return min(candidates) if candidates else None


def _budget_label(budget_bytes: int) -> str:
    return f"{budget_bytes / GIB:g}gib"


@dataclass(frozen=True, slots=True)
class StoredPlan:
    """One stored selection a run made, with what identifies it to a reader."""

    path: Path
    makespan_ns: int
    pool_capacity_bytes: int
    budget_bytes: int | None
    geometry: str  # "<sequences>x<accumulation>_<ordering>", or "" when unknown
    program_digest: str = ""
    #: A resolution the search kept beside its answer: the share of the
    #: flexible groups it recomputes, and whether the search chose it. Empty
    #: for the chosen plan's own record.
    resolution: str = ""
    selected: bool = True
    #: The store key both records sit under.
    key: str = ""


def stored_plans(run_root: Path) -> list[StoredPlan]:
    """Every selection with evidence under the run's plan store, labelled.

    A stored selection names its program and its capacities but not the
    geometry the search called it; that comes from the search report, whose
    points carry the simulated makespan, which the two share.
    """

    budgets: list[int] = []
    by_makespan: dict[int, set[str]] = defaultdict(set)
    search_path = run_root / "search.json"
    if search_path.exists():
        search = json.loads(search_path.read_text())
        budgets = sorted({int(pair[0]) for pair in search.get("budgets", ())})
        for point in search.get("points", ()):
            if point.get("makespan_seconds") is None:
                continue
            geometry = (
                f"{quote(point['candidate'], safe='')}x{point['accumulation_count']}"
                f"_{point['ordering_label']}"
            )
            by_makespan[round(float(point["makespan_seconds"]) * 1e6)].add(geometry)
    else:
        request_path = run_root / "request.json"
        if request_path.exists():
            request = json.loads(request_path.read_text()).get("request", {})
            budgets = sorted(
                int(value * GIB) for value in request.get("run_budget_gib", ())
            )
    plans: list[StoredPlan] = []
    # A report copied from another machine records that machine's path; its own
    # plan_store/ beside it is the same store.
    recorded_store = _request(run_root).get("plan_store")
    store = run_root / "plan_store"
    if recorded_store is not None and Path(recorded_store).is_dir():
        store = Path(recorded_store)
    results = store / f"v{ARTIFACT_VERSION}" / "planning" / "results"
    for path in sorted(results.glob("*/*/selection.json")):
        selection = _stored_selection(path, layout=False)
        simulation = selection.get("simulation_result")
        layout = selection.get("admission_certificate", {}).get("layout", {})
        if simulation is None or "pool_capacity_bytes" not in layout:
            continue
        makespan_ns = int(simulation["makespan_ns"])
        labels = by_makespan.get(round(makespan_ns / 1e3), set())
        capacity = int(layout["pool_capacity_bytes"])
        plans.append(
            StoredPlan(
                path,
                makespan_ns,
                capacity,
                _requested_budget(capacity, budgets),
                next(iter(labels)) if len(labels) == 1 else "",
                str(selection.get("program_digest", "")),
                key=path.parent.name,
            )
        )
    # The resolutions a search kept beside its answer, when it was asked to:
    # one record each under the chosen plan's key, certified like it. One whose
    # layout did not fit has no certificate and no page.
    for path in sorted(results.glob("*/*/resolutions/*/selection.json")):
        selection = _stored_selection(path, layout=False)
        simulation = selection.get("simulation_result")
        layout = selection.get("admission_certificate", {}).get("layout", {})
        resolution = selection.get("resolution") or {}
        if simulation is None or "pool_capacity_bytes" not in layout or not resolution:
            continue
        capacity = int(layout["pool_capacity_bytes"])
        plans.append(
            StoredPlan(
                path,
                int(simulation["makespan_ns"]),
                capacity,
                _requested_budget(capacity, budgets),
                "",
                str(selection.get("program_digest", "")),
                resolution=str(resolution.get("recompute_share", "")),
                selected=bool(resolution.get("selected", False)),
                key=path.parents[2].name,
            )
        )
    # A program is one geometry walked one way, so a plan the makespan could
    # not name takes the name of any plan of the same program that it could.
    by_digest: dict[str, set[str]] = defaultdict(set)
    for record in plans:
        if record.geometry:
            by_digest[record.program_digest].add(record.geometry)
    return [
        (
            replace(record, geometry=next(iter(by_digest[record.program_digest])))
            if not record.geometry
            and len(by_digest.get(record.program_digest, ())) == 1
            else record
        )
        for record in plans
    ]


def _request(run_root: Path) -> dict[str, Any]:
    """What the run was asked for, from its ``request.json``, or nothing."""

    request_path = run_root / "request.json"
    if not request_path.exists():
        return {}
    record = json.loads(request_path.read_text()).get("request", {})
    return dict(record) if isinstance(record, Mapping) else {}


def _tokens_per_step(run_root: Path) -> int | None:
    report = run_root / "search.json"
    if report.is_file():
        metadata = json.loads(report.read_text()).get("metadata", {})
        if metadata.get("unit_label") == "tokens":
            return int(metadata["units_per_step"])
    request = _request(run_root)
    try:
        return int(request["sequence_length"]) * int(request["sequences_per_step"])
    except (KeyError, TypeError, ValueError):
        return None


def _shape(request: Mapping[str, Any]) -> tuple[int | None, int | None]:
    try:
        return int(request["sequence_length"]), int(request["sequences_per_step"])
    except (KeyError, TypeError, ValueError):
        return None, None
