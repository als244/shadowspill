"""Small offline budget/transfer sweeps on the ordinary measured Plan search."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import product
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from shadowspill.artifacts._values import digest, freeze, plain
from shadowspill.artifacts.plan import Plan
from shadowspill.communication import CommunicationInfo
from shadowspill.errors import PlanInfeasibleError
from shadowspill.planning.shared import shared_memory
from shadowspill.training.preparation.offline.search import search
from shadowspill_client import Memory

if TYPE_CHECKING:
    from shadowspill.training.preparation.offline.inputs import PlanningInputs


@dataclass(frozen=True)
class SweepRow:
    """One candidate's predicted result at one point of a `PlanningSweep`.

    `PlanningSweep.rows` holds them. A row with a ``failure`` has no
    prediction: its time and peaks are None.

    Attributes
    ----------
    point : int
        Index of the grid point, from zero. The points take each device
        budget in turn, each spill budget within it, and each transfer scale
        within that.
    device_gib : float
        The point's device budget per worker, in GiB.
    spill_gib : float
        The point's spill budget per worker, in GiB.
    transfer_scale : float
        The point's factor on the measured fetch and eviction rates.
    candidate : str
        The candidate's ID, a key of `PlanningInputs.candidates`.
    operation : str
        The candidate's operation, output set and mode, joined by slashes,
        such as ``"step/metrics/training"``.
    capacities : mapping of str to int
        The candidate's scheduled capacities, padding included, by capacity
        name.
    build : mapping
        The candidate's `BuildCase`, as plain data.
    microbatch_count : int
        Microbatches per run on each worker.
    ordering : mapping or None
        The candidate's `StepDataOrdering`, as plain data; None for a
        forward candidate.
    comparison : str
        Digest of the candidate's operation, declared batch and scheduled
        capacities: rows with the same digest do the same work.
    predicted_seconds : float or None
        Simulated time of one run of the slowest Plan the candidate needs on
        any worker, optimizer-only Plans excluded, in seconds.
    device_peak_bytes : int or None
        Device memory a worker needs for all its Plans, sharing their
        persistent state, in bytes; the largest over the workers.
    spill_peak_bytes : int or None
        Spill memory a worker needs for all its Plans, in bytes; the largest
        over the workers.
    failure : str or None
        Why the candidate cannot run at this point: the planner's reasons
        for its rejected Plans, or a worker's Plans exceeding a budget
        together. None when it can.
    best : bool
        Whether it is the fastest feasible row among those of its point with
        the same ``comparison``; a tie goes to the candidate ID that sorts
        first.
    """

    point: int
    device_gib: float
    spill_gib: float
    transfer_scale: float
    candidate: str
    operation: str
    capacities: Mapping[str, int]
    build: Mapping[str, Any]
    microbatch_count: int | None
    ordering: Mapping[str, Any] | None
    comparison: str
    predicted_seconds: float | None
    device_peak_bytes: int | None
    spill_peak_bytes: int | None
    failure: str | None
    best: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return the row as plain data.

        Returns
        -------
        dict
            Its fields by name.
        """
        return {name: plain(getattr(self, name)) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class PlanningSweep:
    """Predicted results of replanning exported candidates over a grid.

    `PlanningInputs.sweep` returns one. Rows with the same ``comparison`` do
    the same work; within each point, only they are ranked against each
    other.

    Attributes
    ----------
    rows : tuple of SweepRow
        One row per grid point and candidate: by point, and within each, in
        the order of `PlanningInputs.candidates`.
    measurements : mapping
        What the sweep planned from: ``"configuration"``, the preparation's
        resolved settings, and ``"captures"``, each capture template's
        ``"profiled_program"`` digest and unscaled ``"communication"``
        costs, by template ID.
    """

    rows: tuple[SweepRow, ...]
    measurements: Mapping[str, Any] = field(repr=False)
    _plans: Mapping[tuple[int, str], Mapping[int, Mapping[str, Plan]]] = field(
        repr=False
    )

    def write_report(self, path: str | Path) -> Path:
        """Write an HTML page comparing the rows, with a report of each Plan.

        The page groups the rows by ``comparison``. A directory beside it,
        named after it with ``_reports`` appended (``sweep_reports`` for
        ``sweep.html``), holds ``rows.json``, ``measurements.json`` and, linked
        from the page, the `shadowspill.write_report` report of each Plan of
        every feasible row, without plots.

        Parameters
        ----------
        path : str or pathlib.Path
            The page to write; its directory is created.

        Returns
        -------
        pathlib.Path
            The page.
        """
        from shadowspill.reporting.training_sweep import write_report

        return write_report(self, path)


def _values(
    value: float | Sequence[float] | None,
    default: float,
    name: str,
    *,
    zero: bool = False,
) -> tuple[float, ...]:
    if value is None:
        value = default
    values = (value,) if isinstance(value, (int, float)) else tuple(value)
    if not values or any(
        type(v) not in (int, float)
        or not math.isfinite(v)
        or (v < 0 if zero else v <= 0)
        for v in values
    ):
        raise ValueError(
            f"{name} requires finite {'nonnegative' if zero else 'positive'} values"
        )
    if len(set(values)) != len(values):
        raise ValueError(f"{name} requires distinct values")
    return tuple(float(v) for v in values)


def _communication(value: CommunicationInfo, scale: float) -> CommunicationInfo:
    if scale == 1:
        return value
    return replace(
        value,
        fetch_bytes_per_second=value.fetch_bytes_per_second * scale,
        evict_bytes_per_second=value.evict_bytes_per_second * scale,
        fetch_concurrent_bytes_per_second=None
        if value.fetch_concurrent_bytes_per_second is None
        else value.fetch_concurrent_bytes_per_second * scale,
        evict_concurrent_bytes_per_second=None
        if value.evict_concurrent_bytes_per_second is None
        else value.evict_concurrent_bytes_per_second * scale,
        metadata={**value.metadata, "timing_kind": "estimated"},
    )


def _description(inputs: PlanningInputs, candidate: str) -> dict[str, Any]:
    if candidate in inputs.descriptions:
        return dict(inputs.descriptions[candidate])
    # Older exports still have complete executable search inputs. Without their
    # workload labels, keep alternatives separate instead of comparing unequal
    # batch sizes or guessing which declared batch generated a capture.
    requests = [inputs.requests[key] for key in inputs.candidates[candidate]]
    main = next(row for row in requests if row.use.kind != "optim_step")
    binding = inputs.sources[main.use.template].program.bindings[
        "preparation_template"
    ]["binding"]
    return {
        "operation": {**binding["operation"], "operation": main.use.kind},
        "batch": {"uncompared_export_candidate": candidate},
        "capacities": {},
        "build": {},
        "microbatch_count": main.use.count,
        "ordering": None if main.ordering is None else main.ordering.to_dict(),
    }


def sweep(
    inputs: PlanningInputs,
    *,
    device_gib: float | Sequence[float] | None,
    spill_gib: float | Sequence[float] | None,
    transfer_scale: float | Sequence[float],
) -> PlanningSweep:
    devices = _values(device_gib, inputs.config.memory.device_gib, "device_gib")
    spills = _values(spill_gib, inputs.config.memory.spill_gib, "spill_gib", zero=True)
    transfers = _values(transfer_scale, 1, "transfer_scale")
    rows, details = [], {}
    for point, (device, spill, transfer) in enumerate(
        product(devices, spills, transfers)
    ):
        result = search(
            inputs,
            config=replace(inputs.config, memory=Memory(device, spill)),
            communication={
                key: _communication(value, transfer)
                for key, value in inputs.communication.items()
            },
        )
        for candidate, members in inputs.candidates.items():
            description = _description(inputs, candidate)
            operation = description["operation"]
            comparison = digest(
                {key: description[key] for key in ("operation", "batch", "capacities")}
            )
            duration = device_peak = spill_peak = None
            failure = None
            if candidate not in result.candidates:
                failure = "; ".join(
                    dict.fromkeys(
                        result.rejected[key]
                        for key in members
                        if key in result.rejected
                    )
                )
                if not failure:
                    raise ValueError(
                        "rejected sweep candidate has no search diagnostic"
                    )
            else:
                workers: dict[int, dict[str, Plan]] = {}
                for key in members:
                    workers.setdefault(inputs.requests[key].rank, {})[key] = (
                        result.plans[key]
                    )
                try:
                    memory = [
                        shared_memory(
                            plans[first], {k: p for k, p in plans.items() if k != first}
                        )
                        for plans in workers.values()
                        for first in [next(iter(plans))]
                    ]
                except PlanInfeasibleError as error:
                    failure = str(error)
                else:
                    duration = max(
                        result.plans[key].timeline.makespan_ns / 1e9
                        for key in members
                        if inputs.requests[key].use.kind != "optim_step"
                    )
                    device_peak = max(m.execution_bytes for m in memory)
                    spill_peak = max(m.spill_bytes for m in memory)
                    details[point, candidate] = MappingProxyType(
                        {
                            rank: MappingProxyType(plans)
                            for rank, plans in workers.items()
                        }
                    )
            rows.append(
                SweepRow(
                    point,
                    device,
                    spill,
                    transfer,
                    candidate,
                    "/".join(
                        operation[key] for key in ("operation", "outputs", "mode")
                    ),
                    freeze(description["capacities"]),
                    freeze(description["build"]),
                    description["microbatch_count"],
                    freeze(description["ordering"]),
                    comparison,
                    duration,
                    device_peak,
                    spill_peak,
                    failure,
                )
            )
    winners: dict[tuple[int, str], SweepRow] = {}
    for row in rows:
        if row.predicted_seconds is None:
            continue
        comparison_key = row.point, row.comparison
        old = winners.get(comparison_key)
        if (
            old is None
            or old.predicted_seconds is None
            or (row.predicted_seconds, row.candidate)
            < (
                old.predicted_seconds,
                old.candidate,
            )
        ):
            winners[comparison_key] = row
    return PlanningSweep(
        tuple(
            replace(row, best=winners.get((row.point, row.comparison)) is row)
            for row in rows
        ),
        freeze(
            {
                "configuration": inputs.config.to_dict(),
                "captures": {
                    key: {
                        "profiled_program": source.digest,
                        "communication": inputs.communication[key].to_dict(),
                    }
                    for key, source in inputs.sources.items()
                },
            }
        ),
        MappingProxyType(details),
    )
