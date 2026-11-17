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
from shadowspill_client import Memory

from .local_search import search

if TYPE_CHECKING:
    from .local_planning import PlanningInputs


@dataclass(frozen=True)
class SweepRow:
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
        return {name: plain(getattr(self, name)) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class PlanningSweep:
    """Predicted alternatives; only equal declared workloads share a comparison."""

    rows: tuple[SweepRow, ...]
    measurements: Mapping[str, Any] = field(repr=False)
    _plans: Mapping[tuple[int, str], Mapping[int, Mapping[str, Plan]]] = field(
        repr=False
    )

    def write_report(self, path: str | Path) -> Path:
        """Write an offline comparison and adjacent detailed Plan HTML reports."""
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
