"""Shape/count expansion with scheduled-capacity pruning before GPU preparation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from shadowspill.training.components.preparation import (
    PreparationContext,
    ResolvedPreparationComponent,
)
from shadowspill_client._records import integer, number
from shadowspill_client.inputs.coverage import CoverageCertificate, certify_coverage
from shadowspill_client.inputs.shapes import PackingSchedule
from shadowspill_client.inputs.validation import Path
from shadowspill_client.protocol.preparation import ResolvedPreparation


@dataclass(frozen=True)
class ScheduleCandidate:
    batch: int
    case: int
    count: int
    schedule: PackingSchedule


@dataclass(frozen=True)
class RejectedSchedule:
    batch: int
    case: int
    count: int
    capacities: Mapping[str, int]
    reason: str


@dataclass(frozen=True)
class ExpandedPreparation:
    candidates: tuple[ScheduleCandidate, ...]
    rejected: tuple[RejectedSchedule, ...]

    @property
    def build_cases(self) -> int:
        return len({(item.batch, item.case) for item in self.candidates})

    def coverage(self, context: PreparationContext) -> tuple[CoverageCertificate, ...]:
        placement = context.parallelism.inputs
        assert placement is not None
        grouping = context.model.component.batching
        if grouping == "whole_batch":
            raise ValueError(
                "whole-batch dependencies require the lower-level custom graph API"
            )
        return tuple(
            certify_coverage(
                context.schema,
                batch,
                placement,
                [item.schedule for item in self.candidates if item.batch == index],
                grouping=grouping,
            )
            for index, batch in enumerate(context.batches)
        )


def expand_schedules(
    resolved: ResolvedPreparation,
    installed: ResolvedPreparationComponent,
    context: PreparationContext,
    required_paths: Sequence[Path],
    *,
    max_schedule_cases: int = 4096,
    max_buffers: int = 65536,
) -> ExpandedPreparation:
    for name, value in (
        ("max_schedule_cases", max_schedule_cases),
        ("max_buffers", max_buffers),
    ):
        integer(value, name, minimum=1)
    if resolved.preset != installed.component.id or resolved.source != installed.source:
        raise ValueError("resolved preparation component identity changed")
    counts = resolved.planning.microbatch_counts
    assert counts is not None
    if len(resolved.cases) * len(counts) * len(context.batches) > max_schedule_cases:
        raise ValueError(
            f"schedule-case expansion exceeds configured limit {max_schedule_cases}"
        )
    placement = context.parallelism.inputs
    if placement is None:
        raise ValueError("preparation requires explicit logical input placement")
    fields = {tuple(f.path): f for f in context.schema.fields}
    if (
        not required_paths
        or len(set(required_paths)) != len(required_paths)
        or set(required_paths) - fields.keys()
    ):
        raise ValueError("required fields must be a distinct nonempty schema closure")
    candidates, rejected = [], []
    reductions = {cap.name: cap.reduction for cap in context.schema.capacities}
    for batch_index, batch in enumerate(context.batches):
        if batch.limits.keys() != reductions.keys():
            raise ValueError("batch limits must name exactly the declared capacities")
        for case_index, case in enumerate(resolved.cases):
            buffers = tuple(
                installed.component.buffers(
                    context,
                    batch_index,
                    tuple(required_paths),
                    case,
                )
            )
            if len(buffers) != placement.partition_count or any(
                set(b.fields) != set(required_paths) for b in buffers
            ):
                raise ValueError(
                    "preset geometry differs from the input placement or field closure"
                )
            base = PackingSchedule("geometry", [[b] for b in buffers])
            per_count = base.capacities(context.schema)
            for count in counts:
                capacities = {
                    name: value if reductions[name] == "max" else value * count
                    for name, value in per_count.items()
                }
                excess = [
                    f"{name}={value} exceeds {batch.limits[name]}"
                    for name, value in capacities.items()
                    if value > number(batch.limits[name], name)
                ]
                if excess:
                    rejected.append(
                        RejectedSchedule(
                            batch_index,
                            case_index,
                            count,
                            MappingProxyType(capacities),
                            "scheduled padding: " + "; ".join(excess),
                        )
                    )
                    continue
                if count * len(buffers) > max_buffers:
                    raise ValueError(
                        f"schedule buffers exceed configured limit {max_buffers}"
                    )
                schedule = PackingSchedule(
                    f"batch{batch_index}.case{case_index}.count{count}",
                    [tuple(buffer for _ in range(count)) for buffer in buffers],
                )
                schedule.validate(context.schema, batch, placement)
                candidates.append(
                    ScheduleCandidate(batch_index, case_index, count, schedule)
                )
    return ExpandedPreparation(tuple(candidates), tuple(rejected))
