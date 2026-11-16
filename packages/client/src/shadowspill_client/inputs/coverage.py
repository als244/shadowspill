"""Proven finite packing constructions, independent of runtime search success."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from shadowspill_client import BatchConfig
from shadowspill_client.placement import InputPlacement
from shadowspill_client.protocol.inputs import InputSchema

from .coverage_bounds import (
    CoverageError,
    covers_aggregate,
    guaranteed_slots,
    input_bounds,
)
from .packing import PackedAssignment, PackingError, atomic_groups, select_schedule
from .shapes import PackingSchedule
from .validation import ValidatedBatch


@dataclass(frozen=True)
class CoverageCertificate:
    """Geometry proof for one requested schedule and fixed validation contract.

    This does not certify model padding/independence, memory, compiled metadata
    bounds or coordinated collectives. Preparation must qualify those separately.
    Loading rederives the construction; an edited record cannot assert a proof.
    """

    schema: InputSchema
    config: BatchConfig
    placement: InputPlacement
    schedule: PackingSchedule
    grouping: Literal["independent", "grouped"]
    construction: Literal["aggregate", "slots"]
    max_datums: int
    slots: tuple[int, ...]
    aggregate_buffer: int | None = None

    def pack(
        self, batch: ValidatedBatch, *, groups: Sequence[Sequence[int]] | None = None
    ) -> PackedAssignment:
        batch.require_context(self.schema, self.config)
        if len(batch.fields) > self.max_datums:
            raise ValueError("validated batch exceeds its certified record bound")
        paths = self.schedule.buffers[0].fields.keys()
        if any(paths - fields.keys() for fields in batch.fields):
            raise ValueError("coverage requires an absent validated input field")
        atomic = atomic_groups(len(batch.fields), groups)
        if self.grouping == "independent" and any(len(group) != 1 for group in atomic):
            raise ValueError("independent coverage does not admit coupled groups")
        assigned: list[tuple[tuple[int, ...], ...]] = [()] * len(self.slots)
        if self.construction == "aggregate":
            assert self.aggregate_buffer is not None
            assigned[self.aggregate_buffer] = atomic
        else:
            offset = 0
            for index, slots in enumerate(self.slots):
                assigned[index] = atomic[offset : offset + slots]
                offset += slots
            assert offset >= len(atomic)
        return PackedAssignment(
            batch,
            self.schedule,
            tuple(tuple(i for group in groups for i in group) for groups in assigned),
            tuple(assigned),
            "coverage",
        )

    def select(
        self,
        batch: ValidatedBatch,
        candidates: Sequence[PackingSchedule],
        *,
        groups: Sequence[Sequence[int]] | None = None,
        max_repair_nodes: int = 10000,
    ) -> PackedAssignment:
        """Cost-ordered attempts, with a guaranteed in-family construction.

        The caller supplies only jointly admitted candidates. The witness must
        remain in that exact family; it is never silently added at call time.
        """
        fallback = self.pack(batch, groups=groups)
        if self.schedule not in candidates:
            raise ValueError("prepared family is missing its coverage witness")
        if any(
            choice.buffers[0].fields.keys() != self.schedule.buffers[0].fields.keys()
            for choice in candidates
        ):
            raise ValueError("coverage candidates must have the same input closure")
        try:
            return select_schedule(
                batch,
                self.schema,
                self.config,
                self.placement,
                candidates,
                groups=groups,
                max_repair_nodes=max_repair_nodes,
            )
        except PackingError:
            return fallback

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "schema": self.schema.to_dict(),
            "config": self.config.to_dict(),
            "placement": self.placement.to_dict(),
            "schedule": self.schedule.to_dict(),
            "grouping": self.grouping,
            "construction": self.construction,
            "max_datums": self.max_datums,
            "slots": list(self.slots),
            "aggregate_buffer": self.aggregate_buffer,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CoverageCertificate:
        if type(value.get("version")) is not int or value["version"] != 1:
            raise ValueError("unsupported coverage certificate version")
        result = certify_coverage(
            InputSchema.from_dict(value["schema"]),
            BatchConfig.from_dict(value["config"]),
            InputPlacement.from_dict(value["placement"]),
            [PackingSchedule.from_dict(value["schedule"])],
            grouping=value["grouping"],
        )
        if result.to_dict() != value:
            raise ValueError(
                "saved coverage certificate differs from its derived proof"
            )
        return result


def certify_coverage(
    schema: InputSchema,
    config: BatchConfig,
    placement: InputPlacement,
    candidates: Sequence[PackingSchedule],
    *,
    grouping: Literal["independent", "grouped"] = "independent",
) -> CoverageCertificate:
    """Certify every input admitted by limits using a constructive envelope.

    Either one buffer covers aggregate resources and all record slots (including
    arbitrary atomic groups), or independent records fit reserved worst-case
    slots across the schedule. A certificate is an optional fallback; families
    without one rely on actual-batch packing. Additional proofs can extend this
    same interface.
    """
    if grouping not in ("independent", "grouped"):
        raise ValueError("coverage requires independent or grouped contributions")
    if not candidates or len({s.name for s in candidates}) != len(candidates):
        raise ValueError("coverage requires uniquely named requested schedules")
    placement.validate(sum(len(group) for group in placement.partitions.groups))
    fields = candidates[0].buffers[0].fields
    for schedule in candidates:
        schedule.validate(schema, config, placement)
        if schedule.buffers[0].fields.keys() != fields.keys():
            raise ValueError("coverage candidates must have the same input closure")
    bounds = input_bounds(schema, config, fields)
    definitions = {tuple(field.path): field for field in schema.fields}
    failures = []
    for schedule in candidates:
        slots = tuple(
            guaranteed_slots(b, definitions, bounds) for b in schedule.buffers
        )
        aggregate = next(
            (
                i
                for i, b in enumerate(schedule.buffers)
                if covers_aggregate(b, definitions, bounds)
            ),
            None,
        )
        if aggregate is not None or (
            grouping == "independent" and sum(slots) >= bounds.records
        ):
            return CoverageCertificate(
                schema,
                config,
                placement,
                schedule,
                grouping,
                "aggregate" if aggregate is not None else "slots",
                bounds.records,
                slots,
                aggregate,
            )
        failures.append(
            f"{schedule.name}: no buffer covers aggregate resources; "
            f"guaranteed independent slots {sum(slots)} < {bounds.records} required"
            if grouping == "independent"
            else f"{schedule.name}: no buffer covers the advertised indivisible group"
        )
    raise CoverageError(
        "cannot certify advertised input coverage: " + "; ".join(failures)
    )
