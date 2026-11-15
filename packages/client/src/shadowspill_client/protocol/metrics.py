"""Bounded views of durable scalar outcomes, ordered by completion records."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, cast

from .._records import Record, integer, mapping, number, record, text
from .requests import OPERATIONS, Operation, RequestIdentity, identifier


def metric_cursor(value: int, name: str) -> int:
    integer(value, name)
    if value >= 1 << 63:
        raise ValueError(f"{name} exceeds the supported journal range")
    return value


@dataclass(frozen=True)
class MetricAnnotation(Record):
    request_id: str
    values: Mapping[str, float | int]
    metric_id: str

    def __post_init__(self) -> None:
        identifier(self.request_id, "metric request")
        identifier(self.metric_id, "caller metric ID")
        # Bound the number and names before copying caller-owned values. These
        # are scalars, so no arbitrary nested tree or tensor snapshot is retained.
        if not isinstance(self.values, Mapping) or not 1 <= len(self.values) <= 128:
            raise ValueError("caller metrics require 1..128 named scalars")
        owned = {}
        for name, value in self.values.items():
            if len(text(name, "caller metric name")) > 256:
                raise ValueError("caller metric name exceeds 256 characters")
            number(value, f"caller metric {name}")
            owned[name] = value
        object.__setattr__(self, "values", MappingProxyType(owned))


@dataclass(frozen=True)
class MetricRecord(Record):
    position: int
    identity: RequestIdentity
    operation: Operation
    outcome: Literal["completed", "skipped", "failed"]
    history_id: str | None
    completed_updates: int | None
    state_valid: bool
    execution_effect: Literal["not_started", "completed", "unknown"]
    loss: float | int | None = None
    metrics: Mapping[str, float | int] = field(default_factory=dict)
    failure_code: str | None = None
    source: Literal["engine", "caller"] = "engine"
    annotation_id: str | None = None

    def __post_init__(self) -> None:
        if not metric_cursor(self.position, "metric position"):
            raise ValueError("metric positions start at one")
        object.__setattr__(
            self, "identity", record(RequestIdentity, self.identity, "identity")
        )
        if self.operation not in OPERATIONS or self.outcome not in (
            "completed",
            "skipped",
            "failed",
        ):
            raise ValueError("unknown metric operation or outcome")
        if self.history_id is not None:
            identifier(self.history_id, "metric history")
        if self.completed_updates is not None:
            integer(self.completed_updates, "completed updates")
        if type(self.state_valid) is not bool:
            raise TypeError("metric state validity must be boolean")
        if self.execution_effect not in ("not_started", "completed", "unknown") or (
            self.execution_effect == "unknown" and self.state_valid
        ):
            raise ValueError("invalid metric execution effect")
        if self.loss is not None:
            number(self.loss, "metric loss")
        values = mapping(self.metrics, "metrics")
        for key, value in values.items():
            number(value, f"metric {key}")
        object.__setattr__(self, "metrics", cast(Mapping[str, float | int], values))
        if self.failure_code is not None:
            text(self.failure_code, "metric failure code")
        if self.source == "caller":
            if self.annotation_id is None:
                raise ValueError("caller metrics require an annotation identity")
            MetricAnnotation(self.identity.request_id, self.metrics, self.annotation_id)
            if self.loss is not None:
                raise ValueError("caller metrics cannot supply engine loss")
        elif self.source != "engine" or self.annotation_id is not None:
            raise ValueError("invalid metric namespace or annotation identity")
        if self.outcome == "failed":
            if self.failure_code is None or (
                self.source == "engine"
                and (self.metrics or self.loss is not None)
                and (
                    self.execution_effect != "completed"
                    or self.history_id is None
                    or self.completed_updates is None
                )
            ):
                raise ValueError(
                    "failed metrics require a failure code and confirmed scalar effects"
                )
        elif (
            self.failure_code is not None
            or self.history_id is None
            or self.completed_updates is None
            or self.execution_effect != "completed"
        ):
            raise ValueError(
                "successful metrics require a confirmed history and update count"
            )


@dataclass(frozen=True)
class MetricSnapshot(Record):
    session_id: str
    run_id: str
    after: int
    through: int

    def __post_init__(self) -> None:
        identifier(self.session_id, "metric session")
        identifier(self.run_id, "metric run")
        metric_cursor(self.after, "metric start")
        metric_cursor(self.through, "metric snapshot")
        if self.after > self.through:
            raise ValueError("metric start exceeds its snapshot")


@dataclass(frozen=True)
class MetricPage(MetricSnapshot):
    records: Sequence[MetricRecord]
    next_cursor: int | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.records, (tuple, list)) or len(self.records) > 128:
            raise ValueError("metric page requires at most 128 records")
        rows = tuple(record(MetricRecord, row, "metric record") for row in self.records)
        positions = [row.position for row in rows]
        if positions != sorted(set(positions)) or any(
            row.identity.run_id != self.run_id
            or not self.after < row.position <= self.through
            for row in rows
        ):
            raise ValueError("metric page crosses its run, order or snapshot boundary")
        if self.next_cursor is not None:
            metric_cursor(self.next_cursor, "next metric cursor")
            if not rows or self.next_cursor != rows[-1].position:
                raise ValueError("metric cursor must name the final returned record")
        object.__setattr__(self, "records", rows)
