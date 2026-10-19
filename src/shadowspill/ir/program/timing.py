"""Measured resource use inside a task and progress during transfer overlap.

Resource identities describe queues; timing classes identify comparable work
on those queues. Neither requires a particular framework or collective API.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..serialization import JsonValue
from ..validation import (
    expect_integer,
    expect_mapping,
    expect_string,
    field,
    require,
    require_identifier,
    require_non_negative,
    require_tuple,
)


@dataclass(frozen=True, slots=True)
class TaskResourcePhase:
    """Sequential task work; ``None`` means ordinary GPU computation.

    Named phases occupy the same submission-ordered resource as transfer phases.
    Workspace and outputs remain allocated throughout the task, including waits.
    """

    duration_ns: int
    resource: str | None = None

    def __post_init__(self) -> None:
        require_non_negative(self.duration_ns, "task_phase.duration_ns")
        if self.resource is not None:
            require_identifier(self.resource, "task_phase.resource")

    def to_dict(self) -> dict[str, JsonValue]:
        return {"duration_ns": self.duration_ns, "resource": self.resource}

    @classmethod
    def from_value(cls, value: object, path: str) -> TaskResourcePhase:
        data = expect_mapping(value, path)
        resource = field(data, "resource", path)
        return cls(
            expect_integer(field(data, "duration_ns", path), f"{path}.duration_ns"),
            None if resource is None else expect_string(resource, f"{path}.resource"),
        )


@dataclass(frozen=True, slots=True)
class TaskOverlapCost:
    """Fractions of solo progress while this task and a transfer phase overlap.

    For example, 0.5 means that one nanosecond of wall time completes half a
    nanosecond of isolated work. Rates apply only during actual overlap. Samples
    are interpolated in log(message bytes), clamped to the measured endpoints.
    """

    timing_class: str
    bytes: int
    task_rate: float
    transfer_rate: float

    def __post_init__(self) -> None:
        require_identifier(self.timing_class, "overlap.timing_class")
        require_non_negative(self.bytes, "overlap.bytes")
        for name in ("task_rate", "transfer_rate"):
            value = getattr(self, name)
            require(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                and 0 < value <= 1,
                f"overlap.{name}",
                "expected a finite progress rate in (0, 1]",
            )

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "timing_class": self.timing_class,
            "bytes": self.bytes,
            "task_rate": self.task_rate,
            "transfer_rate": self.transfer_rate,
        }

    @classmethod
    def from_value(cls, value: object, path: str) -> TaskOverlapCost:
        data = expect_mapping(value, path)
        rates = []
        for name in ("task_rate", "transfer_rate"):
            value = field(data, name, path)
            require(
                isinstance(value, (int, float)) and not isinstance(value, bool),
                f"{path}.{name}",
                "expected a number",
            )
            assert isinstance(value, (int, float))
            rates.append(float(value))
        return cls(
            expect_string(field(data, "timing_class", path), f"{path}.timing_class"),
            expect_integer(field(data, "bytes", path), f"{path}.bytes"),
            *rates,
        )


def validate_task_timing(
    runtime_ns: int,
    resource_phases: tuple[TaskResourcePhase, ...],
    overlap_costs: tuple[TaskOverlapCost, ...],
) -> None:
    """Shared validation for neutral profiles and frontend measurements."""
    for name, values, kind in (
        ("resource_phases", resource_phases, TaskResourcePhase),
        ("overlap_costs", overlap_costs, TaskOverlapCost),
    ):
        require_tuple(values, f"profile.{name}")
        require(
            all(isinstance(item, kind) for item in values),
            f"profile.{name}",
            "invalid timing measurement",
        )
    require(
        not resource_phases
        or sum(item.duration_ns for item in resource_phases) == runtime_ns,
        "profile.resource_phases",
        "phase durations must sum to runtime_ns",
    )
    keys = [(item.timing_class, item.bytes) for item in overlap_costs]
    require(
        len(keys) == len(set(keys)),
        "profile.overlap_costs",
        "duplicate timing class and message size",
    )
