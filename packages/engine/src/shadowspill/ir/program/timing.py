"""Measured resource use inside a task and progress during transfer overlap.

Resource identities describe queues; timing classes identify comparable work
on those queues. Neither requires a particular framework or collective API.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from shadowspill.serialization.sharing import shared_writer

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
    """One sequential part of a task's run.

    A part named for a resource occupies that serial resource, which timed
    transfer phases of the same name also occupy; a part without one is
    computation.
    The task's workspace and outputs stay allocated through every part, waits
    included.

    Parameters
    ----------
    duration_ns : int
        How long the part takes alone, in nanoseconds. At least zero.
    resource : str, optional
        The serial resource the part occupies; None for computation.

    Raises
    ------
    ValidationError
        If ``duration_ns`` is negative or ``resource`` is empty or padded with
        whitespace.
    """

    duration_ns: int
    resource: str | None = None

    def __post_init__(self) -> None:
        require_non_negative(self.duration_ns, "task_phase.duration_ns")
        if self.resource is not None:
            require_identifier(self.resource, "task_phase.resource")

    @shared_writer
    def to_dict(self) -> dict[str, JsonValue]:
        """Return the part as JSON values.

        Returns
        -------
        dict
            Its fields by name.
        """
        return {"duration_ns": self.duration_ns, "resource": self.resource}

    @classmethod
    def from_value(cls, value: object, path: str) -> TaskResourcePhase:
        """Read a record from its `to_dict` form.

        Parameters
        ----------
        value : object
            The decoded JSON value.
        path : str
            Where the value sits in its document, such as
            ``"program.profiles[0].resource_phases[0]"``.
            Error paths start with it.

        Returns
        -------
        TaskResourcePhase
            The record.

        Raises
        ------
        ValidationError
            If a field is missing, has the wrong type or is invalid.
        """
        data = expect_mapping(value, path)
        resource = field(data, "resource", path)
        return cls(
            expect_integer(field(data, "duration_ns", path), f"{path}.duration_ns"),
            None if resource is None else expect_string(resource, f"{path}.resource"),
        )


@dataclass(frozen=True, slots=True)
class TaskOverlapCost:
    """How fast a task and a timed transfer phase progress while they overlap.

    A rate is the fraction of solo progress made per unit of wall time: at 0.5,
    one nanosecond of overlap does half a nanosecond of the work. Rates apply
    only while the two overlap. Between the samples of one class they are
    interpolated in the logarithm of one plus the message size, and outside them
    the nearest sample's rates hold.

    Parameters
    ----------
    timing_class : str
        The class of transfer work, matched against `TransferPhase.timing_class`.
    bytes : int
        Message size the sample was measured at, in bytes.
    task_rate : float
        The task's rate during the overlap, in (0, 1].
    transfer_rate : float
        The transfer phase's rate during the overlap, in (0, 1].

    Raises
    ------
    ValidationError
        If ``timing_class`` is empty or padded with whitespace, ``bytes`` is
        negative, or a rate is outside (0, 1].
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

    @shared_writer
    def to_dict(self) -> dict[str, JsonValue]:
        """Return the sample as JSON values.

        Returns
        -------
        dict
            Its fields by name.
        """
        return {
            "timing_class": self.timing_class,
            "bytes": self.bytes,
            "task_rate": self.task_rate,
            "transfer_rate": self.transfer_rate,
        }

    @classmethod
    def from_value(cls, value: object, path: str) -> TaskOverlapCost:
        """Read a record from its `to_dict` form.

        Parameters
        ----------
        value : object
            The decoded JSON value.
        path : str
            Where the value sits in its document, such as
            ``"program.profiles[0].overlap_costs[0]"``.
            Error paths start with it.

        Returns
        -------
        TaskOverlapCost
            The record.

        Raises
        ------
        ValidationError
            If a field is missing, has the wrong type or is invalid.
        """
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
