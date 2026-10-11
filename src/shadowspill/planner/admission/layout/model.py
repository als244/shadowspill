"""Immutable records for one dependency-certified execution-pool layout.

A step's layout places leases by the million, so the certificate holds its
placements, its reuse dependencies and its task-allocation leases as columns
rather than as one object per entry: building, writing and reading one is a
handful of array copies. Each column group is also a read-only sequence that
decodes one entry at a time, for a reader that wants names.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, overload

import numpy as np

from shadowspill.schema import artifact_schema
from shadowspill.simulator import SimulationAdmission, SimulationResult

from ..admission_replay import AdmissionReplayPurpose

#: An index column's entry for an entry that names no task, alias or action.
NONE = -1

#: Purpose codes, in the order the library's `ShadowSpillAdmissionPurpose`
#: numbers them. A purpose column holds positions in this tuple.
PURPOSES: tuple[AdmissionReplayPurpose, ...] = (
    AdmissionReplayPurpose.INITIAL_OBJECT,
    AdmissionReplayPurpose.TASK_WORKSPACE,
    AdmissionReplayPurpose.TASK_OUTPUT,
    AdmissionReplayPurpose.MUTATION_REPLACEMENT,
    AdmissionReplayPurpose.RELEASE,
    AdmissionReplayPurpose.EVICTION,
    AdmissionReplayPurpose.FETCH_DESTINATION,
    AdmissionReplayPurpose.TERMINAL_COMPLETION,
)


def column(values: Any, dtype: type[np.generic]) -> np.ndarray:
    """A contiguous, read-only copy of `values` as one column of `dtype`."""

    result = np.array(values, dtype=dtype, copy=True, order="C").reshape(-1)
    result.setflags(write=False)
    return result


def _name(names: tuple[str, ...], index: int) -> str | None:
    return None if index == NONE else names[index]


def _optional(value: int) -> int | None:
    return None if value == NONE else value


class _Columns:
    """Equality, length and list conversion shared by the column groups.

    Two groups are equal when every column and name table is; a group also
    equals a tuple or list of the entries it decodes to, so a reader may
    compare it with the entries it expects.
    """

    __slots__ = ()
    COLUMNS: tuple[str, ...] = ()
    TABLES: tuple[str, ...] = ()

    def __len__(self) -> int:
        return len(getattr(self, self.COLUMNS[0]))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, type(self)):
            return all(
                getattr(self, name) == getattr(other, name) for name in self.TABLES
            ) and all(
                np.array_equal(getattr(self, name), getattr(other, name))
                for name in self.COLUMNS
            )
        if isinstance(other, (tuple, list)):
            return tuple(self) == tuple(other)  # type: ignore[call-overload]
        return NotImplemented

    __hash__ = None  # type: ignore[assignment]

    def to_dict(self) -> dict[str, list[int]]:
        """The columns, named as the certificate stores them."""

        return {name: getattr(self, name).tolist() for name in self.COLUMNS}

    def _rows(self) -> Iterator[tuple[int, ...]]:
        return zip(
            *(getattr(self, name).tolist() for name in self.COLUMNS), strict=True
        )


@dataclass(frozen=True, slots=True)
class LeaseLifetime:
    """One physical lease with predicted and causal lifetime boundaries."""

    lease_id: int
    bytes: int
    alignment: int
    predicted_start_ns: int
    predicted_end_ns: int
    causal_start: int
    causal_end: int
    purpose: AdmissionReplayPurpose
    task_id: str | None = None
    alias_group_id: str | None = None
    action_index: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "action_index": self.action_index,
            "alias_group_id": self.alias_group_id,
            "alignment": self.alignment,
            "bytes": self.bytes,
            "causal_end": self.causal_end,
            "causal_start": self.causal_start,
            "lease_id": self.lease_id,
            "predicted_end_ns": self.predicted_end_ns,
            "predicted_start_ns": self.predicted_start_ns,
            "purpose": self.purpose.value,
            "task_id": self.task_id,
        }


@dataclass(frozen=True, slots=True)
class FixedLayoutPlacement:
    """One lease's exact offset within the admitted execution-pool slice."""

    lease_id: int
    offset: int
    bytes: int
    alignment: int
    predicted_start_ns: int
    predicted_end_ns: int
    causal_start: int
    causal_end: int
    purpose: AdmissionReplayPurpose
    task_id: str | None = None
    alias_group_id: str | None = None
    action_index: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "action_index": self.action_index,
            "alias_group_id": self.alias_group_id,
            "alignment": self.alignment,
            "bytes": self.bytes,
            "causal_end": self.causal_end,
            "causal_start": self.causal_start,
            "lease_id": self.lease_id,
            "offset": self.offset,
            "predicted_end_ns": self.predicted_end_ns,
            "predicted_start_ns": self.predicted_start_ns,
            "purpose": self.purpose.value,
            "task_id": self.task_id,
        }


@dataclass(frozen=True, slots=True)
class FixedLayoutReuse:
    """Completion proof required before a successor may use shared bytes."""

    dependency_id: int
    predecessor_lease_id: int
    predecessor_purpose: AdmissionReplayPurpose
    predecessor_task_id: str
    predecessor_action_index: int | None
    successor_lease_id: int
    successor_task_id: str | None
    successor_action_index: int | None

    def to_dict(self) -> dict[str, object]:
        return {
            "dependency_id": self.dependency_id,
            "predecessor_action_index": self.predecessor_action_index,
            "predecessor_lease_id": self.predecessor_lease_id,
            "predecessor_purpose": self.predecessor_purpose.value,
            "predecessor_task_id": self.predecessor_task_id,
            "successor_action_index": self.successor_action_index,
            "successor_lease_id": self.successor_lease_id,
            "successor_task_id": self.successor_task_id,
        }


@dataclass(frozen=True, slots=True, eq=False)
class FixedLayoutPlacements(_Columns, Sequence[FixedLayoutPlacement]):
    """Every fixed lease's offset, as columns indexed alike.

    `task` and `alias` index `task_ids` and `alias_ids`, and `action` is a
    schedule action's index; each holds `NONE` where the lease names none.
    `purpose` holds positions in `PURPOSES`.
    """

    COLUMNS = (
        "lease_id",
        "offset",
        "bytes",
        "alignment",
        "predicted_start_ns",
        "predicted_end_ns",
        "causal_start",
        "causal_end",
        "purpose",
        "task",
        "alias",
        "action",
    )
    TABLES = ("task_ids", "alias_ids")

    lease_id: np.ndarray
    offset: np.ndarray
    bytes: np.ndarray
    alignment: np.ndarray
    predicted_start_ns: np.ndarray
    predicted_end_ns: np.ndarray
    causal_start: np.ndarray
    causal_end: np.ndarray
    purpose: np.ndarray
    task: np.ndarray
    alias: np.ndarray
    action: np.ndarray
    task_ids: tuple[str, ...]
    alias_ids: tuple[str, ...]

    @overload
    def __getitem__(self, index: int) -> FixedLayoutPlacement: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[FixedLayoutPlacement, ...]: ...

    def __getitem__(self, index: int | slice) -> Any:
        if isinstance(index, slice):
            return tuple(self[row] for row in range(len(self))[index])
        row = range(len(self))[index]
        return self._decode(
            tuple(int(getattr(self, name)[row]) for name in self.COLUMNS)
        )

    def __iter__(self) -> Iterator[FixedLayoutPlacement]:
        return map(self._decode, self._rows())

    def _decode(self, row: tuple[int, ...]) -> FixedLayoutPlacement:
        (
            lease_id,
            offset,
            size,
            alignment,
            start_ns,
            end_ns,
            causal_start,
            causal_end,
            purpose,
            task,
            alias,
            action,
        ) = row
        return FixedLayoutPlacement(
            lease_id=lease_id,
            offset=offset,
            bytes=size,
            alignment=alignment,
            predicted_start_ns=start_ns,
            predicted_end_ns=end_ns,
            causal_start=causal_start,
            causal_end=causal_end,
            purpose=PURPOSES[purpose],
            task_id=_name(self.task_ids, task),
            alias_group_id=_name(self.alias_ids, alias),
            action_index=_optional(action),
        )


@dataclass(frozen=True, slots=True, eq=False)
class FixedLayoutReuses(_Columns, Sequence[FixedLayoutReuse]):
    """Every reuse dependency, as columns indexed alike.

    Ordered by successor lease, then the dependency waited for, then
    predecessor lease. The task columns index `task_ids`: a predecessor
    always names a task, and a successor names a task, an action or both.
    """

    COLUMNS = (
        "dependency_id",
        "predecessor_lease_id",
        "predecessor_purpose",
        "predecessor_task",
        "predecessor_action",
        "successor_lease_id",
        "successor_task",
        "successor_action",
    )
    TABLES = ("task_ids",)

    dependency_id: np.ndarray
    predecessor_lease_id: np.ndarray
    predecessor_purpose: np.ndarray
    predecessor_task: np.ndarray
    predecessor_action: np.ndarray
    successor_lease_id: np.ndarray
    successor_task: np.ndarray
    successor_action: np.ndarray
    task_ids: tuple[str, ...]

    @overload
    def __getitem__(self, index: int) -> FixedLayoutReuse: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[FixedLayoutReuse, ...]: ...

    def __getitem__(self, index: int | slice) -> Any:
        if isinstance(index, slice):
            return tuple(self[row] for row in range(len(self))[index])
        row = range(len(self))[index]
        return self._decode(
            tuple(int(getattr(self, name)[row]) for name in self.COLUMNS)
        )

    def __iter__(self) -> Iterator[FixedLayoutReuse]:
        return map(self._decode, self._rows())

    def _decode(self, row: tuple[int, ...]) -> FixedLayoutReuse:
        (
            dependency_id,
            predecessor,
            purpose,
            predecessor_task,
            predecessor_action,
            successor,
            successor_task,
            successor_action,
        ) = row
        return FixedLayoutReuse(
            dependency_id=dependency_id,
            predecessor_lease_id=predecessor,
            predecessor_purpose=PURPOSES[purpose],
            predecessor_task_id=self.task_ids[predecessor_task],
            predecessor_action_index=_optional(predecessor_action),
            successor_lease_id=successor,
            successor_task_id=_name(self.task_ids, successor_task),
            successor_action_index=_optional(successor_action),
        )


@dataclass(frozen=True, slots=True, eq=False)
class FixedLayoutTaskAllocations(_Columns, Sequence[tuple[str, int, int]]):
    """The lease each task allocation step used, as columns indexed alike.

    In step order: tasks in the order the template selects them, and each
    task's allocation steps in the order its facts declare them. `task`
    indexes `task_ids`. An entry decodes to `(task_id, allocation_ordinal,
    lease_id)`.
    """

    COLUMNS = ("task", "allocation_ordinal", "lease_id")
    TABLES = ("task_ids",)

    task: np.ndarray
    allocation_ordinal: np.ndarray
    lease_id: np.ndarray
    task_ids: tuple[str, ...]

    @overload
    def __getitem__(self, index: int) -> tuple[str, int, int]: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[tuple[str, int, int], ...]: ...

    def __getitem__(self, index: int | slice) -> Any:
        if isinstance(index, slice):
            return tuple(self[row] for row in range(len(self))[index])
        row = range(len(self))[index]
        return (
            self.task_ids[int(self.task[row])],
            int(self.allocation_ordinal[row]),
            int(self.lease_id[row]),
        )

    def __iter__(self) -> Iterator[tuple[str, int, int]]:
        names = self.task_ids
        return ((names[task], ordinal, lease) for task, ordinal, lease in self._rows())


@dataclass(frozen=True, slots=True)
class FixedPhysicalLayout:
    """Complete physical certificate for one selected execution schedule."""

    program_digest: str
    schedule_digest: str
    facts_digest: str
    pool_capacity_bytes: int
    fixed_slice_bytes: int
    #: The packed slice at the end of the fixed slice holding the objects the
    #: planner kept resident; zero when it kept none.
    resident_slice_bytes: int
    dynamic_reserve_bytes: int
    scratch_reserve_bytes: int
    required_bytes: int
    placements: FixedLayoutPlacements
    reuse_dependencies: FixedLayoutReuses
    initial_alias_leases: tuple[tuple[str, int], ...]
    task_allocation_leases: FixedLayoutTaskAllocations
    action_destination_leases: tuple[tuple[int, int], ...]
    dynamic_lifetimes: tuple[LeaseLifetime, ...] = ()

    def __post_init__(self) -> None:
        tasks = self.placements.task_ids
        if (
            self.reuse_dependencies.task_ids != tasks
            or self.task_allocation_leases.task_ids != tasks
        ):
            raise ValueError("a fixed layout's column groups name one task table")

    @property
    def dynamic_lease_ids(self) -> tuple[int, ...]:
        return tuple(item.lease_id for item in self.dynamic_lifetimes)

    @property
    def slack_bytes(self) -> int:
        return self.pool_capacity_bytes - self.required_bytes

    @property
    def digest(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()

    def header(self) -> dict[str, object]:
        """Every field but the three column groups, as `to_dict` writes them."""

        return {
            "action_destination_leases": [
                {"action_index": action, "lease_id": lease}
                for action, lease in self.action_destination_leases
            ],
            "initial_alias_leases": [
                {"alias_group_id": alias, "lease_id": lease}
                for alias, lease in self.initial_alias_leases
            ],
            "alias_ids": list(self.placements.alias_ids),
            "purposes": [item.value for item in PURPOSES],
            "pool_capacity_bytes": self.pool_capacity_bytes,
            "fixed_slice_bytes": self.fixed_slice_bytes,
            "resident_slice_bytes": self.resident_slice_bytes,
            "dynamic_reserve_bytes": self.dynamic_reserve_bytes,
            "scratch_reserve_bytes": self.scratch_reserve_bytes,
            "dynamic_lifetimes": [item.to_dict() for item in self.dynamic_lifetimes],
            "program_digest": self.program_digest,
            "required_bytes": self.required_bytes,
            "schedule_digest": self.schedule_digest,
            "schema": artifact_schema("fixed_physical_layout"),
            "task_ids": list(self.placements.task_ids),
            "facts_digest": self.facts_digest,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self.header(),
            "placements": self.placements.to_dict(),
            "reuse_dependencies": self.reuse_dependencies.to_dict(),
            "task_allocation_leases": self.task_allocation_leases.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class FixedLayoutAdmission:
    """Fixed layout plus its dependency-aware simulator evidence."""

    layout: FixedPhysicalLayout
    simulator_input: SimulationAdmission
    simulation: SimulationResult


class FixedLayoutInfeasibleError(ValueError):
    """The selected schedule has no layout within its physical pool."""

    def __init__(self, required_bytes: int, capacity_bytes: int) -> None:
        self.required_bytes = required_bytes
        self.capacity_bytes = capacity_bytes
        self.additional_bytes = max(0, required_bytes - capacity_bytes)
        super().__init__(
            "fixed physical layout exceeds the execution pool: "
            f"required={required_bytes}, capacity={capacity_bytes}, "
            f"additional={self.additional_bytes}"
        )


__all__ = [
    "NONE",
    "PURPOSES",
    "FixedLayoutAdmission",
    "FixedLayoutInfeasibleError",
    "FixedLayoutPlacement",
    "FixedLayoutPlacements",
    "FixedLayoutReuse",
    "FixedLayoutReuses",
    "FixedLayoutTaskAllocations",
    "FixedPhysicalLayout",
    "LeaseLifetime",
    "column",
]
