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
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, overload

import numpy as np

from shadowspill.planning.memory_search.admission.admission_replay import (
    AdmissionReplayPurpose,
)
from shadowspill.schema import artifact_schema
from shadowspill.simulator import SimulationAdmission, SimulationResult

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


class _Columns(Iterable[object]):
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
            return tuple(self) == tuple(other)
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
    """One lease's size and lifetime, as simulated and in operation order.

    `FixedPhysicalLayout.dynamic_lifetimes` holds one for each lease kept out
    of the fixed slice.

    Attributes
    ----------
    lease_id : int
        The lease.
    bytes : int
        Its size, in bytes.
    alignment : int
        The alignment its offset needs, in bytes.
    predicted_start_ns, predicted_end_ns : int
        The half-open interval over which the simulation has it live, in
        nanoseconds.
    causal_start, causal_end : int
        The sequence numbers of the operations that acquire and retire it.
        A lease may take over another's bytes only when the other's
        ``causal_end`` precedes its ``causal_start``, whatever the timing.
    purpose : AdmissionReplayPurpose
        Why it exists, such as ``"task_workspace"`` or
        ``"fetch_destination"``.
    task_id : str or None
        The task it serves, if any.
    alias_group_id : str or None
        The alias group whose object it holds; None for task workspace.
    action_index : int or None
        The index of the schedule action that made it, if one did.
    """

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
        """Write the lease as JSON-compatible values.

        Returns
        -------
        dict
            Every field by name, ``purpose`` as its value.
        """
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
    """One fixed lease's offset, with its size and lifetime.

    `FixedLayoutPlacements` decodes one per lease. Beside ``offset``, its
    fields are those of `LeaseLifetime`, with the same meanings.

    Attributes
    ----------
    offset : int
        The lease's offset from the start of the fixed slice, in bytes.
    """

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
        """Write the placement as JSON-compatible values.

        Returns
        -------
        dict
            Every field by name, ``purpose`` as its value.
        """
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
    """A lease that takes over another's bytes, and the completion it waits for.

    `FixedLayoutReuses` decodes one per dependency.

    Attributes
    ----------
    dependency_id : int
        The predecessor's retirement dependency, which the successor waits
        for before it uses the bytes.
    predecessor_lease_id : int
        The lease whose bytes are taken over.
    predecessor_purpose : AdmissionReplayPurpose
        Why the predecessor is retired, such as ``"release"`` or
        ``"eviction"``.
    predecessor_task_id : str
        The task that retires the predecessor, or that triggers the action
        that does.
    predecessor_action_index : int or None
        The index of the schedule action that retires it, if one does.
    successor_lease_id : int
        The lease that takes over the bytes.
    successor_task_id : str or None
        The task the successor serves, if any.
    successor_action_index : int or None
        The index of the schedule action that made it, if one did. A
        successor names a task, an action or both.
    """

    dependency_id: int
    predecessor_lease_id: int
    predecessor_purpose: AdmissionReplayPurpose
    predecessor_task_id: str
    predecessor_action_index: int | None
    successor_lease_id: int
    successor_task_id: str | None
    successor_action_index: int | None

    def to_dict(self) -> dict[str, object]:
        """Write the dependency as JSON-compatible values.

        Returns
        -------
        dict
            Every field by name, ``predecessor_purpose`` as its value.
        """
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
    """Every fixed lease's offset, size and lifetime, as columns indexed alike.

    `FixedPhysicalLayout.placements` holds one. It is also a read-only
    sequence that decodes one `FixedLayoutPlacement` per lease, and it equals
    a tuple or list of the entries it decodes to.

    Attributes
    ----------
    lease_id, offset, bytes, alignment : numpy.ndarray
        ``uint64`` columns of the `FixedLayoutPlacement` fields so named.
    predicted_start_ns, predicted_end_ns, causal_start, causal_end : numpy.ndarray
        ``uint64`` columns, likewise.
    purpose : numpy.ndarray
        A ``uint8`` column of purpose codes: positions in the ``purposes`` list
        `FixedPhysicalLayout.header` writes.
    task, alias, action : numpy.ndarray
        ``int32`` columns of indices into ``task_ids``, into ``alias_ids``,
        and of schedule actions; -1 where the lease names none.
    task_ids, alias_ids : tuple of str
        The task and alias group ids the index columns refer to.
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
    """Every reuse dependency of a layout, as columns indexed alike.

    `FixedPhysicalLayout.reuse_dependencies` holds one. Its rows are ordered
    by successor lease, then the dependency waited for, then predecessor
    lease. It is also a read-only sequence that decodes one `FixedLayoutReuse`
    per dependency, and it equals a tuple or list of the entries it decodes
    to.

    Attributes
    ----------
    dependency_id, predecessor_lease_id, successor_lease_id : numpy.ndarray
        ``uint64`` columns of the `FixedLayoutReuse` fields so named.
    predecessor_purpose : numpy.ndarray
        A ``uint8`` column of purpose codes, as in
        `FixedLayoutPlacements.purpose`.
    predecessor_task, successor_task : numpy.ndarray
        ``int32`` columns of indices into ``task_ids``; -1 where a successor
        names no task.
    predecessor_action, successor_action : numpy.ndarray
        ``int32`` columns of schedule action indices; -1 where the lease names
        none.
    task_ids : tuple of str
        The task ids the task columns refer to.
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
    """The lease each allocating step uses, as columns indexed alike.

    `FixedPhysicalLayout.task_allocation_leases` holds one. Its rows are in
    step order: the resolved program's tasks in order, and each task's
    allocating steps in the order its facts declare them. It is also a
    read-only sequence whose entries decode to
    ``(task_id, allocation_ordinal, lease_id)``.

    Attributes
    ----------
    task : numpy.ndarray
        An ``int32`` column of indices into ``task_ids``.
    allocation_ordinal : numpy.ndarray
        A ``uint32`` column of each step's `TaskAllocationStep.allocation_ordinal`.
    lease_id : numpy.ndarray
        A ``uint64`` column of the lease each step uses.
    task_ids : tuple of str
        The task ids ``task`` refers to.
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
    """The certified place of every lease of one schedule in the execution pool.

    `Plan.memory_layout` and `AnnotatedProgramPlan.fixed_layout` hold one.
    Its digests bind it to the program, schedule and pool facts it was
    certified for. The runtime enforces it: each fixed lease takes its
    offset, relative to the start of the fixed slice, and a lease that
    reuses another's bytes waits for the other's retirement.

    Attributes
    ----------
    program_digest : str
        Digest of the program it was certified for.
    schedule_digest : str
        Digest of the schedule it places.
    facts_digest : str
        Digest of the `AdmissionFacts` it was certified against.
    pool_capacity_bytes : int
        Bytes of the execution pool the layout must fit.
    fixed_slice_bytes : int
        Bytes of the fixed slice: the placed leases, then the resident slice.
    resident_slice_bytes : int
        Bytes at the end of the fixed slice that hold the leases of the
        objects kept resident throughout; zero when there are none.
    dynamic_reserve_bytes : int
        Bytes reserved outside the fixed slice for the leases in
        ``dynamic_lifetimes``.
    scratch_reserve_bytes : int
        Bytes reserved for task allocations that match none of their task's
        declared allocation steps.
    required_bytes : int
        Bytes the layout needs: the fixed slice, the dynamic reserve and the
        scratch reserve.
    placements : FixedLayoutPlacements
        Every fixed lease's offset, size and lifetime.
    reuse_dependencies : FixedLayoutReuses
        Every lease that takes over bytes another lease held, with the
        completion it waits for.
    initial_alias_leases : tuple of (str, int)
        The lease each alias group is resident in before the first task, as
        ``(alias_group_id, lease_id)`` pairs sorted by alias group.
    task_allocation_leases : FixedLayoutTaskAllocations
        The lease each allocating step uses, in step order.
    action_destination_leases : tuple of (int, int)
        The lease each fetch lands in, as ``(action_index, lease_id)`` pairs
        sorted by the action's index in the schedule.
    dynamic_lifetimes : tuple of LeaseLifetime
        The final leases of the caller's outputs, sorted by lease id. They
        are allocated dynamically rather than placed, so a caller can keep
        an output across later invocations.
    """

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
        """The lease ids of ``dynamic_lifetimes``, in its order."""
        return tuple(item.lease_id for item in self.dynamic_lifetimes)

    @property
    def slack_bytes(self) -> int:
        """Pool bytes the layout leaves unused.

        ``pool_capacity_bytes`` less ``required_bytes``.
        """
        return self.pool_capacity_bytes - self.required_bytes

    @property
    def digest(self) -> str:
        """SHA-256 hex digest of `FixedPhysicalLayout.to_dict`.

        It is taken over compact JSON with sorted keys.
        """
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()

    def header(self) -> dict[str, object]:
        """Write every field but the three column groups as JSON-compatible values.

        The plan store keeps it beside the columns, so a reader can take
        the byte counts and digests without the columns.

        Returns
        -------
        dict
            The digests and byte counts; ``initial_alias_leases``,
            ``action_destination_leases`` and ``dynamic_lifetimes`` as
            lists of mappings; the ``task_ids`` and ``alias_ids`` name tables
            the columns index; ``purposes``, the purpose names in the order a
            purpose column numbers them; and ``schema``.
        """

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
        """Write the layout as JSON-compatible values.

        Returns
        -------
        dict
            `FixedPhysicalLayout.header`, with ``placements``,
            ``reuse_dependencies`` and ``task_allocation_leases`` each as a
            mapping of column name to a list of integers.
        """
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
    """A schedule's fixed layout needs more bytes than its execution pool holds.

    Certifying a layout raises it, and `memory_search.plan_program` lets it
    through when its answer's layout does not fit; `shadowspill.plan_program`
    raises a `PlanInfeasibleError` of kind ``"physical_layout"`` instead.

    Parameters
    ----------
    required_bytes : int
        Bytes the layout needs.
    capacity_bytes : int
        Bytes the execution pool holds.

    Attributes
    ----------
    additional_bytes : int
        Bytes by which the layout exceeds the pool; zero when it does not.
    """

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
