"""The pool operations a schedule implies, from the planner library."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass

from ._admission import CompiledAdmissionTopology, EncodedIndexedSchedule
from ._capi import (
    CAdmissionOperations,
    CIndexedSchedule,
    load_planner_library,
)


@dataclass(frozen=True, slots=True)
class OperationArrays:
    """The library's own view of one schedule's operations."""

    operations: CAdmissionOperations
    schedule: CIndexedSchedule


@dataclass(frozen=True, slots=True)
class AdmissionOperations:
    """One schedule's operation sequence, with the provenance a layout needs.

    Columns named for operations have `len(kinds)` entries indexed alike; an
    operation's sequence is its index. Columns named for leases are indexed by
    lease id: the alias it carries (`None` for anonymous task workspace), and
    the operations that create and retire it (`None` where a lease outlives
    the step). Those last two let a reader go straight to a lease instead of
    scanning, since most operations touch no lease that matters.
    """

    lease_ids: tuple[int, ...]
    dependency_ids: tuple[int | None, ...]
    bytes: tuple[int, ...]
    alignments: tuple[int, ...]
    kinds: tuple[int, ...]
    purposes: tuple[int, ...]
    boundaries: tuple[int, ...]
    indices: tuple[int, ...]
    allocation_offsets: tuple[int | None, ...]
    lease_aliases: tuple[int | None, ...]
    lease_starts: tuple[int, ...]
    lease_retires: tuple[int | None, ...]
    dependency_count: int
    fetch_bytes: int
    evict_bytes: int
    #: The library's own operation arrays and the schedule it derived them
    #: from, kept so a later library call can read them straight through
    #: instead of re-encoding the columns above.
    arrays: OperationArrays

    @property
    def lease_count(self) -> int:
        return len(self.lease_aliases)


_NO_INDEX = (1 << 32) - 1
_NO_DEPENDENCY = (1 << 64) - 1
_NO_OPERATION = (1 << 64) - 1


def build_admission_operations(
    simulation: object,
    admission: CompiledAdmissionTopology,
    schedule: EncodedIndexedSchedule,
) -> AdmissionOperations:
    """Derive the operations `schedule` implies for this resolved program."""

    library = load_planner_library()
    action_count = len(schedule.action_kinds)
    indexed = CIndexedSchedule(
        action_count=action_count,
        action_trigger_tasks=_u32(schedule.action_trigger_tasks),
        action_aliases=_u32(schedule.action_aliases),
        action_kinds=_u8(schedule.action_kinds),
        initial_count=len(schedule.initial_aliases),
        initial_aliases=_u32(schedule.initial_aliases),
        initial_locations=_u8(schedule.initial_locations),
        final_count=len(schedule.final_aliases),
        final_aliases=_u32(schedule.final_aliases),
        final_locations=_u8(schedule.final_locations),
    )
    program = simulation.program  # type: ignore[attr-defined]

    operation_capacity = ctypes.c_uint64(0)
    lease_capacity = ctypes.c_uint64(0)
    _check(
        library.shadowspill_admission_operation_bounds(
            ctypes.byref(program),
            ctypes.byref(admission.value),
            ctypes.byref(indexed),
            ctypes.byref(operation_capacity),
            ctypes.byref(lease_capacity),
        )
    )
    operations = int(operation_capacity.value)
    leases = int(lease_capacity.value)

    lease_ids = (ctypes.c_uint64 * operations)()
    dependency_ids = (ctypes.c_uint64 * operations)()
    sizes = (ctypes.c_uint64 * operations)()
    alignments = (ctypes.c_uint64 * operations)()
    kinds = (ctypes.c_uint8 * operations)()
    purposes = (ctypes.c_uint8 * operations)()
    boundaries = (ctypes.c_uint8 * operations)()
    indices = (ctypes.c_uint32 * operations)()
    allocation_offsets = (ctypes.c_uint32 * operations)()
    aliases = (ctypes.c_uint32 * leases)()
    lease_starts = (ctypes.c_uint64 * leases)()
    lease_retires = (ctypes.c_uint64 * leases)()
    result = CAdmissionOperations(
        lease_ids=lease_ids,
        dependency_ids=dependency_ids,
        bytes=sizes,
        alignments=alignments,
        kinds=kinds,
        purposes=purposes,
        boundaries=boundaries,
        indices=indices,
        allocation_offsets=allocation_offsets,
        operation_capacity=operations,
        lease_aliases=aliases,
        lease_starts=lease_starts,
        lease_retires=lease_retires,
        lease_capacity=leases,
    )
    _check(
        library.shadowspill_build_admission_operations(
            ctypes.byref(program),
            ctypes.byref(admission.value),
            ctypes.byref(indexed),
            ctypes.byref(result),
        )
    )
    count = int(result.operation_count)
    live = int(result.lease_count)
    return AdmissionOperations(
        lease_ids=tuple(lease_ids[:count]),
        dependency_ids=tuple(
            None if value == _NO_DEPENDENCY else value
            for value in dependency_ids[:count]
        ),
        bytes=tuple(sizes[:count]),
        alignments=tuple(alignments[:count]),
        kinds=tuple(kinds[:count]),
        purposes=tuple(purposes[:count]),
        boundaries=tuple(boundaries[:count]),
        indices=tuple(indices[:count]),
        allocation_offsets=tuple(
            None if value == _NO_INDEX else value
            for value in allocation_offsets[:count]
        ),
        lease_aliases=tuple(
            None if value == _NO_INDEX else value for value in aliases[:live]
        ),
        lease_starts=tuple(lease_starts[:live]),
        lease_retires=tuple(
            None if value == _NO_OPERATION else value
            for value in lease_retires[:live]
        ),
        dependency_count=int(result.dependency_count),
        fetch_bytes=int(result.fetch_bytes),
        evict_bytes=int(result.evict_bytes),
        arrays=OperationArrays(operations=result, schedule=indexed),
    )


def _check(status: int) -> None:
    if int(status) != 0:
        raise RuntimeError(
            f"building admission operations failed with planner status {status}"
        )


def _u32(values: tuple[int, ...]) -> ctypes.Array[ctypes.c_uint32]:
    return (ctypes.c_uint32 * max(len(values), 1))(*values)


def _u8(values: tuple[int, ...]) -> ctypes.Array[ctypes.c_uint8]:
    return (ctypes.c_uint8 * max(len(values), 1))(*values)


__all__ = [
    "AdmissionOperations",
    "OperationArrays",
    "build_admission_operations",
]
