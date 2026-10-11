"""The pool operations a schedule implies, from the planner library."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from functools import cached_property

from ..capi import (
    CAdmissionOperations,
    CIndexedSchedule,
    check_planner_status,
    planner_api,
)
from .indexing import IndexedAdmissionFacts, IndexedMemorySchedule


@dataclass(frozen=True, slots=True)
class OperationArrays:
    """The library's own view of one schedule's operations."""

    operations: CAdmissionOperations
    schedule: CIndexedSchedule


# Not slotted: the columns below are decoded on first read and kept in the
# instance dictionary.
@dataclass(frozen=True)
class AdmissionOperations:
    """One schedule's operation sequence, with the provenance a layout needs.

    Columns named for operations have `len(kinds)` entries indexed alike; an
    operation's sequence is its index. Columns named for leases are indexed by
    lease id: the alias it carries (`None` for anonymous task workspace), and
    the operations that create and retire it (`None` where a lease outlives
    the step). Those last two let a reader go straight to a lease instead of
    scanning, since most operations touch no lease that matters.

    The library's arrays are what the layout's own library calls read; the
    columns are decoded from them only when Python reads one.
    """

    operation_count: int
    live_lease_count: int
    dependency_count: int
    fetch_bytes: int
    evict_bytes: int
    #: The library's own operation arrays and the schedule it derived them
    #: from, read straight through by every later library call.
    arrays: OperationArrays

    def _operations(self, name: str, missing: int | None = None) -> tuple:
        values = getattr(self.arrays.operations, name)[: self.operation_count]
        if missing is None:
            return tuple(values)
        return tuple(None if value == missing else value for value in values)

    def _leases(self, name: str, missing: int | None = None) -> tuple:
        values = getattr(self.arrays.operations, name)[: self.live_lease_count]
        if missing is None:
            return tuple(values)
        return tuple(None if value == missing else value for value in values)

    @cached_property
    def lease_ids(self) -> tuple[int, ...]:
        return self._operations("lease_ids")

    @cached_property
    def dependency_ids(self) -> tuple[int | None, ...]:
        return self._operations("dependency_ids", _NO_DEPENDENCY)

    @cached_property
    def bytes(self) -> tuple[int, ...]:
        return self._operations("bytes")

    @cached_property
    def alignments(self) -> tuple[int, ...]:
        return self._operations("alignments")

    @cached_property
    def kinds(self) -> tuple[int, ...]:
        return self._operations("kinds")

    @cached_property
    def purposes(self) -> tuple[int, ...]:
        return self._operations("purposes")

    @cached_property
    def boundaries(self) -> tuple[int, ...]:
        return self._operations("boundaries")

    @cached_property
    def indices(self) -> tuple[int, ...]:
        return self._operations("indices")

    @cached_property
    def allocation_offsets(self) -> tuple[int | None, ...]:
        return self._operations("allocation_offsets", _NO_INDEX)

    @cached_property
    def lease_aliases(self) -> tuple[int | None, ...]:
        return self._leases("lease_aliases", _NO_INDEX)

    @cached_property
    def lease_starts(self) -> tuple[int, ...]:
        return self._leases("lease_starts")

    @cached_property
    def lease_retires(self) -> tuple[int | None, ...]:
        return self._leases("lease_retires", _NO_OPERATION)

    @property
    def lease_count(self) -> int:
        return self.live_lease_count


_NO_INDEX = (1 << 32) - 1
_NO_DEPENDENCY = (1 << 64) - 1
_NO_OPERATION = (1 << 64) - 1


def build_admission_operations(
    simulation: object,
    admission: IndexedAdmissionFacts,
    schedule: IndexedMemorySchedule,
) -> AdmissionOperations:
    """Derive the operations `schedule` implies for this resolved program."""

    library = planner_api()
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
    check_planner_status(
        library.shadowspill_admission_operation_bounds(
            ctypes.byref(program),
            ctypes.byref(admission.value),
            ctypes.byref(indexed),
            ctypes.byref(operation_capacity),
            ctypes.byref(lease_capacity),
        ),
        "bounding admission operations",
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
    check_planner_status(
        library.shadowspill_build_admission_operations(
            ctypes.byref(program),
            ctypes.byref(admission.value),
            ctypes.byref(indexed),
            ctypes.byref(result),
        ),
        "building admission operations",
    )
    return AdmissionOperations(
        operation_count=int(result.operation_count),
        live_lease_count=int(result.lease_count),
        dependency_count=int(result.dependency_count),
        fetch_bytes=int(result.fetch_bytes),
        evict_bytes=int(result.evict_bytes),
        arrays=OperationArrays(operations=result, schedule=indexed),
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
