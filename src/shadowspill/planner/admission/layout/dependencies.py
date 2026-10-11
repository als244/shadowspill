"""Recover causal proofs for every byte range shared by a fixed layout."""

from __future__ import annotations

import ctypes

import numpy as np

from shadowspill.planner.admission.operations import AdmissionOperations
from shadowspill.planner.capi import (
    CLayoutReuseProblem,
    CLayoutReuseResult,
    CLeaseIdentity,
    CLeaseLifetime,
    check_planner_status,
    planner_api,
)
from shadowspill.simulator import MemoryReuseDependency
from shadowspill.status import ABI_VERSION

from ..admission_replay import AdmissionReplayPurpose
from ..setup import AdmissionSetup
from .lifetimes import LeaseLayout
from .model import NONE, PURPOSES, FixedLayoutReuses, column

_EVICTION = PURPOSES.index(AdmissionReplayPurpose.EVICTION)


#: What the library's `ShadowSpillLayoutReuseRefusal` numbers mean, as
#: the refusal reads.
_REFUSALS = {
    1: (
        "fixed layout overlaps causally live leases: "
        "predecessor={predecessor}, successor={successor}"
    ),
    2: "fixed layout reuses lease {predecessor} without a completion dependency",
    3: "fixed-layout predecessor lacks a task",
    4: "fixed-layout successor lacks a consumer",
}

#: The result's columns, their element types, and the certificate's names.
_RESULT_COLUMNS = (
    ("dependency_ids", np.uint64, "dependency_id"),
    ("predecessor_leases", np.uint64, "predecessor_lease_id"),
    ("predecessor_purposes", np.uint8, "predecessor_purpose"),
    ("predecessor_tasks", np.uint32, "predecessor_task"),
    ("predecessor_actions", np.uint32, "predecessor_action"),
    ("successor_leases", np.uint64, "successor_lease_id"),
    ("successor_tasks", np.uint32, "successor_task"),
    ("successor_actions", np.uint32, "successor_action"),
)

_CTYPES = {
    np.uint64: ctypes.c_uint64,
    np.uint32: ctypes.c_uint32,
    np.uint8: ctypes.c_uint8,
}

#: What each result column is kept as. Task and action columns carry the
#: library's 32-bit no-index marker, which read signed is `NONE`.
_KEPT = {np.uint64: np.uint64, np.uint32: np.int32, np.uint8: np.uint8}


def _kept(values: np.ndarray, dtype: type[np.generic]) -> np.ndarray:
    return column(values.view(np.int32) if dtype is np.uint32 else values, _KEPT[dtype])


def recover_reuse_dependencies(
    operations: AdmissionOperations,
    setup: AdmissionSetup,
    layout: LeaseLayout,
    offsets: np.ndarray,
) -> FixedLayoutReuses:
    """Recover timing-independent predecessors from causal operation order.

    The library takes the placed fixed prefix and the operations it was built
    from and reports every pair of leases sharing an address, the completion
    the later one waits for, and who waits: `shadowspill_recover_layout_reuse`,
    specified readably in `reference/python/admission/reuse.py`. A predecessor
    that is still live when its successor is acquired, or whose retirement
    publishes no completion, refuses the layout here. The library writes the
    columns the certificate keeps, so nothing is decoded per pair.
    """

    count = layout.fixed_count
    if count == 0:
        return FixedLayoutReuses(
            **{
                name: _kept(np.empty(0, dtype=dtype), dtype)
                for _, dtype, name in _RESULT_COLUMNS
            },
            task_ids=setup.task_ids,
        )
    offset_column = np.ascontiguousarray(offsets[:count], dtype=np.uint64)
    problem = CLayoutReuseProblem(
        abi_version=ABI_VERSION,
        operations=ctypes.pointer(operations.arrays.operations),
        schedule=ctypes.pointer(operations.arrays.schedule),
        lifetimes=ctypes.cast(layout.leases.lifetimes, ctypes.POINTER(CLeaseLifetime)),
        identities=ctypes.cast(
            layout.leases.identities, ctypes.POINTER(CLeaseIdentity)
        ),
        offsets=offset_column.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        lifetime_count=count,
    )
    library = planner_api()
    capacity = 2 * count + 16
    while True:
        buffers = {
            field: np.empty(max(capacity, 1), dtype=dtype)
            for field, dtype, _ in _RESULT_COLUMNS
        }
        result = CLayoutReuseResult(
            **{
                field: buffers[field].ctypes.data_as(ctypes.POINTER(_CTYPES[dtype]))
                for field, dtype, _ in _RESULT_COLUMNS
            },
            capacity=capacity,
        )
        status = library.shadowspill_recover_layout_reuse(
            ctypes.byref(problem), ctypes.byref(result)
        )
        if status == 1 and result.count > capacity:
            capacity = result.count
            continue
        break
    if status == 81:
        raise ValueError(
            _REFUSALS.get(result.refusal, "fixed layout refused").format(
                predecessor=result.refusal_predecessor,
                successor=result.refusal_successor,
            )
        )
    check_planner_status(status, "shadowspill_recover_layout_reuse")
    found = int(result.count)
    return FixedLayoutReuses(
        **{
            name: _kept(buffers[field][:found], dtype)
            for field, dtype, name in _RESULT_COLUMNS
        },
        task_ids=setup.task_ids,
    )


def simulator_reuse_dependencies(
    dependencies: FixedLayoutReuses,
) -> tuple[MemoryReuseDependency, ...]:
    """Project cross-lane eviction proofs into simulator dependencies."""

    evictions = dependencies.predecessor_purpose == _EVICTION
    predecessors = dependencies.predecessor_action[evictions]
    if (predecessors == NONE).any():
        raise ValueError("eviction reuse lacks its action identity")
    task_ids = dependencies.task_ids
    edges: set[MemoryReuseDependency] = set()
    for predecessor, action, task in zip(
        predecessors.tolist(),
        dependencies.successor_action[evictions].tolist(),
        dependencies.successor_task[evictions].tolist(),
        strict=True,
    ):
        if action != NONE:
            edges.add(MemoryReuseDependency(predecessor, successor_action_index=action))
        elif task != NONE:
            edges.add(
                MemoryReuseDependency(predecessor, successor_task_id=task_ids[task])
            )
        else:  # pragma: no cover - validated by recovery
            raise ValueError("fixed-layout reuse lacks a successor")
    return tuple(
        sorted(
            edges,
            key=lambda item: (
                item.predecessor_action_index,
                item.successor_task_id or "",
                -1
                if item.successor_action_index is None
                else item.successor_action_index,
            ),
        )
    )


__all__ = ["recover_reuse_dependencies", "simulator_reuse_dependencies"]
