"""Recover causal proofs for every byte range shared by a fixed layout."""

from __future__ import annotations

from dataclasses import dataclass

from shadowspill.planner._operations import AdmissionOperations
from shadowspill.simulator import MemoryReuseDependency

from ..admission_replay import AdmissionReplayPurpose
from ..setup import AdmissionSetup
from .lifetimes import _ACTION_BOUNDARIES, _INITIAL_BOUNDARY, _PURPOSES
from .model import FixedLayoutPlacement, FixedLayoutReuse


@dataclass(frozen=True, slots=True)
class _Retirement:
    """What a reuse of one lease's address must wait for."""

    dependency_id: int | None
    purpose: AdmissionReplayPurpose
    task_id: str | None
    action_index: int | None


_EMPTY = -1
_MIXED = -2


class _LatestOwnerIndex:
    """Range-assignment tree whose leaves name the latest physical owner."""

    def __init__(self, placements: tuple[FixedLayoutPlacement, ...]) -> None:
        coordinates = sorted(
            {
                value
                for item in placements
                for value in (item.offset, item.offset + item.bytes)
            }
        )
        self._positions = {value: index for index, value in enumerate(coordinates)}
        self._length = max(0, len(coordinates) - 1)
        size = 4 * max(1, self._length)
        self._owners = [_EMPTY] * size
        self._lazy: list[int | None] = [None] * size

    def replace(self, start: int, end: int, owner: int) -> set[int]:
        left = self._positions[start]
        right = self._positions[end]
        predecessors: set[int] = set()
        self._query(1, 0, self._length, left, right, predecessors)
        self._assign(1, 0, self._length, left, right, owner)
        return predecessors

    def _push(self, node: int) -> None:
        value = self._lazy[node]
        if value is None:
            return
        for child in (node * 2, node * 2 + 1):
            self._owners[child] = value
            self._lazy[child] = value
        self._lazy[node] = None

    def _query(
        self,
        node: int,
        left: int,
        right: int,
        start: int,
        end: int,
        found: set[int],
    ) -> None:
        if end <= left or right <= start:
            return
        if start <= left and right <= end and self._owners[node] != _MIXED:
            if self._owners[node] >= 0:
                found.add(self._owners[node])
            return
        if right - left == 1:
            return
        self._push(node)
        middle = (left + right) // 2
        self._query(node * 2, left, middle, start, end, found)
        self._query(node * 2 + 1, middle, right, start, end, found)

    def _assign(
        self,
        node: int,
        left: int,
        right: int,
        start: int,
        end: int,
        owner: int,
    ) -> None:
        if end <= left or right <= start:
            return
        if start <= left and right <= end:
            self._owners[node] = owner
            self._lazy[node] = owner
            return
        if right - left == 1:
            self._owners[node] = owner
            self._lazy[node] = owner
            return
        self._push(node)
        middle = (left + right) // 2
        self._assign(node * 2, left, middle, start, end, owner)
        self._assign(node * 2 + 1, middle, right, start, end, owner)
        left_owner = self._owners[node * 2]
        right_owner = self._owners[node * 2 + 1]
        self._owners[node] = left_owner if left_owner == right_owner else _MIXED


def recover_reuse_dependencies(
    operations: AdmissionOperations,
    setup: AdmissionSetup,
    placements: tuple[FixedLayoutPlacement, ...],
) -> tuple[FixedLayoutReuse, ...]:
    """Recover timing-independent predecessors from causal operation order.

    A predecessor's retirement is resolved only when its address is actually
    reused, which is a small fraction of the leases.
    """

    def retirement(lease: int) -> _Retirement | None:
        index = operations.lease_retires[lease]
        if index is None:
            return None
        boundary = operations.boundaries[index]
        position = operations.indices[index]
        if boundary in _ACTION_BOUNDARIES:
            task_id = setup.task_ids[setup.action_trigger_tasks[position]]
            action_index: int | None = position
        elif boundary == _INITIAL_BOUNDARY:
            task_id = None
            action_index = None
        else:
            task_id = setup.task_ids[position]
            action_index = None
        return _Retirement(
            dependency_id=operations.dependency_ids[index],
            purpose=_PURPOSES[operations.purposes[index]],
            task_id=task_id,
            action_index=action_index,
        )

    by_lease = {item.lease_id: item for item in placements}
    latest = _LatestOwnerIndex(placements)
    dependencies: set[FixedLayoutReuse] = set()
    for successor in sorted(
        placements,
        key=lambda item: (item.causal_start, item.lease_id),
    ):
        predecessors = latest.replace(
            successor.offset,
            successor.offset + successor.bytes,
            successor.lease_id,
        )
        for predecessor_id in predecessors:
            predecessor = by_lease[predecessor_id]
            if predecessor.causal_end > successor.causal_start:
                raise ValueError(
                    "fixed layout overlaps causally live leases: "
                    f"predecessor={predecessor_id}, "
                    f"successor={successor.lease_id}"
                )
            retired = retirement(predecessor_id)
            if retired is None or retired.dependency_id is None:
                raise ValueError(
                    f"fixed layout reuses lease {predecessor_id} without a "
                    "completion dependency"
                )
            if retired.task_id is None:
                raise ValueError("fixed-layout predecessor lacks a task")
            if successor.task_id is None and successor.action_index is None:
                raise ValueError("fixed-layout successor lacks a consumer")
            dependencies.add(
                FixedLayoutReuse(
                    dependency_id=retired.dependency_id,
                    predecessor_lease_id=predecessor_id,
                    predecessor_purpose=retired.purpose,
                    predecessor_task_id=retired.task_id,
                    predecessor_action_index=retired.action_index,
                    successor_lease_id=successor.lease_id,
                    successor_task_id=successor.task_id,
                    successor_action_index=successor.action_index,
                )
            )
    return tuple(
        sorted(
            dependencies,
            key=lambda item: (
                item.successor_lease_id,
                item.dependency_id,
                item.predecessor_lease_id,
            ),
        )
    )


def simulator_reuse_dependencies(
    dependencies: tuple[FixedLayoutReuse, ...],
) -> tuple[MemoryReuseDependency, ...]:
    """Project cross-lane eviction proofs into simulator dependencies."""

    edges: set[MemoryReuseDependency] = set()
    for item in dependencies:
        if item.predecessor_purpose is not AdmissionReplayPurpose.EVICTION:
            continue
        predecessor = item.predecessor_action_index
        if predecessor is None:
            raise ValueError("eviction reuse lacks its action identity")
        if item.successor_action_index is not None:
            edges.add(
                MemoryReuseDependency(
                    predecessor,
                    successor_action_index=item.successor_action_index,
                )
            )
        elif item.successor_task_id is not None:
            edges.add(
                MemoryReuseDependency(
                    predecessor,
                    successor_task_id=item.successor_task_id,
                )
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
