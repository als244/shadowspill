"""Readable reuse-dependency recovery; production uses the planner.

Two leases that share an address are sound only if the second cannot begin
before the first has released it. Leases are taken in causal order, a
range-assignment tree names the latest lease to have owned each address, and
every lease it names across a successor's range is a predecessor of that
successor; the predecessor's retirement must precede the successor's
acquisition in operation order and must publish a completion dependency.
This is the algorithm `shadowspill_recover_layout_reuse` implements, written
for reading rather than speed: it is the oracle the compiled recovery is
differentially tested against.

Everything is columns of integers, as the library sees them: a task or action
that is not named is `None`, and a lease that outlives the step retires at
`None`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

#: Operation boundaries, as `ShadowSpillAdmissionBoundaryKind` numbers them.
INITIAL_BOUNDARY = 0
ACTION_BOUNDARIES = frozenset({3, 4})


@dataclass(frozen=True, slots=True)
class PlacedLease:
    """One placed lease: where it lies, when it is held, and who consumes it."""

    lease_id: int
    offset: int
    bytes: int
    causal_start: int
    causal_end: int
    task: int | None
    action: int | None


@dataclass(frozen=True, slots=True)
class Operations:
    """The operation columns the recovery reads, indexed by operation, and the
    retiring operation of each lease, indexed by lease id."""

    dependency_ids: Sequence[int | None]
    purposes: Sequence[int]
    boundaries: Sequence[int]
    indices: Sequence[int]
    lease_retires: Sequence[int | None]
    action_trigger_tasks: Sequence[int]


@dataclass(frozen=True, slots=True)
class Reuse:
    """One recovered dependency, in the order the library reports them."""

    dependency_id: int
    predecessor_lease_id: int
    predecessor_purpose: int
    predecessor_task: int
    predecessor_action: int | None
    successor_lease_id: int
    successor_task: int | None
    successor_action: int | None


class LayoutRefused(ValueError):
    """The layout is not safe to run; `kind` says why, as the library's
    `ShadowSpillLayoutReuseRefusal` numbers it."""

    def __init__(self, kind: int, predecessor: int, successor: int, message: str):
        super().__init__(message)
        self.kind = kind
        self.predecessor = predecessor
        self.successor = successor


_EMPTY = -1
_MIXED = -2


class _LatestOwnerIndex:
    """Range-assignment tree whose leaves name the latest owner of an address."""

    def __init__(self, leases: Sequence[PlacedLease]) -> None:
        coordinates = sorted(
            {
                value
                for item in leases
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

    def _query(self, node, left, right, start, end, found: set[int]) -> None:
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

    def _assign(self, node, left, right, start, end, owner: int) -> None:
        if end <= left or right <= start:
            return
        if (start <= left and right <= end) or right - left == 1:
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


def _retirement(operations: Operations, lease_id: int):
    """(dependency, purpose, task, action) of a lease's retirement, or a refusal."""

    index = operations.lease_retires[lease_id]
    if index is None or operations.dependency_ids[index] is None:
        return 2, None
    boundary = operations.boundaries[index]
    position = operations.indices[index]
    if boundary in ACTION_BOUNDARIES:
        task, action = operations.action_trigger_tasks[position], position
    elif boundary == INITIAL_BOUNDARY:
        return 3, None
    else:
        task, action = position, None
    return 0, (
        operations.dependency_ids[index],
        operations.purposes[index],
        task,
        action,
    )


def recover_layout_reuse(
    leases: Sequence[PlacedLease], operations: Operations
) -> tuple[Reuse, ...]:
    """Every dependency a layout's shared addresses imply, or `LayoutRefused`."""

    by_id = {item.lease_id: item for item in leases}
    latest = _LatestOwnerIndex(leases)
    found: set[Reuse] = set()
    for successor in sorted(
        leases, key=lambda item: (item.causal_start, item.lease_id)
    ):
        predecessors = latest.replace(
            successor.offset, successor.offset + successor.bytes, successor.lease_id
        )
        for predecessor_id in sorted(predecessors):
            predecessor = by_id[predecessor_id]
            if predecessor.causal_end > successor.causal_start:
                raise LayoutRefused(
                    1,
                    predecessor_id,
                    successor.lease_id,
                    "fixed layout overlaps causally live leases: "
                    f"predecessor={predecessor_id}, successor={successor.lease_id}",
                )
            kind, retired = _retirement(operations, predecessor_id)
            if kind == 2:
                raise LayoutRefused(
                    2,
                    predecessor_id,
                    successor.lease_id,
                    f"fixed layout reuses lease {predecessor_id} without a "
                    "completion dependency",
                )
            if kind == 3:
                raise LayoutRefused(
                    3,
                    predecessor_id,
                    successor.lease_id,
                    "fixed-layout predecessor lacks a task",
                )
            if successor.task is None and successor.action is None:
                raise LayoutRefused(
                    4,
                    predecessor_id,
                    successor.lease_id,
                    "fixed-layout successor lacks a consumer",
                )
            dependency_id, purpose, task, action = retired
            found.add(
                Reuse(
                    dependency_id=dependency_id,
                    predecessor_lease_id=predecessor_id,
                    predecessor_purpose=purpose,
                    predecessor_task=task,
                    predecessor_action=action,
                    successor_lease_id=successor.lease_id,
                    successor_task=successor.task,
                    successor_action=successor.action,
                )
            )
    return tuple(
        sorted(
            found,
            key=lambda item: (
                item.successor_lease_id,
                item.dependency_id,
                item.predecessor_lease_id,
            ),
        )
    )


__all__ = [
    "LayoutRefused",
    "Operations",
    "PlacedLease",
    "Reuse",
    "recover_layout_reuse",
]
