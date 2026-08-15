"""Deterministic interval placement for a fixed execution-pool slice."""

from __future__ import annotations

from .model import FixedLayoutPlacement, LeaseLifetime


class _TemporalIndex:
    """Segment tree returning placed lifetimes overlapping a time range."""

    def __init__(self, length: int) -> None:
        self._length = length
        self._nodes: dict[int, list[int]] = {}

    def add(self, start: int, end: int, interval_id: int) -> None:
        self._add(1, 0, self._length, start, end, interval_id)

    def overlapping(self, start: int, end: int) -> set[int]:
        found: set[int] = set()
        self._query(1, 0, self._length, start, end, found)
        return found

    def _add(
        self,
        node: int,
        left: int,
        right: int,
        start: int,
        end: int,
        interval_id: int,
    ) -> None:
        if start <= left and right <= end:
            self._nodes.setdefault(node, []).append(interval_id)
            return
        middle = (left + right) // 2
        if start < middle:
            self._add(node * 2, left, middle, start, end, interval_id)
        if middle < end:
            self._add(node * 2 + 1, middle, right, start, end, interval_id)

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
        found.update(self._nodes.get(node, ()))
        if right - left == 1:
            return
        middle = (left + right) // 2
        if start < middle:
            self._query(node * 2, left, middle, start, end, found)
        if middle < end:
            self._query(node * 2 + 1, middle, right, start, end, found)


def place_lifetimes(
    lifetimes: tuple[LeaseLifetime, ...],
) -> tuple[tuple[FixedLayoutPlacement, ...], int]:
    """Place larger and longer-lived leases first, then choose lowest fit."""

    times = sorted(
        {
            value
            for item in lifetimes
            for value in (item.predicted_start_ns, item.predicted_end_ns)
        }
    )
    rank = {value: index for index, value in enumerate(times)}
    offsets = [0] * len(lifetimes)
    index = _TemporalIndex(max(1, len(times)))
    order = sorted(
        range(len(lifetimes)),
        key=lambda item: (
            -lifetimes[item].bytes,
            -(
                rank[lifetimes[item].predicted_end_ns]
                - rank[lifetimes[item].predicted_start_ns]
            ),
            lifetimes[item].predicted_start_ns,
            lifetimes[item].lease_id,
        ),
    )
    required = 0
    for interval_id in order:
        interval = lifetimes[interval_id]
        occupied = sorted(
            (
                offsets[other],
                offsets[other] + lifetimes[other].bytes,
            )
            for other in index.overlapping(
                rank[interval.predicted_start_ns],
                rank[interval.predicted_end_ns],
            )
        )
        cursor = 0
        for left, right in occupied:
            if right <= cursor:
                continue
            candidate = _align_up(cursor, interval.alignment)
            if candidate + interval.bytes <= left:
                break
            cursor = max(cursor, right)
        offset = _align_up(cursor, interval.alignment)
        offsets[interval_id] = offset
        required = max(required, offset + interval.bytes)
        index.add(
            rank[interval.predicted_start_ns],
            rank[interval.predicted_end_ns],
            interval_id,
        )
    placements = tuple(
        FixedLayoutPlacement(
            lease_id=item.lease_id,
            offset=offsets[index_],
            bytes=item.bytes,
            alignment=item.alignment,
            predicted_start_ns=item.predicted_start_ns,
            predicted_end_ns=item.predicted_end_ns,
            causal_start=item.causal_start,
            causal_end=item.causal_end,
            purpose=item.purpose,
            task_id=item.task_id,
            alias_group_id=item.alias_group_id,
            action_index=item.action_index,
        )
        for index_, item in enumerate(lifetimes)
    )
    return placements, required


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


__all__ = ["place_lifetimes"]
