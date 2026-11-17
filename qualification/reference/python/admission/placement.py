"""Readable fixed-offset placement; production uses the planner.

Leases are placed largest first, longest-lived first among equals, and each
takes the lowest aligned offset that clears every lease it overlaps in time.
This is the algorithm `shadowspill_place_lifetimes` implements, written for
reading rather than speed: it is the oracle the compiled placement is
differentially tested against, and the baseline its speedup is measured
against.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class Lifetime(Protocol):
    """The four numbers placement reads from one lease."""

    @property
    def bytes(self) -> int: ...

    @property
    def alignment(self) -> int: ...

    @property
    def predicted_start_ns(self) -> int: ...

    @property
    def predicted_end_ns(self) -> int: ...


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
        self, node: int, left: int, right: int, start: int, end: int, item: int
    ) -> None:
        if start <= left and right <= end:
            self._nodes.setdefault(node, []).append(item)
            return
        middle = (left + right) // 2
        if start < middle:
            self._add(node * 2, left, middle, start, end, item)
        if middle < end:
            self._add(node * 2 + 1, middle, right, start, end, item)

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


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def place_lifetimes(
    items: Sequence[Lifetime],
    excluded: Sequence[bool] | None = None,
) -> tuple[tuple[int, ...], int]:
    """Return each lease's offset, in input order, and the bytes required.

    An excluded lease is placed elsewhere: it gets offset zero here and is
    outside the bytes required.
    """

    if not items:
        return (), 0
    if excluded is not None and any(excluded):
        kept = [source for source, skip in enumerate(excluded) if not skip]
        placed, required = place_lifetimes([items[source] for source in kept])
        offsets = [0] * len(items)
        for position, source in enumerate(kept):
            offsets[source] = placed[position]
        return tuple(offsets), required
    sizes = [item.bytes for item in items]
    alignments = [item.alignment for item in items]
    starts = [item.predicted_start_ns for item in items]
    ends = [item.predicted_end_ns for item in items]

    times = sorted({value for value in (*starts, *ends)})
    rank = {value: index for index, value in enumerate(times)}
    offsets = [0] * len(items)
    index = _TemporalIndex(max(1, len(times)))
    order = sorted(
        range(len(items)),
        key=lambda item: (
            -sizes[item],
            -(rank[ends[item]] - rank[starts[item]]),
            starts[item],
            item,
        ),
    )
    required = 0
    for interval_id in order:
        occupied = sorted(
            (offsets[other], offsets[other] + sizes[other])
            for other in index.overlapping(
                rank[starts[interval_id]], rank[ends[interval_id]]
            )
        )
        cursor = 0
        for left, right in occupied:
            if right <= cursor:
                continue
            candidate = _align_up(cursor, alignments[interval_id])
            if candidate + sizes[interval_id] <= left:
                break
            cursor = max(cursor, right)
        offset = _align_up(cursor, alignments[interval_id])
        offsets[interval_id] = offset
        required = max(required, offset + sizes[interval_id])
        index.add(rank[starts[interval_id]], rank[ends[interval_id]], interval_id)
    return tuple(offsets), required


__all__ = ["Lifetime", "place_lifetimes"]
