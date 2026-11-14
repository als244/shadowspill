"""Assign nonoverlapping task temporaries to reusable fixed-address slots."""

from bisect import bisect_left
from collections.abc import Mapping
from dataclasses import dataclass, field, replace

from shadowspill.task.allocations.events import (
    TaskAllocationEvent,
    TaskAllocationOperation,
)


@dataclass
class _Slot:
    bytes: int
    stream: int
    alignment: int
    intervals: list[tuple[int, int, int]] = field(default_factory=list)

    def insertion(self, start: int, end: int) -> int | None:
        position = bisect_left(self.intervals, (start,))
        if position and self.intervals[position - 1][1] > start:
            return None
        if position < len(self.intervals) and end > self.intervals[position][0]:
            return None
        return position


def _cost(trace: tuple[TaskAllocationEvent, ...]) -> tuple[int, int]:
    roots: dict[int, int] = {}
    sizes: dict[int, int] = {}
    for event in trace:
        if event.operation is not TaskAllocationOperation.ALLOCATE:
            continue
        ordinal = event.allocation_ordinal
        root = ordinal if event.reuses_ordinal is None else roots[event.reuses_ordinal]
        roots[ordinal] = root
        aligned = (
            -(-event.charged_bytes // event.alignment_bytes) * event.alignment_bytes
        )
        sizes[root] = max(sizes.get(root, 0), aligned)
    return sum(sizes.values()), len(sizes)


def assign_temporary_slots(
    trace: tuple[TaskAllocationEvent, ...], streams: Mapping[int, int]
) -> tuple[TaskAllocationEvent, ...]:
    """Compare exact-size slots with a largest-first interval assignment.

    Each slot retains one address and reserves its largest request. Reuse stays
    on the observed allocation stream; runtime joins recorded private uses at
    that address. Escaping storage gets its own slot in the interval candidate
    so a small returned value cannot keep a large scratch slot resident.
    Neither candidate changes requests, frees, output ownership or live peaks.
    """
    starts = {}
    ends = {}
    for position, event in enumerate(trace):
        if event.operation is TaskAllocationOperation.ALLOCATE:
            starts[event.allocation_ordinal] = (position, event)
        else:
            ends[event.allocation_ordinal] = position
    slots: list[_Slot] = []
    for ordinal, (start, event) in sorted(
        starts.items(), key=lambda item: (-item[1][1].charged_bytes, item[1][0])
    ):
        # Returned roots and persistent in-pool state outlive this invocation.
        if event.output_leaf_indices or ordinal not in ends:
            continue
        end = ends[ordinal]
        candidates = []
        for index, slot in enumerate(slots):
            if (
                slot.stream != streams[ordinal]
                or slot.alignment != event.alignment_bytes
            ):
                continue
            insertion = slot.insertion(start, end)
            if insertion is not None:
                candidates.append((slot.bytes, index, insertion))
        if candidates:
            _, index, insertion = min(candidates)
            slots[index].intervals.insert(insertion, (start, end, ordinal))
        else:
            slots.append(
                _Slot(
                    event.charged_bytes,
                    streams[ordinal],
                    event.alignment_bytes,
                    [(start, end, ordinal)],
                )
            )
    reuse = {}
    for slot in slots:
        for previous, current in zip(slot.intervals, slot.intervals[1:], strict=False):
            reuse[current[2]] = previous[2]
    candidate = tuple(
        replace(event, reuses_ordinal=reuse.get(event.allocation_ordinal))
        if event.operation is TaskAllocationOperation.ALLOCATE
        else event
        for event in trace
    )
    return candidate if _cost(candidate) < _cost(trace) else trace
