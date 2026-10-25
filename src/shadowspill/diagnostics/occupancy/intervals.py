"""Intervals for pool-occupancy diagnostics."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

GIB = float(1 << 30)


ROLE_PRIORITY = {
    "parameter": 0,
    "optimizer_state": 1,
    "gradient": 2,
    "buffer": 3,
    "activation": 4,
    "control": 5,
    "output": 6,
}


WORKSPACE = "task workspace"


UNATTRIBUTED = "unattributed"


CATEGORY_ORDER = (
    "weights",
    "optimizer state",
    "model gradients",
    "saved activations",
    "recomputed activations",
    "tangents",
    "inputs",
    "buffers",
    "control",
    "outputs",
    WORKSPACE,
    UNATTRIBUTED,
)


@dataclass(frozen=True, slots=True)
class ObjectFacts:
    """One object of the program, with what the executing tasks make of it."""

    object_id: str
    alias_group_id: str
    offset_bytes: int
    size_bytes: int
    role: str
    persistence: str
    producer: str | None
    consumers: tuple[str, ...]
    mutated: bool
    category: str


@dataclass(frozen=True, slots=True)
class Interval:
    """Bytes occupying one pool from ``start_ns`` to ``end_ns``."""

    start_ns: int
    end_ns: int
    bytes: int
    object_id: str | None
    category: str
    role: str


@dataclass(frozen=True, slots=True)
class TaskSpan:
    task_id: str
    name: str
    phase: str
    start_ns: int
    end_ns: int
    #: The regeneration this task carries: for the backward task of an
    #: alternative chosen as ``recompute``, the group's profiled time beyond
    #: its ``save`` option's. None when either profile is unavailable;
    #: zero for every other task. This is not a measured kernel boundary.
    overhead_ns: int | None = 0


@dataclass(frozen=True, slots=True)
class TransferSpan:
    direction: str
    start_ns: int
    end_ns: int
    bytes: int
    alias_group_id: str
    object_id: str | None
    category: str
    trigger_task_id: str
    #: Full execution object served by this transfer; ``bytes`` above counts
    #: only the host-copy payload. Neither measures collective fabric traffic.
    execution_bytes: int


def _category_rank(name: str) -> tuple[int, str]:
    try:
        return CATEGORY_ORDER.index(name), name
    except ValueError:
        return len(CATEGORY_ORDER), name


@dataclass(frozen=True, slots=True)
class Occupancy:
    """One pool's occupancy over the step, as intervals."""

    pool: str
    intervals: tuple[Interval, ...]
    #: What the planner itself reported for this pool's peak, for the reader
    #: to check the walk against.
    reported_peak_bytes: int | None = None
    capacity_bytes: int | None = None
    #: For the execution pool, what the admitted layout needs with every lease
    #: at its fixed offset: at or above the walk's peak, which packs perfectly.
    required_bytes: int | None = None

    def at(self, time_ns: int, *, by: str = "category") -> dict[str, int]:
        """Bytes per category (or role) resident at ``time_ns``."""

        totals: dict[str, int] = defaultdict(int)
        for item in self.intervals:
            if item.start_ns <= time_ns < item.end_ns:
                totals[item.category if by == "category" else item.role] += item.bytes
        return dict(totals)

    def total_at(self, time_ns: int) -> int:
        return sum(self.at(time_ns).values())

    def peak(self) -> tuple[int, int]:
        """The largest total and the earliest time it holds, as ``(bytes, time_ns)``."""

        events: list[tuple[int, int]] = []
        for item in self.intervals:
            if item.end_ns > item.start_ns:
                events.append((item.start_ns, item.bytes))
                events.append((item.end_ns, -item.bytes))
        # At one instant, frees before allocations: the simulator frees a
        # copy when its transfer completes before it charges the next.
        events.sort(key=lambda event: (event[0], event[1]))
        total, peak, when = 0, 0, 0
        for time_ns, delta in events:
            total += delta
            if total > peak:
                peak, when = total, time_ns
        return peak, when

    def peaks_by(self, by: str = "category") -> dict[str, tuple[int, int]]:
        """Each category's own peak, as ``{name: (bytes, time_ns)}``."""

        groups: dict[str, list[Interval]] = defaultdict(list)
        for item in self.intervals:
            groups[item.category if by == "category" else item.role].append(item)
        return {
            name: Occupancy(self.pool, tuple(items)).peak()
            for name, items in groups.items()
        }

    def series(
        self, *, by: str = "category", points: int = 400, end_ns: int | None = None
    ) -> tuple[list[float], dict[str, list[float]]]:
        """Occupancy sampled at ``points`` instants: seconds, and GiB per name.

        Sampled by one sweep over the interval edges rather than a query per
        instant, so a step of ten thousand intervals renders in a moment.
        """

        end = max((item.end_ns for item in self.intervals), default=0)
        if end_ns is not None:
            end = max(end, end_ns)
        times = [end * index / max(points - 1, 1) for index in range(points)]

        def key(item: Interval) -> str:
            return item.category if by == "category" else item.role

        names = sorted({key(item) for item in self.intervals}, key=_category_rank)
        rows = {name: [0.0] * points for name in names}
        events: list[tuple[int, int, str]] = []
        for item in self.intervals:
            if item.end_ns > item.start_ns:
                events.append((item.start_ns, item.bytes, key(item)))
                events.append((item.end_ns, -item.bytes, key(item)))
        events.sort(key=lambda event: (event[0], event[1]))
        running: dict[str, int] = defaultdict(int)
        cursor = 0
        for index, time_ns in enumerate(times):
            while cursor < len(events) and events[cursor][0] <= time_ns:
                _, delta, name = events[cursor]
                running[name] += delta
                cursor += 1
            for name in names:
                rows[name][index] = running[name] / GIB
        return [t / 1e9 for t in times], rows
