"""Collective execution on the runtime trace's shared device-time axis."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class CollectiveRecord:
    collective_id: str
    operation: str
    group: str
    group_ranks: tuple[int, ...]
    group_rank: int
    group_size: int
    group_sequence: int
    input_bytes: int
    output_bytes: int
    source: str
    task_id: str | None
    execution_task_id: str | None
    object_id: int | None
    alias_group_id: str | None
    action_ordinal: int | None
    issued_at_ns: int | None
    started_at_ns: int | None
    finished_at_ns: int | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class HostCopyRecord:
    """The copy inside one transfer that did more than copy, as its lane timed it.

    The runtime's lane marks the stream around the copy alone and nothing else
    does, so the interval holds the copy and nothing a trace added.
    """

    copy_id: str
    source: str
    bytes: int
    task_id: str | None
    execution_task_id: str | None
    object_id: int | None
    alias_group_id: str | None
    action_ordinal: int | None
    issued_at_ns: int | None
    started_at_ns: int | None
    finished_at_ns: int | None


@dataclass(frozen=True, slots=True)
class HostCopyTrace:
    """Every such copy of a step, from the runtime's own trace.

    A record exists only with both of its instants. A transfer whose copy the
    lane could not time has none, and stays unsplit where it is drawn.
    """

    records: tuple[HostCopyRecord, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CollectiveTrace:
    """Bounded collective capture; missing timing is explicit, never zero.

    Issue is a host-clock sample mapped to the trace origin. Start and finish
    are GPU events on the collective's stream, after its input/queue waits.
    They bracket the collective operation, including time waiting for peers
    inside that operation. Byte counts are per-rank input/output payloads, not an
    estimate of fabric traffic. Task collectives are already included in task
    durations; transfer collectives are already included in transfer durations.
    """

    enabled: bool = False
    capacity: int = 0
    dropped: int = 0
    errors: int = 0
    missing_timings: int = 0
    records: tuple[CollectiveRecord, ...] = ()
    # None when the runtime trace ran out of room and its events are not joined.
    host_copies: HostCopyTrace | None = None

    @property
    def complete(self) -> bool:
        return not (self.dropped or self.errors or self.missing_timings)

    def as_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "capacity": self.capacity,
            "dropped": self.dropped,
            "errors": self.errors,
            "missing_timings": self.missing_timings,
            "complete": self.complete,
            "records": [record.as_dict() for record in self.records],
            "host_copies": (
                None if self.host_copies is None else self.host_copies.as_dict()
            ),
        }
