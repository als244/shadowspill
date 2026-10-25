"""Spill for pool-occupancy diagnostics."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from .clock import Clock
from .facts import ProgramFacts
from .intervals import UNATTRIBUTED, Interval, Occupancy, TransferSpan


def _interval(
    facts: ProgramFacts,
    clock: Clock,
    alias_id: str,
    start_ns: int,
    end_ns: int,
    size: int,
) -> Interval:
    holder = facts.occupant(alias_id, clock.task_start_ns, start_ns)
    return Interval(
        start_ns,
        max(end_ns, start_ns),
        size,
        None if holder is None else holder.object_id,
        UNATTRIBUTED if holder is None else holder.category,
        UNATTRIBUTED if holder is None else holder.role,
    )


def spill_occupancy(
    facts: ProgramFacts,
    schedule: Mapping[str, Any],
    clock: Clock,
    *,
    reported_peak_bytes: int | None = None,
    capacity_bytes: int | None = None,
) -> tuple[Occupancy, tuple[TransferSpan, ...]]:
    """The spill pool over the step under the simulator's rules, and every
    transfer the walk saw, as lane intervals."""

    intervals: list[Interval] = []
    spans: list[TransferSpan] = []
    opened: dict[str, int] = {}  # alias -> start of its current spill copy
    cursors: dict[tuple[str, str, str], int] = defaultdict(int)

    def close(alias_id: str, end_ns: int) -> None:
        start_ns = opened.pop(alias_id, None)
        if start_ns is not None:
            intervals.append(
                _interval(
                    facts,
                    clock,
                    alias_id,
                    start_ns,
                    end_ns,
                    facts.alias_spill_size[alias_id],
                )
            )

    def transfer(alias_id: str, kind: str, trigger: str) -> tuple[int, int] | None:
        key = (alias_id, kind, trigger)
        times = clock.transfers.get(key, ())
        index = cursors[key]
        cursors[key] += 1
        if index >= len(times):
            return None
        start_ns, end_ns = times[index]
        holder = facts.occupant(alias_id, clock.task_start_ns, start_ns)
        spans.append(
            TransferSpan(
                "fetch" if kind == "fetch" else "evict",
                start_ns,
                end_ns,
                facts.alias_copy_bytes[
                    (alias_id, "fetch" if kind == "fetch" else "evict")
                ],
                alias_id,
                None if holder is None else holder.object_id,
                UNATTRIBUTED if holder is None else holder.category,
                trigger,
                facts.alias_size[alias_id],
            )
        )
        return start_ns, end_ns

    for alias_id in facts.retained:
        opened[alias_id] = 0
    for item in schedule.get("initial_residency", ()):
        if item["location"] == "host":
            opened.setdefault(item["alias_group_id"], 0)
    for action in schedule["actions"]:
        alias_id, kind, trigger = (
            action["alias_group_id"],
            action["kind"],
            action["trigger_task_id"],
        )
        if facts.alias_size.get(alias_id, 0) == 0:
            continue
        if kind in ("evict", "write_back"):
            times = transfer(alias_id, kind, trigger)
            opened.setdefault(
                alias_id, times[0] if times else clock.task_end_ns.get(trigger, 0)
            )
        elif kind == "fetch":
            times = transfer(alias_id, kind, trigger)
            if alias_id not in facts.retained:
                close(
                    alias_id, times[1] if times else clock.task_end_ns.get(trigger, 0)
                )
        elif kind == "release":
            if alias_id not in facts.retained:
                close(alias_id, clock.task_end_ns.get(trigger, 0))
    for alias_id in list(opened):
        close(alias_id, clock.makespan_ns)
    return (
        Occupancy("spill", tuple(intervals), reported_peak_bytes, capacity_bytes),
        tuple(spans),
    )
