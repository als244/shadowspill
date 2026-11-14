"""Clock for pool-occupancy diagnostics."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Clock:
    """When each task runs and each transfer is issued, starts and ends, on
    one clock."""

    task_start_ns: Mapping[str, int]
    task_end_ns: Mapping[str, int]
    #: ``(alias_group_id, kind, trigger_task_id)`` -> the matching transfers'
    #: ``(issued_ns, start_ns, end_ns)`` in issue order. A transfer is issued
    #: when the simulator submits it, its ``ready_ns``, or on the device at
    #: its trigger task's end, and starts when its lane takes it up, which
    #: can be much later.
    transfers: Mapping[tuple[str, str, str], Sequence[tuple[int, int, int]]]
    makespan_ns: int
    #: What a traced step calls each task, when there is one.
    task_names: Mapping[str, str]
    #: ``(direction, sequence)`` -> that transfer's ``(start_ns, end_ns)``, by
    #: the simulated transfer's sequence number, which its traced copy keeps.
    transfer_times: Mapping[tuple[str, int], tuple[int, int]] = field(
        default_factory=dict
    )
    #: On a traced clock, the simulated transfers the trace could not time:
    #: a record whose lane reported no start or finish, or no record at all.
    #: They are off the lanes, and a walk that needs their time takes their
    #: trigger task's end instead.
    untimed_transfers: int = 0

    @classmethod
    def simulated(cls, simulation: Mapping[str, Any]) -> Clock:
        intervals = simulation["task_intervals"]
        task_start = {item["task_id"]: int(item["start_ns"]) for item in intervals}
        task_end = {item["task_id"]: int(item["end_ns"]) for item in intervals}
        transfers: dict[tuple[str, str, str], list[tuple[int, int, int]]] = defaultdict(
            list
        )
        times: dict[tuple[str, int], tuple[int, int]] = {}
        for item in sorted(
            simulation["transfer_intervals"], key=lambda item: int(item["sequence"])
        ):
            span = (int(item["start_ns"]), int(item["end_ns"]))
            # a record without its ready time was submitted at its trigger's end
            issued = (
                int(item["ready_ns"])
                if "ready_ns" in item
                else task_end[item["trigger_task_id"]]
            )
            transfers[
                (item["alias_group_id"], item["kind"], item["trigger_task_id"])
            ].append((issued, *span))
            times[(item["direction"], int(item["sequence"]))] = span
        return cls(
            task_start, task_end, transfers, int(simulation["makespan_ns"]), {}, times
        )

    @classmethod
    def traced(
        cls, simulation: Mapping[str, Any], diagnostics: Mapping[str, Any]
    ) -> Clock:
        """The simulated clock replaced by the device's, transfer by transfer.

        Traced transfers carry the simulated transfer's ``sequence``, which
        is how the two are paired; a traced task carries its program task id.
        A transfer is issued at its trigger task's traced end, where the
        runtime reserves its destination, or, when that task was not traced,
        at its start on the lane.
        """

        seconds = 1e9
        records = list(diagnostics["tasks"].values())
        task_start = {
            record["task_id"]: int(
                record["compute"]["compute_started_at_seconds"] * seconds
            )
            for record in records
        }
        task_end = {
            record["task_id"]: int(
                record["compute"]["compute_finished_at_seconds"] * seconds
            )
            for record in records
        }
        names = {
            record["task_id"]: str(record.get("semantic_name", ""))
            for record in records
        }
        lanes: dict[tuple[str, int], tuple[int, int]] = {}
        for direction in ("fetch", "evict"):
            for record in diagnostics["transfers"][direction].values():
                lane = record["lane"]
                started = lane.get("lane_started_at_seconds")
                finished = lane.get("lane_finished_at_seconds")
                if started is None or finished is None:
                    continue  # the lane reported no timing for this copy
                # The lane's occupancy: from the transfer's start on the lane
                # to its finish. Issue comes earlier, and a fetch issued far
                # ahead waits in the queue for most of a step.
                lanes[(direction, int(record["sequence"]))] = (
                    int(started * seconds),
                    int(finished * seconds),
                )
        transfers: dict[tuple[str, str, str], list[tuple[int, int, int]]] = defaultdict(
            list
        )
        untimed = 0
        for item in sorted(
            simulation["transfer_intervals"], key=lambda item: int(item["sequence"])
        ):
            key = (item["direction"], int(item["sequence"]))
            if key not in lanes:
                untimed += 1
                continue
            start_ns, end_ns = lanes[key]
            transfers[
                (item["alias_group_id"], item["kind"], item["trigger_task_id"])
            ].append(
                (task_end.get(item["trigger_task_id"], start_ns), start_ns, end_ns)
            )
        makespan = max(
            [end for ends in transfers.values() for _, _, end in ends]
            + list(task_end.values()),
            default=0,
        )
        return cls(task_start, task_end, transfers, makespan, names, lanes, untimed)


def _busy_ns(intervals: Iterable[tuple[int, int]]) -> int:
    """Union of one lane's intervals; batched trace events can overlap."""
    total = end = 0
    for start, stop in sorted(intervals):
        total += max(0, stop - max(start, end))
        end = max(end, stop)
    return total
