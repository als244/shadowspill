"""Execution for pool-occupancy diagnostics."""

from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .clock import Clock
from .facts import ProgramFacts
from .intervals import WORKSPACE, Interval, Occupancy, TaskSpan
from .spill import _interval


def placement_rows(layout: Mapping[str, Any]) -> list[dict[str, Any]]:
    """A stored certificate's placements, one mapping per lease, keyed as a
    single placement serializes. The certificate keeps them as columns,
    with task and alias names in tables and purposes as codes."""

    columns = layout["placements"]
    tasks, aliases = layout["task_ids"], layout["alias_ids"]
    purposes = layout["purposes"]
    numbers = (
        "lease_id",
        "offset",
        "bytes",
        "alignment",
        "predicted_start_ns",
        "predicted_end_ns",
        "causal_start",
        "causal_end",
    )
    return [
        {
            **dict(zip(numbers, values[:8], strict=True)),
            "purpose": purposes[values[8]],
            "task_id": None if values[9] < 0 else tasks[values[9]],
            "alias_group_id": None if values[10] < 0 else aliases[values[10]],
            "action_index": None if values[11] < 0 else values[11],
        }
        for values in zip(
            *(columns[name] for name in numbers),
            columns["purpose"],
            columns["task"],
            columns["alias"],
            columns["action"],
            strict=True,
        )
    ]


def storage_handoffs(
    facts: ProgramFacts, layout: Mapping[str, Any], simulated: Clock
) -> dict[int, tuple[str, str]]:
    """Leases a task's output took over from one of its inputs, by index into
    the layout's placements: ``{index: (output object, task)}``.

    The layout records a storage handoff as no lease of the output's own:
    the input's lease runs on and is retired by the output's evict or
    release. So an executing task's output without a ``task_output`` lease,
    beside exactly one lease of an input of the output's size that outlives
    the task, is that lease's next occupant. Predicted times are the
    simulator's, so the test is made on the simulated clock. The store keeps
    no other record of a handoff.
    """

    placements = layout["placements"]
    by_alias: dict[str, list[int]] = defaultdict(list)
    for index, lease in enumerate(placements):
        alias_id = lease.get("alias_group_id")
        if alias_id is not None and lease["purpose"] != "task_workspace":
            by_alias[alias_id].append(index)
    handoffs: dict[int, tuple[str, str]] = {}
    for task_id in facts.task_order:
        start_ns = simulated.task_start_ns.get(task_id)
        end_ns = simulated.task_end_ns.get(task_id)
        if start_ns is None or end_ns is None:
            continue
        input_aliases = {
            facts.objects[object_id].alias_group_id
            for object_id in facts.task_inputs.get(task_id, ())
            if object_id in facts.objects
        }
        for object_id in facts.task_outputs.get(task_id, ()):
            holder = facts.objects.get(object_id)
            if holder is None:
                continue
            size = facts.alias_size.get(holder.alias_group_id, 0)
            if size == 0 or any(
                placements[index]["task_id"] == task_id
                and placements[index]["purpose"] == "task_output"
                for index in by_alias[holder.alias_group_id]
            ):
                continue
            candidates = [
                index
                for alias_id in input_aliases
                for index in by_alias[alias_id]
                if int(placements[index]["bytes"]) == size
                and int(placements[index]["predicted_start_ns"]) <= start_ns
                and int(placements[index]["predicted_end_ns"]) >= end_ns
            ]
            if len(candidates) == 1:
                handoffs[candidates[0]] = (object_id, task_id)
    return handoffs


@dataclass(slots=True)
class LeaseClock:
    """Where the layout's leases fall on a clock other than the simulator's.

    A lease's predicted times are the simulator's, and each is an event's:
    a task's start or end, a fetch's issue at its trigger task's end, an
    evict's or write-back's completion, a release at its trigger task's
    end, or the step's end. Placed on the device's clock, a lease opens and
    closes at those events' traced times. An instant no event names is
    interpolated between the task boundaries around it, and counted.
    """

    simulated: Clock
    target: Clock
    actions: Sequence[Mapping[str, Any]]
    #: ``(alias, simulated start)`` -> the fetch of it issued then
    fetch_starts: Mapping[tuple[str, int], tuple[str, int]]
    #: ``(alias, simulated end)`` -> the evict or write-back completing then
    transfer_ends: Mapping[tuple[str, int], tuple[str, int]]
    #: ``(alias, simulated end)`` -> the task whose end releases it then
    release_ends: Mapping[tuple[str, int], str]
    #: Task boundaries as ``(simulated, target)`` pairs, for interpolation.
    anchors: Sequence[tuple[int, int]]
    interpolated: int = 0

    @classmethod
    def build(
        cls,
        simulation: Mapping[str, Any],
        schedule: Mapping[str, Any],
        simulated: Clock,
        target: Clock,
    ) -> LeaseClock:
        fetch_starts: dict[tuple[str, int], tuple[str, int]] = {}
        transfer_ends: dict[tuple[str, int], tuple[str, int]] = {}
        for item in simulation["transfer_intervals"]:
            key = (item["direction"], int(item["sequence"]))
            if item["kind"] == "fetch":
                fetch_starts[(item["alias_group_id"], int(item["start_ns"]))] = key
            else:
                transfer_ends[(item["alias_group_id"], int(item["end_ns"]))] = key
        release_ends: dict[tuple[str, int], str] = {}
        for action in schedule["actions"]:
            trigger = action["trigger_task_id"]
            if action["kind"] == "release" and trigger in simulated.task_end_ns:
                release_ends[
                    (action["alias_group_id"], simulated.task_end_ns[trigger])
                ] = trigger
        anchors = {(0, 0), (simulated.makespan_ns, target.makespan_ns)}
        for task_id, start_ns in simulated.task_start_ns.items():
            if task_id in target.task_start_ns:
                anchors.add((start_ns, target.task_start_ns[task_id]))
        for task_id, end_ns in simulated.task_end_ns.items():
            if task_id in target.task_end_ns:
                anchors.add((end_ns, target.task_end_ns[task_id]))
        return cls(
            simulated,
            target,
            schedule["actions"],
            fetch_starts,
            transfer_ends,
            release_ends,
            sorted(anchors),
        )

    def opens(self, lease: Mapping[str, Any]) -> int:
        purpose = lease["purpose"]
        task_id = str(lease.get("task_id") or "")
        start_ns = int(lease["predicted_start_ns"])
        if purpose == "initial_object":
            return 0
        if purpose in ("task_output", "task_workspace"):
            if (
                start_ns == self.simulated.task_start_ns.get(task_id)
                and task_id in self.target.task_start_ns
            ):
                return self.target.task_start_ns[task_id]
        elif purpose == "fetch_destination":
            # issued at its trigger task's end, its destination held from then
            index = lease.get("action_index")
            trigger = None if index is None else self.actions[index]["trigger_task_id"]
            if (
                trigger is not None
                and start_ns == self.simulated.task_end_ns.get(trigger)
                and trigger in self.target.task_end_ns
            ):
                return self.target.task_end_ns[trigger]
            key = self.fetch_starts.get((lease.get("alias_group_id", ""), start_ns))
            if key is not None and key in self.target.transfer_times:
                return self.target.transfer_times[key][0]
        return self.interpolate(start_ns)

    def closes(self, lease: Mapping[str, Any], aliases: Sequence[str]) -> int:
        task_id = str(lease.get("task_id") or "")
        end_ns = int(lease["predicted_end_ns"])
        if (
            lease["purpose"] == "task_workspace"
            and end_ns == self.simulated.task_end_ns.get(task_id)
            and task_id in self.target.task_end_ns
        ):
            return self.target.task_end_ns[task_id]
        if end_ns == self.simulated.makespan_ns:
            return self.target.makespan_ns
        for alias_id in aliases:
            key = self.transfer_ends.get((alias_id, end_ns))
            if key is not None and key in self.target.transfer_times:
                return self.target.transfer_times[key][1]
            trigger = self.release_ends.get((alias_id, end_ns))
            if trigger is not None and trigger in self.target.task_end_ns:
                return self.target.task_end_ns[trigger]
        return self.interpolate(end_ns)

    def interpolate(self, time_ns: int) -> int:
        """Between the task boundaries around ``time_ns``, linearly."""

        self.interpolated += 1
        anchors = self.anchors
        index = bisect_left(anchors, (time_ns, -1))
        if index == len(anchors):
            last_simulated, last_target = anchors[-1]
            return last_target + (time_ns - last_simulated)
        after_simulated, after_target = anchors[index]
        if after_simulated == time_ns or index == 0:
            return after_target
        before_simulated, before_target = anchors[index - 1]
        return before_target + (time_ns - before_simulated) * (
            after_target - before_target
        ) // max(after_simulated - before_simulated, 1)


def execution_occupancy(
    facts: ProgramFacts,
    layout: Mapping[str, Any],
    clock: Clock,
    *,
    placed: LeaseClock | None = None,
    reported_peak_bytes: int | None = None,
) -> Occupancy:
    """The execution pool over the step, lease by lease, as admitted.

    On the simulator's clock a lease runs from its predicted start to its
    predicted end; ``placed`` puts each on another clock instead. A lease an
    output took over from an input (``storage_handoffs``) counts for the
    input until that task starts and for the output from then on.
    """

    simulated = clock if placed is None else placed.simulated
    handoffs = storage_handoffs(facts, layout, simulated)
    intervals: list[Interval] = []
    for index, lease in enumerate(layout["placements"]):
        alias_id = lease.get("alias_group_id")
        size = int(lease["bytes"])
        taken = handoffs.get(index)
        if placed is None:
            start_ns = int(lease["predicted_start_ns"])
            end_ns = int(lease["predicted_end_ns"])
        else:
            aliases = [] if alias_id is None else [alias_id]
            if taken is not None:
                aliases.insert(0, facts.objects[taken[0]].alias_group_id)
            start_ns, end_ns = placed.opens(lease), placed.closes(lease, aliases)
        end_ns = max(end_ns, start_ns)
        if lease["purpose"] == "task_workspace" or alias_id is None:
            intervals.append(
                Interval(start_ns, end_ns, size, None, WORKSPACE, WORKSPACE)
            )
        elif taken is None:
            intervals.append(_interval(facts, clock, alias_id, start_ns, end_ns, size))
        else:
            object_id, task_id = taken
            split = min(
                max(clock.task_start_ns.get(task_id, start_ns), start_ns), end_ns
            )
            if split > start_ns:
                intervals.append(
                    _interval(facts, clock, alias_id, start_ns, split, size)
                )
            holder = facts.objects[object_id]
            intervals.append(
                Interval(split, end_ns, size, object_id, holder.category, holder.role)
            )
    return Occupancy(
        "execution",
        tuple(intervals),
        reported_peak_bytes,
        int(layout["pool_capacity_bytes"]) if "pool_capacity_bytes" in layout else None,
        int(layout["required_bytes"]) if "required_bytes" in layout else None,
    )


def task_spans(facts: ProgramFacts, clock: Clock) -> tuple[TaskSpan, ...]:
    return tuple(
        TaskSpan(
            task_id,
            clock.task_names.get(task_id) or task_id,
            facts.task_phase.get(task_id, ""),
            clock.task_start_ns[task_id],
            clock.task_end_ns.get(task_id, clock.task_start_ns[task_id]),
            facts.overhead_by_task.get(task_id, 0),
        )
        for task_id in facts.task_order
        if task_id in clock.task_start_ns
    )
