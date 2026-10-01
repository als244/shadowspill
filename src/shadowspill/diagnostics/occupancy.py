"""Pool occupancy over a step, attributed to objects and their roles.

A plan the store holds says what occupies each pool and when: the schedule's
memory actions and the simulated transfer intervals give the spill pool under
the simulator's own rules, and the admitted layout's leases give the
execution pool as it was placed. This module walks both and attributes every
byte to the object holding it at that moment -- an alias group is a storage
slot that several objects may occupy over a step, so a slot counts for
whichever of its objects is live -- and from the object to its role and to a
category that says what the object is for: a weight, optimizer state, a
saved activation, a tangent, a model gradient.

The spill rules are the simulator's. A retained alias group (checkpoint
state) holds a spill copy for the whole step. Any other group holds one from
the moment its evict or write-back is issued until the fetch that brings it
back completes, or until it is released; a group the schedule starts on the
host holds one from the start. The execution rules are the layout's: each
lease occupies its bytes from its predicted start to its predicted end, a
task's workspace is its own category, and a lease that a task's output
takes over from one of its inputs -- a storage handoff, which the layout
records as no lease of the output's own -- is the output's from that
task's start. The same walk gives the compute, fetch and evict lanes as
intervals, for the pages this module writes.

With a traced step's diagnostics the walk also runs on the device's clock:
the spill pool and the lanes from the traced tasks and transfers, and the
execution pool with each lease placed at the device times of the events
that open and close it.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from bisect import bisect_left
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from fractions import Fraction
from html import escape
from itertools import pairwise
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote

GIB = float(1 << 30)

#: Which object of several sharing one storage slot at the same time names
#: the slot: state over what is derived from it, and the larger over the
#: smaller.
ROLE_PRIORITY = {
    "parameter": 0,
    "optimizer_state": 1,
    "gradient": 2,
    "activation": 3,
    "buffer": 4,
    "control": 5,
    "output": 6,
}

WORKSPACE = "task workspace"
UNATTRIBUTED = "unattributed"

#: The categories in the order the page stacks them, state at the bottom.
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
    #: its ``save`` option's. Zero for every other task.
    overhead_ns: int = 0


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


# --- reading a program --------------------------------------------------------


def selected_task_ids(
    program: Mapping[str, Any], selections: Iterable[Mapping[str, str]]
) -> list[str]:
    """The tasks that execute under ``selections``, in program order."""

    chosen = {item["group_id"]: item["option_id"] for item in selections}
    in_groups: set[str] = set()
    active: set[str] = set()
    for group in program.get("task_alternative_groups", ()):
        for option in group["options"]:
            in_groups.update(option["active_task_ids"])
            if chosen.get(group["group_id"]) == option["option_id"]:
                active.update(option["active_task_ids"])
    return [
        task["task_id"]
        for task in program["tasks"]
        if task["task_id"] not in in_groups or task["task_id"] in active
    ]


def _mutated(task: Mapping[str, Any]) -> list[str]:
    return [
        item["object_id"] if isinstance(item, Mapping) else str(item)
        for item in task.get("mutations", ())
    ]


def categorize(
    role: str, producer_phase: str | None, consumer_phases: Sequence[str], mutated: bool
) -> str:
    """What an object is for, from its role and the phases that touch it.

    An activation the forward made is a saved activation whatever reads it
    later -- the next stage after the other microbatches have had theirs, or
    the backward -- since the pool holds it the same way for either. One a
    backward-phase task made is a recomputed activation.
    """

    if role == "parameter":
        return "weights"
    if role == "optimizer_state":
        return "optimizer state"
    if role == "gradient":
        if mutated or "optimizer" in consumer_phases:
            return "model gradients"
        return "tangents"
    if role == "activation":
        if producer_phase is None:
            return "inputs"
        return (
            "saved activations"
            if producer_phase == "forward"
            else "recomputed activations"
        )
    return {"buffer": "buffers", "control": "control", "output": "outputs"}.get(
        role, role
    )


@dataclass(slots=True)
class ProgramFacts:
    """A program indexed for attribution.

    Only the tasks that execute count: a step object that no executing task
    makes or reads belongs to an alternative that was not selected and is
    left out, so a storage slot is never attributed to a value that never
    existed. A task that first writes an object in place is its producer.
    """

    alias_size: dict[str, int]
    retained: set[str]
    objects: dict[str, ObjectFacts]
    #: Per alias group, its generations in production order: ``(producer
    #: order, owning object)``. An initial generation, at order -1, exists
    #: only for a group no task produces into: an object without a producer
    #: in a produced group is a view of a produced object, not a value of
    #: its own.
    generations: dict[str, list[tuple[int, str]]]
    task_order: dict[str, int]
    task_phase: dict[str, str]
    #: What each executing task reads and makes, as object ids.
    task_inputs: dict[str, tuple[str, ...]]
    task_outputs: dict[str, tuple[str, ...]]
    #: The executing tasks that regenerate a value rather than read a saved
    #: one: the tasks of every alternative the selection chose as ``recompute``.
    recompute_tasks: frozenset[str] = frozenset()
    #: What regenerating costs, by the program's own profiles: for every
    #: alternative chosen as ``recompute``, its tasks' profiled time less the
    #: ``save`` option's. The planner's recomputation overhead.
    recompute_overhead_ns: int = 0
    #: The same, per task: each group's overhead on its backward task (the
    #: one that re-runs the forward), or its last task when it has none.
    overhead_by_task: dict[str, int] = field(default_factory=dict)

    @classmethod
    def build(
        cls, program: Mapping[str, Any], selections: Iterable[Mapping[str, str]] = ()
    ) -> ProgramFacts:
        selections = list(selections)
        active_ids = selected_task_ids(program, selections)
        chosen = {item["group_id"]: item["option_id"] for item in selections}
        tasks = {task["task_id"]: task for task in program["tasks"]}
        runtime = {
            profile["profile_id"]: int(profile.get("runtime_ns", 0))
            for profile in program.get("profiles", ())
        }

        def profiled(task_ids: Iterable[str]) -> int:
            return sum(
                runtime.get(str(tasks[task_id].get("profile_id")), 0)
                for task_id in task_ids
                if task_id in tasks
            )

        recompute: set[str] = set()
        overhead = 0
        overhead_by_task: dict[str, int] = {}
        for group in program.get("task_alternative_groups", ()):
            if chosen.get(group["group_id"]) != "recompute":
                continue
            options = {
                option["option_id"]: option["active_task_ids"]
                for option in group["options"]
            }
            regenerating = list(options.get("recompute", ()))
            recompute.update(regenerating)
            extra = profiled(regenerating) - profiled(options.get("save", ()))
            overhead += extra
            carriers = [
                task_id
                for task_id in regenerating
                if str(tasks.get(task_id, {}).get("phase")) == "backward"
            ] or regenerating[-1:]
            if carriers and extra > 0:
                overhead_by_task[carriers[-1]] = (
                    overhead_by_task.get(carriers[-1], 0) + extra
                )
        task_order = {task_id: index for index, task_id in enumerate(active_ids)}
        task_phase = {
            task_id: str(tasks[task_id].get("phase")) for task_id in active_ids
        }
        producer: dict[str, str] = {}
        first_writer: dict[str, str] = {}
        consumers: dict[str, list[str]] = defaultdict(list)
        mutated: set[str] = set()
        touched: set[str] = set()
        task_inputs: dict[str, tuple[str, ...]] = {}
        task_outputs: dict[str, tuple[str, ...]] = {}
        for task_id in active_ids:
            task = tasks[task_id]
            task_inputs[task_id] = tuple(task.get("inputs", ()))
            task_outputs[task_id] = tuple(task.get("outputs", ()))
            for object_id in task.get("outputs", ()):
                producer.setdefault(object_id, task_id)
                touched.add(object_id)
            for object_id in task.get("inputs", ()):
                consumers[object_id].append(task_id)
                touched.add(object_id)
            for object_id in _mutated(task):
                consumers[object_id].append(task_id)
                first_writer.setdefault(object_id, task_id)
                mutated.add(object_id)
                touched.add(object_id)
        objects: dict[str, ObjectFacts] = {}
        by_alias: dict[str, list[ObjectFacts]] = defaultdict(list)
        for item in program["objects"]:
            object_id = item["object_id"]
            persistence = str(item.get("persistence", ""))
            if object_id not in touched and persistence == "step":
                continue
            made_by = producer.get(object_id)
            if made_by is None and persistence == "step":
                made_by = first_writer.get(object_id)
            readers = tuple(consumers.get(object_id, ()))
            facts = ObjectFacts(
                object_id=object_id,
                alias_group_id=item["alias_group_id"],
                offset_bytes=int(item.get("offset_bytes", 0)),
                size_bytes=int(item["size_bytes"]),
                role=str(item["role"]),
                persistence=persistence,
                producer=made_by,
                consumers=readers,
                mutated=object_id in mutated,
                category=categorize(
                    str(item["role"]),
                    None if made_by is None else task_phase.get(made_by),
                    [task_phase.get(task_id, "") for task_id in readers],
                    object_id in mutated,
                ),
            )
            objects[object_id] = facts
            by_alias[facts.alias_group_id].append(facts)
        generations: dict[str, list[tuple[int, str]]] = {}
        for alias_id, members in by_alias.items():
            by_order: dict[int, list[ObjectFacts]] = defaultdict(list)
            for facts in members:
                order = (
                    -1 if facts.producer is None else task_order.get(facts.producer, -1)
                )
                by_order[order].append(facts)
            if len(by_order) > 1:
                by_order.pop(-1, None)
            generations[alias_id] = [
                (
                    order,
                    max(
                        group,
                        key=lambda facts: (
                            facts.size_bytes,
                            -facts.offset_bytes,
                            -ROLE_PRIORITY.get(facts.role, len(ROLE_PRIORITY)),
                        ),
                    ).object_id,
                )
                for order, group in sorted(by_order.items())
            ]
        return cls(
            alias_size={
                group["alias_group_id"]: int(group["size_bytes"])
                for group in program["alias_groups"]
            },
            retained={
                group["alias_group_id"]
                for group in program["alias_groups"]
                if group.get("retain_spill_copy")
            },
            objects=objects,
            generations=generations,
            task_order=task_order,
            task_phase=task_phase,
            task_inputs=task_inputs,
            task_outputs=task_outputs,
            recompute_tasks=frozenset(recompute),
            recompute_overhead_ns=max(overhead, 0),
            overhead_by_task=overhead_by_task,
        )

    def occupant(
        self, alias_id: str, task_start_ns: Mapping[str, int], time_ns: int
    ) -> ObjectFacts | None:
        """The object holding ``alias_id`` at ``time_ns``: the last generation
        whose producer has started by then -- a task's output holds its
        storage from the task's start -- else the earliest one."""

        generations = self.generations.get(alias_id)
        if not generations:
            return None
        chosen = generations[0][1]
        for order, object_id in generations:
            if order < 0:
                chosen = object_id
                continue
            producer = self.objects[object_id].producer
            if (
                producer is not None
                and task_start_ns.get(producer, sys.maxsize) <= time_ns
            ):
                chosen = object_id
        return self.objects[chosen]


# --- the clock ----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Clock:
    """When each task runs and each transfer starts and ends, on one clock."""

    task_start_ns: Mapping[str, int]
    task_end_ns: Mapping[str, int]
    #: ``(alias_group_id, kind, trigger_task_id)`` -> the matching transfers'
    #: ``(start_ns, end_ns)`` in issue order.
    transfers: Mapping[tuple[str, str, str], Sequence[tuple[int, int]]]
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
        transfers: dict[tuple[str, str, str], list[tuple[int, int]]] = defaultdict(list)
        times: dict[tuple[str, int], tuple[int, int]] = {}
        for item in sorted(
            simulation["transfer_intervals"], key=lambda item: int(item["sequence"])
        ):
            span = (int(item["start_ns"]), int(item["end_ns"]))
            transfers[
                (item["alias_group_id"], item["kind"], item["trigger_task_id"])
            ].append(span)
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
        transfers: dict[tuple[str, str, str], list[tuple[int, int]]] = defaultdict(list)
        untimed = 0
        for item in sorted(
            simulation["transfer_intervals"], key=lambda item: int(item["sequence"])
        ):
            key = (item["direction"], int(item["sequence"]))
            if key not in lanes:
                untimed += 1
                continue
            transfers[
                (item["alias_group_id"], item["kind"], item["trigger_task_id"])
            ].append(lanes[key])
        makespan = max(
            [end for ends in transfers.values() for _, end in ends]
            + list(task_end.values()),
            default=0,
        )
        return cls(task_start, task_end, transfers, makespan, names, lanes, untimed)


# --- the walks ----------------------------------------------------------------


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
                    facts, clock, alias_id, start_ns, end_ns, facts.alias_size[alias_id]
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
                facts.alias_size[alias_id],
                alias_id,
                None if holder is None else holder.object_id,
                UNATTRIBUTED if holder is None else holder.category,
                trigger,
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


# --- reading a stored plan ----------------------------------------------------


def program_path_for(selection_path: Path, selection: Mapping[str, Any]) -> Path:
    """Where the store keeps the program a selection was made for."""

    digest = str(selection["program_digest"])
    # <store>/v1/planning/results/<xx>/<key>/selection.json -> <store>/v1/planning
    planning = selection_path.resolve().parents[3]
    return planning / "programs" / digest[:2] / digest / "program.json"


@dataclass(frozen=True, slots=True)
class PlanOccupancy:
    """One view of a plan: its pools, its lanes, and the clock they are on."""

    view: str
    spill: Occupancy
    execution: Occupancy | None
    tasks: tuple[TaskSpan, ...]
    transfers: tuple[TransferSpan, ...]
    facts: ProgramFacts
    clock: Clock
    #: What the plan was priced against: the simulation config's fetch and
    #: evict bandwidths (bytes per second) and latencies (ns), by lane.
    assumed: Mapping[str, float] = field(default_factory=dict)
    #: On a traced view, how many lease instants no traced event named, so
    #: they were interpolated between task boundaries.
    interpolated_leases: int = 0

    @property
    def untimed_transfers(self) -> int:
        """Simulated transfers the trace could not time, so they are off the lanes."""

        return self.clock.untimed_transfers

    @property
    def pools(self) -> tuple[Occupancy, ...]:
        """The execution pool, when the plan has a layout, then the spill
        pool: the order the page stacks them."""

        return ((self.execution,) if self.execution is not None else ()) + (self.spill,)

    @property
    def clock_label(self) -> str:
        if self.view == "traced":
            return "device clock (traced step)"
        if self.view == "unconstrained":
            return "unconstrained: every object resident, tasks at their profiled floor"
        if self.view == "all_save":
            return (
                "all save: every alternative at its cheapest, every object resident,"
                " tasks at their profiled floor"
            )
        return "simulated clock"


def attribute(
    selection: Mapping[str, Any],
    program: Mapping[str, Any],
    *,
    diagnostics: Mapping[str, Any] | None = None,
) -> PlanOccupancy:
    """Both pools and the three lanes of a stored plan, on the simulated clock
    or, with a traced step's diagnostics, on the device's."""

    if "simulation_result" not in selection:
        raise ValueError(
            "this selection holds no simulation result: the store names a plan"
            " here without its evidence, so there is nothing to walk"
        )
    facts = ProgramFacts.build(program, selection.get("selections", ()))
    simulation = selection["simulation_result"]
    clock = (
        Clock.simulated(simulation)
        if diagnostics is None
        else Clock.traced(simulation, diagnostics)
    )
    config = selection.get("simulation") or {}
    spill, transfers = spill_occupancy(
        facts,
        selection["schedule"],
        clock,
        reported_peak_bytes=int(simulation["spill_peak_bytes"]),
        capacity_bytes=(
            int(config["spill_capacity_bytes"])
            if "spill_capacity_bytes" in config
            else None
        ),
    )
    execution = None
    placed = None
    layout = selection.get("admission_certificate", {}).get("layout")
    if layout is not None:
        peaks = simulation.get("device_peaks") or ()
        reported = int(peaks[0]["total_bytes"]) if peaks else None
        if diagnostics is not None:
            placed = LeaseClock.build(
                simulation, selection["schedule"], Clock.simulated(simulation), clock
            )
        execution = execution_occupancy(
            facts, layout, clock, placed=placed, reported_peak_bytes=reported
        )
    device = (config.get("devices") or [{}])[0]
    assumed = {
        key: float(device[key])
        for key in (
            "fetch_bandwidth_bytes_per_second",
            "evict_bandwidth_bytes_per_second",
            "fetch_latency_ns",
            "evict_latency_ns",
        )
        if device.get(key) is not None
    }
    return PlanOccupancy(
        "traced" if diagnostics is not None else "simulated",
        spill,
        execution,
        task_spans(facts, clock),
        transfers,
        facts,
        clock,
        assumed,
        0 if placed is None else placed.interpolated,
    )


def cheapest_selections(program: Mapping[str, Any]) -> list[dict[str, str]]:
    """Every alternative group at its cheapest option by the program's
    profiles -- the selection beneath the planner's compute floor -- in the
    form a stored plan records its ``selections``."""

    runtime = {
        profile["profile_id"]: int(profile.get("runtime_ns", 0))
        for profile in program.get("profiles", ())
    }
    task_ns = {
        task["task_id"]: runtime.get(str(task.get("profile_id")), 0)
        for task in program["tasks"]
    }
    chosen: list[dict[str, str]] = []
    for group in program.get("task_alternative_groups", ()):
        cheapest = min(
            group["options"],
            key=lambda option: sum(
                task_ns.get(task_id, 0) for task_id in option["active_task_ids"]
            ),
        )
        chosen.append(
            {"group_id": group["group_id"], "option_id": cheapest["option_id"]}
        )
    return chosen


def unconstrained(
    program: Mapping[str, Any], selections: Iterable[Mapping[str, str]]
) -> PlanOccupancy:
    """A plan's own floor: its alternatives as ``selections`` fixes them,
    with nothing else to plan around. Every task back to back at its
    profiled time, every object resident from its production, or from the
    step's start for checkpoint state, to its last use, and each task's
    profiled workspace while it runs. Nothing spills and the lanes are
    empty; the execution pool's peak is what the plan would need to run
    this way."""

    return _floor(program, list(selections), "unconstrained")


def all_save(program: Mapping[str, Any]) -> PlanOccupancy:
    """The program's floor with every alternative at its cheapest option --
    save, for a save-or-recompute choice -- which is the step the planner
    reports as unconstrained. The same walk as `unconstrained` otherwise."""

    return _floor(program, cheapest_selections(program), "all_save")


def _floor(
    program: Mapping[str, Any], selections: Sequence[Mapping[str, str]], view: str
) -> PlanOccupancy:
    facts = ProgramFacts.build(program, selections)
    tasks = {task["task_id"]: task for task in program["tasks"]}
    profiles = {
        profile["profile_id"]: profile for profile in program.get("profiles", ())
    }
    task_start: dict[str, int] = {}
    task_end: dict[str, int] = {}
    intervals: list[Interval] = []
    now = 0
    for task_id in facts.task_order:
        profile = profiles.get(str(tasks[task_id].get("profile_id")), {})
        task_start[task_id] = now
        now += int(profile.get("runtime_ns", 0))
        task_end[task_id] = now
        scratch = int(profile.get("workspace_bytes", 0))
        if scratch > 0 and now > task_start[task_id]:
            intervals.append(
                Interval(task_start[task_id], now, scratch, None, WORKSPACE, WORKSPACE)
            )
    clock = Clock(task_start, task_end, {}, now, {})
    by_alias: dict[str, list[ObjectFacts]] = defaultdict(list)
    for item in facts.objects.values():
        by_alias[item.alias_group_id].append(item)
    for alias_id, members in by_alias.items():
        size = facts.alias_size.get(alias_id, 0)
        if size == 0:
            continue
        # state the step keeps is resident throughout; a step object holds its
        # slot from its first production to its last use, an output to the end
        checkpoint = any(item.persistence != "step" for item in members)
        starts = sorted(
            {
                task_start[item.producer]
                for item in members
                if item.producer is not None and item.producer in task_start
            }
        )
        opens = 0 if checkpoint or not starts else starts[0]
        uses = [
            task_end[task_id]
            for item in members
            for task_id in item.consumers
            if task_id in task_end
        ] + [
            task_end[item.producer]
            for item in members
            if item.producer is not None and item.producer in task_end
        ]
        closes = (
            now
            if checkpoint or any(item.role == "output" for item in members)
            else max(uses, default=now)
        )
        edges = [opens, *(start for start in starts if opens < start < closes), closes]
        for begin, end in pairwise(edges):
            if end > begin:
                intervals.append(_interval(facts, clock, alias_id, begin, end, size))
    return PlanOccupancy(
        view,
        Occupancy("spill", ()),
        Occupancy("execution", tuple(intervals)),
        task_spans(facts, clock),
        (),
        facts,
        clock,
        {},
    )


# --- the summary --------------------------------------------------------------


def summarize(
    result: PlanOccupancy, *, tokens_per_step: int | None = None
) -> dict[str, float | None]:
    """The step in a few numbers, from the same spans the pages draw.

    Seconds are the step's span on this view's clock; idle is the share of
    it with nothing computing; recompute is the plan's regeneration overhead
    -- what its recompute alternatives cost beyond their save alternatives,
    by the program's profiles, the number the planner reports -- as a share
    of the span; a lane's utilisation is the share of the span it is busy,
    its bytes the total it moved, and its rate those bytes over that time,
    beside the rate the plan was priced at.
    """

    tasks = result.tasks
    end_ns = max(
        [result.clock.makespan_ns]
        + [span.end_ns for span in tasks]
        + [span.end_ns for span in result.transfers]
    )
    start_ns = min(
        [span.start_ns for span in tasks]
        + [span.start_ns for span in result.transfers],
        default=0,
    )
    span_ns = max(end_ns - start_ns, 1)
    compute_ns = sum(span.end_ns - span.start_ns for span in tasks)
    recompute_ns = result.facts.recompute_overhead_ns
    lane_ns = {
        direction: sum(
            span.end_ns - span.start_ns
            for span in result.transfers
            if span.direction == direction
        )
        for direction in ("fetch", "evict")
    }
    lane_bytes = {
        direction: sum(
            span.bytes for span in result.transfers if span.direction == direction
        )
        for direction in ("fetch", "evict")
    }
    pool = result.execution
    return {
        "seconds": span_ns / 1e9,
        "tokens_per_second": (
            None if tokens_per_step is None else tokens_per_step / (span_ns / 1e9)
        ),
        "spill_peak_gib": result.spill.peak()[0] / GIB,
        "execution_peak_gib": None if pool is None else pool.peak()[0] / GIB,
        "idle_percent": 100.0 * max(span_ns - compute_ns, 0) / span_ns,
        "recompute_percent": 100.0 * recompute_ns / span_ns,
        "fetch_utilization_percent": 100.0 * lane_ns["fetch"] / span_ns,
        "evict_utilization_percent": 100.0 * lane_ns["evict"] / span_ns,
        "fetch_gib": lane_bytes["fetch"] / GIB,
        "evict_gib": lane_bytes["evict"] / GIB,
        # the rate this view achieved, bytes over lane time, against the rate
        # the plan was priced at
        "fetch_gbps": (
            None if lane_ns["fetch"] == 0 else lane_bytes["fetch"] / lane_ns["fetch"]
        ),
        "evict_gbps": (
            None if lane_ns["evict"] == 0 else lane_bytes["evict"] / lane_ns["evict"]
        ),
        "assumed_fetch_gbps": _gbps(
            result.assumed.get("fetch_bandwidth_bytes_per_second")
        ),
        "assumed_evict_gbps": _gbps(
            result.assumed.get("evict_bandwidth_bytes_per_second")
        ),
        "assumed_fetch_latency_us": _us(result.assumed.get("fetch_latency_ns")),
        "assumed_evict_latency_us": _us(result.assumed.get("evict_latency_ns")),
    }


def _gbps(bytes_per_second: float | None) -> float | None:
    return None if bytes_per_second is None else bytes_per_second / 1e9


def _us(nanoseconds: float | None) -> float | None:
    return None if nanoseconds is None else nanoseconds / 1e3


# --- outputs ------------------------------------------------------------------


def table(occupancy: Occupancy, by: str, at_ns: Sequence[int] = ()) -> str:
    """One pool at its peak and at the times asked for, by category or role."""

    peak_bytes, peak_ns = occupancy.peak()
    columns = [("peak", peak_ns), *((f"{t / 1e9:.2f} s", t) for t in at_ns)]
    snapshots = [occupancy.at(t, by=by) for _, t in columns]
    names = sorted(
        {name for snapshot in snapshots for name in snapshot},
        key=lambda n: -snapshots[0].get(n, 0),
    )
    own_peaks = occupancy.peaks_by(by)
    width = max([len(n) for n in names] + [12])
    reported = (
        f" (planner reported {occupancy.reported_peak_bytes / GIB:.2f} GiB)"
        if occupancy.reported_peak_bytes is not None
        else ""
    )
    capacity = (
        f", capacity {occupancy.capacity_bytes / GIB:.2f} GiB"
        if occupancy.capacity_bytes is not None
        else ""
    )
    required = (
        f", layout requires {occupancy.required_bytes / GIB:.2f} GiB"
        if occupancy.required_bytes is not None
        else ""
    )
    lines = [
        f"{occupancy.pool} pool: peak {peak_bytes / GIB:.2f} GiB"
        f" at {peak_ns / 1e9:.2f} s{reported}{capacity}{required}",
        "  "
        + f"{by:<{width}}"
        + "".join(f"{label:>16}" for label, _ in columns)
        + f"{'own peak':>16}",
    ]
    for name in names:
        row = f"  {name:<{width}}"
        for snapshot in snapshots:
            value = snapshot.get(name, 0)
            share = f" {100 * value / peak_bytes:3.0f}%" if peak_bytes else "    "
            row += f"{value / GIB:>10.2f} GiB" + share
        own, own_at = own_peaks[name]
        row += f"{own / GIB:>10.2f} GiB {own_at / 1e9:5.1f}s"
        lines.append(row)
    totals = [sum(snapshot.values()) for snapshot in snapshots]
    lines.append(
        "  "
        + f"{'total':<{width}}"
        + "".join(f"{value / GIB:>10.2f} GiB     " for value in totals)
    )
    return "\n".join(lines)


def page_data(
    result: PlanOccupancy,
    *,
    by: str = "category",
    points: int = 1200,
    tokens_per_step: int | None = None,
    plan: str = "",
) -> dict[str, Any]:
    """What a page draws: the step's summary, the pools sampled over the
    step, execution then spill, and the lanes, all on this view's clock;
    ``plan`` is the line under the title that says whose step this is."""

    end_ns = max(
        [result.clock.makespan_ns]
        + [span.end_ns for span in result.tasks]
        + [span.end_ns for span in result.transfers]
    )
    data: dict[str, Any] = {
        "view": result.view,
        "clock": result.clock_label,
        "plan": plan,
        "untimed_transfers": result.untimed_transfers,
        "end_seconds": end_ns / 1e9,
        "summary": summarize(result, tokens_per_step=tokens_per_step),
    }
    pools = []
    for occupancy in result.pools:
        if not occupancy.intervals:
            continue  # the unconstrained view spills nothing
        times, rows = occupancy.series(by=by, points=points, end_ns=end_ns)
        peak_bytes, peak_ns = occupancy.peak()
        note = ""
        if occupancy is result.execution and result.view == "traced":
            note = "leases at the device times of the events that open and close them"
            if result.interpolated_leases:
                note += (
                    f", {result.interpolated_leases} lease instants interpolated"
                    " between task boundaries"
                )
        pools.append(
            {
                "name": occupancy.pool,
                "note": note,
                "times": [round(t, 4) for t in times],
                "rows": {
                    name: [round(v, 4) for v in values] for name, values in rows.items()
                },
                "peak_gib": peak_bytes / GIB,
                "peak_seconds": peak_ns / 1e9,
                "reported_peak_gib": (
                    None
                    if occupancy.reported_peak_bytes is None
                    else occupancy.reported_peak_bytes / GIB
                ),
                "capacity_gib": (
                    None
                    if occupancy.capacity_bytes is None
                    else occupancy.capacity_bytes / GIB
                ),
            }
        )
    data["pools"] = pools
    # In start order on this clock, which on a traced step is not the
    # schedule's order.
    data["compute"] = [
        [
            span.start_ns / 1e9,
            span.end_ns / 1e9,
            span.phase,
            span.name,
            span.overhead_ns / 1e9,
        ]
        for span in sorted(result.tasks, key=lambda span: span.start_ns)
    ]
    for direction in ("fetch", "evict"):
        data[direction] = [
            [
                span.start_ns / 1e9,
                span.end_ns / 1e9,
                span.bytes,
                span.category,
                span.alias_group_id,
            ]
            for span in sorted(result.transfers, key=lambda span: span.start_ns)
            if span.direction == direction
        ]
    return data


def _gib(value: float) -> str:
    return f"{value / GIB:.2f}".rstrip("0").rstrip(".")


def _describe_geometry(label: str) -> str:
    """Describe a named candidate, accumulation and ordering."""
    head, _, ordering = label.rpartition("_")
    candidate, separator, count = head.rpartition("x")
    if not (separator and count.isdigit() and ordering):
        return label
    return f"candidate {unquote(candidate)}, {count} microbatches, ordering {ordering}"


def describe_plan(
    selection: Mapping[str, Any],
    *,
    model: str = "",
    geometry: str = "",
    budget: str = "",
    sequence_length: int | None = None,
    sequences_per_step: int | None = None,
    unconstrained: bool = False,
    resolution: str = "",
    selected: bool = False,
) -> str:
    """The line under a page's title: whose step this is and what it was
    planned within -- the model, the geometry, the tokens a step, the
    execution budget and the pool it resolved to, the spill capacity, the
    resolution when the page is one of several the search kept -- or,
    ``unconstrained``, that nothing constrained it."""

    parts: list[str] = []
    if model:
        parts.append(model)
    if geometry:
        parts.append(_describe_geometry(geometry))
    if sequence_length and sequences_per_step:
        parts.append(
            f"{sequence_length} tokens by {sequences_per_step} sequences"
            f" = {sequence_length * sequences_per_step} tokens a step"
        )
    layout = selection.get("admission_certificate", {}).get("layout", {})
    if "pool_capacity_bytes" in layout:
        pool = f"pool {_gib(int(layout['pool_capacity_bytes']))} GiB"
        parts.append(
            f"execution budget {budget} ({pool})" if budget else f"execution {pool}"
        )
    spill = (selection.get("simulation") or {}).get("spill_capacity_bytes")
    if spill is not None:
        parts.append(f"spill pool {_gib(int(spill))} GiB")
    if resolution:
        parts.append(
            f"resolution: {resolution} of the flexible groups recompute"
            + (", the search's choice" if selected else "")
        )
    if unconstrained:
        # whose alternatives the floor is taken at: a resolution's, a plan's
        # (a selection given), or the program's cheapest
        parts.append(
            "unconstrained: every object resident, nothing spilled, "
            + (
                "alternatives as this resolution fixes them"
                if resolution
                else "alternatives as this plan fixes them"
                if selection.get("selections")
                else "every alternative at its cheapest"
            )
        )
    return " · ".join(parts)


def write_pages(
    views: Sequence[PlanOccupancy],
    directory: Path,
    *,
    title: str,
    by: str = "category",
    tokens_per_step: int | None = None,
    plan: str = "",
    plan_by_view: Mapping[str, str] | None = None,
    related: Sequence[tuple[str, str]] = (),
) -> list[Path]:
    """Write one page per view -- ``simulated.html``, ``traced.html`` -- each
    with the summary, both pools and the lanes on one zoom, and an index
    that links them and carries the tables. ``plan`` names the step under
    every title, and ``plan_by_view`` names it differently for a view;
    ``related`` is ``(label, href)`` links the index adds, to the folders
    around this one. Returns the pages written, the index first."""

    directory.mkdir(parents=True, exist_ok=True)
    template = Path(__file__).with_name("occupancy_page.html").read_text()
    written: list[Path] = []
    links: list[str] = []
    tables: list[str] = []
    for result in views:
        name = f"{result.view}.html"
        named = (plan_by_view or {}).get(result.view, plan)
        payload = json.dumps(
            page_data(result, by=by, tokens_per_step=tokens_per_step, plan=named),
            separators=(",", ":"),
        )
        page = template.replace(
            "__TITLE__", escape(f"{title} · {result.view}")
        ).replace("__OCCUPANCY_DATA__", payload)
        (directory / name).write_text(page)
        written.append(directory / name)
        links.append(f'<li><a href="{name}">{result.clock_label}</a></li>')
        tables.append(
            f"<h2>{escape(result.clock_label)}</h2>"
            + "".join(
                f"<pre>{escape(table(occupancy, by))}</pre>"
                for occupancy in result.pools
                if occupancy.intervals
            )
        )
    index = directory / "index.html"
    around = "".join(
        f'<li><a href="{escape(href)}">{escape(label)}</a></li>'
        for label, href in related
    )
    index.write_text(
        _page(
            f"Occupancy {title}",
            f"<h1>Occupancy over the step: {escape(title)}</h1>"
            f"<p>{escape(plan)}</p><ul>{''.join(links)}</ul>"
            + (f"<p>Around this plan:</p><ul>{around}</ul>" if around else "")
            + "".join(tables),
        )
    )
    return [index, *written]


# --- a whole run --------------------------------------------------------------


def _requested_budget(pool_capacity_bytes: int, budgets: Sequence[int]) -> int | None:
    """The requested execution budget a plan's pool capacity resolved from:
    the smallest budget at or above the capacity, since a slab is a budget
    less the leeway the pool keeps."""

    candidates = [budget for budget in budgets if budget >= pool_capacity_bytes]
    return min(candidates) if candidates else None


def _budget_label(budget_bytes: int) -> str:
    return f"{budget_bytes / GIB:g}gib"


@dataclass(frozen=True, slots=True)
class StoredPlan:
    """One stored selection a run made, with what identifies it to a reader."""

    path: Path
    makespan_ns: int
    pool_capacity_bytes: int
    budget_bytes: int | None
    geometry: str  # "<sequences>x<accumulation>_<ordering>", or "" when unknown
    program_digest: str = ""
    #: A resolution the search kept beside its answer: the share of the
    #: flexible groups it recomputes, and whether the search chose it. Empty
    #: for the chosen plan's own record.
    resolution: str = ""
    selected: bool = True
    #: The store key both records sit under.
    key: str = ""


def stored_plans(run_root: Path) -> list[StoredPlan]:
    """Every selection with evidence under the run's plan store, labelled.

    A stored selection names its program and its capacities but not the
    geometry the search called it; that comes from the search report, whose
    points carry the simulated makespan, which the two share.
    """

    budgets: list[int] = []
    by_makespan: dict[int, set[str]] = defaultdict(set)
    search_path = run_root / "search.json"
    if search_path.exists():
        search = json.loads(search_path.read_text())
        budgets = sorted({int(pair[0]) for pair in search.get("budgets", ())})
        for point in search.get("points", ()):
            if point.get("makespan_seconds") is None:
                continue
            geometry = (
                f"{quote(point['candidate'], safe='')}x{point['accumulation_count']}"
                f"_{point['ordering_label']}"
            )
            by_makespan[round(float(point["makespan_seconds"]) * 1e6)].add(geometry)
    else:
        request_path = run_root / "request.json"
        if request_path.exists():
            request = json.loads(request_path.read_text()).get("request", {})
            budgets = sorted(
                int(value * GIB) for value in request.get("run_budget_gib", ())
            )
    plans: list[StoredPlan] = []
    recorded_store = _request(run_root).get("plan_store")
    store = (
        Path(recorded_store) if recorded_store is not None else run_root / "plan_store"
    )
    results = store / "v1" / "planning" / "results"
    for path in sorted(results.glob("*/*/selection.json")):
        selection = json.loads(path.read_text())
        simulation = selection.get("simulation_result")
        layout = selection.get("admission_certificate", {}).get("layout", {})
        if simulation is None or "pool_capacity_bytes" not in layout:
            continue
        makespan_ns = int(simulation["makespan_ns"])
        labels = by_makespan.get(round(makespan_ns / 1e3), set())
        capacity = int(layout["pool_capacity_bytes"])
        plans.append(
            StoredPlan(
                path,
                makespan_ns,
                capacity,
                _requested_budget(capacity, budgets),
                next(iter(labels)) if len(labels) == 1 else "",
                str(selection.get("program_digest", "")),
                key=path.parent.name,
            )
        )
    # The resolutions a search kept beside its answer, when it was asked to:
    # one record each under the chosen plan's key, certified like it. One whose
    # layout did not fit has no certificate and no page.
    for path in sorted(results.glob("*/*/resolutions/*/selection.json")):
        selection = json.loads(path.read_text())
        simulation = selection.get("simulation_result")
        layout = selection.get("admission_certificate", {}).get("layout", {})
        resolution = selection.get("resolution") or {}
        if simulation is None or "pool_capacity_bytes" not in layout or not resolution:
            continue
        capacity = int(layout["pool_capacity_bytes"])
        plans.append(
            StoredPlan(
                path,
                int(simulation["makespan_ns"]),
                capacity,
                _requested_budget(capacity, budgets),
                "",
                str(selection.get("program_digest", "")),
                resolution=str(resolution.get("recompute_share", "")),
                selected=bool(resolution.get("selected", False)),
                key=path.parents[2].name,
            )
        )
    # A program is one geometry walked one way, so a plan the makespan could
    # not name takes the name of any plan of the same program that it could.
    by_digest: dict[str, set[str]] = defaultdict(set)
    for record in plans:
        if record.geometry:
            by_digest[record.program_digest].add(record.geometry)
    return [
        (
            replace(record, geometry=next(iter(by_digest[record.program_digest])))
            if not record.geometry
            and len(by_digest.get(record.program_digest, ())) == 1
            else record
        )
        for record in plans
    ]


def _request(run_root: Path) -> dict[str, Any]:
    """What the run was asked for, from its ``request.json``, or nothing."""

    request_path = run_root / "request.json"
    if not request_path.exists():
        return {}
    record = json.loads(request_path.read_text()).get("request", {})
    return dict(record) if isinstance(record, Mapping) else {}


def _tokens_per_step(run_root: Path) -> int | None:
    report = run_root / "search.json"
    if report.is_file():
        metadata = json.loads(report.read_text()).get("metadata", {})
        if metadata.get("unit_label") == "tokens":
            return int(metadata["units_per_step"])
    request = _request(run_root)
    try:
        return int(request["sequence_length"]) * int(request["sequences_per_step"])
    except (KeyError, TypeError, ValueError):
        return None


def _shape(request: Mapping[str, Any]) -> tuple[int | None, int | None]:
    try:
        return int(request["sequence_length"]), int(request["sequences_per_step"])
    except (KeyError, TypeError, ValueError):
        return None, None


def _floor_line(
    selection: Mapping[str, Any],
    model: str,
    record: StoredPlan,
    sequence_length: int | None,
    sequences_per_step: int | None,
) -> str:
    """The line under a plan's own floor page: the plan's alternatives with
    nothing to plan around, so the capacities are left off."""

    return describe_plan(
        {"selections": selection.get("selections", ())},
        model=model,
        geometry=record.geometry,
        sequence_length=sequence_length,
        sequences_per_step=sequences_per_step,
        unconstrained=True,
    )


def _share(text: str) -> float:
    """A recompute share as a number, for ordering; an unreadable one last."""

    try:
        return float(Fraction(text))
    except (ValueError, ZeroDivisionError):
        return 2.0


def _row(cells: Sequence[str]) -> str:
    return "<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>"


def _links(relative: Path, views: Sequence[str]) -> str:
    parts = [f'<a href="{relative}/{view}.html">{view}</a>' for view in views]
    parts.append(f'<a href="{relative}/index.html">tables</a>')
    return " · ".join(parts)


#: What one row of the run's summary table carries, in this order.
SUMMARY_COLUMNS = (
    "kind",  # chosen, resolution, all_save
    "geometry",
    "budget",
    "resolution",  # the share of the flexible groups recomputed
    "selected",
    "view",
    "page",
    "step_seconds",
    "tokens_per_second",
    "spill_peak_gib",
    "execution_peak_gib",
    "idle_percent",
    "recompute_percent",
    "fetch_utilization_percent",
    "evict_utilization_percent",
    "fetch_gib",
    "evict_gib",
    "fetch_gbps",
    "evict_gbps",
    "assumed_fetch_gbps",
    "assumed_evict_gbps",
    "untimed_transfers",
    "interpolated_leases",
    "pool_capacity_gib",
    "spill_capacity_gib",
)


@dataclass(slots=True)
class _Entry:
    """One folder of pages in a run's tree: what it is, and its views."""

    kind: str  # chosen, resolution, all_save
    geometry: str
    budget: str
    resolution: str
    selected: bool
    directory: Path
    views: list[PlanOccupancy]
    pool_capacity_bytes: int | None = None
    spill_capacity_bytes: int | None = None

    @property
    def simulated(self) -> PlanOccupancy:
        return self.views[0]

    @property
    def traced(self) -> PlanOccupancy | None:
        return next((view for view in self.views if view.view == "traced"), None)

    def label(self) -> str:
        """The resolution as the indexes name it."""

        return escape(self.resolution) + (
            " (the search's choice)" if self.selected else ""
        )


_STYLE = (
    "<style>body{font:14px/1.4 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;"
    "margin:24px;max-width:1200px}table{border-collapse:collapse;margin:8px 0 20px}"
    "td,th{border-bottom:1px solid #d2d2d7;padding:4px 10px;text-align:left;"
    "font-variant-numeric:tabular-nums;vertical-align:top}th{font-weight:600}"
    "pre{background:#f4f4f6;padding:10px;border-radius:8px;overflow-x:auto;"
    "font-size:12px}nav a{margin-right:14px}</style>"
)


def _page(title: str, body: str) -> str:
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        f"<title>{escape(title)}</title>{_STYLE}</head><body>{body}</body></html>"
    )


def _table(head: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    return (
        "<table><tr>"
        + "".join(f"<th>{cell}</th>" for cell in head)
        + "</tr>"
        + "".join(_row(cells) for cells in rows)
        + "</table>"
    )


def _gib_of(occupancy: Occupancy | None) -> str:
    return "" if occupancy is None else f"{occupancy.peak()[0] / GIB:.2f} GiB"


def _seconds(result: PlanOccupancy | None) -> str:
    return "" if result is None else f"{summarize(result)['seconds']:.3f} s"


def _summary_rows(
    entry: _Entry, out: Path, tokens: int | None
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for view in entry.views:
        summary = summarize(view, tokens_per_step=tokens)
        rows.append(
            {
                "kind": entry.kind,
                "geometry": entry.geometry,
                "budget": entry.budget,
                "resolution": entry.resolution,
                "selected": entry.selected,
                "view": view.view,
                "page": str((entry.directory / f"{view.view}.html").relative_to(out)),
                "step_seconds": summary["seconds"],
                "tokens_per_second": summary["tokens_per_second"],
                "spill_peak_gib": summary["spill_peak_gib"],
                "execution_peak_gib": summary["execution_peak_gib"],
                "idle_percent": summary["idle_percent"],
                "recompute_percent": summary["recompute_percent"],
                "fetch_utilization_percent": summary["fetch_utilization_percent"],
                "evict_utilization_percent": summary["evict_utilization_percent"],
                "fetch_gib": summary["fetch_gib"],
                "evict_gib": summary["evict_gib"],
                "fetch_gbps": summary["fetch_gbps"],
                "evict_gbps": summary["evict_gbps"],
                "assumed_fetch_gbps": summary["assumed_fetch_gbps"],
                "assumed_evict_gbps": summary["assumed_evict_gbps"],
                "untimed_transfers": view.untimed_transfers,
                "interpolated_leases": view.interpolated_leases,
                "pool_capacity_gib": (
                    None
                    if entry.pool_capacity_bytes is None
                    else entry.pool_capacity_bytes / GIB
                ),
                "spill_capacity_gib": (
                    None
                    if entry.spill_capacity_bytes is None
                    else entry.spill_capacity_bytes / GIB
                ),
            }
        )
    return rows


def recompute_share_of(
    program: Mapping[str, Any], selections: Iterable[Mapping[str, str]]
) -> str:
    """The share of the program's flexible groups ``selections`` recomputes,
    by the planner's own rule, as the store names a kept resolution by it;
    empty when the program is not one the planner's IR reads."""

    from shadowspill.ir import ShadowSpillProgram, TaskAlternativeChoice
    from shadowspill.planner.search.toolkit.resolution import recompute_share

    try:
        parsed = ShadowSpillProgram.from_dict(program)
        choices = tuple(
            TaskAlternativeChoice.from_value(item, "selections") for item in selections
        )
    except Exception:
        return ""
    return str(recompute_share(parsed, choices))


def _share_folder(share: str) -> str:
    """``recompute_<share>``, the store's own naming of a kept resolution."""

    numerator, _, denominator = share.partition("/")
    return f"recompute_{numerator}" + (f"of{denominator}" if denominator else "")


def _traced_steps(
    run_root: Path, plans: Sequence[StoredPlan]
) -> tuple[dict[Path, Mapping[str, Any]], list[tuple[str, str]]]:
    """Each traced step's diagnostics, keyed by the plan it ran, found by
    the makespan the step records; and the steps no stored plan matches."""

    matched: dict[Path, Mapping[str, Any]] = {}
    missing: list[tuple[str, str]] = []
    for trace in sorted((run_root / "steps").glob("*gib.json")):
        diagnostics = json.loads(trace.read_text())
        makespan_ns = round(
            float(diagnostics["summary"]["simulator_makespan_seconds"]) * 1e9
        )
        budget_bytes = round(float(trace.stem.removesuffix("gib")) * GIB)
        candidates = [
            record
            for record in plans
            if not record.resolution and abs(record.makespan_ns - makespan_ns) <= 1_000
        ]
        exact = [record for record in candidates if record.budget_bytes == budget_bytes]
        chosen = (exact or candidates)[:1]
        if chosen:
            matched[chosen[0].path] = diagnostics
        else:
            missing.append(
                (
                    trace.stem,
                    "no stored plan has this step's simulated makespan"
                    f" ({makespan_ns / 1e9:.3f} s)",
                )
            )
    return matched, missing


def write_run_timelines(
    run_root: Path,
    out: Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Pages for every plan a run made, budget by budget.

    ``timelines/<budget>/<geometry>/recompute_<share>/`` holds the plan the
    search chose for that geometry at that budget, named by the share of
    the flexible groups it recomputes: ``simulated.html``,
    ``unconstrained.html``, its own floor, and ``traced.html`` on the
    device's clock when this budget ran with this geometry, found by the
    makespan the traced step records; that traced page is copied to
    ``timelines/<budget>/traced.html``, since one geometry runs per budget.
    When the search kept its resolutions, the others sit beside the choice
    under their own shares, each with ``simulated.html`` and its own
    ``unconstrained.html``. ``timelines/all_save/<geometry>/all_save.html``
    is each geometry at its cheapest alternatives, which no budget bounds.
    Each page carries the summary, the pools and the lanes on one zoom.

    The root ``index.html`` is the table of contents for everything, and
    every level below has its own: each budget, each geometry within it,
    each plan's folder, and ``all_save/``. ``summary.csv`` at the root holds
    one row per page, the summary the page's cards show with what
    identifies the plan (`SUMMARY_COLUMNS`). ``progress`` hears one line at
    the start and one per budget finished, since a tour's pages take
    minutes to write. Returns the root index.
    """

    out = out or (run_root / "timelines")
    plans = stored_plans(run_root)
    if progress is not None:
        kept_count = sum(1 for record in plans if record.resolution)
        progress(
            f"timelines: writing pages for {len(plans) - kept_count} plans"
            + (f" and {kept_count} kept resolutions" if kept_count else "")
            + f" under {out}"
        )
    tokens = _tokens_per_step(run_root)
    request = _request(run_root)
    model = str(request.get("model") or "")
    sequence_length, sequences_per_step = _shape(request)
    traced_for, missing = _traced_steps(run_root, plans)
    entries: list[_Entry] = []
    taken: set[Path] = set()
    programs: dict[str, tuple[str, Mapping[str, Any]]] = {}
    kept_by_key: dict[str, list[StoredPlan]] = defaultdict(list)
    for record in plans:
        if record.resolution:
            kept_by_key[record.key].append(record)
    ordered = sorted(
        (record for record in plans if not record.resolution),
        key=lambda item: (item.budget_bytes or 0, item.geometry, item.makespan_ns),
    )

    def naming(
        record: StoredPlan, budget: str, kept: StoredPlan | None = None
    ) -> dict[str, Any]:
        return dict(
            model=model,
            geometry=record.geometry,
            budget=budget,
            sequence_length=sequence_length,
            sequences_per_step=sequences_per_step,
            resolution="" if kept is None else kept.resolution,
            selected=False if kept is None else kept.selected,
        )

    finished: str | None = None
    for record in ordered:
        selection = json.loads(record.path.read_text())
        program = json.loads(program_path_for(record.path, selection).read_text())
        geometry = record.geometry or f"plan_{record.path.parent.name[:8]}"
        budget = (
            _budget_label(record.budget_bytes)
            if record.budget_bytes
            else f"{record.pool_capacity_bytes / GIB:.2f}gib_slab"
        )
        if progress is not None and finished not in (None, budget):
            progress(f"timelines: {finished} written")
        finished = budget
        directory = out / budget / geometry
        if directory in taken:
            directory = out / budget / f"{geometry}_{record.path.parent.name[:8]}"
        taken.add(directory)
        programs.setdefault(record.program_digest, (geometry, program))
        kept_records = sorted(
            kept_by_key.get(record.key, ()), key=lambda item: _share(item.resolution)
        )
        # the choice is one of the kept resolutions when there are any, and
        # its folder is theirs; otherwise the planner's rule names its share
        chosen_record = next((kept for kept in kept_records if kept.selected), None)
        if chosen_record is not None:
            chosen_share, chosen_folder = (
                chosen_record.resolution,
                chosen_record.path.parent.name,
            )
        else:
            chosen_share = recompute_share_of(program, selection.get("selections", ()))
            chosen_folder = _share_folder(chosen_share) if chosen_share else "chosen"
        others = [kept for kept in kept_records if not kept.selected]
        spill_capacity = (selection.get("simulation") or {}).get("spill_capacity_bytes")
        spill_bytes = None if spill_capacity is None else int(spill_capacity)
        views = [attribute(selection, program)]
        diagnostics = traced_for.get(record.path)
        if diagnostics is not None:
            views.append(attribute(selection, program, diagnostics=diagnostics))
        views.append(unconstrained(program, selection.get("selections", ())))
        pages = write_pages(
            views,
            directory / chosen_folder,
            title=f"{geometry} at {budget}, recompute {chosen_share or '?'}",
            tokens_per_step=tokens,
            plan=describe_plan(
                selection,
                **{
                    **naming(record, budget),
                    "resolution": chosen_share,
                    "selected": True,
                },
            ),
            plan_by_view={
                "unconstrained": _floor_line(
                    selection, model, record, sequence_length, sequences_per_step
                )
            },
            related=[
                (
                    f"resolution {kept.resolution}",
                    f"../{kept.path.parent.name}/index.html",
                )
                for kept in others
            ]
            + [
                (
                    f"{geometry} at its cheapest, all save",
                    f"../../../all_save/{geometry}/index.html",
                ),
                (f"all of {budget}", "../../index.html"),
                ("everything", "../../../index.html"),
            ],
        )
        if diagnostics is not None:
            # the budget's traced step, a copy at the budget's own level: the
            # page is self-contained, so the bytes are the same anywhere
            traced_page = next(page for page in pages if page.name == "traced.html")
            (out / budget / "traced.html").write_bytes(traced_page.read_bytes())
        entries.append(
            _Entry(
                "chosen",
                geometry,
                budget,
                chosen_share,
                True,
                directory / chosen_folder,
                views,
                record.pool_capacity_bytes,
                spill_bytes,
            )
        )
        for kept in others:
            folder = directory / kept.path.parent.name
            kept_selection = json.loads(kept.path.read_text())
            kept_views = [
                attribute(kept_selection, program),
                unconstrained(program, kept_selection.get("selections", ())),
            ]
            write_pages(
                kept_views,
                folder,
                title=f"{geometry} at {budget}, recompute {kept.resolution}",
                tokens_per_step=tokens,
                plan=describe_plan(kept_selection, **naming(record, budget, kept)),
                # the resolution's own floor: nothing to plan around, the
                # alternatives fixed as this resolution fixes them
                plan_by_view={
                    "unconstrained": describe_plan(
                        {}, unconstrained=True, **naming(record, "", kept)
                    )
                },
                related=[
                    (
                        f"the search's choice, recompute {chosen_share or '?'}",
                        f"../{chosen_folder}/index.html",
                    ),
                    (f"all of {budget}", "../../index.html"),
                    ("everything", "../../../index.html"),
                ],
            )
            entries.append(
                _Entry(
                    "resolution",
                    geometry,
                    budget,
                    kept.resolution,
                    kept.selected,
                    folder,
                    kept_views,
                    kept.pool_capacity_bytes,
                    spill_bytes,
                )
            )
        _write_plan_index(out, directory, entries)
    if progress is not None and finished is not None:
        progress(f"timelines: {finished} written")

    if progress is not None:
        progress("timelines: the geometries' all-save floors")
    for geometry, program in programs.values():
        result = all_save(program)
        directory = out / "all_save" / geometry
        write_pages(
            [result],
            directory,
            title=f"{geometry} all save",
            tokens_per_step=tokens,
            plan=describe_plan(
                {},
                model=model,
                geometry=geometry,
                sequence_length=sequence_length,
                sequences_per_step=sequences_per_step,
                unconstrained=True,
            ),
            related=[
                ("every geometry's floor", "../index.html"),
                ("everything", "../../index.html"),
            ],
        )
        entries.append(_Entry("all_save", geometry, "", "", False, directory, [result]))

    _write_run_indexes(out, run_root, entries, missing, tokens)
    return out / "index.html"


def _pages_of(entry: _Entry, base: Path) -> str:
    return _links(
        entry.directory.relative_to(base), [view.view for view in entry.views]
    )


def _others_of(entries: Sequence[_Entry], entry: _Entry, base: Path) -> str:
    kept = [
        item
        for item in entries
        if item.kind == "resolution"
        and item.budget == entry.budget
        and item.geometry == entry.geometry
    ]
    return "<br>".join(f"{item.label()}: {_pages_of(item, base)}" for item in kept)


def _geometry_rows(entries: Sequence[_Entry], budget: str, base: Path) -> str:
    """One row per geometry planned at ``budget``: the choice and what was kept."""

    rows = []
    for entry in entries:
        if entry.kind != "chosen" or entry.budget != budget:
            continue
        rows.append(
            [
                escape(entry.geometry),
                escape(entry.resolution or "?"),
                _seconds(entry.simulated),
                _seconds(entry.traced),
                _gib_of(entry.simulated.spill),
                _gib_of(entry.simulated.execution),
                _pages_of(entry, base),
                _others_of(entries, entry, base),
            ]
        )
    return _table(
        [
            "geometry",
            "recompute share chosen",
            "simulated step",
            "traced step",
            "spill peak",
            "execution peak",
            "the choice's pages",
            "other resolutions the search kept",
        ],
        rows,
    )


def _write_plan_index(out: Path, directory: Path, entries: Sequence[_Entry]) -> None:
    """``<budget>/<geometry>/index.html``: the choice and the others kept."""

    here = [entry for entry in entries if entry.directory.parent == directory]
    answer = next(entry for entry in here if entry.kind == "chosen")
    kept = [entry for entry in here if entry.kind == "resolution"]
    rows = [
        [
            answer.label(),
            _seconds(answer.simulated),
            _seconds(answer.traced),
            _gib_of(answer.simulated.spill),
            _gib_of(answer.simulated.execution),
            _pages_of(answer, directory),
        ]
    ] + [
        [
            item.label(),
            _seconds(item.simulated),
            "",
            _gib_of(item.simulated.spill),
            _gib_of(item.simulated.execution),
            _pages_of(item, directory),
        ]
        for item in kept
    ]
    (directory / "index.html").write_text(
        _page(
            f"Timelines: {answer.geometry} at {answer.budget}",
            f"<h1>{escape(answer.geometry)} at {escape(answer.budget)}</h1>"
            f'<p><a href="../index.html">all of {escape(answer.budget)}</a> · '
            f'<a href="../../all_save/{escape(answer.geometry)}/index.html">'
            f"{escape(answer.geometry)} at its cheapest, all save</a> · "
            '<a href="../../index.html">everything</a></p>'
            + _table(
                [
                    "recompute share",
                    "simulated step",
                    "traced step",
                    "spill peak",
                    "execution peak",
                    "pages",
                ],
                rows,
            ),
        )
    )


def _write_run_indexes(
    out: Path,
    run_root: Path,
    entries: Sequence[_Entry],
    missing: Sequence[tuple[str, str]],
    tokens: int | None,
) -> None:
    """The root table of contents, one per budget, ``all_save/``, the summary."""

    out.mkdir(parents=True, exist_ok=True)
    with (out / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for entry in entries:
            for row in _summary_rows(entry, out, tokens):
                writer.writerow(row)
    budgets = sorted(
        {entry.budget for entry in entries if entry.kind == "chosen"},
        key=lambda label: (
            _share(label.removesuffix("gib")) if label.endswith("gib") else 1e9,
            label,
        ),
    )
    floors = [entry for entry in entries if entry.kind == "all_save"]

    def ran_rows(base: Path) -> str:
        rows = [
            [
                escape(entry.budget),
                escape(entry.geometry),
                escape(entry.resolution or "?"),
                _seconds(entry.simulated),
                _seconds(entry.traced),
                _gib_of(entry.traced.spill if entry.traced else None),
                _gib_of(entry.simulated.execution),
                f'<a href="{escape(entry.budget)}/traced.html">traced.html</a>',
                _pages_of(entry, base),
            ]
            for entry in entries
            if entry.kind == "chosen" and entry.traced is not None
        ] + [
            [escape(stem), escape(why), "", "", "", "", "", "", ""]
            for stem, why in missing
        ]
        return _table(
            [
                "budget",
                "geometry that ran",
                "recompute share chosen",
                "simulated step",
                "traced step",
                "spill peak, traced",
                "execution peak",
                "traced page, at the budget",
                "the choice's pages",
            ],
            rows,
        )

    def floor_rows(base: Path) -> str:
        return _table(
            ["geometry", "compute floor, all save", "resident peak", "pages"],
            [
                [
                    escape(entry.geometry),
                    _seconds(entry.simulated),
                    _gib_of(entry.simulated.execution),
                    _pages_of(entry, base),
                ]
                for entry in floors
            ],
        )

    intro = (
        "<p>One page per plan and view: the step's summary, the spill and"
        " execution pools over the step by category, and the fetch, compute and"
        " evict lanes, on one zoom. Under each budget, each geometry's folder"
        " holds the plan the search chose there, under the share of the"
        " flexible groups it recomputes: its simulated page, its own"
        " unconstrained page -- its alternatives with nothing to plan around:"
        " tasks back to back at their profiled floor, every object resident,"
        " nothing spilled -- and, for the geometry that ran at that budget, the"
        " traced page on the device's clock. A search asked to keep its"
        " resolutions has the others beside it under their own shares, each with"
        " its simulated page and its own unconstrained page. Each budget that"
        " ran also carries a copy of its traced page at its own level. Each"
        " geometry also has its all-save page, every alternative at its"
        " cheapest, the floor the planner reports as unconstrained.</p>"
    )
    body = (
        f"<h1>Timelines: {escape(str(run_root))}</h1>"
        + intro
        + '<nav><a href="summary.csv">summary.csv, one row per page</a>'
        '<a href="all_save/index.html">all-save floors</a>'
        + "".join(
            f'<a href="{escape(budget)}/index.html">{escape(budget)}</a>'
            for budget in budgets
        )
        + "</nav><h2>Budgets that ran</h2>"
        + ran_rows(out)
        + "".join(
            f'<h2 id="{escape(budget)}"><a href="{escape(budget)}/index.html">'
            f"{escape(budget)}</a></h2>" + _geometry_rows(entries, budget, out)
            for budget in budgets
        )
        + "<h2>Each geometry at its cheapest alternatives, all save</h2>"
        + floor_rows(out)
    )
    (out / "index.html").write_text(_page(f"Timelines {run_root.name}", body))
    for budget in budgets:
        base = out / budget
        ran = next(
            (
                entry
                for entry in entries
                if entry.kind == "chosen"
                and entry.budget == budget
                and entry.traced is not None
            ),
            None,
        )
        (base / "index.html").write_text(
            _page(
                f"Timelines: {budget}",
                f"<h1>{escape(budget)}</h1>"
                '<p><a href="../index.html">everything</a> · '
                '<a href="../all_save/index.html">all-save floors</a></p>'
                + (
                    f"<p>This budget ran with {escape(ran.geometry)}, recompute"
                    f" {escape(ran.resolution or '?')}: "
                    f'<a href="traced.html">traced.html</a>, its step on the'
                    " device's clock, a copy of the one in that plan's folder.</p>"
                    if ran is not None
                    else ""
                )
                + _geometry_rows(entries, budget, base),
            )
        )
    floor_dir = out / "all_save"
    floor_dir.mkdir(parents=True, exist_ok=True)
    (floor_dir / "index.html").write_text(
        _page(
            "Timelines: all-save floors",
            "<h1>Every geometry at its cheapest alternatives, all save</h1>"
            '<p><a href="../index.html">everything</a></p>' + floor_rows(floor_dir),
        )
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m shadowspill.diagnostics.occupancy",
        description="What occupies each pool over a step, by object category or role,"
        " and the fetch, compute and evict lanes.",
    )
    parser.add_argument(
        "selection",
        type=Path,
        nargs="?",
        help="a plan store's planning/results/<..>/selection.json",
    )
    parser.add_argument(
        "--run",
        type=Path,
        metavar="RUN",
        help="a quickstart run directory (holding search.json, steps/ and"
        " plan_store/): write pages for every plan it made under its timelines/"
        " (or --html), the budgets that ran on both clocks",
    )
    parser.add_argument(
        "--program",
        type=Path,
        help="its program.json; found beside it in the store by default",
    )
    parser.add_argument(
        "--step",
        type=Path,
        help="the traced step's diagnostics JSON: adds the view on the device's"
        " clock, both pools and the lanes",
    )
    parser.add_argument(
        "--unconstrained",
        action="store_true",
        help="also the plan's own floor: its alternatives as it fixed them, with"
        " nothing else to plan around -- tasks back to back at their profiled"
        " floor, every object resident, nothing spilled",
    )
    parser.add_argument(
        "--all-save",
        action="store_true",
        help="also the program at its cheapest alternatives (all save for"
        " save-or-recompute choices), the floor the planner reports as"
        " unconstrained; with --program and no selection, only that view",
    )
    parser.add_argument("--by", choices=("category", "role"), default="category")
    parser.add_argument(
        "--tokens-per-step",
        type=int,
        help="the step's tokens, for the tokens-per-second figure in the summary",
    )
    parser.add_argument(
        "--at",
        default="",
        help="comma-separated seconds into the step to add columns for",
    )
    parser.add_argument(
        "--html",
        type=Path,
        metavar="DIRECTORY",
        help="write the pages there: one per view, with the summary, both pools"
        " and the lanes, and an index",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the snapshots as JSON instead of tables",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.run is not None:
        index = write_run_timelines(arguments.run, arguments.html)
        print(f"timelines: {index}")
        return 0
    if arguments.selection is None and not (arguments.program and arguments.all_save):
        raise SystemExit(
            "give a selection.json, --run with a run directory, or --program"
            " with --all-save"
        )
    selection = (
        {}
        if arguments.selection is None
        else json.loads(arguments.selection.read_text())
    )
    program_path = arguments.program or program_path_for(arguments.selection, selection)
    program = json.loads(program_path.read_text())
    try:
        views = [] if arguments.selection is None else [attribute(selection, program)]
        if arguments.step:
            diagnostics = json.loads(arguments.step.read_text())
            views.append(attribute(selection, program, diagnostics=diagnostics))
        if arguments.unconstrained and arguments.selection is not None:
            views.append(unconstrained(program, selection.get("selections", ())))
        if arguments.all_save:
            views.append(all_save(program))
    except ValueError as error:
        raise SystemExit(str(error)) from error
    at_ns = [
        int(float(value) * 1e9) for value in arguments.at.split(",") if value.strip()
    ]
    if arguments.json:
        payload = {
            result.view: {
                occupancy.pool: {
                    "peak_bytes": occupancy.peak()[0],
                    "peak_ns": occupancy.peak()[1],
                    "reported_peak_bytes": occupancy.reported_peak_bytes,
                    "capacity_bytes": occupancy.capacity_bytes,
                    "at_peak": occupancy.at(occupancy.peak()[1], by=arguments.by),
                    "at": {str(t): occupancy.at(t, by=arguments.by) for t in at_ns},
                    "own_peaks": {
                        name: list(value)
                        for name, value in occupancy.peaks_by(arguments.by).items()
                    },
                }
                for occupancy in result.pools
                if occupancy.intervals
            }
            for result in views
        }
        print(json.dumps(payload, indent=2))
    else:
        first = views[0]
        print(
            f"{arguments.selection or program_path}: {len(first.facts.objects)}"
            f" objects, {len(first.facts.task_order)} executing tasks"
        )
        for result in views:
            print(f"\n== {result.clock_label} ==")
            for occupancy in result.pools:
                if occupancy.intervals:
                    print()
                    print(table(occupancy, arguments.by, at_ns))
    if arguments.html:
        source = arguments.selection or program_path
        pages = write_pages(
            views,
            arguments.html,
            title=source.parent.name[:12],
            by=arguments.by,
            tokens_per_step=arguments.tokens_per_step,
            plan=describe_plan(selection),
            plan_by_view={
                "unconstrained": describe_plan(
                    {"selections": selection.get("selections", ())}, unconstrained=True
                ),
                "all_save": describe_plan({}, unconstrained=True),
            },
        )
        print(f"\npages: {pages[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
