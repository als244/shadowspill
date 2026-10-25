"""Facts for pool-occupancy diagnostics."""

from __future__ import annotations

import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .intervals import ROLE_PRIORITY, ObjectFacts


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
    alias_spill_size: dict[str, int]
    alias_copy_bytes: dict[tuple[str, str], int]
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
    #: ``save`` option's. None if any selected recompute group lacks either
    #: profile; an unavailable baseline must never be priced as zero.
    recompute_overhead_ns: int | None = 0
    #: The same, per task: each group's overhead on its backward task (the
    #: one that re-runs the forward), or its last task when it has none.
    overhead_by_task: dict[str, int | None] = field(default_factory=dict)

    @classmethod
    def build(
        cls, program: Mapping[str, Any], selections: Iterable[Mapping[str, str]] = ()
    ) -> ProgramFacts:
        selections = list(selections)
        active_ids = selected_task_ids(program, selections)
        chosen = {item["group_id"]: item["option_id"] for item in selections}
        tasks = {task["task_id"]: task for task in program["tasks"]}
        runtime = {
            profile["profile_id"]: int(profile["runtime_ns"])
            for profile in program.get("profiles", ())
            if profile.get("runtime_ns") is not None
        }

        def profiled(task_ids: Iterable[str]) -> int | None:
            total = 0
            for task_id in task_ids:
                task = tasks.get(task_id)
                profile_id = None if task is None else task.get("profile_id")
                if profile_id not in runtime:
                    return None
                total += runtime[profile_id]
            return total

        recompute: set[str] = set()
        overhead = 0
        overhead_known = True
        overhead_by_task: dict[str, int | None] = {}
        for group in program.get("task_alternative_groups", ()):
            if chosen.get(group["group_id"]) != "recompute":
                continue
            options = {
                option["option_id"]: option["active_task_ids"]
                for option in group["options"]
            }
            regenerating = list(options.get("recompute", ()))
            recompute.update(regenerating)
            selected_ns = profiled(regenerating)
            save_ns = profiled(options["save"]) if "save" in options else None
            extra = (
                None
                if selected_ns is None or save_ns is None
                else selected_ns - save_ns
            )
            if extra is None:
                overhead_known = False
            else:
                overhead += extra
            carriers = [
                task_id
                for task_id in regenerating
                if str(tasks.get(task_id, {}).get("phase")) == "backward"
            ] or regenerating[-1:]
            if carriers and (extra is None or extra > 0):
                previous = overhead_by_task.get(carriers[-1], 0)
                overhead_by_task[carriers[-1]] = (
                    None if previous is None or extra is None else previous + extra
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
            alias_spill_size={
                group["alias_group_id"]: int(
                    (group.get("transfer") or {}).get(
                        "spill_size_bytes", group["size_bytes"]
                    )
                )
                for group in program["alias_groups"]
            },
            alias_copy_bytes={
                (group["alias_group_id"], direction): (
                    sum(
                        int(phase["bytes"])
                        for phase in group["transfer"][direction]
                        if phase["kind"] == "copy"
                    )
                    if group.get("transfer")
                    else int(group["size_bytes"])
                )
                for group in program["alias_groups"]
                for direction in ("fetch", "evict")
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
            recompute_overhead_ns=max(overhead, 0) if overhead_known else None,
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
