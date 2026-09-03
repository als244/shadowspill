"""Explicit memory residency schedules."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from shadowspill.schema import artifact_schema

from .program import Program, RecomputationSelection, TaskSpec
from .serialization import JsonValue, canonical_json, digest_json, parse_json
from .validation import (
    expect_list,
    expect_mapping,
    expect_string,
    fail,
    field,
    index_unique,
    require,
    require_identifier,
    require_tuple,
)

SCHEDULE_SCHEMA = artifact_schema("memory_schedule")


class MemoryLocation(StrEnum):
    DEVICE = "device"
    # The serialized value stays "host": the saved-Program corpus is verified
    # against digests taken over it.
    SPILL = "host"


class MemoryActionKind(StrEnum):
    RELEASE = "release"
    EVICT = "evict"
    FETCH = "fetch"


@dataclass(frozen=True, slots=True)
class ResidencySpec:
    alias_group_id: str
    location: MemoryLocation

    def __post_init__(self) -> None:
        require_identifier(self.alias_group_id, "residency.alias_group_id")
        require(
            isinstance(self.location, MemoryLocation),
            "residency.location",
            "invalid location",
        )

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "alias_group_id": self.alias_group_id,
            "location": self.location.value,
        }

    @classmethod
    def from_value(cls, value: object, path: str) -> ResidencySpec:
        data = expect_mapping(value, path)
        location_value = expect_string(
            field(data, "location", path), f"{path}.location"
        )
        try:
            location = MemoryLocation(location_value)
        except ValueError:
            fail(f"{path}.location", f"unknown location {location_value!r}")
        return cls(
            alias_group_id=expect_string(
                field(data, "alias_group_id", path), f"{path}.alias_group_id"
            ),
            location=location,
        )


@dataclass(frozen=True, slots=True)
class MemoryAction:
    trigger_task_id: str
    alias_group_id: str
    kind: MemoryActionKind

    def __post_init__(self) -> None:
        require_identifier(self.trigger_task_id, "action.trigger_task_id")
        require_identifier(self.alias_group_id, "action.alias_group_id")
        require(isinstance(self.kind, MemoryActionKind), "action.kind", "invalid kind")

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "alias_group_id": self.alias_group_id,
            "kind": self.kind.value,
            "trigger_task_id": self.trigger_task_id,
        }

    @classmethod
    def from_value(cls, value: object, path: str) -> MemoryAction:
        data = expect_mapping(value, path)
        kind_value = expect_string(field(data, "kind", path), f"{path}.kind")
        try:
            kind = MemoryActionKind(kind_value)
        except ValueError:
            fail(f"{path}.kind", f"unknown memory action {kind_value!r}")
        return cls(
            trigger_task_id=expect_string(
                field(data, "trigger_task_id", path), f"{path}.trigger_task_id"
            ),
            alias_group_id=expect_string(
                field(data, "alias_group_id", path), f"{path}.alias_group_id"
            ),
            kind=kind,
        )


@dataclass(frozen=True, slots=True)
class MemorySchedule:
    initial_residency: tuple[ResidencySpec, ...]
    actions: tuple[MemoryAction, ...]
    final_residency: tuple[ResidencySpec, ...] = ()

    def __post_init__(self) -> None:
        require_tuple(self.initial_residency, "schedule.initial_residency")
        require_tuple(self.actions, "schedule.actions")
        require_tuple(self.final_residency, "schedule.final_residency")
        index_unique(
            (item.alias_group_id for item in self.initial_residency),
            "schedule.initial_residency",
        )
        index_unique(
            (item.alias_group_id for item in self.final_residency),
            "schedule.final_residency",
        )

    @property
    def digest(self) -> str:
        return digest_json(self.to_dict())

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "actions": [action.to_dict() for action in self.actions],
            "final_residency": [item.to_dict() for item in self.final_residency],
            "initial_residency": [item.to_dict() for item in self.initial_residency],
            "schema": SCHEDULE_SCHEMA,
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> MemorySchedule:
        data = expect_mapping(value, "schedule")
        schema = expect_string(field(data, "schema", "schedule"), "schedule.schema")
        require(
            schema == SCHEDULE_SCHEMA,
            "schedule.schema",
            f"unsupported schema {schema!r}",
        )
        initial = expect_list(
            field(data, "initial_residency", "schedule"),
            "schedule.initial_residency",
        )
        actions = expect_list(field(data, "actions", "schedule"), "schedule.actions")
        final = expect_list(
            field(data, "final_residency", "schedule"),
            "schedule.final_residency",
        )
        return cls(
            initial_residency=tuple(
                ResidencySpec.from_value(item, f"schedule.initial_residency[{index}]")
                for index, item in enumerate(initial)
            ),
            actions=tuple(
                MemoryAction.from_value(item, f"schedule.actions[{index}]")
                for index, item in enumerate(actions)
            ),
            final_residency=tuple(
                ResidencySpec.from_value(item, f"schedule.final_residency[{index}]")
                for index, item in enumerate(final)
            ),
        )

    @classmethod
    def from_json(cls, payload: str) -> MemorySchedule:
        return cls.from_dict(parse_json(payload))

    def validate(
        self,
        program: Program,
        selections: tuple[RecomputationSelection, ...] = (),
    ) -> None:
        self._validate_selected(program, program.selected_tasks(selections))

    def _validate_selected(
        self,
        program: Program,
        active_tasks: tuple[TaskSpec, ...],
    ) -> None:
        """Validate against an already-validated recomputation projection."""

        task_order = {task.task_id: index for index, task in enumerate(active_tasks)}
        alias_ids = {group.alias_group_id for group in program.alias_groups}
        object_alias = {item.object_id: item.alias_group_id for item in program.objects}

        alias_by_id = {group.alias_group_id: group for group in program.alias_groups}
        shared_aliases = {
            group.alias_group_id
            for group in program.alias_groups
            if group.shared_residency is not None
        }
        zero_size_aliases = {
            group.alias_group_id
            for group in program.alias_groups
            if group.size_bytes == 0
        }
        device_resident: set[str] = set(shared_aliases)
        spill_resident = {
            group.alias_group_id
            for group in program.alias_groups
            if group.retain_spill_copy
        }
        spill_current = set(spill_resident)
        for index, residency in enumerate(self.initial_residency):
            require(
                residency.alias_group_id in alias_ids,
                f"schedule.initial_residency[{index}].alias_group_id",
                f"unknown alias group {residency.alias_group_id!r}",
            )
            require(
                residency.alias_group_id not in shared_aliases,
                f"schedule.initial_residency[{index}].alias_group_id",
                "shared residency is owned by the runtime, not the schedule",
            )
            if residency.alias_group_id in zero_size_aliases:
                continue
            if residency.location is MemoryLocation.DEVICE:
                device_resident.add(residency.alias_group_id)
            else:
                spill_resident.add(residency.alias_group_id)
                spill_current.add(residency.alias_group_id)

        actions_by_task: dict[str, list[tuple[int, MemoryAction]]] = {}
        previous_trigger = -1
        for index, action in enumerate(self.actions):
            path = f"schedule.actions[{index}]"
            require(
                action.trigger_task_id in task_order,
                f"{path}.trigger_task_id",
                f"unknown or inactive task {action.trigger_task_id!r}",
            )
            require(
                action.alias_group_id in alias_ids,
                f"{path}.alias_group_id",
                f"unknown alias group {action.alias_group_id!r}",
            )
            require(
                action.alias_group_id not in zero_size_aliases,
                f"{path}.alias_group_id",
                "zero-size alias groups cannot have physical memory actions",
            )
            require(
                action.alias_group_id not in shared_aliases,
                f"{path}.alias_group_id",
                "shared alias groups cannot have plan-owned memory actions",
            )
            trigger = task_order[action.trigger_task_id]
            require(
                trigger >= previous_trigger,
                path,
                "actions must be ordered by trigger task",
            )
            previous_trigger = trigger
            actions_by_task.setdefault(action.trigger_task_id, []).append(
                (index, action)
            )

        for task in active_tasks:
            for object_id in task.inputs:
                alias_id = object_alias[object_id]
                if alias_id in zero_size_aliases:
                    continue
                require(
                    alias_id in device_resident,
                    f"schedule.task[{task.task_id}].inputs",
                    f"alias group {alias_id!r} is not device resident",
                )
            for object_id in task.outputs:
                alias_id = object_alias[object_id]
                if alias_id in zero_size_aliases:
                    continue
                device_resident.add(alias_id)
                spill_current.discard(alias_id)
            for mutation in task.mutations:
                alias_id = object_alias[mutation.object_id]
                if alias_id not in zero_size_aliases:
                    spill_current.discard(alias_id)
            for action_index, action in actions_by_task.get(task.task_id, []):
                path = f"schedule.actions[{action_index}]"
                alias_id = action.alias_group_id
                if action.kind is MemoryActionKind.RELEASE:
                    require(
                        alias_id in device_resident,
                        path,
                        "release requires device residency",
                    )
                    device_resident.remove(alias_id)
                    if not alias_by_id[alias_id].retain_spill_copy:
                        spill_resident.discard(alias_id)
                        spill_current.discard(alias_id)
                elif action.kind is MemoryActionKind.EVICT:
                    require(
                        alias_id in device_resident,
                        path,
                        "evict requires device residency",
                    )
                    spill_resident.add(alias_id)
                    spill_current.add(alias_id)
                    device_resident.remove(alias_id)
                else:
                    require(
                        alias_id in spill_resident and alias_id in spill_current,
                        path,
                        "fetch requires current host residency",
                    )
                    require(
                        alias_id not in device_resident,
                        path,
                        "fetch requires absent device residency",
                    )
                    device_resident.add(alias_id)
                    if not alias_by_id[alias_id].retain_spill_copy:
                        spill_resident.remove(alias_id)
                        spill_current.remove(alias_id)

        for index, residency in enumerate(self.final_residency):
            require(
                residency.alias_group_id in alias_ids,
                f"schedule.final_residency[{index}].alias_group_id",
                f"unknown alias group {residency.alias_group_id!r}",
            )
            require(
                residency.alias_group_id not in shared_aliases,
                f"schedule.final_residency[{index}].alias_group_id",
                "shared residency is owned by the runtime, not the schedule",
            )
            if residency.alias_group_id in zero_size_aliases:
                continue
            if residency.location is MemoryLocation.DEVICE:
                reached = residency.alias_group_id in device_resident
            else:
                reached = (
                    residency.alias_group_id in spill_resident
                    and residency.alias_group_id in spill_current
                )
            require(
                reached,
                f"schedule.final_residency[{index}]",
                f"required current {residency.location.value} residency "
                "was not reached",
            )


def first_use_initial_order(
    program: Program, schedule: MemorySchedule
) -> tuple[str, ...]:
    """The schedule's initial device aliases, ordered by first consuming task.

    The runtime realizes initial residency as one FIFO transfer batch, so
    the batch's order decides how long the earliest tasks wait for their
    inputs. The schedule emits the set in alias order, which strands a
    first task's input arbitrarily deep in the queue; ordering by the
    program's task sequence lets every task's inputs arrive no later than
    the work ahead of them requires. Aliases first consumed by the same
    task follow that task's own input order; aliases no task consumes keep
    their emitted relative order after all consumed ones.
    """

    emitted = tuple(
        item.alias_group_id
        for item in schedule.initial_residency
        if item.location is MemoryLocation.DEVICE
    )
    wanted = set(emitted)
    alias_of = {
        item.object_id: item.alias_group_id
        for item in program.objects
        if item.alias_group_id in wanted
    }
    rank: dict[str, int] = {}
    for task in program.tasks:
        consumed = tuple(task.inputs) + tuple(
            mutation.object_id for mutation in task.mutations
        )
        for object_id in consumed:
            alias = alias_of.get(object_id)
            if alias is not None and alias not in rank:
                rank[alias] = len(rank)
    unused = len(rank)
    position = {alias: index for index, alias in enumerate(emitted)}
    return tuple(
        sorted(emitted, key=lambda alias: (rank.get(alias, unused), position[alias]))
    )


__all__ = [
    "MemoryAction",
    "MemoryActionKind",
    "MemoryLocation",
    "MemorySchedule",
    "ResidencySpec",
    "first_use_initial_order",
]
