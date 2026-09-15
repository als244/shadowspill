"""Explicit memory residency schedules."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from shadowspill.schema import artifact_schema

from .program import (
    AliasGroupSpec,
    ShadowSpillProgram,
    TaskAlternativeChoice,
    TaskSpec,
)
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

#: Refusal for a schedule that claims residency the runtime, not the plan, owns.
_RUNTIME_OWNED = "shared residency is owned by the runtime, not the schedule"


class MemoryLocation(StrEnum):
    DEVICE = "device"
    # Serialized as "host": stored programs carry that spelling and their
    # digests are taken over it.
    SPILL = "host"


class MemoryActionKind(StrEnum):
    """What a memory action does to an alias group's two copies.

    `FETCH` copies spill to execution. `WRITE_BACK` copies execution to spill
    and keeps the execution copy, so the spill copy is current again and a
    later `RELEASE` costs nothing. `RELEASE` drops the execution copy, which
    requires the spill copy to be current. `EVICT` is the two in one: a
    write-back where the spill copy is stale, then the release.
    """

    RELEASE = "release"
    EVICT = "evict"
    FETCH = "fetch"
    WRITE_BACK = "write_back"


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
        program: ShadowSpillProgram,
        selections: tuple[TaskAlternativeChoice, ...] = (),
    ) -> None:
        self._validate_selected(program, program.selected_tasks(selections))

    def _validate_selected(
        self,
        program: ShadowSpillProgram,
        active_tasks: tuple[TaskSpec, ...],
    ) -> None:
        """Validate against an already-validated recomputation projection."""

        facts = _alias_facts(program, active_tasks, self.final_residency)
        residency = self._initial_residency(facts)
        self._walk_tasks(active_tasks, facts, residency)
        self._final_residency_reached(facts, residency)

    def _initial_residency(self, facts: _AliasFacts) -> _Residency:
        """Read the starting residency, which may not claim shared groups."""

        retained = {
            group_id
            for group_id, group in facts.alias_by_id.items()
            if group.retain_spill_copy
        }
        residency = _Residency(
            device=set(facts.shared_aliases),
            spill=set(retained),
            spill_current=set(retained),
        )
        for index, item in enumerate(self.initial_residency):
            path = f"schedule.initial_residency[{index}].alias_group_id"
            _require_plan_owned(item.alias_group_id, facts, path, _RUNTIME_OWNED)
            if item.alias_group_id in facts.zero_size_aliases:
                continue
            if item.location is MemoryLocation.DEVICE:
                residency.device.add(item.alias_group_id)
            else:
                residency.spill.add(item.alias_group_id)
                residency.spill_current.add(item.alias_group_id)
        return residency

    def _ordered_actions(
        self, facts: _AliasFacts
    ) -> dict[str, list[tuple[int, MemoryAction]]]:
        """Group the actions by trigger task, rejecting any that cannot run."""

        by_task: dict[str, list[tuple[int, MemoryAction]]] = {}
        previous_trigger = -1
        for index, action in enumerate(self.actions):
            path = f"schedule.actions[{index}]"
            require(
                action.trigger_task_id in facts.task_order,
                f"{path}.trigger_task_id",
                f"unknown or inactive task {action.trigger_task_id!r}",
            )
            _require_plan_owned(
                action.alias_group_id,
                facts,
                f"{path}.alias_group_id",
                "shared alias groups cannot have plan-owned memory actions",
            )
            require(
                action.alias_group_id not in facts.zero_size_aliases,
                f"{path}.alias_group_id",
                "zero-size alias groups cannot have physical memory actions",
            )
            trigger = facts.task_order[action.trigger_task_id]
            require(
                trigger >= previous_trigger,
                path,
                "actions must be ordered by trigger task",
            )
            previous_trigger = trigger
            by_task.setdefault(action.trigger_task_id, []).append((index, action))
        return by_task

    def _walk_tasks(
        self,
        active_tasks: tuple[TaskSpec, ...],
        facts: _AliasFacts,
        residency: _Residency,
    ) -> None:
        """Run the schedule forward: every input resident when its task reads it."""

        by_task = self._ordered_actions(facts)
        for task in active_tasks:
            _require_inputs_resident(task, facts, residency)
            _record_task_writes(task, facts, residency)
            for index, action in by_task.get(task.task_id, []):
                _apply_action(
                    action, f"schedule.actions[{index}]", task, facts, residency
                )

    def _final_residency_reached(
        self, facts: _AliasFacts, residency: _Residency
    ) -> None:
        """Every location the schedule promised to end in must have been reached."""

        for index, item in enumerate(self.final_residency):
            path = f"schedule.final_residency[{index}]"
            _require_plan_owned(
                item.alias_group_id, facts, f"{path}.alias_group_id", _RUNTIME_OWNED
            )
            if item.alias_group_id in facts.zero_size_aliases:
                continue
            if item.location is MemoryLocation.DEVICE:
                reached = item.alias_group_id in residency.device
            else:
                reached = (
                    item.alias_group_id in residency.spill
                    and item.alias_group_id in residency.spill_current
                )
            require(
                reached,
                path,
                f"required current {item.location.value} residency was not reached",
            )


@dataclass(frozen=True, slots=True)
class _AliasFacts:
    """What the program says about its alias groups, indexed for the walk."""

    alias_by_id: dict[str, AliasGroupSpec]
    object_alias: dict[str, str]
    shared_aliases: set[str]
    zero_size_aliases: set[str]
    task_order: dict[str, int]
    last_reader: dict[str, int]
    final_aliases: set[str]


@dataclass(slots=True)
class _Residency:
    """Where each alias group's copies are, as the walk reaches each task.

    ``spill`` is every group with a spill copy; ``spill_current`` is the subset
    whose spill copy still matches the execution copy. A release is only free
    when the group is in both.
    """

    device: set[str]
    spill: set[str]
    spill_current: set[str]


def _alias_facts(
    program: ShadowSpillProgram,
    active_tasks: tuple[TaskSpec, ...],
    final_residency: tuple[ResidencySpec, ...],
) -> _AliasFacts:
    """Index the program once, including who reads each alias group last."""

    object_alias = {item.object_id: item.alias_group_id for item in program.objects}
    last_reader: dict[str, int] = {}
    for index, task in enumerate(active_tasks):
        for object_id in (
            *task.inputs,
            *(mutation.object_id for mutation in task.mutations),
        ):
            last_reader[object_alias[object_id]] = index
    return _AliasFacts(
        alias_by_id={group.alias_group_id: group for group in program.alias_groups},
        object_alias=object_alias,
        shared_aliases={
            group.alias_group_id
            for group in program.alias_groups
            if group.shared_residency is not None
        },
        zero_size_aliases={
            group.alias_group_id
            for group in program.alias_groups
            if group.size_bytes == 0
        },
        task_order={task.task_id: index for index, task in enumerate(active_tasks)},
        last_reader=last_reader,
        final_aliases={item.alias_group_id for item in final_residency},
    )


def _require_plan_owned(
    alias_group_id: str, facts: _AliasFacts, path: str, refusal: str
) -> None:
    """The group must exist, and must not be one whose residency the runtime owns."""

    require(
        alias_group_id in facts.alias_by_id,
        path,
        f"unknown alias group {alias_group_id!r}",
    )
    require(alias_group_id not in facts.shared_aliases, path, refusal)


def _require_inputs_resident(
    task: TaskSpec, facts: _AliasFacts, residency: _Residency
) -> None:
    """A task may only read what is on the device when it runs."""

    for object_id in task.inputs:
        alias_id = facts.object_alias[object_id]
        if alias_id in facts.zero_size_aliases:
            continue
        require(
            alias_id in residency.device,
            f"schedule.task[{task.task_id}].inputs",
            f"alias group {alias_id!r} is not device resident",
        )


def _record_task_writes(
    task: TaskSpec, facts: _AliasFacts, residency: _Residency
) -> None:
    """Outputs become device resident; anything written makes its spill copy stale."""

    for object_id in task.outputs:
        alias_id = facts.object_alias[object_id]
        if alias_id in facts.zero_size_aliases:
            continue
        residency.device.add(alias_id)
        residency.spill_current.discard(alias_id)
    for mutation in task.mutations:
        alias_id = facts.object_alias[mutation.object_id]
        if alias_id not in facts.zero_size_aliases:
            residency.spill_current.discard(alias_id)


def _apply_action(
    action: MemoryAction,
    path: str,
    task: TaskSpec,
    facts: _AliasFacts,
    residency: _Residency,
) -> None:
    """Move one alias group's copies, refusing a move its residency cannot make."""

    alias_id = action.alias_group_id
    if action.kind is MemoryActionKind.FETCH:
        require(
            alias_id in residency.spill and alias_id in residency.spill_current,
            path,
            "fetch requires current host residency",
        )
        require(
            alias_id not in residency.device,
            path,
            "fetch requires absent device residency",
        )
        residency.device.add(alias_id)
        if not facts.alias_by_id[alias_id].retain_spill_copy:
            residency.spill.remove(alias_id)
            residency.spill_current.remove(alias_id)
        return

    require(
        alias_id in residency.device,
        path,
        f"{action.kind.value.replace('_', '-')} requires device residency",
    )
    if action.kind is MemoryActionKind.RELEASE:
        require(
            alias_id in residency.spill_current
            or (
                alias_id not in facts.final_aliases
                and facts.last_reader.get(alias_id, -1)
                <= facts.task_order[task.task_id]
            ),
            path,
            "release drops the only current copy of a value still needed",
        )
        residency.device.remove(alias_id)
        if not facts.alias_by_id[alias_id].retain_spill_copy:
            residency.spill.discard(alias_id)
            residency.spill_current.discard(alias_id)
        return

    residency.spill.add(alias_id)
    residency.spill_current.add(alias_id)
    if action.kind is MemoryActionKind.EVICT:
        residency.device.remove(alias_id)


def first_use_initial_order(
    program: ShadowSpillProgram, schedule: MemorySchedule
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
