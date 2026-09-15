"""One ShadowSpillProgram: what a plan is written against.

The vocabularies it speaks are in ``enums``, what it is made of in ``specs``,
the choices it offers in ``alternatives``, and what must be true of it in
``checks``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from dataclasses import field as dataclass_field

from shadowspill.schema import artifact_schema

from ..serialization import (
    JsonValue,
    canonical_json,
    digest_json,
    parse_json,
)
from ..validation import (
    ValidationError,
    expect_list,
    expect_mapping,
    expect_string,
    field,
    require,
    require_tuple,
)
from .alternatives import (
    TaskAlternativeChoice,
    TaskAlternativeGroup,
    TaskAlternativeOption,
)
from .checks import (
    check_alias_groups,
    check_alternative_groups,
    check_objects,
    check_tasks,
    index_identities,
    require_shapes,
)
from .enums import ObjectRole, Persistence, ResourceKind, SharedResidencyPolicy
from .specs import (
    AliasGroupSpec,
    DeviceSpec,
    MutationSpec,
    ObjectSpec,
    ResourceSpec,
    TaskProfile,
    TaskSpec,
)

PROGRAM_SCHEMA = artifact_schema("program")


@dataclass(frozen=True, slots=True)
class ShadowSpillProgram:
    devices: tuple[DeviceSpec, ...]
    alias_groups: tuple[AliasGroupSpec, ...]
    objects: tuple[ObjectSpec, ...]
    profiles: tuple[TaskProfile, ...]
    tasks: tuple[TaskSpec, ...]
    task_alternative_groups: tuple[TaskAlternativeGroup, ...] = ()

    def __post_init__(self) -> None:
        require_shapes(self)
        identities = index_identities(self)
        check_alias_groups(self, identities)
        check_objects(self, identities)
        exclusive_pairs = check_alternative_groups(self, identities)
        check_tasks(self, identities, exclusive_pairs)

    #: The digest, once computed: a program is immutable and its digest is asked
    #: for by every store key, record and check that names it, and computing it
    #: serializes the whole program.
    _digest_cache: list[str] = dataclass_field(
        default_factory=list, init=False, repr=False, compare=False
    )

    @property
    def digest(self) -> str:
        if not self._digest_cache:
            self._digest_cache.append(digest_json(self.to_dict()))
        return self._digest_cache[0]

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "alias_groups": [group.to_dict() for group in self.alias_groups],
            "devices": [device.to_dict() for device in self.devices],
            "objects": [item.to_dict() for item in self.objects],
            "profiles": [profile.to_dict() for profile in self.profiles],
            "task_alternative_groups": [
                group.to_dict() for group in self.task_alternative_groups
            ],
            "schema": PROGRAM_SCHEMA,
            "tasks": [task.to_dict() for task in self.tasks],
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> ShadowSpillProgram:
        data = expect_mapping(value, "program")
        schema = expect_string(field(data, "schema", "program"), "program.schema")
        require(
            schema == PROGRAM_SCHEMA, "program.schema", f"unsupported schema {schema!r}"
        )

        def records(name: str) -> list[object]:
            return expect_list(field(data, name, "program"), f"program.{name}")

        devices = records("devices")
        aliases = records("alias_groups")
        objects = records("objects")
        profiles = records("profiles")
        tasks = records("tasks")
        groups = records("task_alternative_groups")
        return cls(
            devices=tuple(
                DeviceSpec.from_value(item, f"program.devices[{index}]")
                for index, item in enumerate(devices)
            ),
            alias_groups=tuple(
                AliasGroupSpec.from_value(item, f"program.alias_groups[{index}]")
                for index, item in enumerate(aliases)
            ),
            objects=tuple(
                ObjectSpec.from_value(item, f"program.objects[{index}]")
                for index, item in enumerate(objects)
            ),
            profiles=tuple(
                TaskProfile.from_value(item, f"program.profiles[{index}]")
                for index, item in enumerate(profiles)
            ),
            tasks=tuple(
                TaskSpec.from_value(item, f"program.tasks[{index}]")
                for index, item in enumerate(tasks)
            ),
            task_alternative_groups=tuple(
                TaskAlternativeGroup.from_value(
                    item, f"program.task_alternative_groups[{index}]"
                )
                for index, item in enumerate(groups)
            ),
        )

    @classmethod
    def from_json(cls, payload: str) -> ShadowSpillProgram:
        return cls.from_dict(parse_json(payload))

    def selected_tasks(
        self, selections: tuple[TaskAlternativeChoice, ...]
    ) -> tuple[TaskSpec, ...]:
        require_tuple(selections, "selections")
        selection_by_group = {selection.group_id: selection for selection in selections}
        require(
            len(selection_by_group) == len(selections),
            "selections",
            "contains duplicate group IDs",
        )
        expected_groups = {group.group_id for group in self.task_alternative_groups}
        require(
            set(selection_by_group) == expected_groups,
            "selections",
            "must select exactly one option from every task-alternative group",
        )
        variant_tasks: set[str] = set()
        active_variant_tasks: set[str] = set()
        for group in self.task_alternative_groups:
            options = {option.option_id: option for option in group.options}
            selection = selection_by_group[group.group_id]
            require(
                selection.option_id in options,
                "selections",
                f"unknown option {selection.option_id!r} for group {group.group_id!r}",
            )
            for option in group.options:
                variant_tasks.update(option.active_task_ids)
            active_variant_tasks.update(options[selection.option_id].active_task_ids)
        selected_ids = {
            task.task_id
            for task in self.tasks
            if task.task_id not in variant_tasks or task.task_id in active_variant_tasks
        }
        selected = tuple(
            replace(
                task,
                dependencies=tuple(
                    dependency
                    for dependency in task.dependencies
                    if dependency in selected_ids
                ),
            )
            for task in self.tasks
            if task.task_id in selected_ids
        )
        produced_by: dict[str, str] = {}
        for task in selected:
            for output in task.outputs:
                require(
                    output not in produced_by,
                    "selections",
                    f"selected tasks produce object {output!r} more than once",
                )
                produced_by[output] = task.task_id
            for input_id in task.inputs:
                producer = produced_by.get(input_id)
                if producer is not None:
                    require(
                        producer in task.dependencies,
                        "selections",
                        f"task {task.task_id!r} omits active producer {producer!r}",
                    )
        return selected


__all__ = [
    "AliasGroupSpec",
    "DeviceSpec",
    "MutationSpec",
    "ObjectRole",
    "ObjectSpec",
    "Persistence",
    "ResourceKind",
    "ResourceSpec",
    "ShadowSpillProgram",
    "SharedResidencyPolicy",
    "TaskAlternativeChoice",
    "TaskAlternativeGroup",
    "TaskAlternativeOption",
    "TaskProfile",
    "TaskSpec",
    "ValidationError",
    "canonical_index",
]


def canonical_index(value: str, prefix: str) -> int:
    """The dense index a canonical identity such as ``task_000012`` carries."""

    suffix = value.removeprefix(prefix)
    if not value.startswith(prefix) or not suffix.isdigit():
        raise ValueError(f"identity {value!r} is not {prefix}<index>")
    return int(suffix)
