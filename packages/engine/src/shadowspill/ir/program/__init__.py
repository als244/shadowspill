"""One ShadowSpillProgram: what a plan is written against.

The vocabularies it speaks are in ``enums``, what it is made of in ``specs``,
the choices it offers in ``alternatives``, and what must be true of it in
``checks``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from dataclasses import field as dataclass_field

from shadowspill.schema import artifact_schema
from shadowspill.serialization.sharing import read_record, shared_writer

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
from .timing import TaskOverlapCost, TaskResourcePhase
from .transfers import TransferPhase, TransferSpec

PROGRAM_SCHEMA = artifact_schema("program")


@dataclass(frozen=True, slots=True)
class ShadowSpillProgram:
    """One step of work: tasks over objects, with what they cost.

    A program says what the work is, not where it starts, what it runs on or how
    much memory it has. Construction checks every cross-reference, so a program
    that exists is consistent.

    Parameters
    ----------
    devices : tuple of DeviceSpec
        The devices; at least one.
    alias_groups : tuple of AliasGroupSpec
        The storages objects occupy.
    objects : tuple of ObjectSpec
        The values tasks read and write.
    profiles : tuple of TaskProfile
        The measurements tasks refer to.
    tasks : tuple of TaskSpec
        The tasks, in execution order.
    task_alternative_groups : tuple of TaskAlternativeGroup, default ()
        Choices between sets of tasks, one option of each taken by a plan;
        empty for a program with one set of tasks.

    Raises
    ------
    ValidationError
        If an identity is unknown or repeated, an object lies outside its group,
        a dependency does not precede its task, a task does not depend on the
        producer of an input, two writers of one object can be active together,
        or a shared group is used against its policy.
    """

    devices: tuple[DeviceSpec, ...]
    alias_groups: tuple[AliasGroupSpec, ...]
    objects: tuple[ObjectSpec, ...]
    profiles: tuple[TaskProfile, ...]
    tasks: tuple[TaskSpec, ...]
    task_alternative_groups: tuple[TaskAlternativeGroup, ...] = ()

    def __post_init__(self) -> None:
        require_shapes(self)
        require_tuple(self.profiles, "program.profiles")
        identities = index_identities(self, tuple(p.profile_id for p in self.profiles))
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

    #: The most recent selection's tasks, once derived: deriving them rebuilds
    #: every selected task, and planning and execution ask for the same
    #: selection many times over. One entry, so a program never holds more
    #: than one resolved copy of its tasks.
    _selected_cache: list[
        tuple[tuple[TaskAlternativeChoice, ...], tuple[TaskSpec, ...]]
    ] = dataclass_field(default_factory=list, init=False, repr=False, compare=False)

    @property
    def digest(self) -> str:
        """SHA-256 of the program's canonical JSON, in hexadecimal.

        Computed once.
        """
        if not self._digest_cache:
            self._digest_cache.append(digest_json(self.to_dict()))
        return self._digest_cache[0]

    @shared_writer
    def to_dict(self) -> dict[str, JsonValue]:
        """Return the program as JSON values.

        Returns
        -------
        dict
            Its records by kind, and its schema label.
        """
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
        """Return the program's canonical JSON.

        Returns
        -------
        str
            The `to_dict` form with sorted keys and no whitespace.
        """
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> ShadowSpillProgram:
        """Read a program from its `to_dict` form.

        Parameters
        ----------
        value : mapping
            The decoded JSON value.

        Returns
        -------
        ShadowSpillProgram
            The program.

        Raises
        ------
        ValidationError
            If the schema label is not this version's, or a record is malformed or
            the records are inconsistent.
        """
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
                read_record(DeviceSpec.from_value, item, f"program.devices[{index}]")
                for index, item in enumerate(devices)
            ),
            alias_groups=tuple(
                read_record(
                    AliasGroupSpec.from_value, item, f"program.alias_groups[{index}]"
                )
                for index, item in enumerate(aliases)
            ),
            objects=tuple(
                read_record(ObjectSpec.from_value, item, f"program.objects[{index}]")
                for index, item in enumerate(objects)
            ),
            profiles=tuple(
                read_record(TaskProfile.from_value, item, f"program.profiles[{index}]")
                for index, item in enumerate(profiles)
            ),
            tasks=tuple(
                read_record(TaskSpec.from_value, item, f"program.tasks[{index}]")
                for index, item in enumerate(tasks)
            ),
            task_alternative_groups=tuple(
                read_record(
                    TaskAlternativeGroup.from_value,
                    item,
                    f"program.task_alternative_groups[{index}]",
                )
                for index, item in enumerate(groups)
            ),
        )

    @classmethod
    def from_json(cls, payload: str) -> ShadowSpillProgram:
        """Read a program from its JSON text.

        Parameters
        ----------
        payload : str
            The text, as `to_json` writes it.

        Returns
        -------
        ShadowSpillProgram
            The program.

        Raises
        ------
        ValueError
            If ``payload`` is not JSON.
        ValidationError
            If the decoded record is invalid, as for `from_dict`.
        """
        return cls.from_dict(parse_json(payload))

    def selected_tasks(
        self, selections: tuple[TaskAlternativeChoice, ...]
    ) -> tuple[TaskSpec, ...]:
        """Return the tasks that run under one choice per alternative group.

        A task in no group always runs; a task in a group runs when the chosen
        option activates it. Dependencies on tasks that do not run are dropped.

        Parameters
        ----------
        selections : tuple of TaskAlternativeChoice
            One choice for every group of the program, in any order.

        Returns
        -------
        tuple of TaskSpec
            The tasks that run, in program order.

        Raises
        ------
        ValidationError
            If ``selections`` is not a tuple, misses or repeats a group, or names an
            unknown option, or if the tasks it selects produce one object twice or
            leave a task without a dependency on the producer of an input.
        """
        require_tuple(selections, "selections")
        cached = self._selected_cache
        if cached and cached[0][0] == selections:
            return cached[0][1]
        selected = self._select_tasks(selections)
        cached[:] = [(selections, selected)]
        return selected

    def _select_tasks(
        self, selections: tuple[TaskAlternativeChoice, ...]
    ) -> tuple[TaskSpec, ...]:
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
            _without_inactive_dependencies(task, selected_ids)
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


def _without_inactive_dependencies(task: TaskSpec, active: set[str]) -> TaskSpec:
    """The task with its dependencies on inactive tasks dropped: the task
    itself when it has none, which is every task of a program without
    alternatives, since rebuilding one validates it again."""

    dependencies = tuple(
        dependency for dependency in task.dependencies if dependency in active
    )
    if len(dependencies) == len(task.dependencies):
        return task
    return replace(task, dependencies=dependencies)


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
    "TaskOverlapCost",
    "TaskProfile",
    "TaskResourcePhase",
    "TaskSpec",
    "TransferPhase",
    "TransferSpec",
    "ValidationError",
    "canonical_index",
]


def canonical_index(value: str, prefix: str) -> int:
    """The dense index a canonical identity such as ``task_000012`` carries."""

    suffix = value.removeprefix(prefix)
    if not value.startswith(prefix) or not suffix.isdigit():
        raise ValueError(f"identity {value!r} is not {prefix}<index>")
    return int(suffix)
