"""Immutable logical program records."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum

from ._serialization import (
    JsonValue,
    canonical_json,
    digest_json,
    parse_json,
    string_list,
)
from ._validation import (
    ValidationError,
    expect_boolean,
    expect_integer,
    expect_list,
    expect_mapping,
    expect_string,
    fail,
    field,
    index_unique,
    require,
    require_identifier,
    require_non_negative,
    require_positive,
    require_tuple,
)

PROGRAM_SCHEMA = "shadowspill.program/v1"


class ObjectRole(StrEnum):
    INPUT = "input"
    PARAMETER = "parameter"
    BUFFER = "buffer"
    ACTIVATION = "activation"
    GRADIENT = "gradient"
    OPTIMIZER_STATE = "optimizer_state"
    OUTPUT = "output"
    OTHER = "other"


class Persistence(StrEnum):
    STEP = "step"
    RUN = "run"
    CHECKPOINT = "checkpoint"


class ResourceKind(StrEnum):
    COMPUTE = "compute"
    COMMUNICATION = "communication"
    CONTROL = "control"


@dataclass(frozen=True, slots=True)
class DeviceSpec:
    device_id: str
    process_id: str
    kind: str
    index: int

    def __post_init__(self) -> None:
        require_identifier(self.device_id, "device.device_id")
        require_identifier(self.process_id, "device.process_id")
        require_identifier(self.kind, "device.kind")
        require_non_negative(self.index, "device.index")

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "device_id": self.device_id,
            "index": self.index,
            "kind": self.kind,
            "process_id": self.process_id,
        }

    @classmethod
    def from_value(cls, value: object, path: str) -> DeviceSpec:
        data = expect_mapping(value, path)
        return cls(
            device_id=expect_string(
                field(data, "device_id", path), f"{path}.device_id"
            ),
            process_id=expect_string(
                field(data, "process_id", path), f"{path}.process_id"
            ),
            kind=expect_string(field(data, "kind", path), f"{path}.kind"),
            index=expect_integer(field(data, "index", path), f"{path}.index"),
        )


@dataclass(frozen=True, slots=True)
class ResourceSpec:
    device_id: str
    kind: ResourceKind
    lane: int = 0

    def __post_init__(self) -> None:
        require_identifier(self.device_id, "resource.device_id")
        require(isinstance(self.kind, ResourceKind), "resource.kind", "invalid kind")
        require_non_negative(self.lane, "resource.lane")

    def to_dict(self) -> dict[str, JsonValue]:
        return {"device_id": self.device_id, "kind": self.kind.value, "lane": self.lane}

    @classmethod
    def from_value(cls, value: object, path: str) -> ResourceSpec:
        data = expect_mapping(value, path)
        kind_value = expect_string(field(data, "kind", path), f"{path}.kind")
        try:
            kind = ResourceKind(kind_value)
        except ValueError:
            fail(f"{path}.kind", f"unknown resource kind {kind_value!r}")
        return cls(
            device_id=expect_string(
                field(data, "device_id", path), f"{path}.device_id"
            ),
            kind=kind,
            lane=expect_integer(field(data, "lane", path), f"{path}.lane"),
        )


@dataclass(frozen=True, slots=True)
class AliasGroupSpec:
    alias_group_id: str
    device_id: str
    size_bytes: int
    initial_version: int = 0
    retain_host_backing: bool = False

    def __post_init__(self) -> None:
        require_identifier(self.alias_group_id, "alias_group.alias_group_id")
        require_identifier(self.device_id, "alias_group.device_id")
        require_non_negative(self.size_bytes, "alias_group.size_bytes")
        require_non_negative(self.initial_version, "alias_group.initial_version")
        require(
            isinstance(self.retain_host_backing, bool),
            "alias_group.retain_host_backing",
            "must be a boolean",
        )

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "alias_group_id": self.alias_group_id,
            "device_id": self.device_id,
            "initial_version": self.initial_version,
            "retain_host_backing": self.retain_host_backing,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_value(cls, value: object, path: str) -> AliasGroupSpec:
        data = expect_mapping(value, path)
        return cls(
            alias_group_id=expect_string(
                field(data, "alias_group_id", path), f"{path}.alias_group_id"
            ),
            device_id=expect_string(
                field(data, "device_id", path), f"{path}.device_id"
            ),
            size_bytes=expect_integer(
                field(data, "size_bytes", path), f"{path}.size_bytes"
            ),
            initial_version=expect_integer(
                field(data, "initial_version", path), f"{path}.initial_version"
            ),
            retain_host_backing=expect_boolean(
                field(data, "retain_host_backing", path),
                f"{path}.retain_host_backing",
            ),
        )


@dataclass(frozen=True, slots=True)
class ObjectSpec:
    object_id: str
    alias_group_id: str
    offset_bytes: int
    size_bytes: int
    role: ObjectRole = ObjectRole.OTHER
    persistence: Persistence = Persistence.STEP

    def __post_init__(self) -> None:
        require_identifier(self.object_id, "object.object_id")
        require_identifier(self.alias_group_id, "object.alias_group_id")
        require_non_negative(self.offset_bytes, "object.offset_bytes")
        require_non_negative(self.size_bytes, "object.size_bytes")
        require(isinstance(self.role, ObjectRole), "object.role", "invalid role")
        require(
            isinstance(self.persistence, Persistence),
            "object.persistence",
            "invalid persistence",
        )

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "alias_group_id": self.alias_group_id,
            "object_id": self.object_id,
            "offset_bytes": self.offset_bytes,
            "persistence": self.persistence.value,
            "role": self.role.value,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_value(cls, value: object, path: str) -> ObjectSpec:
        data = expect_mapping(value, path)
        role_value = expect_string(field(data, "role", path), f"{path}.role")
        persistence_value = expect_string(
            field(data, "persistence", path), f"{path}.persistence"
        )
        try:
            role = ObjectRole(role_value)
        except ValueError:
            fail(f"{path}.role", f"unknown object role {role_value!r}")
        try:
            persistence = Persistence(persistence_value)
        except ValueError:
            fail(
                f"{path}.persistence",
                f"unknown persistence {persistence_value!r}",
            )
        return cls(
            object_id=expect_string(
                field(data, "object_id", path), f"{path}.object_id"
            ),
            alias_group_id=expect_string(
                field(data, "alias_group_id", path), f"{path}.alias_group_id"
            ),
            offset_bytes=expect_integer(
                field(data, "offset_bytes", path), f"{path}.offset_bytes"
            ),
            size_bytes=expect_integer(
                field(data, "size_bytes", path), f"{path}.size_bytes"
            ),
            role=role,
            persistence=persistence,
        )


@dataclass(frozen=True, slots=True)
class TaskProfile:
    profile_id: str
    runtime_ns: int
    workspace_bytes: int
    compatibility_digest: str

    def __post_init__(self) -> None:
        require_identifier(self.profile_id, "profile.profile_id")
        require_non_negative(self.runtime_ns, "profile.runtime_ns")
        require_non_negative(self.workspace_bytes, "profile.workspace_bytes")
        require_identifier(self.compatibility_digest, "profile.compatibility_digest")

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "compatibility_digest": self.compatibility_digest,
            "profile_id": self.profile_id,
            "runtime_ns": self.runtime_ns,
            "workspace_bytes": self.workspace_bytes,
        }

    @classmethod
    def from_value(cls, value: object, path: str) -> TaskProfile:
        data = expect_mapping(value, path)
        return cls(
            profile_id=expect_string(
                field(data, "profile_id", path), f"{path}.profile_id"
            ),
            runtime_ns=expect_integer(
                field(data, "runtime_ns", path), f"{path}.runtime_ns"
            ),
            workspace_bytes=expect_integer(
                field(data, "workspace_bytes", path), f"{path}.workspace_bytes"
            ),
            compatibility_digest=expect_string(
                field(data, "compatibility_digest", path),
                f"{path}.compatibility_digest",
            ),
        )


@dataclass(frozen=True, slots=True)
class MutationSpec:
    object_id: str
    version_delta: int = 1

    def __post_init__(self) -> None:
        require_identifier(self.object_id, "mutation.object_id")
        require_positive(self.version_delta, "mutation.version_delta")

    def to_dict(self) -> dict[str, JsonValue]:
        return {"object_id": self.object_id, "version_delta": self.version_delta}

    @classmethod
    def from_value(cls, value: object, path: str) -> MutationSpec:
        data = expect_mapping(value, path)
        return cls(
            object_id=expect_string(
                field(data, "object_id", path), f"{path}.object_id"
            ),
            version_delta=expect_integer(
                field(data, "version_delta", path), f"{path}.version_delta"
            ),
        )


@dataclass(frozen=True, slots=True)
class TaskSpec:
    task_id: str
    resource: ResourceSpec
    profile_id: str
    dependencies: tuple[str, ...] = ()
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    mutations: tuple[MutationSpec, ...] = ()
    phase: str = "compute"
    requires_entrypoint: bool = True

    def __post_init__(self) -> None:
        require_identifier(self.task_id, "task.task_id")
        require(
            isinstance(self.resource, ResourceSpec), "task.resource", "invalid resource"
        )
        require_identifier(self.profile_id, "task.profile_id")
        require_tuple(self.dependencies, "task.dependencies")
        require_tuple(self.inputs, "task.inputs")
        require_tuple(self.outputs, "task.outputs")
        require_tuple(self.mutations, "task.mutations")
        index_unique(self.dependencies, "task.dependencies")
        index_unique(self.inputs, "task.inputs")
        index_unique(self.outputs, "task.outputs")
        index_unique(
            (mutation.object_id for mutation in self.mutations),
            "task.mutations",
        )
        require_identifier(self.phase, "task.phase")
        require(
            isinstance(self.requires_entrypoint, bool),
            "task.requires_entrypoint",
            "must be a boolean",
        )

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "dependencies": string_list(self.dependencies),
            "inputs": string_list(self.inputs),
            "mutations": [mutation.to_dict() for mutation in self.mutations],
            "outputs": string_list(self.outputs),
            "phase": self.phase,
            "profile_id": self.profile_id,
            "requires_entrypoint": self.requires_entrypoint,
            "resource": self.resource.to_dict(),
            "task_id": self.task_id,
        }

    @classmethod
    def from_value(cls, value: object, path: str) -> TaskSpec:
        data = expect_mapping(value, path)
        dependencies = expect_list(
            field(data, "dependencies", path), f"{path}.dependencies"
        )
        inputs = expect_list(field(data, "inputs", path), f"{path}.inputs")
        outputs = expect_list(field(data, "outputs", path), f"{path}.outputs")
        mutations = expect_list(field(data, "mutations", path), f"{path}.mutations")
        return cls(
            task_id=expect_string(field(data, "task_id", path), f"{path}.task_id"),
            resource=ResourceSpec.from_value(
                field(data, "resource", path), f"{path}.resource"
            ),
            profile_id=expect_string(
                field(data, "profile_id", path), f"{path}.profile_id"
            ),
            dependencies=tuple(
                expect_string(item, f"{path}.dependencies[{index}]")
                for index, item in enumerate(dependencies)
            ),
            inputs=tuple(
                expect_string(item, f"{path}.inputs[{index}]")
                for index, item in enumerate(inputs)
            ),
            outputs=tuple(
                expect_string(item, f"{path}.outputs[{index}]")
                for index, item in enumerate(outputs)
            ),
            mutations=tuple(
                MutationSpec.from_value(item, f"{path}.mutations[{index}]")
                for index, item in enumerate(mutations)
            ),
            phase=expect_string(field(data, "phase", path), f"{path}.phase"),
            requires_entrypoint=expect_boolean(
                field(data, "requires_entrypoint", path),
                f"{path}.requires_entrypoint",
            ),
        )


@dataclass(frozen=True, slots=True)
class RecomputationOption:
    option_id: str
    active_task_ids: tuple[str, ...]
    retained_alias_group_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        require_identifier(self.option_id, "recomputation_option.option_id")
        require_tuple(self.active_task_ids, "recomputation_option.active_task_ids")
        require_tuple(
            self.retained_alias_group_ids,
            "recomputation_option.retained_alias_group_ids",
        )
        index_unique(self.active_task_ids, "recomputation_option.active_task_ids")
        index_unique(
            self.retained_alias_group_ids,
            "recomputation_option.retained_alias_group_ids",
        )

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "active_task_ids": string_list(self.active_task_ids),
            "option_id": self.option_id,
            "retained_alias_group_ids": string_list(self.retained_alias_group_ids),
        }

    @classmethod
    def from_value(cls, value: object, path: str) -> RecomputationOption:
        data = expect_mapping(value, path)
        active = expect_list(
            field(data, "active_task_ids", path), f"{path}.active_task_ids"
        )
        retained = expect_list(
            field(data, "retained_alias_group_ids", path),
            f"{path}.retained_alias_group_ids",
        )
        return cls(
            option_id=expect_string(
                field(data, "option_id", path), f"{path}.option_id"
            ),
            active_task_ids=tuple(
                expect_string(item, f"{path}.active_task_ids[{index}]")
                for index, item in enumerate(active)
            ),
            retained_alias_group_ids=tuple(
                expect_string(item, f"{path}.retained_alias_group_ids[{index}]")
                for index, item in enumerate(retained)
            ),
        )


@dataclass(frozen=True, slots=True)
class RecomputationGroup:
    group_id: str
    options: tuple[RecomputationOption, ...]

    def __post_init__(self) -> None:
        require_identifier(self.group_id, "recomputation_group.group_id")
        require_tuple(self.options, "recomputation_group.options")
        require(bool(self.options), "recomputation_group.options", "must not be empty")
        index_unique(
            (option.option_id for option in self.options),
            "recomputation_group.options",
        )

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "group_id": self.group_id,
            "options": [option.to_dict() for option in self.options],
        }

    @classmethod
    def from_value(cls, value: object, path: str) -> RecomputationGroup:
        data = expect_mapping(value, path)
        options = expect_list(field(data, "options", path), f"{path}.options")
        return cls(
            group_id=expect_string(field(data, "group_id", path), f"{path}.group_id"),
            options=tuple(
                RecomputationOption.from_value(item, f"{path}.options[{index}]")
                for index, item in enumerate(options)
            ),
        )


@dataclass(frozen=True, slots=True)
class RecomputationSelection:
    group_id: str
    option_id: str

    def __post_init__(self) -> None:
        require_identifier(self.group_id, "selection.group_id")
        require_identifier(self.option_id, "selection.option_id")

    def to_dict(self) -> dict[str, JsonValue]:
        return {"group_id": self.group_id, "option_id": self.option_id}

    @classmethod
    def from_value(cls, value: object, path: str) -> RecomputationSelection:
        data = expect_mapping(value, path)
        return cls(
            group_id=expect_string(field(data, "group_id", path), f"{path}.group_id"),
            option_id=expect_string(
                field(data, "option_id", path), f"{path}.option_id"
            ),
        )


@dataclass(frozen=True, slots=True)
class Program:
    devices: tuple[DeviceSpec, ...]
    alias_groups: tuple[AliasGroupSpec, ...]
    objects: tuple[ObjectSpec, ...]
    profiles: tuple[TaskProfile, ...]
    tasks: tuple[TaskSpec, ...]
    recomputation_groups: tuple[RecomputationGroup, ...] = ()

    def __post_init__(self) -> None:
        for name, value in (
            ("devices", self.devices),
            ("alias_groups", self.alias_groups),
            ("objects", self.objects),
            ("profiles", self.profiles),
            ("tasks", self.tasks),
            ("recomputation_groups", self.recomputation_groups),
        ):
            require_tuple(value, f"program.{name}")
        require(bool(self.devices), "program.devices", "must not be empty")
        device_ids = index_unique(
            (device.device_id for device in self.devices), "program.devices"
        )
        alias_ids = index_unique(
            (group.alias_group_id for group in self.alias_groups),
            "program.alias_groups",
        )
        object_ids = index_unique(
            (item.object_id for item in self.objects), "program.objects"
        )
        profile_ids = index_unique(
            (profile.profile_id for profile in self.profiles), "program.profiles"
        )
        task_ids = index_unique((task.task_id for task in self.tasks), "program.tasks")
        group_ids = index_unique(
            (group.group_id for group in self.recomputation_groups),
            "program.recomputation_groups",
        )

        alias_by_id = {
            alias_group.alias_group_id: alias_group for alias_group in self.alias_groups
        }
        for index, alias_group in enumerate(self.alias_groups):
            require(
                alias_group.device_id in device_ids,
                f"program.alias_groups[{index}].device_id",
                f"unknown device {alias_group.device_id!r}",
            )
        for index, item in enumerate(self.objects):
            path = f"program.objects[{index}]"
            if item.alias_group_id not in alias_by_id:
                fail(
                    f"{path}.alias_group_id",
                    f"unknown alias group {item.alias_group_id!r}",
                )
            extent = alias_by_id[item.alias_group_id].size_bytes
            require(
                item.offset_bytes + item.size_bytes <= extent,
                path,
                f"span exceeds alias-group extent of {extent} bytes",
            )

        tasks_in_groups: set[str] = set()
        exclusive_pairs: set[frozenset[str]] = set()
        for group_index, recomputation_group in enumerate(self.recomputation_groups):
            group_path = f"program.recomputation_groups[{group_index}]"
            require(
                recomputation_group.group_id in group_ids,
                group_path,
                "invalid group identity",
            )
            group_tasks: set[str] = set()
            option_tasks: list[set[str]] = []
            for option_index, option in enumerate(recomputation_group.options):
                option_path = f"{group_path}.options[{option_index}]"
                active = set(option.active_task_ids)
                option_tasks.append(active)
                for task_id in active:
                    require(
                        task_id in task_ids,
                        f"{option_path}.active_task_ids",
                        f"unknown task {task_id!r}",
                    )
                    group_tasks.add(task_id)
                for alias_id in option.retained_alias_group_ids:
                    require(
                        alias_id in alias_ids,
                        f"{option_path}.retained_alias_group_ids",
                        f"unknown alias group {alias_id!r}",
                    )
            overlap = tasks_in_groups & group_tasks
            require(
                not overlap,
                group_path,
                f"tasks occur in several recomputation groups: {sorted(overlap)}",
            )
            tasks_in_groups.update(group_tasks)
            for left_index, left in enumerate(option_tasks):
                for right in option_tasks[left_index + 1 :]:
                    for left_task in left - right:
                        for right_task in right - left:
                            exclusive_pairs.add(frozenset((left_task, right_task)))

        produced_by: dict[str, list[str]] = {}
        seen_tasks: set[str] = set()
        for index, task in enumerate(self.tasks):
            path = f"program.tasks[{index}]"
            require(
                task.resource.device_id in device_ids,
                f"{path}.resource.device_id",
                f"unknown device {task.resource.device_id!r}",
            )
            require(
                task.profile_id in profile_ids,
                f"{path}.profile_id",
                f"unknown profile {task.profile_id!r}",
            )
            for dependency in task.dependencies:
                require(
                    dependency in seen_tasks,
                    f"{path}.dependencies",
                    f"dependency {dependency!r} must precede the task",
                )
            for relation, values in (
                ("inputs", task.inputs),
                ("outputs", task.outputs),
            ):
                for object_id in values:
                    require(
                        object_id in object_ids,
                        f"{path}.{relation}",
                        f"unknown object {object_id!r}",
                    )
            for output in task.outputs:
                previous_writers = produced_by.setdefault(output, [])
                require(
                    all(
                        frozenset((writer, task.task_id)) in exclusive_pairs
                        for writer in previous_writers
                    ),
                    f"{path}.outputs",
                    f"object {output!r} has simultaneously active writers",
                )
                previous_writers.append(task.task_id)
            for input_id in task.inputs:
                producers = produced_by.get(input_id, [])
                if producers:
                    require(
                        any(producer in task.dependencies for producer in producers),
                        f"{path}.dependencies",
                        (
                            f"must include a producer for input {input_id!r}; "
                            f"candidates are {producers}"
                        ),
                    )
            for mutation_index, mutation in enumerate(task.mutations):
                require(
                    mutation.object_id in task.inputs,
                    f"{path}.mutations[{mutation_index}].object_id",
                    "mutated object must be an input",
                )
            seen_tasks.add(task.task_id)

    @property
    def digest(self) -> str:
        return digest_json(self.to_dict())

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "alias_groups": [group.to_dict() for group in self.alias_groups],
            "devices": [device.to_dict() for device in self.devices],
            "objects": [item.to_dict() for item in self.objects],
            "profiles": [profile.to_dict() for profile in self.profiles],
            "recomputation_groups": [
                group.to_dict() for group in self.recomputation_groups
            ],
            "schema": PROGRAM_SCHEMA,
            "tasks": [task.to_dict() for task in self.tasks],
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> Program:
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
        groups = records("recomputation_groups")
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
            recomputation_groups=tuple(
                RecomputationGroup.from_value(
                    item, f"program.recomputation_groups[{index}]"
                )
                for index, item in enumerate(groups)
            ),
        )

    @classmethod
    def from_json(cls, payload: str) -> Program:
        return cls.from_dict(parse_json(payload))

    def selected_tasks(
        self, selections: tuple[RecomputationSelection, ...]
    ) -> tuple[TaskSpec, ...]:
        require_tuple(selections, "selections")
        selection_by_group = {selection.group_id: selection for selection in selections}
        require(
            len(selection_by_group) == len(selections),
            "selections",
            "contains duplicate group IDs",
        )
        expected_groups = {group.group_id for group in self.recomputation_groups}
        require(
            set(selection_by_group) == expected_groups,
            "selections",
            "must select exactly one option from every recomputation group",
        )
        variant_tasks: set[str] = set()
        active_variant_tasks: set[str] = set()
        for group in self.recomputation_groups:
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
    "Program",
    "RecomputationGroup",
    "RecomputationOption",
    "RecomputationSelection",
    "ResourceKind",
    "ResourceSpec",
    "TaskProfile",
    "TaskSpec",
    "ValidationError",
]
