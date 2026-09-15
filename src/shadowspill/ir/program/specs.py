"""What a program is made of: devices, alias groups, objects, profiles, tasks."""

from __future__ import annotations

from dataclasses import dataclass

from shadowspill.schema import artifact_schema

from ..serialization import (
    JsonValue,
    string_list,
)
from ..validation import (
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
from .enums import ObjectRole, Persistence, ResourceKind, SharedResidencyPolicy

PROGRAM_SCHEMA = artifact_schema("program")


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
    retain_spill_copy: bool = False
    shared_residency: SharedResidencyPolicy | None = None

    def __post_init__(self) -> None:
        require_identifier(self.alias_group_id, "alias_group.alias_group_id")
        require_identifier(self.device_id, "alias_group.device_id")
        require_non_negative(self.size_bytes, "alias_group.size_bytes")
        require_non_negative(self.initial_version, "alias_group.initial_version")
        require(
            isinstance(self.retain_spill_copy, bool),
            "alias_group.retain_spill_copy",
            "must be a boolean",
        )
        require(
            self.shared_residency is None
            or isinstance(self.shared_residency, SharedResidencyPolicy),
            "alias_group.shared_residency",
            "invalid shared residency policy",
        )

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "alias_group_id": self.alias_group_id,
            "device_id": self.device_id,
            "initial_version": self.initial_version,
            "retain_spill_copy": self.retain_spill_copy,
            "shared_residency": (
                None if self.shared_residency is None else self.shared_residency.value
            ),
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_value(cls, value: object, path: str) -> AliasGroupSpec:
        data = expect_mapping(value, path)
        shared_value = field(data, "shared_residency", path)
        if shared_value is None:
            shared_residency = None
        else:
            shared_name = expect_string(shared_value, f"{path}.shared_residency")
            try:
                shared_residency = SharedResidencyPolicy(shared_name)
            except ValueError:
                fail(
                    f"{path}.shared_residency",
                    f"unknown shared residency policy {shared_name!r}",
                )
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
            retain_spill_copy=expect_boolean(
                field(data, "retain_spill_copy", path),
                f"{path}.retain_spill_copy",
            ),
            shared_residency=shared_residency,
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
