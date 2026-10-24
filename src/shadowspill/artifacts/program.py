"""The static, unprofiled result of building one concrete computation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from types import MappingProxyType
from typing import Any, ClassVar

from shadowspill.ir import (
    AliasGroupSpec,
    DeviceSpec,
    ObjectSpec,
    ResidencySpec,
    TaskAlternativeGroup,
    TaskSpec,
)
from shadowspill.ir.program.checks import (
    check_alias_groups,
    check_alternative_groups,
    check_objects,
    check_tasks,
    index_identities,
    require_shapes,
)

from ._values import digest, freeze, nonnegative_int, plain
from .archive import Artifact


@dataclass(frozen=True)
class TaskDefinition:
    """One reusable task form. None means unresolved scratch, never zero."""

    profile_key: str
    compatibility_digest: str
    workspace_bytes: int | None = None
    name: str = ""
    module_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.profile_key or not self.compatibility_digest:
            raise ValueError("task key and compatibility digest must be nonempty")
        if self.workspace_bytes is not None:
            nonnegative_int(self.workspace_bytes, "workspace_bytes")
        object.__setattr__(self, "module_paths", tuple(self.module_paths))


@dataclass(frozen=True)
class Program(Artifact):
    """Graph/storage structure, independent of costs, pools and real devices.

    Builders populate these IR records. The public fields remain inspectable;
    profile keys identify reusable task forms, task IDs identify occurrences.
    """

    artifact_kind: ClassVar[str] = "program"
    devices: tuple[DeviceSpec, ...]
    alias_groups: tuple[AliasGroupSpec, ...]
    objects: tuple[ObjectSpec, ...]
    tasks: tuple[TaskSpec, ...]
    task_definitions: tuple[TaskDefinition, ...]
    initial_residency: tuple[ResidencySpec, ...] = ()
    final_residency: tuple[ResidencySpec, ...] = ()
    task_alternative_groups: tuple[TaskAlternativeGroup, ...] = ()
    name: str = ""
    setup: Mapping[str, Any] = field(default_factory=dict)
    sources: Mapping[str, Any] = field(default_factory=dict)
    fixed_device_bytes: int | None = None
    memory_estimated: bool = False
    executables: Mapping[str, bytes] = field(default_factory=dict, repr=False)
    bindings: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_shapes(self)
        if not isinstance(self.task_definitions, tuple):
            raise TypeError("task_definitions must be a tuple")
        identities = index_identities(
            self, tuple(item.profile_key for item in self.task_definitions)
        )
        check_alias_groups(self, identities)
        check_objects(self, identities)
        check_tasks(self, identities, check_alternative_groups(self, identities))
        for name in ("initial_residency", "final_residency"):
            entries = getattr(self, name)
            if not isinstance(entries, tuple):
                raise TypeError(f"{name} must be a tuple")
            keys = [item.alias_group_id for item in entries]
            if len(set(keys)) != len(keys) or set(keys) - identities.alias_ids.keys():
                raise ValueError(f"{name} contains duplicate or unknown storage IDs")
        if self.fixed_device_bytes is not None:
            nonnegative_int(self.fixed_device_bytes, "fixed_device_bytes")
        object.__setattr__(self, "setup", freeze(self.setup))
        object.__setattr__(self, "sources", freeze(self.sources))
        if any(not isinstance(value, bytes) for value in self.executables.values()):
            raise TypeError("executable payloads must be immutable bytes")
        object.__setattr__(
            self, "executables", MappingProxyType(dict(self.executables))
        )
        object.__setattr__(self, "bindings", freeze(self.bindings))

    @property
    def digest(self) -> str:
        return digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            **{
                name: [item.to_dict() for item in getattr(self, name)]
                for name in (
                    "devices",
                    "alias_groups",
                    "objects",
                    "tasks",
                    "initial_residency",
                    "final_residency",
                    "task_alternative_groups",
                )
            },
            "task_definitions": [asdict(item) for item in self.task_definitions],
            "name": self.name,
            "setup": plain(self.setup),
            "sources": plain(self.sources),
            "fixed_device_bytes": self.fixed_device_bytes,
            "memory_estimated": self.memory_estimated,
            "executables": {
                key: sha256(value).hexdigest()
                for key, value in self.executables.items()
            },
            "bindings": plain(self.bindings),
        }

    def _archive_payloads(self) -> Mapping[str, bytes]:
        return {sha256(value).hexdigest(): value for value in self.executables.values()}

    @classmethod
    def from_dict(
        cls, value: dict[str, Any], *, payloads: Mapping[str, bytes] | None = None
    ) -> Program:
        records: dict[str, Callable[[object, str], Any]] = {
            "devices": DeviceSpec.from_value,
            "alias_groups": AliasGroupSpec.from_value,
            "objects": ObjectSpec.from_value,
            "tasks": TaskSpec.from_value,
            "initial_residency": ResidencySpec.from_value,
            "final_residency": ResidencySpec.from_value,
            "task_alternative_groups": TaskAlternativeGroup.from_value,
        }
        data = dict(value)
        for name, record in records.items():
            data[name] = tuple(
                record(item, f"program.{name}[{index}]")
                for index, item in enumerate(data[name])
            )
        data["task_definitions"] = tuple(
            TaskDefinition(**item) for item in data["task_definitions"]
        )
        try:
            data["executables"] = {
                key: (payloads or {})[checksum]
                for key, checksum in data.get("executables", {}).items()
            }
        except KeyError as error:
            raise ValueError("program executable payload is missing") from error
        return cls(**data)
