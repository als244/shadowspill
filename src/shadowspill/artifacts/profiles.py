"""Editable profile collections with plain numeric per-task records."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar

from shadowspill.ir import TaskOverlapCost, TaskResourcePhase

from ._values import freeze, nonnegative_int, plain, seconds
from .archive import Artifact


@dataclass(frozen=True)
class TaskProfile:
    """None leaves a field unspecified; manual values default to estimates.

    Metadata may mark ``duration_kind`` or ``memory_kind`` as ``measured``
    and record ordinary source information. A memory peak alone is not an
    allocation/lifetime contract for execution.
    """

    duration_seconds: float | None = None
    workspace_bytes: int | None = None
    compatibility_digest: str | None = None
    resource_phases: tuple[TaskResourcePhase, ...] | None = None
    overlap_costs: tuple[TaskOverlapCost, ...] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.duration_seconds is not None:
            seconds(self.duration_seconds, "duration_seconds")
        if self.workspace_bytes is not None:
            nonnegative_int(self.workspace_bytes, "workspace_bytes")
        object.__setattr__(self, "metadata", freeze(self.metadata))
        for key in ("duration_kind", "memory_kind"):
            if self.metadata.get(key, "estimated") not in (
                "measured",
                "estimated",
                "static",
            ):
                raise ValueError(f"{key} must be measured, estimated or static")

    def to_dict(self) -> dict[str, Any]:
        return {
            "duration_seconds": self.duration_seconds,
            "workspace_bytes": self.workspace_bytes,
            "compatibility_digest": self.compatibility_digest,
            "resource_phases": None
            if self.resource_phases is None
            else [item.to_dict() for item in self.resource_phases],
            "overlap_costs": None
            if self.overlap_costs is None
            else [item.to_dict() for item in self.overlap_costs],
            "metadata": plain(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TaskProfile:
        data = dict(value)
        for key, record in (
            ("resource_phases", TaskResourcePhase),
            ("overlap_costs", TaskOverlapCost),
        ):
            if data.get(key) is not None:
                data[key] = tuple(
                    record.from_value(item, f"profile.{key}[{index}]")
                    for index, item in enumerate(data[key])
                )
        return cls(**data)

    def merged(self, later: TaskProfile) -> TaskProfile:
        if (
            self.compatibility_digest is not None
            and later.compatibility_digest is not None
            and self.compatibility_digest != later.compatibility_digest
        ):
            raise ValueError("cannot merge profiles for different task definitions")
        metadata = dict(self.metadata)
        for field_name, kind in (
            ("duration_seconds", "duration_kind"),
            ("workspace_bytes", "memory_kind"),
        ):
            if getattr(later, field_name) is not None:
                metadata[kind] = later.metadata.get(kind, "estimated")
        metadata.update(later.metadata)
        values = {
            key: getattr(later, key)
            for key in (
                "duration_seconds",
                "workspace_bytes",
                "compatibility_digest",
                "resource_phases",
                "overlap_costs",
            )
            if getattr(later, key) is not None
        }
        return replace(self, **values, metadata=metadata)


@dataclass
class Profiles(Artifact):
    """One setup, many task forms. Later supplied fields overwrite earlier ones."""

    artifact_kind: ClassVar[str] = "profiles"
    setup: dict[str, Any] = field(default_factory=dict)
    tasks: dict[str, TaskProfile] = field(default_factory=dict)
    fixed_device_bytes: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def merge(self, *others: Profiles) -> Profiles:
        """Return a new collection, preserving unspecified fields."""
        result = Profiles.from_dict(self.to_dict())
        for other in others:
            if freeze(result.setup) != freeze(other.setup):
                raise ValueError("profiles describe different hardware/runtime setups")
            for key, record in other.tasks.items():
                result.tasks[key] = (
                    result.tasks[key].merged(record) if key in result.tasks else record
                )
            if other.fixed_device_bytes is not None:
                result.fixed_device_bytes = other.fixed_device_bytes
                result.metadata["memory_kind"] = other.metadata.get(
                    "memory_kind", "estimated"
                )
            result.metadata.update(other.metadata)
        return result

    def to_dict(self) -> dict[str, Any]:
        if self.fixed_device_bytes is not None:
            nonnegative_int(self.fixed_device_bytes, "fixed_device_bytes")
        return {
            "setup": plain(freeze(self.setup)),
            "tasks": {key: record.to_dict() for key, record in self.tasks.items()},
            "fixed_device_bytes": self.fixed_device_bytes,
            "metadata": plain(freeze(self.metadata)),
        }

    @classmethod
    def from_dict(
        cls, value: dict[str, Any], *, payloads: Mapping[str, bytes] | None = None
    ) -> Profiles:
        if payloads:
            raise ValueError("Profiles cannot contain executable payloads")
        return cls(
            setup=plain(freeze(value["setup"])),
            tasks={
                key: TaskProfile.from_dict(item) for key, item in value["tasks"].items()
            },
            fixed_device_bytes=value["fixed_device_bytes"],
            metadata=plain(freeze(value["metadata"])),
        )
