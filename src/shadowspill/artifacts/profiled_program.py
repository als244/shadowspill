"""An immutable snapshot of structure and complete supplied task costs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, ClassVar

from shadowspill.ir import ShadowSpillProgram
from shadowspill.ir import TaskProfile as CostedTaskProfile

from ._values import digest, freeze, nonnegative_int, plain
from .archive import Artifact
from .profiles import Profiles, TaskProfile
from .program import Program


@dataclass(frozen=True)
class ProfiledProgram(Artifact):
    artifact_kind: ClassVar[str] = "profiled_program"
    program: Program
    task_profiles: Mapping[str, TaskProfile]
    setup: Mapping[str, Any]
    fixed_device_bytes: int
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        nonnegative_int(self.fixed_device_bytes, "fixed_device_bytes")
        if set(self.task_profiles) != {
            d.profile_key for d in self.program.task_definitions
        }:
            raise ValueError(
                "applied profiles must cover exactly the Program task forms"
            )
        for key, profile in self.task_profiles.items():
            if profile.duration_seconds is None or profile.workspace_bytes is None:
                raise ValueError(f"incomplete applied profile: {key}")
        object.__setattr__(
            self, "task_profiles", MappingProxyType(dict(self.task_profiles))
        )
        object.__setattr__(self, "setup", freeze(self.setup))
        object.__setattr__(self, "metadata", freeze(self.metadata))

    @property
    def profiles(self) -> Profiles:
        """An editable copy; editing it never changes this artifact."""
        return Profiles(
            setup=plain(self.setup),
            tasks=dict(self.task_profiles),
            fixed_device_bytes=self.fixed_device_bytes,
            metadata=plain(self.metadata),
        )

    @property
    def digest(self) -> str:
        return digest(self.to_dict())

    @property
    def estimated_memory(self) -> tuple[str, ...]:
        result = ["program storage"] if self.program.memory_estimated else []
        if self.metadata.get("memory_kind", "estimated") == "estimated":
            result.append("fixed device overhead")
        for key, item in self.task_profiles.items():
            if item.metadata.get("memory_kind", "estimated") == "estimated":
                result.append(key)
        return tuple(result)

    def _costed_ir(self) -> ShadowSpillProgram:
        """The single internal representation consumed by search/simulation."""
        definitions = {item.profile_key: item for item in self.program.task_definitions}
        profiles = []
        for key, item in self.task_profiles.items():
            assert (
                item.duration_seconds is not None and item.workspace_bytes is not None
            )
            profiles.append(
                CostedTaskProfile(
                    key,
                    round(item.duration_seconds * 1e9),
                    item.workspace_bytes,
                    definitions[key].compatibility_digest,
                    item.resource_phases or (),
                    item.overlap_costs or (),
                )
            )
        from shadowspill.planning.workspace import transient_output_workspace

        by_key = {profile.profile_id: profile for profile in profiles}
        tasks = []
        transient = self.program.bindings.get("transient_outputs", {})
        manifests = self.program.bindings.get("manifests", {})
        derived = {}
        for task in self.program.tasks:
            leaves = transient.get(task.task_id, ())
            extra = transient_output_workspace(
                leaves,
                manifests.get(task.profile_id, {}),
                self.task_profiles[task.profile_id].metadata,
            )
            if extra:
                key = f"{task.profile_id}.workspace_{extra}"
                derived[key] = replace(
                    by_key[task.profile_id],
                    profile_id=key,
                    workspace_bytes=by_key[task.profile_id].workspace_bytes + extra,
                )
                task = replace(task, profile_id=key)
            tasks.append(task)
        return ShadowSpillProgram(
            devices=self.program.devices,
            alias_groups=self.program.alias_groups,
            objects=self.program.objects,
            tasks=tuple(tasks),
            task_alternative_groups=self.program.task_alternative_groups,
            profiles=(*profiles, *derived.values()),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"program": self.program.to_dict(), "profiles": self.profiles.to_dict()}

    def _archive_payloads(self) -> Mapping[str, bytes]:
        return self.program._archive_payloads()

    @classmethod
    def from_dict(
        cls, value: dict[str, Any], *, payloads: Mapping[str, bytes] | None = None
    ) -> ProfiledProgram:
        from shadowspill.profiling.apply import apply_profiles

        return apply_profiles(
            Program.from_dict(value["program"], payloads=payloads),
            Profiles.from_dict(value["profiles"]),
        )
