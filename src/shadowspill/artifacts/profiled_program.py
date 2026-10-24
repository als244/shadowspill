"""Compute profiles and memory evidence; communication is priced during planning."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar

from shadowspill.communication.task_costs import task_collective_cost, task_collectives
from shadowspill.ir import ShadowSpillProgram, TaskResourcePhase
from shadowspill.ir import TaskProfile as CostedTaskProfile

from ._values import digest, freeze, nonnegative_int, plain
from .archive import Artifact
from .profiles import Profiles, TaskProfile
from .program import Program

if TYPE_CHECKING:
    from shadowspill.communication.info import CommunicationInfo


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
        communication = task_collectives(self.program)
        for key, profile in self.task_profiles.items():
            if key in communication and (
                profile.duration_seconds is not None
                or profile.resource_phases
                or profile.overlap_costs
            ):
                raise ValueError(
                    f"{key}: collective timing belongs to CommunicationInfo"
                )
            if profile.workspace_bytes is None or (
                profile.duration_seconds is None and key not in communication
            ):
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

    def _compute_profiles(self) -> tuple[CostedTaskProfile, ...]:
        """Validate supplied timing, without assigning placeholder collective costs."""
        definitions = {item.profile_key: item for item in self.program.task_definitions}
        profiles = []
        for key, item in self.task_profiles.items():
            if item.duration_seconds is None:
                continue  # Framework communication is priced only during planning.
            assert item.workspace_bytes is not None
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
        return tuple(profiles)

    def _costed_ir(
        self, communication_info: CommunicationInfo | None = None
    ) -> ShadowSpillProgram:
        """The single internal representation consumed by search/simulation."""
        profiles = list(self._compute_profiles())
        definitions = {item.profile_key: item for item in self.program.task_definitions}
        for key, requests in task_collectives(self.program).items():
            if communication_info is None:
                raise ValueError("explicit collective tasks require CommunicationInfo")
            cost = task_collective_cost(
                requests, communication_info, self.program.setup
            )
            workspace = self.task_profiles[key].workspace_bytes
            assert workspace is not None
            profiles.append(
                CostedTaskProfile(
                    key,
                    round(cost.duration_seconds * 1e9),
                    workspace,
                    definitions[key].compatibility_digest,
                    # Framework reductions and sharded-weight gathers share an
                    # ordered communication stream. Keep that contention visible.
                    (
                        TaskResourcePhase(
                            round(cost.duration_seconds * 1e9), "sharded_collectives"
                        ),
                    ),
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
