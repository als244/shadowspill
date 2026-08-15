"""Immutable records for one dependency-certified execution-pool layout."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from shadowspill.simulator import SimulationAdmission, SimulationResult

from ..admission_replay import AdmissionReplayPurpose


@dataclass(frozen=True, slots=True)
class LeaseLifetime:
    """One physical lease with predicted and causal lifetime boundaries."""

    lease_id: int
    bytes: int
    alignment: int
    predicted_start_ns: int
    predicted_end_ns: int
    causal_start: int
    causal_end: int
    purpose: AdmissionReplayPurpose
    task_id: str | None = None
    alias_group_id: str | None = None
    action_index: int | None = None


@dataclass(frozen=True, slots=True)
class FixedLayoutPlacement:
    """One lease's exact offset within the admitted execution-pool slice."""

    lease_id: int
    offset: int
    bytes: int
    alignment: int
    predicted_start_ns: int
    predicted_end_ns: int
    causal_start: int
    causal_end: int
    purpose: AdmissionReplayPurpose
    task_id: str | None = None
    alias_group_id: str | None = None
    action_index: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "action_index": self.action_index,
            "alias_group_id": self.alias_group_id,
            "alignment": self.alignment,
            "bytes": self.bytes,
            "causal_end": self.causal_end,
            "causal_start": self.causal_start,
            "lease_id": self.lease_id,
            "offset": self.offset,
            "predicted_end_ns": self.predicted_end_ns,
            "predicted_start_ns": self.predicted_start_ns,
            "purpose": self.purpose.value,
            "task_id": self.task_id,
        }


@dataclass(frozen=True, slots=True)
class FixedLayoutReuse:
    """Completion proof required before a successor may use shared bytes."""

    dependency_id: int
    predecessor_lease_id: int
    predecessor_purpose: AdmissionReplayPurpose
    predecessor_task_id: str
    predecessor_action_index: int | None
    successor_lease_id: int
    successor_task_id: str | None
    successor_action_index: int | None

    def to_dict(self) -> dict[str, object]:
        return {
            "dependency_id": self.dependency_id,
            "predecessor_action_index": self.predecessor_action_index,
            "predecessor_lease_id": self.predecessor_lease_id,
            "predecessor_purpose": self.predecessor_purpose.value,
            "predecessor_task_id": self.predecessor_task_id,
            "successor_action_index": self.successor_action_index,
            "successor_lease_id": self.successor_lease_id,
            "successor_task_id": self.successor_task_id,
        }


@dataclass(frozen=True, slots=True)
class FixedPhysicalLayout:
    """Complete physical certificate for one selected execution schedule."""

    program_digest: str
    schedule_digest: str
    topology_digest: str
    pool_capacity_bytes: int
    required_bytes: int
    placements: tuple[FixedLayoutPlacement, ...]
    reuse_dependencies: tuple[FixedLayoutReuse, ...]
    initial_alias_leases: tuple[tuple[str, int], ...]
    task_allocation_leases: tuple[tuple[str, int, int], ...]
    action_destination_leases: tuple[tuple[int, int], ...]

    @property
    def slack_bytes(self) -> int:
        return self.pool_capacity_bytes - self.required_bytes

    @property
    def digest(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {
            "action_destination_leases": [
                {"action_index": action, "lease_id": lease}
                for action, lease in self.action_destination_leases
            ],
            "initial_alias_leases": [
                {"alias_group_id": alias, "lease_id": lease}
                for alias, lease in self.initial_alias_leases
            ],
            "placements": [item.to_dict() for item in self.placements],
            "pool_capacity_bytes": self.pool_capacity_bytes,
            "program_digest": self.program_digest,
            "required_bytes": self.required_bytes,
            "reuse_dependencies": [item.to_dict() for item in self.reuse_dependencies],
            "schedule_digest": self.schedule_digest,
            "schema": "shadowspill.fixed_physical_layout/v1",
            "task_allocation_leases": [
                {
                    "allocation_ordinal": ordinal,
                    "lease_id": lease,
                    "task_id": task,
                }
                for task, ordinal, lease in self.task_allocation_leases
            ],
            "topology_digest": self.topology_digest,
        }


@dataclass(frozen=True, slots=True)
class FixedLayoutAdmission:
    """Fixed layout plus its dependency-aware simulator evidence."""

    layout: FixedPhysicalLayout
    simulator_input: SimulationAdmission
    simulation: SimulationResult


class FixedLayoutInfeasibleError(ValueError):
    """The selected schedule has no layout within its physical pool."""

    def __init__(self, required_bytes: int, capacity_bytes: int) -> None:
        self.required_bytes = required_bytes
        self.capacity_bytes = capacity_bytes
        self.additional_bytes = max(0, required_bytes - capacity_bytes)
        super().__init__(
            "fixed physical layout exceeds the execution pool: "
            f"required={required_bytes}, capacity={capacity_bytes}, "
            f"additional={self.additional_bytes}"
        )


__all__ = [
    "FixedLayoutAdmission",
    "FixedLayoutInfeasibleError",
    "FixedLayoutPlacement",
    "FixedLayoutReuse",
    "FixedPhysicalLayout",
    "LeaseLifetime",
]
