"""Schedule-invariant physical facts consumed by admission-aware planning.

The logical :class:`~shadowspill.ir.Program` deliberately does not encode how
a compiled task returns storage.  This module carries the small additional
physical contract needed to evaluate dynamic slab admission without importing
PyTorch or consulting runtime allocator telemetry in the candidate loop.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from shadowspill.ir import Program


@dataclass(frozen=True, slots=True)
class StorageHandoff:
    """Transfer one live lease between two logical alias identities."""

    source_alias_group_id: str
    destination_alias_group_id: str

    def __post_init__(self) -> None:
        if not self.source_alias_group_id or not self.destination_alias_group_id:
            raise ValueError("storage-handoff alias IDs must be non-empty")
        if self.source_alias_group_id == self.destination_alias_group_id:
            raise ValueError("storage-handoff source and destination must differ")

    def to_dict(self) -> dict[str, str]:
        return {
            "destination_alias_group_id": self.destination_alias_group_id,
            "source_alias_group_id": self.source_alias_group_id,
        }


@dataclass(frozen=True, slots=True)
class TaskAdmissionSpec:
    """Physical ownership transitions for one executable task."""

    task_id: str
    workspace_extents: tuple[int, ...] = ()
    fresh_output_aliases: tuple[str, ...] = ()
    replacement_aliases: tuple[str, ...] = ()
    storage_handoffs: tuple[StorageHandoff, ...] = ()

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("task admission ID must be non-empty")
        if any(value <= 0 for value in self.workspace_extents):
            raise ValueError("task admission workspace extents must be positive")
        for field, values in (
            ("fresh_output_aliases", self.fresh_output_aliases),
            ("replacement_aliases", self.replacement_aliases),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"task admission {field} must be unique")
            if any(not value for value in values):
                raise ValueError(f"task admission {field} must be non-empty")
        destinations = tuple(
            item.destination_alias_group_id for item in self.storage_handoffs
        )
        sources = tuple(item.source_alias_group_id for item in self.storage_handoffs)
        if len(destinations) != len(set(destinations)):
            raise ValueError("task storage-handoff destinations must be unique")
        if len(sources) != len(set(sources)):
            raise ValueError("task storage-handoff sources must be unique")
        output_kinds = (
            set(self.fresh_output_aliases),
            set(self.replacement_aliases),
            set(destinations),
        )
        if any(
            left & right
            for index, left in enumerate(output_kinds)
            for right in output_kinds[index + 1 :]
        ):
            raise ValueError(
                "fresh, replacement, and handoff destination aliases must be disjoint"
            )

    @property
    def workspace_bytes(self) -> int:
        """Return total simultaneously-live anonymous workspace bytes."""

        return sum(self.workspace_extents)

    def to_dict(self) -> dict[str, object]:
        return {
            "fresh_output_aliases": list(self.fresh_output_aliases),
            "replacement_aliases": list(self.replacement_aliases),
            "storage_handoffs": [item.to_dict() for item in self.storage_handoffs],
            "task_id": self.task_id,
            "workspace_extents": list(self.workspace_extents),
        }


@dataclass(frozen=True, slots=True)
class AdmissionTopology:
    """Immutable physical topology reused by every PressureFit candidate.

    ``pool_capacity_bytes`` is the complete execution-pool capacity certified
    by the production range allocator. ``object_capacity_bytes`` is the
    conservative residency capacity used by PressureFit before exact task and
    transfer deltas are evaluated.  The current runtime admits one execution
    pool; the device identity is explicit so extending this record to several
    pools does not require model-specific policy.
    """

    device_id: str
    pool_capacity_bytes: int
    object_capacity_bytes: int
    minimum_alignment: int
    tasks: tuple[TaskAdmissionSpec, ...]

    def __post_init__(self) -> None:
        if not self.device_id:
            raise ValueError("admission device ID must be non-empty")
        if self.pool_capacity_bytes <= 0:
            raise ValueError("admission pool capacity must be positive")
        if not 0 < self.object_capacity_bytes <= self.pool_capacity_bytes:
            raise ValueError(
                "admission object capacity must be positive and no larger than "
                "the pool capacity"
            )
        if self.minimum_alignment <= 0:
            raise ValueError("admission minimum alignment must be positive")
        task_ids = tuple(item.task_id for item in self.tasks)
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("admission task IDs must be unique")

    @property
    def digest(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {
            "device_id": self.device_id,
            "minimum_alignment": self.minimum_alignment,
            "object_capacity_bytes": self.object_capacity_bytes,
            "pool_capacity_bytes": self.pool_capacity_bytes,
            "schema": "shadowspill.admission_topology/v1",
            "tasks": [item.to_dict() for item in self.tasks],
        }

    def validate(self, program: Program) -> None:
        """Validate exact task and alias coverage against ``program``."""

        devices = {item.device_id for item in program.devices}
        if self.device_id not in devices:
            raise ValueError(
                f"admission topology names unknown device {self.device_id!r}"
            )
        if len(program.devices) != 1:
            raise ValueError(
                "one AdmissionTopology currently describes exactly one execution "
                f"pool; Program has {len(program.devices)} devices"
            )
        expected_tasks = tuple(item.task_id for item in program.tasks)
        actual_tasks = tuple(item.task_id for item in self.tasks)
        if actual_tasks != expected_tasks:
            raise ValueError(
                "admission tasks must exactly follow Program task order: "
                f"expected={expected_tasks!r}, actual={actual_tasks!r}"
            )
        aliases = {item.alias_group_id for item in program.alias_groups}
        for task in self.tasks:
            referenced = (
                *task.fresh_output_aliases,
                *task.replacement_aliases,
                *(item.source_alias_group_id for item in task.storage_handoffs),
                *(item.destination_alias_group_id for item in task.storage_handoffs),
            )
            unknown = sorted(set(referenced) - aliases)
            if unknown:
                raise ValueError(
                    f"task admission {task.task_id!r} references unknown aliases "
                    f"{unknown}"
                )


__all__ = ["AdmissionTopology", "StorageHandoff", "TaskAdmissionSpec"]
