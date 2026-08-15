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
from enum import StrEnum

from shadowspill.ir import Program


class TaskAllocationStepKind(StrEnum):
    """One allocator transition observed inside a compiled task."""

    ALLOCATE = "allocate"
    RELEASE = "release"


@dataclass(frozen=True, slots=True)
class TaskAllocationStep:
    """Neutral projection of one profiled task-local allocator transition.

    ``allocation_ordinal`` relates a release to its earlier allocation.  An
    output alias is present only when that allocation becomes a persistent
    Program object after the task.  The trace is admission evidence, not a
    runtime callback-order contract.
    """

    allocation_ordinal: int
    kind: TaskAllocationStepKind
    charged_bytes: int = 0
    output_alias_group_id: str | None = None
    reuses_allocation_ordinal: int | None = None

    def __post_init__(self) -> None:
        if self.allocation_ordinal < 0:
            raise ValueError("task allocation ordinal must be non-negative")
        if not isinstance(self.kind, TaskAllocationStepKind):
            raise TypeError("task allocation step kind is invalid")
        if self.kind is TaskAllocationStepKind.ALLOCATE:
            if self.charged_bytes <= 0:
                raise ValueError("task allocation bytes must be positive")
            if self.output_alias_group_id == "":
                raise ValueError("task allocation output alias must be non-empty")
            if self.reuses_allocation_ordinal is not None and (
                self.reuses_allocation_ordinal < 0
                or self.reuses_allocation_ordinal == self.allocation_ordinal
            ):
                raise ValueError("task allocation reuse ordinal is invalid")
        elif (
            self.charged_bytes != 0
            or self.output_alias_group_id is not None
            or self.reuses_allocation_ordinal is not None
        ):
            raise ValueError("task release carries allocation-only fields")

    def to_dict(self) -> dict[str, object]:
        return {
            "allocation_ordinal": self.allocation_ordinal,
            "charged_bytes": self.charged_bytes,
            "kind": self.kind.value,
            "output_alias_group_id": self.output_alias_group_id,
            "reuses_allocation_ordinal": self.reuses_allocation_ordinal,
        }

    @classmethod
    def from_value(cls, value: object, path: str) -> TaskAllocationStep:
        data = _mapping(value, path)
        return cls(
            allocation_ordinal=_integer(
                data.get("allocation_ordinal"), f"{path}.allocation_ordinal"
            ),
            kind=_enum(
                TaskAllocationStepKind,
                data.get("kind"),
                f"{path}.kind",
            ),
            charged_bytes=_integer(data.get("charged_bytes"), f"{path}.charged_bytes"),
            output_alias_group_id=_optional_string(
                data.get("output_alias_group_id"),
                f"{path}.output_alias_group_id",
            ),
            reuses_allocation_ordinal=_optional_integer(
                data.get("reuses_allocation_ordinal"),
                f"{path}.reuses_allocation_ordinal",
            ),
        )


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

    @classmethod
    def from_value(cls, value: object, path: str) -> StorageHandoff:
        data = _mapping(value, path)
        return cls(
            source_alias_group_id=_string(
                data.get("source_alias_group_id"),
                f"{path}.source_alias_group_id",
            ),
            destination_alias_group_id=_string(
                data.get("destination_alias_group_id"),
                f"{path}.destination_alias_group_id",
            ),
        )


@dataclass(frozen=True, slots=True)
class TaskAdmissionSpec:
    """Physical ownership transitions for one executable task."""

    task_id: str
    workspace_extents: tuple[int, ...] = ()
    fresh_output_aliases: tuple[str, ...] = ()
    replacement_aliases: tuple[str, ...] = ()
    storage_handoffs: tuple[StorageHandoff, ...] = ()
    allocation_steps: tuple[TaskAllocationStep, ...] = ()

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
        self._validate_allocation_steps()

    def _validate_allocation_steps(self) -> None:
        if not self.allocation_steps:
            return
        live: set[int] = set()
        allocations: dict[int, TaskAllocationStep] = {}
        retired: set[int] = set()
        reused: set[int] = set()
        output_aliases: set[str] = set()
        for step in self.allocation_steps:
            ordinal = step.allocation_ordinal
            if step.kind is TaskAllocationStepKind.ALLOCATE:
                if ordinal in allocations:
                    raise ValueError(
                        "task allocation trace allocates one ordinal twice"
                    )
                source = step.reuses_allocation_ordinal
                if source is not None:
                    if source not in retired:
                        raise ValueError(
                            "task allocation trace reuses a non-retired ordinal"
                        )
                    if source in reused:
                        raise ValueError(
                            "task allocation trace reuses one ordinal twice"
                        )
                    reused.add(source)
                    if allocations[source].charged_bytes != step.charged_bytes:
                        raise ValueError(
                            "task allocation trace reuse changes charged bytes"
                        )
                allocations[ordinal] = step
                live.add(ordinal)
                if step.output_alias_group_id is not None:
                    if step.output_alias_group_id in output_aliases:
                        raise ValueError(
                            "task allocation trace binds one output alias twice"
                        )
                    output_aliases.add(step.output_alias_group_id)
                continue
            if ordinal not in live:
                raise ValueError("task allocation trace releases an unknown ordinal")
            live.remove(ordinal)
            retired.add(ordinal)
        expected_outputs = set(self.fresh_output_aliases) | set(
            self.replacement_aliases
        )
        if output_aliases != expected_outputs:
            raise ValueError(
                "task allocation trace output aliases disagree with task ownership: "
                f"trace={sorted(output_aliases)!r}, "
                f"expected={sorted(expected_outputs)!r}"
            )
        persistent_ordinals = {
            step.allocation_ordinal
            for step in self.allocation_steps
            if step.output_alias_group_id is not None
        }
        if live != persistent_ordinals:
            raise ValueError(
                "task allocation trace must release every anonymous allocation"
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
            "allocation_steps": [item.to_dict() for item in self.allocation_steps],
        }

    @classmethod
    def from_value(cls, value: object, path: str) -> TaskAdmissionSpec:
        data = _mapping(value, path)
        return cls(
            task_id=_string(data.get("task_id"), f"{path}.task_id"),
            workspace_extents=_integer_tuple(
                data.get("workspace_extents"), f"{path}.workspace_extents"
            ),
            fresh_output_aliases=_string_tuple(
                data.get("fresh_output_aliases"),
                f"{path}.fresh_output_aliases",
            ),
            replacement_aliases=_string_tuple(
                data.get("replacement_aliases"),
                f"{path}.replacement_aliases",
            ),
            storage_handoffs=tuple(
                StorageHandoff.from_value(item, f"{path}.storage_handoffs[{index}]")
                for index, item in enumerate(
                    _list(data.get("storage_handoffs"), f"{path}.storage_handoffs")
                )
            ),
            allocation_steps=tuple(
                TaskAllocationStep.from_value(item, f"{path}.allocation_steps[{index}]")
                for index, item in enumerate(
                    _list(data.get("allocation_steps"), f"{path}.allocation_steps")
                )
            ),
        )


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
            "schema": "shadowspill.admission_topology/v2",
            "tasks": [item.to_dict() for item in self.tasks],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, value: object) -> AdmissionTopology:
        data = _mapping(value, "admission")
        schema = _string(data.get("schema"), "admission.schema")
        if schema != "shadowspill.admission_topology/v2":
            raise ValueError(f"admission.schema: unsupported schema {schema!r}")
        return cls(
            device_id=_string(data.get("device_id"), "admission.device_id"),
            pool_capacity_bytes=_integer(
                data.get("pool_capacity_bytes"), "admission.pool_capacity_bytes"
            ),
            object_capacity_bytes=_integer(
                data.get("object_capacity_bytes"),
                "admission.object_capacity_bytes",
            ),
            minimum_alignment=_integer(
                data.get("minimum_alignment"), "admission.minimum_alignment"
            ),
            tasks=tuple(
                TaskAdmissionSpec.from_value(item, f"admission.tasks[{index}]")
                for index, item in enumerate(
                    _list(data.get("tasks"), "admission.tasks")
                )
            ),
        )

    @classmethod
    def from_json(cls, payload: str) -> AdmissionTopology:
        try:
            value = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError("admission JSON is invalid") from exc
        return cls.from_dict(value)

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


def _mapping(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{path}: expected an object")
    return value


def _list(value: object, path: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{path}: expected a list")
    return value


def _integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path}: expected an integer")
    return value


def _string(value: object, path: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{path}: expected a string")
    return value


def _optional_string(value: object, path: str) -> str | None:
    if value is None:
        return None
    return _string(value, path)


def _optional_integer(value: object, path: str) -> int | None:
    if value is None:
        return None
    return _integer(value, path)


def _integer_tuple(value: object, path: str) -> tuple[int, ...]:
    return tuple(
        _integer(item, f"{path}[{index}]")
        for index, item in enumerate(_list(value, path))
    )


def _string_tuple(value: object, path: str) -> tuple[str, ...]:
    return tuple(
        _string(item, f"{path}[{index}]")
        for index, item in enumerate(_list(value, path))
    )


def _enum(
    enum_type: type[TaskAllocationStepKind], value: object, path: str
) -> TaskAllocationStepKind:
    raw = _string(value, path)
    try:
        return enum_type(raw)
    except ValueError as exc:
        raise ValueError(f"{path}: unknown value {raw!r}") from exc


__all__ = [
    "AdmissionTopology",
    "StorageHandoff",
    "TaskAdmissionSpec",
    "TaskAllocationStep",
    "TaskAllocationStepKind",
]
