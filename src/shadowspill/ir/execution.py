"""Resolved execution-plan records consumed by runtime admission."""

from __future__ import annotations

from dataclasses import dataclass

from shadowspill.schema import artifact_schema

from .program import Program, TaskAlternativeChoice
from .schedule import MemorySchedule
from .serialization import JsonValue, canonical_json, digest_json, parse_json
from .validation import (
    expect_integer,
    expect_list,
    expect_mapping,
    expect_string,
    field,
    index_unique,
    require,
    require_identifier,
    require_non_negative,
    require_tuple,
)

EXECUTION_PLAN_SCHEMA = artifact_schema("execution_plan")


@dataclass(frozen=True, slots=True)
class EntrypointSpec:
    task_id: str
    entrypoint_id: str
    executor_id: str
    contract_digest: str

    def __post_init__(self) -> None:
        require_identifier(self.task_id, "entrypoint.task_id")
        require_identifier(self.entrypoint_id, "entrypoint.entrypoint_id")
        require_identifier(self.executor_id, "entrypoint.executor_id")
        require_identifier(self.contract_digest, "entrypoint.contract_digest")

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "contract_digest": self.contract_digest,
            "entrypoint_id": self.entrypoint_id,
            "executor_id": self.executor_id,
            "task_id": self.task_id,
        }

    @classmethod
    def from_value(cls, value: object, path: str) -> EntrypointSpec:
        data = expect_mapping(value, path)
        return cls(
            task_id=expect_string(field(data, "task_id", path), f"{path}.task_id"),
            entrypoint_id=expect_string(
                field(data, "entrypoint_id", path), f"{path}.entrypoint_id"
            ),
            executor_id=expect_string(
                field(data, "executor_id", path), f"{path}.executor_id"
            ),
            contract_digest=expect_string(
                field(data, "contract_digest", path), f"{path}.contract_digest"
            ),
        )


@dataclass(frozen=True, slots=True)
class PhysicalAdmission:
    """Physical execution and host admission for one selected plan.

    ``workspace_reserve_bytes`` is the contiguous task-workspace
    allowance the execution pool must be able to serve; it is validated
    against ``slab_bytes`` and is NOT subtracted from the pool. Task
    workspace is charged per boundary during planning and placed inside
    the admitted fixed slice, so this value does not define PressureFit's
    object capacity — see ``pytorch.planning.common.simulation_capacity``
    for the capacity actually presented to the planner.
    """

    device_budget_bytes: int
    spill_budget_bytes: int
    baseline_bytes: int
    provider_headroom_bytes: int
    slab_bytes: int
    workspace_reserve_bytes: int
    spill_reservation_bytes: int
    predicted_fragmentation_bytes: int = 0

    def __post_init__(self) -> None:
        byte_fields = (
            ("device_budget_bytes", self.device_budget_bytes),
            ("spill_budget_bytes", self.spill_budget_bytes),
            ("baseline_bytes", self.baseline_bytes),
            ("provider_headroom_bytes", self.provider_headroom_bytes),
            ("slab_bytes", self.slab_bytes),
            ("workspace_reserve_bytes", self.workspace_reserve_bytes),
            ("spill_reservation_bytes", self.spill_reservation_bytes),
            ("predicted_fragmentation_bytes", self.predicted_fragmentation_bytes),
        )
        for name, value in byte_fields:
            require_non_negative(value, f"admission.{name}")
        require(
            self.baseline_bytes + self.provider_headroom_bytes + self.slab_bytes
            <= self.device_budget_bytes,
            "admission.device_budget_bytes",
            "baseline, provider headroom, and slab exceed the physical cap",
        )
        require(
            self.workspace_reserve_bytes <= self.slab_bytes,
            "admission.workspace_reserve_bytes",
            "cannot exceed slab bytes",
        )
        require(
            self.predicted_fragmentation_bytes <= self.slab_bytes,
            "admission.predicted_fragmentation_bytes",
            "cannot exceed slab bytes",
        )
        require(
            self.spill_reservation_bytes <= self.spill_budget_bytes,
            "admission.spill_reservation_bytes",
            "cannot exceed host budget",
        )

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "baseline_bytes": self.baseline_bytes,
            "device_budget_bytes": self.device_budget_bytes,
            "spill_budget_bytes": self.spill_budget_bytes,
            "spill_reservation_bytes": self.spill_reservation_bytes,
            "predicted_fragmentation_bytes": self.predicted_fragmentation_bytes,
            "provider_headroom_bytes": self.provider_headroom_bytes,
            "slab_bytes": self.slab_bytes,
            "workspace_reserve_bytes": self.workspace_reserve_bytes,
        }

    @classmethod
    def from_value(cls, value: object, path: str) -> PhysicalAdmission:
        data = expect_mapping(value, path)

        def integer(name: str) -> int:
            return expect_integer(field(data, name, path), f"{path}.{name}")

        return cls(
            device_budget_bytes=integer("device_budget_bytes"),
            spill_budget_bytes=integer("spill_budget_bytes"),
            baseline_bytes=integer("baseline_bytes"),
            provider_headroom_bytes=integer("provider_headroom_bytes"),
            slab_bytes=integer("slab_bytes"),
            workspace_reserve_bytes=integer("workspace_reserve_bytes"),
            spill_reservation_bytes=integer("spill_reservation_bytes"),
            predicted_fragmentation_bytes=integer("predicted_fragmentation_bytes"),
        )


@dataclass(frozen=True, slots=True)
class PlanPrediction:
    device_peak_bytes: int
    spill_peak_bytes: int
    makespan_ns: int

    def __post_init__(self) -> None:
        require_non_negative(self.device_peak_bytes, "prediction.device_peak_bytes")
        require_non_negative(self.spill_peak_bytes, "prediction.spill_peak_bytes")
        require_non_negative(self.makespan_ns, "prediction.makespan_ns")

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "device_peak_bytes": self.device_peak_bytes,
            "spill_peak_bytes": self.spill_peak_bytes,
            "makespan_ns": self.makespan_ns,
        }

    @classmethod
    def from_value(cls, value: object, path: str) -> PlanPrediction:
        data = expect_mapping(value, path)
        return cls(
            device_peak_bytes=expect_integer(
                field(data, "device_peak_bytes", path),
                f"{path}.device_peak_bytes",
            ),
            spill_peak_bytes=expect_integer(
                field(data, "spill_peak_bytes", path), f"{path}.spill_peak_bytes"
            ),
            makespan_ns=expect_integer(
                field(data, "makespan_ns", path), f"{path}.makespan_ns"
            ),
        )


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    program: Program
    schedule: MemorySchedule
    selections: tuple[TaskAlternativeChoice, ...]
    entrypoints: tuple[EntrypointSpec, ...]
    admission: PhysicalAdmission
    prediction: PlanPrediction

    def __post_init__(self) -> None:
        require(isinstance(self.program, Program), "plan.program", "invalid program")
        require(
            isinstance(self.schedule, MemorySchedule),
            "plan.schedule",
            "invalid schedule",
        )
        require_tuple(self.selections, "plan.selections")
        require_tuple(self.entrypoints, "plan.entrypoints")
        active_tasks = self.program.selected_tasks(self.selections)
        self.schedule.validate(self.program, self.selections)
        entrypoint_tasks = index_unique(
            (entrypoint.task_id for entrypoint in self.entrypoints),
            "plan.entrypoints",
        )
        required = {task.task_id for task in active_tasks if task.requires_entrypoint}
        require(
            set(entrypoint_tasks) == required,
            "plan.entrypoints",
            "must bind every active task that requires an entrypoint exactly once",
        )
        require(
            self.prediction.device_peak_bytes <= self.admission.device_budget_bytes,
            "plan.prediction.device_peak_bytes",
            "exceeds physical device budget",
        )
        require(
            self.prediction.spill_peak_bytes <= self.admission.spill_budget_bytes,
            "plan.prediction.spill_peak_bytes",
            "exceeds host budget",
        )

    @property
    def digest(self) -> str:
        return digest_json(self.to_dict())

    @property
    def scheduling_digest(self) -> str:
        value: dict[str, JsonValue] = {
            "program": self.program.to_dict(),
            "schedule": self.schedule.to_dict(),
            "selections": [selection.to_dict() for selection in self.selections],
        }
        return digest_json(value)

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "admission": self.admission.to_dict(),
            "entrypoints": [entrypoint.to_dict() for entrypoint in self.entrypoints],
            "prediction": self.prediction.to_dict(),
            "program": self.program.to_dict(),
            "schedule": self.schedule.to_dict(),
            "schema": EXECUTION_PLAN_SCHEMA,
            "selections": [selection.to_dict() for selection in self.selections],
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> ExecutionPlan:
        data = expect_mapping(value, "plan")
        schema = expect_string(field(data, "schema", "plan"), "plan.schema")
        require(
            schema == EXECUTION_PLAN_SCHEMA,
            "plan.schema",
            f"unsupported schema {schema!r}",
        )
        selections = expect_list(field(data, "selections", "plan"), "plan.selections")
        entrypoints = expect_list(
            field(data, "entrypoints", "plan"), "plan.entrypoints"
        )
        return cls(
            program=Program.from_dict(field(data, "program", "plan")),
            schedule=MemorySchedule.from_dict(field(data, "schedule", "plan")),
            selections=tuple(
                TaskAlternativeChoice.from_value(item, f"plan.selections[{index}]")
                for index, item in enumerate(selections)
            ),
            entrypoints=tuple(
                EntrypointSpec.from_value(item, f"plan.entrypoints[{index}]")
                for index, item in enumerate(entrypoints)
            ),
            admission=PhysicalAdmission.from_value(
                field(data, "admission", "plan"), "plan.admission"
            ),
            prediction=PlanPrediction.from_value(
                field(data, "prediction", "plan"), "plan.prediction"
            ),
        )

    @classmethod
    def from_json(cls, payload: str) -> ExecutionPlan:
        return cls.from_dict(parse_json(payload))


__all__ = [
    "EntrypointSpec",
    "ExecutionPlan",
    "PhysicalAdmission",
    "PlanPrediction",
]
