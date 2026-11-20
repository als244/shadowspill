"""Resolved execution-plan records consumed by runtime admission."""

from __future__ import annotations

from dataclasses import dataclass

from shadowspill.schema import artifact_schema

from .program import ShadowSpillProgram, TaskAlternativeChoice
from .schedule import MemorySchedule
from .serialization import JsonValue, canonical_json, digest_json, parse_json
from .validation import (
    expect_boolean,
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
    """Which function runs one active task.

    Parameters
    ----------
    task_id : str
        The task.
    entrypoint_id : str
        The function's identity.
    executor_id : str
        Identity of the executor that runs the function.
    contract_digest : str
        Digest of the task contract the function was compiled against.

    Raises
    ------
    ValidationError
        If an identity is empty or padded with whitespace.
    """

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
        """Return the record as JSON values.

        Returns
        -------
        dict
            Its fields by name.
        """
        return {
            "contract_digest": self.contract_digest,
            "entrypoint_id": self.entrypoint_id,
            "executor_id": self.executor_id,
            "task_id": self.task_id,
        }

    @classmethod
    def from_value(cls, value: object, path: str) -> EntrypointSpec:
        """Read a record from its `to_dict` form.

        Parameters
        ----------
        value : object
            The decoded JSON value.
        path : str
            Where the value sits in its document, such as
            ``"plan.entrypoints[0]"``.
            Error paths start with it.

        Returns
        -------
        EntrypointSpec
            The record.

        Raises
        ------
        ValidationError
            If a field is missing, has the wrong type or is invalid.
        """
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
    """The memory one plan is admitted under, in bytes.

    Parameters
    ----------
    device_budget_bytes : int
        Device memory the process may use: at least ``baseline_bytes``,
        ``external_headroom_bytes`` and ``slab_bytes`` together.
    spill_budget_bytes : int
        Spill memory the plan may use.
    baseline_bytes : int
        Device memory the process holds before the execution pool exists.
    external_headroom_bytes : int
        Allowance for device memory the process uses outside the execution pool.
    slab_bytes : int
        Size of the execution pool the plan's layout is placed in.
    workspace_reserve_bytes : int
        Contiguous task workspace the pool must be able to serve; at most
        ``slab_bytes``, and not subtracted from it.
    spill_reservation_bytes : int
        Spill memory reserved for the plan; at most ``spill_budget_bytes``.
    predicted_fragmentation_bytes : int, default 0
        Declared estimate of slab bytes the plan's lifetimes leave unusable; at
        most ``slab_bytes``.
    reject_overbudget : bool, default False
        Whether external or whole-process memory beyond the budget is refused
        rather than reported.

    Raises
    ------
    ValidationError
        If a size is negative, the baseline, headroom and slab exceed
        ``device_budget_bytes``, a reserve exceeds what holds it, or
        ``reject_overbudget`` is not a bool.
    """

    device_budget_bytes: int
    spill_budget_bytes: int
    baseline_bytes: int
    external_headroom_bytes: int
    slab_bytes: int
    workspace_reserve_bytes: int
    spill_reservation_bytes: int
    predicted_fragmentation_bytes: int = 0
    reject_overbudget: bool = False

    def __post_init__(self) -> None:
        require(
            isinstance(self.reject_overbudget, bool),
            "admission.reject_overbudget",
            "must be a bool",
        )
        byte_fields = (
            ("device_budget_bytes", self.device_budget_bytes),
            ("spill_budget_bytes", self.spill_budget_bytes),
            ("baseline_bytes", self.baseline_bytes),
            ("external_headroom_bytes", self.external_headroom_bytes),
            ("slab_bytes", self.slab_bytes),
            ("workspace_reserve_bytes", self.workspace_reserve_bytes),
            ("spill_reservation_bytes", self.spill_reservation_bytes),
            ("predicted_fragmentation_bytes", self.predicted_fragmentation_bytes),
        )
        for name, value in byte_fields:
            require_non_negative(value, f"admission.{name}")
        require(
            self.baseline_bytes + self.external_headroom_bytes + self.slab_bytes
            <= self.device_budget_bytes,
            "admission.device_budget_bytes",
            "baseline, external headroom, and slab exceed the physical cap",
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
        """Return the record as JSON values.

        Returns
        -------
        dict
            Its fields by name.
        """
        return {
            "baseline_bytes": self.baseline_bytes,
            "device_budget_bytes": self.device_budget_bytes,
            "spill_budget_bytes": self.spill_budget_bytes,
            "spill_reservation_bytes": self.spill_reservation_bytes,
            "predicted_fragmentation_bytes": self.predicted_fragmentation_bytes,
            "external_headroom_bytes": self.external_headroom_bytes,
            "reject_overbudget": self.reject_overbudget,
            "slab_bytes": self.slab_bytes,
            "workspace_reserve_bytes": self.workspace_reserve_bytes,
        }

    @classmethod
    def from_value(cls, value: object, path: str) -> PhysicalAdmission:
        """Read a record from its `to_dict` form.

        Parameters
        ----------
        value : object
            The decoded JSON value.
        path : str
            Where the value sits in its document, such as
            ``"plan.admission"``.
            Error paths start with it.

        Returns
        -------
        PhysicalAdmission
            The record.

        Raises
        ------
        ValidationError
            If a field is missing, has the wrong type or is invalid.
        """
        data = expect_mapping(value, path)

        def integer(name: str) -> int:
            return expect_integer(field(data, name, path), f"{path}.{name}")

        return cls(
            device_budget_bytes=integer("device_budget_bytes"),
            spill_budget_bytes=integer("spill_budget_bytes"),
            baseline_bytes=integer("baseline_bytes"),
            external_headroom_bytes=integer("external_headroom_bytes"),
            reject_overbudget=expect_boolean(
                field(data, "reject_overbudget", path), f"{path}.reject_overbudget"
            ),
            slab_bytes=integer("slab_bytes"),
            workspace_reserve_bytes=integer("workspace_reserve_bytes"),
            spill_reservation_bytes=integer("spill_reservation_bytes"),
            predicted_fragmentation_bytes=integer("predicted_fragmentation_bytes"),
        )


@dataclass(frozen=True, slots=True)
class PlanPrediction:
    """What a plan is expected to use and take.

    Parameters
    ----------
    device_peak_bytes : int
        Peak device memory of the process, in bytes.
    spill_peak_bytes : int
        Peak spill memory, in bytes.
    makespan_ns : int
        Duration of the plan, in nanoseconds.

    Raises
    ------
    ValidationError
        If a value is negative.
    """

    device_peak_bytes: int
    spill_peak_bytes: int
    makespan_ns: int

    def __post_init__(self) -> None:
        require_non_negative(self.device_peak_bytes, "prediction.device_peak_bytes")
        require_non_negative(self.spill_peak_bytes, "prediction.spill_peak_bytes")
        require_non_negative(self.makespan_ns, "prediction.makespan_ns")

    def to_dict(self) -> dict[str, JsonValue]:
        """Return the record as JSON values.

        Returns
        -------
        dict
            Its fields by name.
        """
        return {
            "device_peak_bytes": self.device_peak_bytes,
            "spill_peak_bytes": self.spill_peak_bytes,
            "makespan_ns": self.makespan_ns,
        }

    @classmethod
    def from_value(cls, value: object, path: str) -> PlanPrediction:
        """Read a record from its `to_dict` form.

        Parameters
        ----------
        value : object
            The decoded JSON value.
        path : str
            Where the value sits in its document, such as
            ``"plan.prediction"``.
            Error paths start with it.

        Returns
        -------
        PlanPrediction
            The record.

        Raises
        ------
        ValidationError
            If a field is missing, has the wrong type or is invalid.
        """
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
    """A schedule made runnable: alternatives fixed, functions bound, memory admitted.

    `ProgramPlanResult.to_execution_plan` builds one from a search's answer.

    Parameters
    ----------
    program : ShadowSpillProgram
        The program.
    schedule : MemorySchedule
        The schedule over it.
    selections : tuple of TaskAlternativeChoice
        One choice for every alternative group of the program.
    entrypoints : tuple of EntrypointSpec
        One for each active task that requires an entrypoint, and no others.
    admission : PhysicalAdmission
        The memory the plan is admitted under.
    prediction : PlanPrediction
        What the plan is expected to use and take.

    Raises
    ------
    ValidationError
        If the schedule cannot run on the selected tasks, the entrypoints do not
        match the tasks that require one, or a predicted peak exceeds its
        budget.
    """

    program: ShadowSpillProgram
    schedule: MemorySchedule
    selections: tuple[TaskAlternativeChoice, ...]
    entrypoints: tuple[EntrypointSpec, ...]
    admission: PhysicalAdmission
    prediction: PlanPrediction

    def __post_init__(self) -> None:
        require(
            isinstance(self.program, ShadowSpillProgram),
            "plan.program",
            "invalid program",
        )
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
        """SHA-256 of the plan's canonical JSON, in hexadecimal."""
        return digest_json(self.to_dict())

    @property
    def scheduling_digest(self) -> str:
        """SHA-256 over the program, schedule and selections, in hexadecimal.

        Plans that differ only in entrypoints, admission or prediction share it.
        """
        value: dict[str, JsonValue] = {
            "program": self.program.to_dict(),
            "schedule": self.schedule.to_dict(),
            "selections": [selection.to_dict() for selection in self.selections],
        }
        return digest_json(value)

    def to_dict(self) -> dict[str, JsonValue]:
        """Return the plan as JSON values.

        Returns
        -------
        dict
            Its parts by name, and its schema label.
        """
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
        """Return the plan's canonical JSON.

        Returns
        -------
        str
            The `to_dict` form with sorted keys and no whitespace.
        """
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> ExecutionPlan:
        """Read a plan from its `to_dict` form.

        Parameters
        ----------
        value : mapping
            The decoded JSON value.

        Returns
        -------
        ExecutionPlan
            The plan.

        Raises
        ------
        ValidationError
            If the schema label is not this version's, a record is malformed, or the
            parts do not agree.
        """
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
            program=ShadowSpillProgram.from_dict(field(data, "program", "plan")),
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
        """Read a plan from its JSON text.

        Parameters
        ----------
        payload : str
            The text, as `to_json` writes it.

        Returns
        -------
        ExecutionPlan
            The plan.

        Raises
        ------
        ValueError
            If ``payload`` is not JSON.
        ValidationError
            If the decoded record is invalid, as for `from_dict`.
        """
        return cls.from_dict(parse_json(payload))


__all__ = [
    "EntrypointSpec",
    "ExecutionPlan",
    "PhysicalAdmission",
    "PlanPrediction",
]
