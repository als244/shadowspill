"""Physically admitted PressureFit selection and simulator evidence."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING

from shadowspill.ir import MemorySchedule, RecomputationSelection, ResidencySpec
from shadowspill.planner import AdmissionTopology, PressureFitResult
from shadowspill.simulator import SimulationAdmission, SimulationResult

from .program_inputs import MemoryBudgets, PressureFitProgram, TransferBandwidths
from .program_serialization import (
    _boolean,
    _canonical_json,
    _digest,
    _fixed_layout_from_value,
    _integer,
    _list,
    _mapping,
    _options_from_value,
    _options_to_dict,
    _pressurefit_diagnostics_from_value,
    _simulation_admission_from_value,
    _simulation_result_from_value,
    _string,
)

if TYPE_CHECKING:
    from .planning.admission import FixedLayoutAttempt, FixedPhysicalLayout

_ANNOTATED_PROGRAM_PLAN_SCHEMA = "shadowspill.annotated_program_plan/v1"


@dataclass(frozen=True, slots=True)
class AnnotatedProgramPlan:
    """PressureFit winner plus exact fixed-layout and simulator evidence."""

    program: PressureFitProgram
    memory_budgets: MemoryBudgets
    transfer_bandwidths: TransferBandwidths
    result: PressureFitResult
    effective_topology: AdmissionTopology
    fixed_layout: FixedPhysicalLayout
    simulation_admission: SimulationAdmission
    simulation: SimulationResult
    attempts: tuple[FixedLayoutAttempt, ...]
    pressurefit_cache_hit: bool
    wall_time_ns: int

    @property
    def digest(self) -> str:
        """Stable selected-plan identity, excluding cache and wall-time evidence."""

        value = self.to_dict()
        selection = dict(_mapping(value["selection"], "selection"))
        selection.pop("cache_hit")
        value["selection"] = selection
        value.pop("timing")
        return _digest(value)

    @property
    def execution_budget_bytes(self) -> int:
        """Compatibility shorthand for the physical execution budget."""

        return self.memory_budgets.execution_bytes

    @property
    def spill_budget_bytes(self) -> int:
        """Compatibility shorthand for the physical spill budget."""

        return self.memory_budgets.spill_bytes

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": _ANNOTATED_PROGRAM_PLAN_SCHEMA,
            "source_program": self.program.to_dict(),
            "memory_budgets": self.memory_budgets.to_dict(),
            "transfer_bandwidths": self.transfer_bandwidths.to_dict(),
            "selection": {
                "cache_hit": self.pressurefit_cache_hit,
                "diagnostics": asdict(self.result.diagnostics),
                "final_residency": [
                    item.to_dict() for item in self.result.final_residency
                ],
                "initial_residency": [
                    item.to_dict() for item in self.result.initial_residency
                ],
                "options": _options_to_dict(self.result.options),
                "schedule": self.result.schedule.to_dict(),
                "selections": [item.to_dict() for item in self.result.selections],
            },
            "simulation": {
                "result": asdict(self.simulation),
                "admission": asdict(self.simulation_admission),
            },
            "physical_admission": {
                "effective_topology": self.effective_topology.to_dict(),
                "fixed_layout": self.fixed_layout.to_dict(),
                "fixed_layout_digest": self.fixed_layout.digest,
                "attempts": [asdict(item) for item in self.attempts],
            },
            "timing": {"pressurefit_and_admission_wall_time_ns": self.wall_time_ns},
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> AnnotatedProgramPlan:
        from .planning.admission import FixedLayoutAttempt

        data = _mapping(value, "annotated_program_plan")
        if data.get("schema") != _ANNOTATED_PROGRAM_PLAN_SCHEMA:
            raise ValueError("annotated_program_plan.schema: unsupported schema")
        program = PressureFitProgram.from_value(
            data.get("source_program"), "annotated_program_plan.source_program"
        )
        budgets = MemoryBudgets.from_value(
            data.get("memory_budgets"),
            "annotated_program_plan.memory_budgets",
        )
        selection = _mapping(data.get("selection"), "annotated_program_plan.selection")
        simulation_value = _mapping(
            data.get("simulation"), "annotated_program_plan.simulation"
        )
        physical = _mapping(
            data.get("physical_admission"),
            "annotated_program_plan.physical_admission",
        )
        timing = _mapping(data.get("timing"), "annotated_program_plan.timing")
        topology = AdmissionTopology.from_dict(physical.get("effective_topology"))
        layout = _fixed_layout_from_value(
            physical.get("fixed_layout"),
            "annotated_program_plan.physical_admission.fixed_layout",
        )
        expected_layout_digest = _string(
            physical.get("fixed_layout_digest"),
            "annotated_program_plan.physical_admission.fixed_layout_digest",
        )
        if layout.digest != expected_layout_digest:
            raise ValueError("annotated physical-layout digest mismatch")
        simulation = _simulation_result_from_value(
            simulation_value.get("result"),
            "annotated_program_plan.simulation.result",
        )
        simulation_admission = _simulation_admission_from_value(
            simulation_value.get("admission"),
            "annotated_program_plan.simulation.admission",
        )
        schedule = MemorySchedule.from_dict(selection.get("schedule"))
        selection_values = _list(
            selection.get("selections"), "annotated_program_plan.selection.selections"
        )
        selections = tuple(
            RecomputationSelection.from_value(
                item, f"annotated_program_plan.selection.selections[{index}]"
            )
            for index, item in enumerate(selection_values)
        )
        schedule.validate(program.program, selections)
        diagnostics = _pressurefit_diagnostics_from_value(
            selection.get("diagnostics"),
            "annotated_program_plan.selection.diagnostics",
        )
        initial_residency = tuple(
            ResidencySpec.from_value(
                item,
                f"annotated_program_plan.selection.initial_residency[{index}]",
            )
            for index, item in enumerate(
                _list(
                    selection.get("initial_residency"),
                    "annotated_program_plan.selection.initial_residency",
                )
            )
        )
        final_residency = tuple(
            ResidencySpec.from_value(
                item,
                f"annotated_program_plan.selection.final_residency[{index}]",
            )
            for index, item in enumerate(
                _list(
                    selection.get("final_residency"),
                    "annotated_program_plan.selection.final_residency",
                )
            )
        )
        if initial_residency != program.initial_residency:
            raise ValueError("annotated initial residency differs from source Program")
        if final_residency != program.final_residency:
            raise ValueError("annotated final residency differs from source Program")
        config, _requested_topology = program.pressurefit_inputs(
            execution_budget_bytes=budgets.execution_bytes,
            spill_budget_bytes=budgets.spill_bytes,
            transfer_bandwidths=TransferBandwidths.from_value(
                data.get("transfer_bandwidths"),
                "annotated_program_plan.transfer_bandwidths",
            ),
        )
        if layout.program_digest != program.program.digest:
            raise ValueError("annotated layout names a different Program")
        if layout.schedule_digest != schedule.digest:
            raise ValueError("annotated layout names a different schedule")
        if layout.topology_digest != topology.digest:
            raise ValueError("annotated layout names a different topology")
        if diagnostics.selected_makespan_ns != simulation.makespan_ns:
            raise ValueError("annotated diagnostics and simulation makespans differ")
        result = PressureFitResult(
            program=program.program,
            options=_options_from_value(
                selection.get("options"),
                "annotated_program_plan.selection.options",
            ),
            initial_residency=initial_residency,
            final_residency=final_residency,
            simulation_config=replace(
                config,
                devices=tuple(
                    replace(item, capacity_bytes=topology.object_capacity_bytes)
                    if item.device_id == topology.device_id
                    else item
                    for item in config.devices
                ),
            ),
            schedule=schedule,
            selections=selections,
            simulation=simulation,
            diagnostics=diagnostics,
            admission_topology=topology,
        )
        attempts_value = _list(
            physical.get("attempts"),
            "annotated_program_plan.physical_admission.attempts",
        )
        transfer = TransferBandwidths.from_value(
            data.get("transfer_bandwidths"),
            "annotated_program_plan.transfer_bandwidths",
        )
        return cls(
            program=program,
            memory_budgets=budgets,
            transfer_bandwidths=transfer,
            result=result,
            effective_topology=topology,
            fixed_layout=layout,
            simulation_admission=simulation_admission,
            simulation=simulation,
            attempts=tuple(
                FixedLayoutAttempt(
                    requested_object_capacity_bytes=_integer(
                        item.get("requested_object_capacity_bytes"),
                        "annotated_program_plan.physical_admission."
                        f"attempts[{index}].requested_object_capacity_bytes",
                    ),
                    effective_object_capacity_bytes=_integer(
                        item.get("effective_object_capacity_bytes"),
                        "annotated_program_plan.physical_admission."
                        f"attempts[{index}].effective_object_capacity_bytes",
                    ),
                    required_bytes=_integer(
                        item.get("required_bytes"),
                        "annotated_program_plan.physical_admission."
                        f"attempts[{index}].required_bytes",
                    ),
                    pool_capacity_bytes=_integer(
                        item.get("pool_capacity_bytes"),
                        "annotated_program_plan.physical_admission."
                        f"attempts[{index}].pool_capacity_bytes",
                    ),
                    accepted=_boolean(
                        item.get("accepted"),
                        "annotated_program_plan.physical_admission."
                        f"attempts[{index}].accepted",
                    ),
                )
                for index, raw in enumerate(attempts_value)
                for item in (
                    _mapping(
                        raw,
                        f"annotated_program_plan.physical_admission.attempts[{index}]",
                    ),
                )
            ),
            pressurefit_cache_hit=_boolean(
                selection.get("cache_hit"),
                "annotated_program_plan.selection.cache_hit",
            ),
            wall_time_ns=_integer(
                timing.get("pressurefit_and_admission_wall_time_ns"),
                "annotated_program_plan.timing.pressurefit_and_admission_wall_time_ns",
            ),
        )

    @classmethod
    def from_json(cls, payload: str) -> AnnotatedProgramPlan:
        try:
            value = json.loads(payload)
        except json.JSONDecodeError as error:
            raise ValueError("annotated Program plan JSON is invalid") from error
        return cls.from_dict(value)
