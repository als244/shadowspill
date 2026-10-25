"""Inspectable planning result; simulation evidence is not execution admission."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, ClassVar

from shadowspill.ir import MemorySchedule, ShadowSpillProgram, TaskAlternativeChoice
from shadowspill.simulator import SimulationResult, TransferDirection

from .archive import Artifact
from .profiled_program import ProfiledProgram
from .program import Program

if TYPE_CHECKING:
    from shadowspill.communication import CommunicationInfo
    from shadowspill.planner.admission import AdmissionFacts
    from shadowspill.planner.admission.layout import FixedPhysicalLayout
    from shadowspill.planner.diagnostics import PlanningDiagnostics
    from shadowspill.planning.budgets import MemoryBudgets


@dataclass(frozen=True)
class Plan(Artifact):
    artifact_kind: ClassVar[str] = "plan"
    profiled_program: ProfiledProgram
    memory_budgets: MemoryBudgets
    communication_info: CommunicationInfo
    schedule: MemorySchedule
    selections: tuple[TaskAlternativeChoice, ...]
    timeline: SimulationResult
    diagnostics: PlanningDiagnostics
    _costed_program: ShadowSpillProgram
    memory_layout: FixedPhysicalLayout | None = None
    _admission_facts: AdmissionFacts | None = None
    _accumulation_plan: Plan | None = None

    def __post_init__(self) -> None:
        layout, facts = self.memory_layout, self._admission_facts
        if (layout is None) != (facts is None):
            raise ValueError("physical layout and admission facts must travel together")
        if layout is not None:
            assert facts is not None
            if (
                layout.program_digest != self._costed_program.digest
                or layout.schedule_digest != self.schedule.digest
                or layout.facts_digest != facts.digest
            ):
                raise ValueError("physical layout belongs to different plan inputs")
            if self.profiled_program.estimated_memory:
                raise ValueError("estimated memory cannot carry a physical layout")
            if (
                layout.required_bytes + self.fixed_device_bytes
                > self.memory_budgets.execution_bytes
            ):
                raise ValueError("physical layout exceeds the total GPU allowance")

        if self._accumulation_plan is not None:
            other = self._accumulation_plan
            if (
                self.program.bindings.get("kind") != "forward_backward"
                or other._accumulation_plan is not None
                or not other.program.bindings.get("accumulate_gradients")
                or self.program.objects != other.program.objects
                or self.program.alias_groups != other.program.alias_groups
                or self.program.setup != other.program.setup
                or self.memory_budgets != other.memory_budgets
            ):
                raise ValueError("incompatible gradient-accumulation execution path")

    @property
    def program(self) -> Program:
        return self.profiled_program.program

    @cached_property
    def _communication_memory(self) -> tuple[int, tuple[str, ...]]:
        from shadowspill.communication.contracts import collective_measurement
        from shadowspill.communication.memory import (
            collective_memory,
            communication_overhead,
        )
        from shadowspill.communication.task_costs import task_collectives

        fixed, overhead_reasons = communication_overhead(
            self.program, self.communication_info
        )
        reasons = list(overhead_reasons)
        for key, requests in task_collectives(self.program).items():
            profile = self.profiled_program.task_profiles[key]
            _, task_reasons = collective_memory(
                requests,
                self.communication_info,
                self.program.setup,
                fallback_bytes=profile.workspace_bytes,
                fallback_kind=profile.metadata.get("memory_kind", "estimated"),
            )
            reasons.extend(task_reasons)
            if self.program.executables:
                try:
                    collective_measurement(self.program, key, self.communication_info)
                except ValueError as error:
                    reasons.append(str(error))
        return fixed, tuple(dict.fromkeys(reasons))

    @property
    def fixed_device_bytes(self) -> int:
        """Compute/provider overhead plus communication resources and reserves."""
        return self.profiled_program.fixed_device_bytes + self._communication_memory[0]

    @property
    def simulation_only_reasons(self) -> tuple[str, ...]:
        reasons = tuple(
            f"estimated memory: {key}" for key in self.profiled_program.estimated_memory
        )
        # Workspace totals support logical simulation. Do not fabricate allocator
        # lifetimes, physical placement or executable task payloads from totals.
        missing_code = not self.program.executables
        return (
            *reasons,
            *self._communication_memory[1],
            *(
                tuple(
                    f"accumulation: {r}"
                    for r in self._accumulation_plan.simulation_only_reasons
                )
                if self._accumulation_plan is not None
                else ()
            ),
            *(("executable task payloads absent",) if missing_code else ()),
            *(
                ("physical allocation evidence absent",)
                if self.memory_layout is None
                else ()
            ),
        )

    @property
    def summary(self) -> dict[str, Any]:
        result = self.timeline
        aliases = {
            item.alias_group_id: item for item in self._costed_program.alias_groups
        }
        volumes = {}
        for direction in (TransferDirection.FETCH, TransferDirection.EVICT):
            intervals = [
                item
                for item in result.transfer_intervals
                if item.direction == direction
            ]
            host = sum(item.bytes for item in intervals)
            collective = 0
            for item in intervals:
                transfer = aliases[item.alias_group_id].transfer
                if transfer is not None:
                    collective += sum(
                        phase.bytes
                        for phase in getattr(transfer, direction.value)
                        if phase.kind == "timed"
                    )
            volumes[f"{direction.value}_bytes"] = host + collective
            volumes[f"{direction.value}_host_bytes"] = host
            volumes[f"{direction.value}_collective_bytes"] = collective
        return {
            "duration_seconds": result.makespan_ns / 1e9,
            "execution_peak_bytes": max(p.total_bytes for p in result.device_peaks)
            + self.fixed_device_bytes,
            "fixed_device_bytes": self.fixed_device_bytes,
            "communication_fixed_bytes": self._communication_memory[0],
            "spill_peak_bytes": result.spill_peak_bytes,
            **volumes,
            "simulation_only": bool(self.simulation_only_reasons),
            "logical_ranks": 1
            if self.program.parallelism is None
            else self.program.parallelism.world_size,
        }

    def to_dict(self) -> dict[str, Any]:
        from shadowspill.planner.serialization.simulation_result import (
            simulation_result_value,
        )

        return {
            "profiled_program": self.profiled_program.to_dict(),
            "memory_budgets": self.memory_budgets.to_dict(),
            "communication_info": self.communication_info.to_dict(),
            "schedule": self.schedule.to_dict(),
            "selections": [item.to_dict() for item in self.selections],
            "timeline": simulation_result_value(self.timeline),
            "diagnostics": self.diagnostics.to_dict(),
            "costed_program": self._costed_program.to_dict(),
            "memory_layout": None
            if self.memory_layout is None
            else self.memory_layout.to_dict(),
            "accumulation_plan": None
            if self._accumulation_plan is None
            else self._accumulation_plan.to_dict(),
            "admission_facts": None
            if self._admission_facts is None
            else self._admission_facts.to_dict(),
        }

    def _archive_payloads(self) -> Mapping[str, bytes]:
        payloads = dict(self.program._archive_payloads())
        if self._accumulation_plan is not None:
            payloads.update(self._accumulation_plan._archive_payloads())
        return payloads

    @classmethod
    def from_dict(
        cls, value: dict[str, Any], *, payloads: Mapping[str, bytes] | None = None
    ) -> Plan:
        from shadowspill.communication import CommunicationInfo
        from shadowspill.planner.admission import AdmissionFacts
        from shadowspill.planner.diagnostics import PlanningDiagnostics
        from shadowspill.planner.serialization import _simulation_result_from_value
        from shadowspill.planner.serialization.physical_layout import (
            _fixed_layout_from_value,
        )
        from shadowspill.planning.budgets import MemoryBudgets

        return cls(
            ProfiledProgram.from_dict(value["profiled_program"], payloads=payloads),
            MemoryBudgets(**value["memory_budgets"]),
            CommunicationInfo.from_dict(value["communication_info"]),
            MemorySchedule.from_dict(value["schedule"]),
            tuple(
                TaskAlternativeChoice.from_value(item, "plan.selections")
                for item in value["selections"]
            ),
            _simulation_result_from_value(value["timeline"], "plan.timeline"),
            PlanningDiagnostics.from_value(value["diagnostics"], "plan.diagnostics"),
            ShadowSpillProgram.from_dict(value["costed_program"]),
            (
                None
                if value.get("memory_layout") is None
                else _fixed_layout_from_value(
                    value["memory_layout"], "plan.memory_layout"
                )
            ),
            (
                None
                if value.get("admission_facts") is None
                else AdmissionFacts.from_dict(value["admission_facts"])
            ),
            (
                None
                if value.get("accumulation_plan") is None
                else cls.from_dict(value["accumulation_plan"], payloads=payloads)
            ),
        )
