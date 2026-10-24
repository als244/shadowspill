"""Inspectable planning result; simulation evidence is not execution admission."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from shadowspill.ir import MemorySchedule, ShadowSpillProgram, TaskAlternativeChoice
from shadowspill.simulator import SimulationResult, TransferDirection

from .archive import Artifact
from .profiled_program import ProfiledProgram
from .program import Program

if TYPE_CHECKING:
    from shadowspill.communication import CommunicationInfo
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
    memory_layout: None = None

    @property
    def program(self) -> Program:
        return self.profiled_program.program

    @property
    def simulation_only_reasons(self) -> tuple[str, ...]:
        reasons = tuple(
            f"estimated memory: {key}" for key in self.profiled_program.estimated_memory
        )
        # Workspace totals support logical simulation. Do not fabricate allocator
        # lifetimes, physical placement or executable task payloads from totals.
        return (
            *reasons,
            "executable task payloads and physical allocation evidence absent",
        )

    @property
    def summary(self) -> dict[str, Any]:
        result = self.timeline
        return {
            "duration_seconds": result.makespan_ns / 1e9,
            "execution_peak_bytes": max(p.total_bytes for p in result.device_peaks)
            + self.profiled_program.fixed_device_bytes,
            "spill_peak_bytes": result.spill_peak_bytes,
            "fetch_bytes": sum(
                t.bytes
                for t in result.transfer_intervals
                if t.direction == TransferDirection.FETCH
            ),
            "evict_bytes": sum(
                t.bytes
                for t in result.transfer_intervals
                if t.direction == TransferDirection.EVICT
            ),
            "simulation_only": bool(self.simulation_only_reasons),
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
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Plan:
        from shadowspill.communication import CommunicationInfo
        from shadowspill.planner.diagnostics import PlanningDiagnostics
        from shadowspill.planner.serialization import _simulation_result_from_value
        from shadowspill.planning.budgets import MemoryBudgets

        return cls(
            ProfiledProgram.from_dict(value["profiled_program"]),
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
        )
