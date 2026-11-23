"""Inspectable planning result; simulation evidence is not execution admission."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, ClassVar

from shadowspill.ir import MemorySchedule, ShadowSpillProgram, TaskAlternativeChoice
from shadowspill.serialization.sharing import shared_writer
from shadowspill.simulator import SimulationResult, TransferDirection

from .archive import Artifact
from .pool_memory import PoolMemoryProfile
from .profiled_program import ProfiledProgram
from .program import Program

if TYPE_CHECKING:
    from shadowspill.artifacts.budgets import MemoryBudgets
    from shadowspill.communication import CommunicationInfo
    from shadowspill.planning.memory_search.admission import AdmissionFacts
    from shadowspill.planning.memory_search.admission.layout import FixedPhysicalLayout
    from shadowspill.planning.memory_search.diagnostics import PlanningDiagnostics


@dataclass(frozen=True)
class Plan(Artifact):
    """A schedule for one Program within memory budgets, with its timeline.

    `plan_program` returns one, and ``Plan.load`` reads a saved one; an
    `Executor` runs it. A Plan holds its task code, so running it needs
    neither the model nor the build caches. A Plan whose
    `simulation_only_reasons` is not empty predicts a timeline but cannot run.
    A Plan of a forward/backward Program holds two schedules on one memory
    layout: one starting gradients fresh, one adding to the stored gradients.

    Attributes
    ----------
    profiled_program : ProfiledProgram
        The planned Program and its profiles.
    memory_budgets : MemoryBudgets
        The budgets it was planned within.
    communication_info : CommunicationInfo
        The transfer and collective costs it was planned with.
    schedule : MemorySchedule
        Where storage starts and ends, and the residency actions around each
        task.
    selections : tuple of TaskAlternativeChoice
        The option chosen in each task alternative group.
    timeline : SimulationResult
        The simulated invocation: task and transfer intervals and memory
        peaks.
    diagnostics : PlanningDiagnostics
        What the search did.
    memory_layout : FixedPhysicalLayout or None
        Each allocation's place in the execution pool, certified against the
        schedule; None when the memory evidence does not allow one.
    pool_memory : PoolMemoryProfile or None
        The spill pool memory measurement it was planned with.
    reserved_spill_bytes : int
        Spill held for persistent state the other operations of its run keep
        while it runs; it was planned within the spill budget less this.
    """

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
    pool_memory: PoolMemoryProfile | None = None
    reserved_spill_bytes: int = 0

    def __post_init__(self) -> None:
        if (
            type(self.reserved_spill_bytes) is not int
            or not 0 <= self.reserved_spill_bytes <= self.memory_budgets.spill_bytes
        ):
            raise ValueError(
                "reserved_spill_bytes must be an int from 0 to the spill budget"
            )
        if self.pool_memory is not None:
            self.pool_memory.validate_program(
                self.memory_budgets.spill_bytes,
                1
                if self.program.parallelism is None
                else self.program.parallelism.world_size,
                gpu=self.program.setup.get("gpu"),
                spill_kind=self.communication_info.metadata.get("spill_kind"),
            )
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
                > self.effective_execution_bytes
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
                or self.pool_memory != other.pool_memory
                or self.reserved_spill_bytes != other.reserved_spill_bytes
            ):
                raise ValueError("incompatible gradient-accumulation execution path")

    @property
    def program(self) -> Program:
        """The planned `Program`."""
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
        """Device memory the worker needs outside the execution pool, in bytes.

        The profiled overhead, communicators and collective workspaces, and
        the spill pool's measured process reservation.
        """
        return (
            self.profiled_program.fixed_device_bytes
            + self._communication_memory[0]
            + (
                0
                if self.pool_memory is None
                else self.pool_memory.process_reservation_bytes
            )
        )

    @property
    def effective_execution_bytes(self) -> int:
        """The device budget, capped by `pool_memory`'s available memory, in bytes."""
        return (
            min(
                self.memory_budgets.execution_bytes,
                self.pool_memory.physical_limit_bytes,
            )
            if self.pool_memory is not None
            else self.memory_budgets.execution_bytes
        )

    @property
    def simulation_only_reasons(self) -> tuple[str, ...]:
        """Why the Plan cannot run; empty when it can.

        Estimated memory, unresolved communication memory, absent task code or
        an absent memory layout, in this Plan or its accumulating schedule.
        """
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
        """Headline figures of the simulated invocation, as a dict.

        ``duration_seconds``, the simulated time; ``execution_peak_bytes``,
        the device peak with the fixed overhead, and ``spill_peak_bytes``;
        ``fixed_device_bytes`` and its ``communication_fixed_bytes`` part;
        ``declared_execution_bytes`` and ``effective_execution_bytes``;
        ``pool_process_reservation_bytes``; ``fetch_bytes`` and
        ``evict_bytes``, each with its ``_host_bytes`` and
        ``_collective_bytes`` parts; ``simulation_only``; and
        ``logical_ranks``.
        """
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
            "declared_execution_bytes": self.memory_budgets.execution_bytes,
            "effective_execution_bytes": self.effective_execution_bytes,
            "pool_process_reservation_bytes": 0
            if self.pool_memory is None
            else self.pool_memory.process_reservation_bytes,
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

    @shared_writer
    def to_dict(self) -> dict[str, Any]:
        """Return the Plan as plain data.

        Returns
        -------
        dict
            Every field as plain records, with the Program's code named by
            SHA-256 hex digest.
        """
        from shadowspill.planning.serialization.simulation_result import (
            simulation_result_value,
        )

        return {
            "profiled_program": self.profiled_program.to_dict(),
            "pool_memory": None
            if self.pool_memory is None
            else self.pool_memory.to_dict(),
            "memory_budgets": self.memory_budgets.to_dict(),
            "reserved_spill_bytes": self.reserved_spill_bytes,
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
        """Rebuild a Plan from `to_dict` data and its binary payloads.

        Parameters
        ----------
        value : dict
            The Plan, in `to_dict` form.
        payloads : mapping of str to bytes, optional
            The Program's binary payloads, by SHA-256 hex digest.

        Returns
        -------
        Plan
            The Plan.

        Raises
        ------
        ValueError
            If a payload is missing, or the Plan is invalid.
        """
        from shadowspill.artifacts.budgets import MemoryBudgets
        from shadowspill.communication import CommunicationInfo
        from shadowspill.planning.memory_search.admission import AdmissionFacts
        from shadowspill.planning.memory_search.diagnostics import PlanningDiagnostics
        from shadowspill.planning.serialization import _simulation_result_from_value
        from shadowspill.planning.serialization.physical_layout import (
            _fixed_layout_from_value,
        )

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
            None
            if value.get("pool_memory") is None
            else PoolMemoryProfile.from_dict(value["pool_memory"]),
            value["reserved_spill_bytes"],
        )
