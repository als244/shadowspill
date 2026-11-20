"""What a search gives back, including the two ways it can decline."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from shadowspill.errors import PlanInfeasibleError, PlanSearchExhaustedError
from shadowspill.ir import (
    EntrypointSpec,
    ExecutionPlan,
    MemorySchedule,
    PhysicalAdmission,
    PlanPrediction,
    ResidencySpec,
    ShadowSpillProgram,
    TaskAlternativeChoice,
)
from shadowspill.planning.memory_search.diagnostics import PlanningDiagnostics
from shadowspill.simulator import SimulationConfig, SimulationResult

if TYPE_CHECKING:
    from shadowspill.planning.memory_search.admission import AdmissionFacts
    from shadowspill.planning.memory_search.admission.layout.model import (
        FixedLayoutAdmission,
    )
    from shadowspill.planning.memory_search.algorithm import SearchOptions


@dataclass(frozen=True, slots=True)
class ResidentSlice:
    """The part of the fixed layout reserved for objects that stay resident.

    Objects under `GenericPlanningOptions.minimum_object_bytes_evict_eligible`
    are never evicted, so each of their leases gets a fixed home in a slice
    at the end of the layout. The planner sizes the slice before the search
    and takes it out of the capacity the search plans against.

    Parameters
    ----------
    bytes : int
        Bytes reserved: the sum of those homes. At least zero.
    aliases : tuple of str
        The alias groups whose leases the slice holds, sorted and distinct.

    Raises
    ------
    ValueError
        If ``bytes`` is negative, or ``aliases`` is not sorted and distinct.
    """

    bytes: int
    aliases: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.bytes < 0:
            raise ValueError("resident slice bytes must be non-negative")
        if self.aliases != tuple(sorted(set(self.aliases))):
            raise ValueError("resident slice aliases must be sorted and distinct")

    def to_dict(self) -> dict[str, object]:
        """Write the slice as a JSON-compatible mapping.

        Returns
        -------
        dict
            ``bytes``, and ``aliases`` as a list.
        """
        return {"bytes": self.bytes, "aliases": list(self.aliases)}


@dataclass(frozen=True, slots=True)
class ResolutionPlan:
    """The best plan one resolved program reached, kept beside the answer.

    A search fixes the program's alternatives several ways and answers with
    one plan. When asked, it keeps what each way found, decoded and
    simulated like the answer, in `ProgramPlanResult.resolutions`.

    Parameters
    ----------
    selection_id : str
        The resolved program, as the diagnostics name it.
    selections : tuple of TaskAlternativeChoice
        The option it chose for each task-alternative group.
    candidate_id : str
        The candidate policy whose plan this is.
    schedule : MemorySchedule
        The plan's schedule.
    simulation : SimulationResult
        Its simulation.
    resident_slice : ResidentSlice, default ResidentSlice(0, ())
        The objects it kept resident throughout.
    """

    selection_id: str
    selections: tuple[TaskAlternativeChoice, ...]
    candidate_id: str
    schedule: MemorySchedule
    simulation: SimulationResult
    resident_slice: ResidentSlice = ResidentSlice(0, ())


@dataclass(frozen=True, slots=True)
class ProgramPlanResult:
    """What a search answers with: a schedule and the simulation that priced it.

    A `SearchAlgorithm` returns one, which `answer_no_worse_than` may
    replace with the plan the search was handed.

    Parameters
    ----------
    program : ShadowSpillProgram
        The program planned, its task alternatives open.
    search_options : SearchOptions
        Everything the search was told: the generic options, the search and
        its own options.
    initial_residency : tuple of ResidencySpec
        Where each alias group is before the first task.
    final_residency : tuple of ResidencySpec
        Where each alias group must be after the last task.
    simulation_config : SimulationConfig
        The machine the schedule was priced on.
    schedule : MemorySchedule
        The chosen memory actions and residency boundaries.
    selections : tuple of TaskAlternativeChoice
        The option chosen for each task-alternative group.
    simulation : SimulationResult
        The simulation that priced the schedule.
    diagnostics : PlanningDiagnostics
        What the search did, and which candidate answered.
    resident_slice : ResidentSlice, default ResidentSlice(0, ())
        The objects kept resident throughout, and the bytes reserved for
        them.
    admission_facts : AdmissionFacts, optional
        The physical facts the plan was planned or admitted under, if any.
    placement_facts : AdmissionFacts, optional
        The pool the plan's layout was measured against, if any.
    resolutions : tuple of ResolutionPlan, default ()
        Every resolved program's best plan, the answer's among them, when
        the search was asked to keep them; empty otherwise.
    """

    program: ShadowSpillProgram
    #: Everything the search was told: the generic options, the algorithm
    #: that ran, and that algorithm's own options. The planner never reads
    #: it back; a record replaying this plan needs it, and so does anything
    #: asking why one search answered differently from another.
    search_options: SearchOptions
    initial_residency: tuple[ResidencySpec, ...]
    final_residency: tuple[ResidencySpec, ...]
    simulation_config: SimulationConfig
    schedule: MemorySchedule
    selections: tuple[TaskAlternativeChoice, ...]
    simulation: SimulationResult
    diagnostics: PlanningDiagnostics
    resident_slice: ResidentSlice = ResidentSlice(0, ())
    admission_facts: AdmissionFacts | None = None
    placement_facts: AdmissionFacts | None = None
    #: Every resolved program's best plan, when the search was asked to keep
    #: them; the answer is among them. A store restores retained plans when
    #: requested. Empty otherwise.
    resolutions: tuple[ResolutionPlan, ...] = ()

    def to_execution_plan(
        self,
        *,
        entrypoints: tuple[EntrypointSpec, ...],
        admission: PhysicalAdmission | None = None,
    ) -> ExecutionPlan:
        """Bind entry points and a physical admission to this result.

        Parameters
        ----------
        entrypoints : tuple of EntrypointSpec
            The frontend's entry points into the plan.
        admission : PhysicalAdmission, optional
            The physical budgets the plan runs under. None builds one from
            the single device's configuration: its capacity as the device
            budget and slab, the spill capacity as the spill budget, the
            simulated spill peak as the spill reservation, the largest task
            workspace as the workspace reserve, and no baseline or external
            headroom.

        Returns
        -------
        ExecutionPlan
            The program, schedule, selections, entry points and admission.
            Its prediction is the simulated makespan and spill peak, and a
            device peak of the admission's baseline, headroom and slab.

        Raises
        ------
        ValueError
            If ``admission`` is None for more than one device, or the simulated
            device or spill peak exceeds what ``admission`` reserves.
        """

        if admission is None:
            if len(self.simulation_config.devices) != 1:
                raise ValueError(
                    "admission is required for a multi-device execution plan"
                )
            device = self.simulation_config.devices[0]
            workspace = max(
                (profile.workspace_bytes for profile in self.program.profiles),
                default=0,
            )
            admission = PhysicalAdmission(
                device_budget_bytes=device.capacity_bytes,
                spill_budget_bytes=self.simulation_config.spill_capacity_bytes,
                baseline_bytes=0,
                external_headroom_bytes=0,
                slab_bytes=device.capacity_bytes,
                workspace_reserve_bytes=min(workspace, device.capacity_bytes),
                spill_reservation_bytes=self.simulation.spill_peak_bytes,
            )
        logical_peak = sum(peak.total_bytes for peak in self.simulation.device_peaks)
        if logical_peak > admission.slab_bytes:
            raise ValueError(
                "simulated device peak exceeds the admitted slab: "
                f"{logical_peak} > {admission.slab_bytes}"
            )
        if self.simulation.spill_peak_bytes > admission.spill_reservation_bytes:
            raise ValueError(
                "simulated host peak exceeds the admitted host reservation: "
                f"{self.simulation.spill_peak_bytes} > "
                f"{admission.spill_reservation_bytes}"
            )
        physical_peak = (
            admission.baseline_bytes
            + admission.external_headroom_bytes
            + admission.slab_bytes
        )
        return ExecutionPlan(
            program=self.program,
            schedule=self.schedule,
            selections=self.selections,
            entrypoints=entrypoints,
            admission=admission,
            prediction=PlanPrediction(
                device_peak_bytes=physical_peak,
                spill_peak_bytes=self.simulation.spill_peak_bytes,
                makespan_ns=self.simulation.makespan_ns,
            ),
        )


__all__ = [
    "PlanInfeasibleError",
    "PlanSearchExhaustedError",
    "ProgramPlanResult",
    "ResidentSlice",
    "ResolutionPlan",
]


def certified_result(
    result: ProgramPlanResult, certificate: FixedLayoutAdmission
) -> ProgramPlanResult:
    """The plan as its certificate re-simulated it: what a caller is handed.

    The search's own simulation is logical. The certificate ran the same
    schedule over the fixed layout, so the two agree on everything the
    schedule decides, and the certified one is what the answer reports.
    """

    return replace(
        result,
        simulation=certificate.simulation,
        diagnostics=result.diagnostics.replace_selected_makespan(
            certificate.simulation.makespan_ns
        ),
    )
