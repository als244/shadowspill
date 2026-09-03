"""What PressureFit gives back, including the two ways it can decline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from shadowspill.ir import (
    EntrypointSpec,
    ExecutionPlan,
    MemorySchedule,
    PhysicalAdmission,
    PlanPrediction,
    Program,
    RecomputationSelection,
    ResidencySpec,
)
from shadowspill.simulator import SimulationConfig, SimulationResult

from .diagnostics import (
    CandidateDiagnostic,
    PressureFitDiagnostics,
)
from .request import PressureFitOptions

if TYPE_CHECKING:
    from .admission import AdmissionFacts


class PressureFitInfeasibleError(ValueError):
    """No candidate satisfied the declared residency and capacity constraints."""

    def __init__(
        self,
        message: str,
        *,
        kind: str,
        device_id: str | None = None,
        boundary_task_id: str | None = None,
        required_bytes: int | None = None,
        capacity_bytes: int | None = None,
        diagnostics: tuple[CandidateDiagnostic, ...] = (),
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.device_id = device_id
        self.boundary_task_id = boundary_task_id
        self.required_bytes = required_bytes
        self.capacity_bytes = capacity_bytes
        self.diagnostics = diagnostics


class PressureFitSearchExhaustedError(RuntimeError):
    """A bounded candidate search stopped before proving feasibility.

    This is deliberately distinct from ``PressureFitInfeasibleError``.  A
    repairable candidate that reaches its evaluation ceiling has not proved
    that no legal schedule exists.
    """

    def __init__(
        self,
        message: str,
        *,
        diagnostics: tuple[CandidateDiagnostic, ...] = (),
    ) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


@dataclass(frozen=True, slots=True)
class ResidentSlice:
    """The slice reserved for the objects the planner kept resident.

    Objects under `minimum_object_bytes_evict_eligible` are never cut, so
    every lease of theirs gets a static home in a slice at the end of the
    fixed layout rather than a place among the leases that come and go.
    `bytes` is what the planner reserved for it -- the sum of those homes,
    sized before the search and taken out of the capacity the search plans
    against -- and `aliases` names the alias groups whose leases it holds.
    An empty slice has no bytes and no aliases.
    """

    bytes: int
    aliases: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.bytes < 0:
            raise ValueError("resident slice bytes must be non-negative")
        if self.aliases != tuple(sorted(set(self.aliases))):
            raise ValueError("resident slice aliases must be sorted and distinct")

    def to_dict(self) -> dict[str, object]:
        return {"bytes": self.bytes, "aliases": list(self.aliases)}


@dataclass(frozen=True, slots=True)
class PressureFitResult:
    """Selected logical schedule plus exact simulator evidence."""

    program: Program
    options: PressureFitOptions
    initial_residency: tuple[ResidencySpec, ...]
    final_residency: tuple[ResidencySpec, ...]
    simulation_config: SimulationConfig
    schedule: MemorySchedule
    selections: tuple[RecomputationSelection, ...]
    simulation: SimulationResult
    diagnostics: PressureFitDiagnostics
    resident_slice: ResidentSlice = ResidentSlice(0, ())
    admission_facts: AdmissionFacts | None = None
    placement_facts: AdmissionFacts | None = None

    def to_execution_plan(
        self,
        *,
        entrypoints: tuple[EntrypointSpec, ...],
        admission: PhysicalAdmission | None = None,
    ) -> ExecutionPlan:
        """Bind frontend entrypoints and physical admission to this result."""

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
                provider_headroom_bytes=0,
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
            + admission.provider_headroom_bytes
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
    "PressureFitInfeasibleError",
    "PressureFitResult",
    "PressureFitSearchExhaustedError",
    "ResidentSlice",
]
