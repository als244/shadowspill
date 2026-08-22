"""Public immutable PressureFit inputs, results, and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
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
    AdmissionRefinement,
    CandidateDiagnostic,
    PressureFitDiagnostics,
    PressureFitRepairDiagnostics,
    PressureFitWorkDiagnostics,
    RecomputationChoiceDiagnostic,
    RecomputationContextDiagnostics,
)

if TYPE_CHECKING:
    from .admission import AdmissionTopology


class InitialPlacement(StrEnum):
    """How host-origin objects may be placed before the first task."""

    REQUIRED = "required"
    GREEDY = "greedy"


@dataclass(frozen=True, slots=True)
class PressureFitOptions:
    """Bounded heuristic portfolio configuration.

    Worker count changes only evaluation concurrency. It never enters candidate
    identity or tie-breaking. Zero selects all available logical CPUs; one
    forces serial evaluation.
    """

    initial_placement: InitialPlacement = InitialPlacement.GREEDY
    # relaxed-stall (byte-identical to tight-stall) and interval-entry
    # never carry a winner: a 435-point regression replay reproduced
    # every schedule digest exactly without them at ~1.25x less
    # portfolio work. Both remain valid explicit options.
    residency_strategies: tuple[str, ...] = (
        "headroom-stall",
        "headroom-transfer",
        "tight-stall",
        "tight-transfer",
    )
    prefetch_rules: tuple[str, ...] = (
        "packed-fifo",
        "packed-fit",
        "latest-safe",
        "demand",
    )
    evaluate_coalesced: bool = True
    max_repair_attempts: int = 64
    workers: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.initial_placement, InitialPlacement):
            raise ValueError("initial_placement is invalid")
        if not self.residency_strategies:
            raise ValueError("residency_strategies must not be empty")
        if not self.prefetch_rules:
            raise ValueError("prefetch_rules must not be empty")
        if len(set(self.residency_strategies)) != len(self.residency_strategies):
            raise ValueError("residency_strategies contains duplicates")
        if len(set(self.prefetch_rules)) != len(self.prefetch_rules):
            raise ValueError("prefetch_rules contains duplicates")
        known_strategies = {
            "headroom-stall",
            "headroom-transfer",
            "tight-stall",
            "tight-transfer",
            "relaxed-stall",
        }
        known_prefetch = {
            "packed-fifo",
            "packed-fit",
            "interval-entry",
            "latest-safe",
            "demand",
        }
        unknown_strategies = set(self.residency_strategies) - known_strategies
        unknown_prefetch = set(self.prefetch_rules) - known_prefetch
        if unknown_strategies:
            raise ValueError(
                f"unknown residency strategies: {sorted(unknown_strategies)}"
            )
        if unknown_prefetch:
            raise ValueError(f"unknown prefetch rules: {sorted(unknown_prefetch)}")
        if (
            isinstance(self.max_repair_attempts, bool)
            or not isinstance(self.max_repair_attempts, int)
            or self.max_repair_attempts < 0
        ):
            raise ValueError("max_repair_attempts must be a non-negative integer")
        if (
            isinstance(self.workers, bool)
            or not isinstance(self.workers, int)
            or self.workers < 0
        ):
            raise ValueError("workers must be a non-negative integer")


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
    admission_topology: AdmissionTopology | None = None

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
                context_bytes=0,
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
            admission.context_bytes
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
    "AdmissionRefinement",
    "CandidateDiagnostic",
    "InitialPlacement",
    "PressureFitDiagnostics",
    "PressureFitInfeasibleError",
    "PressureFitOptions",
    "PressureFitRepairDiagnostics",
    "PressureFitResult",
    "PressureFitSearchExhaustedError",
    "PressureFitWorkDiagnostics",
    "RecomputationChoiceDiagnostic",
    "RecomputationContextDiagnostics",
]
