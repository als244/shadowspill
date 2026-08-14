"""Task-envelope construction and cross-task physical admission."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

from shadowspill.planner import PressureFitResult
from shadowspill.pytorch.profiling import (
    TaskAllocationOperation,
    TaskMeasurement,
)
from shadowspill.pytorch.runtime_adapter.bridge import TaskMemoryEnvelope
from shadowspill.runtime import AdmissionReplayResult
from shadowspill.simulator import SimulationAdmission, SimulationResult, simulate

from .admission_replay import AdmissionReplay, replay_admission
from .bindings import TaskOutputBinding, output_bindings_for_entrypoints
from .simulation import simulation_admission_from_replay


@dataclass(frozen=True, slots=True)
class SelectedAdmission:
    """Physical admission evidence plus runtime task-allocation limits."""

    admission: AdmissionReplay
    task_envelopes: tuple[tuple[str, TaskMemoryEnvelope], ...]
    simulation_admission: SimulationAdmission
    simulation: SimulationResult

    @property
    def replay(self) -> AdmissionReplayResult:
        """Return the exact production-pool result used for admission."""

        return self.admission.pool

    def envelopes_by_task(self) -> dict[str, TaskMemoryEnvelope]:
        return dict(self.task_envelopes)

    def apply_prediction(self, selected: PressureFitResult) -> PressureFitResult:
        """Return the selection with admission-aware simulator evidence."""

        diagnostics = selected.diagnostics
        candidates = tuple(
            replace(item, makespan_ns=self.simulation.makespan_ns)
            if item.candidate_id == diagnostics.selected_candidate_id
            and item.selection_id == diagnostics.selected_selection_id
            else item
            for item in diagnostics.candidates
        )
        return replace(
            selected,
            simulation=self.simulation,
            diagnostics=replace(
                diagnostics,
                selected_makespan_ns=self.simulation.makespan_ns,
                candidates=candidates,
            ),
        )


def admit_selected_schedule(
    selected: PressureFitResult,
    *,
    execution_pool_bytes: int,
    alignment: int = 256,
    output_bindings: Mapping[str, tuple[TaskOutputBinding, ...]] | None = None,
) -> AdmissionReplay:
    """Run timing-free cross-task admission for one selected schedule."""

    return replay_admission(
        selected.program,
        selected.schedule,
        execution_pool_bytes=execution_pool_bytes,
        selections=selected.selections,
        output_bindings=output_bindings,
        alignment=alignment,
    )


def build_selected_admission(
    selected: PressureFitResult,
    measurements: Mapping[str, TaskMeasurement],
    *,
    execution_pool_bytes: int,
    alignment: int = 256,
    output_bindings: Mapping[str, tuple[TaskOutputBinding, ...]] | None = None,
) -> SelectedAdmission:
    """Combine cross-task admission with fail-closed task envelopes."""

    admission = admit_selected_schedule(
        selected,
        execution_pool_bytes=execution_pool_bytes,
        alignment=alignment,
        output_bindings=output_bindings,
    )
    simulation_admission = simulation_admission_from_replay(
        admission,
        selected.program,
        selected.schedule,
        selections=selected.selections,
    )
    return SelectedAdmission(
        admission=admission,
        task_envelopes=_selected_task_envelopes(selected, measurements),
        simulation_admission=simulation_admission,
        simulation=simulate(
            selected.program,
            selected.schedule,
            selections=selected.selections,
            config=selected.simulation_config,
            admission=simulation_admission,
        ),
    )


def _selected_task_envelopes(
    selected: PressureFitResult,
    measurements: Mapping[str, TaskMeasurement],
) -> tuple[tuple[str, TaskMemoryEnvelope], ...]:
    profiles = {item.profile_id: item for item in selected.program.profiles}
    return tuple(
        (
            task.task_id,
            _task_memory_envelope(
                _measurement_for_digest(
                    measurements,
                    profiles[task.profile_id].compatibility_digest,
                )
            ),
        )
        for task in selected.program.selected_tasks(selected.selections)
    )


def _measurement_for_digest(
    measurements: Mapping[str, TaskMeasurement],
    compatibility_digest: str,
) -> TaskMeasurement:
    try:
        return measurements[compatibility_digest]
    except KeyError as error:
        raise ValueError(
            f"task-envelope admission lacks measurement "
            f"{compatibility_digest!r}"
        ) from error


def _task_memory_envelope(measurement: TaskMeasurement) -> TaskMemoryEnvelope:
    live: dict[int, tuple[int, int]] = {}
    live_requested = 0
    live_charged = 0
    peak_requested = 0
    peak_charged = 0
    maximum_requested = 0
    maximum_charged = 0
    for event in measurement.allocation_trace:
        sizes = event.requested_bytes, event.charged_bytes
        if event.operation is TaskAllocationOperation.ALLOCATE:
            live[event.allocation_ordinal] = sizes
            live_requested += sizes[0]
            live_charged += sizes[1]
            peak_requested = max(peak_requested, live_requested)
            peak_charged = max(peak_charged, live_charged)
            maximum_requested = max(maximum_requested, sizes[0])
            maximum_charged = max(maximum_charged, sizes[1])
        else:
            prior = live.pop(event.allocation_ordinal)
            live_requested -= prior[0]
            live_charged -= prior[1]
    return TaskMemoryEnvelope(
        maximum_requested_allocation_bytes=maximum_requested,
        maximum_charged_allocation_bytes=maximum_charged,
        live_requested_allocation_limit_bytes=_envelope_limit(peak_requested),
        live_charged_allocation_limit_bytes=_envelope_limit(peak_charged),
    )


def _envelope_limit(profiled_bytes: int) -> int:
    if profiled_bytes == 0:
        return 0
    two_mib = 2 << 20
    with_headroom = (profiled_bytes * 5 + 3) // 4
    return ((with_headroom + two_mib - 1) // two_mib) * two_mib


__all__ = [
    "SelectedAdmission",
    "TaskOutputBinding",
    "admit_selected_schedule",
    "build_selected_admission",
    "output_bindings_for_entrypoints",
]
