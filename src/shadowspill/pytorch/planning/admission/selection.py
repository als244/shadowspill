"""Task-envelope construction and cross-task physical admission."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

from shadowspill.planner import AdmissionTopology, PressureFitResult
from shadowspill.pytorch.profiling import (
    TaskAllocationOperation,
    TaskMeasurement,
)
from shadowspill.pytorch.runtime_adapter.bridge import TaskMemoryEnvelope
from shadowspill.runtime import AdmissionReplayResult
from shadowspill.simulator import SimulationAdmission, SimulationResult, simulate

from .admission_replay import AdmissionReplay, replay_admission
from .bindings import TaskOutputBinding, output_bindings_for_entrypoints
from .layout import (
    DynamicTaskAllocationPolicy,
    FixedLayoutAdmission,
    FixedPhysicalLayout,
)
from .simulation import simulation_admission_from_replay


@dataclass(frozen=True, slots=True)
class SelectedAdmission:
    """Physical admission evidence plus runtime task-allocation limits."""

    task_envelopes: tuple[tuple[str, TaskMemoryEnvelope], ...]
    simulation_admission: SimulationAdmission
    simulation: SimulationResult
    admission: AdmissionReplay | None = None
    fixed_layout: FixedPhysicalLayout | None = None

    def __post_init__(self) -> None:
        if (self.admission is None) == (self.fixed_layout is None):
            raise ValueError(
                "selected admission requires exactly one physical strategy"
            )

    @property
    def replay(self) -> AdmissionReplayResult:
        """Return the exact production-pool result used for admission."""

        if self.admission is None:
            raise ValueError("fixed-layout admission has no dynamic replay")
        return self.admission.pool

    @property
    def predicted_fragmentation_bytes(self) -> int:
        """Return fragmentation charged by the selected physical strategy."""

        return (
            0
            if self.admission is None
            else self.admission.pool.peak_fragmentation_bytes
        )

    def envelopes_by_task(self) -> dict[str, TaskMemoryEnvelope]:
        return dict(self.task_envelopes)

    def dynamic_provider_allocations(
        self,
    ) -> tuple[DynamicTaskAllocationPolicy, ...]:
        """Return bounded provider-owned allocations excluded from the layout."""

        result: list[DynamicTaskAllocationPolicy] = []
        for task_id, envelope in self.task_envelopes:
            contract = envelope.allocation_contract
            if contract is None:
                continue
            result.extend(
                DynamicTaskAllocationPolicy(
                    task_id,
                    step.allocation_ordinal,
                    step.charged_bytes,
                    step.alignment_bytes,
                )
                for step in contract.steps
                if step.operation is TaskAllocationOperation.ALLOCATE
                and step.persistent_after_task
                and not step.output_leaf_indices
            )
        return tuple(result)

    def apply_prediction(self, selected: PressureFitResult) -> PressureFitResult:
        """Return the selection with admission-aware simulator evidence."""

        return replace(
            selected,
            simulation=self.simulation,
            diagnostics=selected.diagnostics.with_selected_makespan(
                self.simulation.makespan_ns
            ),
        )


def admit_selected_schedule(
    selected: PressureFitResult,
    *,
    execution_pool_bytes: int,
    alignment: int = 256,
    topology: AdmissionTopology | None = None,
    output_bindings: Mapping[str, tuple[TaskOutputBinding, ...]] | None = None,
) -> AdmissionReplay:
    """Run timing-free cross-task admission for one selected schedule."""

    return replay_admission(
        selected.program,
        selected.schedule,
        execution_pool_bytes=execution_pool_bytes,
        selections=selected.selections,
        topology=topology,
        output_bindings=output_bindings,
        alignment=alignment,
    )


def build_selected_admission(
    selected: PressureFitResult,
    measurements: Mapping[str, TaskMeasurement],
    *,
    execution_pool_bytes: int,
    alignment: int = 256,
    topology: AdmissionTopology | None = None,
    output_bindings: Mapping[str, tuple[TaskOutputBinding, ...]] | None = None,
) -> SelectedAdmission:
    """Combine cross-task admission with fail-closed task envelopes."""

    admission = admit_selected_schedule(
        selected,
        execution_pool_bytes=execution_pool_bytes,
        alignment=alignment,
        topology=topology,
        output_bindings=output_bindings,
    )
    simulation_admission = simulation_admission_from_replay(
        admission,
        selected.program,
        selected.schedule,
        selections=selected.selections,
        device_capacity_bytes=execution_pool_bytes,
    )
    return SelectedAdmission(
        task_envelopes=_selected_task_envelopes(
            selected,
            measurements,
            output_bindings=output_bindings,
        ),
        simulation_admission=simulation_admission,
        simulation=simulate(
            selected.program,
            selected.schedule,
            selections=selected.selections,
            config=selected.simulation_config,
            admission=simulation_admission,
        ),
        admission=admission,
    )


def build_fixed_selected_admission(
    selected: PressureFitResult,
    measurements: Mapping[str, TaskMeasurement],
    *,
    fixed_admission: FixedLayoutAdmission,
    output_bindings: Mapping[str, tuple[TaskOutputBinding, ...]] | None = None,
) -> SelectedAdmission:
    """Bind task envelopes to an already-certified fixed layout."""

    return SelectedAdmission(
        task_envelopes=_selected_task_envelopes(
            selected,
            measurements,
            output_bindings=output_bindings,
            minimum_scratch_reserve_bytes=(
                fixed_admission.layout.scratch_reserve_bytes
            ),
        ),
        simulation_admission=fixed_admission.simulator_input,
        simulation=fixed_admission.simulation,
        fixed_layout=fixed_admission.layout,
    )


def _selected_task_envelopes(
    selected: PressureFitResult,
    measurements: Mapping[str, TaskMeasurement],
    *,
    output_bindings: Mapping[str, tuple[TaskOutputBinding, ...]] | None = None,
    minimum_scratch_reserve_bytes: int = 0,
) -> tuple[tuple[str, TaskMemoryEnvelope], ...]:
    profiles = {item.profile_id: item for item in selected.program.profiles}
    bindings_by_task = dict(output_bindings or {})
    return tuple(
        (
            task.task_id,
            _task_memory_envelope(
                _measurement_for_digest(
                    measurements,
                    profiles[task.profile_id].compatibility_digest,
                ),
                retained_output_leaves=tuple(
                    item.leaf_index for item in bindings_by_task.get(task.task_id, ())
                ),
                minimum_scratch_reserve_bytes=minimum_scratch_reserve_bytes,
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
            f"task-envelope admission lacks measurement {compatibility_digest!r}"
        ) from error


def _task_memory_envelope(
    measurement: TaskMeasurement,
    *,
    retained_output_leaves: tuple[int, ...] = (),
    minimum_scratch_reserve_bytes: int = 0,
) -> TaskMemoryEnvelope:
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
    allocation_contract = measurement.allocation_contract
    if allocation_contract is not None:
        allocation_contract = allocation_contract.for_retained_output_leaves(
            retained_output_leaves
        )
    scratch_peak_requested = max(
        measurement.dynamic_scratch_peak_requested_bytes,
        minimum_scratch_reserve_bytes,
    )
    scratch_peak_charged = max(
        measurement.dynamic_scratch_peak_charged_bytes,
        minimum_scratch_reserve_bytes,
    )
    return TaskMemoryEnvelope(
        maximum_requested_allocation_bytes=max(
            maximum_requested,
            measurement.dynamic_scratch_maximum_requested_bytes,
            minimum_scratch_reserve_bytes,
        ),
        maximum_charged_allocation_bytes=max(
            maximum_charged,
            measurement.dynamic_scratch_maximum_charged_bytes,
            minimum_scratch_reserve_bytes,
        ),
        live_requested_allocation_limit_bytes=_envelope_limit(
            peak_requested + scratch_peak_requested
        ),
        live_charged_allocation_limit_bytes=_envelope_limit(
            peak_charged + scratch_peak_charged
        ),
        dynamic_scratch_maximum_allocation_bytes=max(
            measurement.dynamic_scratch_maximum_charged_bytes,
            minimum_scratch_reserve_bytes,
        ),
        dynamic_scratch_live_limit_bytes=_envelope_limit(scratch_peak_charged),
        allocation_path_digests=tuple(
            item.compatibility_digest
            for item in measurement.allocation_path_observations
        ),
        allocation_contract=allocation_contract,
    )


def _envelope_limit(profiled_bytes: int) -> int:
    if profiled_bytes == 0:
        return 0
    two_mib = 2 << 20
    with_headroom = (profiled_bytes * 5 + 3) // 4
    return ((with_headroom + two_mib - 1) // two_mib) * two_mib


def dynamic_scratch_reserve_bytes(
    measurements: Mapping[str, TaskMeasurement],
    *,
    minimum_bytes: int = 0,
) -> int:
    """Return one conservative reserve shared by sequential task probes."""

    if minimum_bytes < 0:
        raise ValueError("dynamic scratch minimum must be non-negative")
    profiled = max(
        (
            _envelope_limit(item.dynamic_scratch_peak_charged_bytes)
            for item in measurements.values()
        ),
        default=0,
    )
    return max(minimum_bytes, profiled)


__all__ = [
    "SelectedAdmission",
    "TaskOutputBinding",
    "admit_selected_schedule",
    "build_fixed_selected_admission",
    "build_selected_admission",
    "dynamic_scratch_reserve_bytes",
    "output_bindings_for_entrypoints",
]
