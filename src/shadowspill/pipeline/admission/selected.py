"""What a plan selected, physically: its envelopes, its replay, its layout.

One admitted plan's physical evidence, and the per-task allocation limits the
runtime will hold it to. Building it from a frontend's task entrypoints is
the frontend's own admission package; the record and its arithmetic are here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

from shadowspill.planner import ProgramPlanResult
from shadowspill.planner.admission.admission_replay import AdmissionReplay
from shadowspill.planner.admission.layout import (
    FixedPhysicalLayout,
)
from shadowspill.runtime import AdmissionReplayResult
from shadowspill.runtime.plan import TaskMemoryEnvelope
from shadowspill.simulator import SimulationAdmission, SimulationResult
from shadowspill.task.allocations import TaskAllocationOperation
from shadowspill.task.profiles import TaskMeasurement

from .layout_runtime import DynamicTaskAllocationPolicy


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

    def apply_prediction(self, selected: ProgramPlanResult) -> ProgramPlanResult:
        """Return the selection with admission-aware simulator evidence."""

        return replace(
            selected,
            simulation=self.simulation,
            diagnostics=selected.diagnostics.replace_selected_makespan(
                self.simulation.makespan_ns
            ),
        )


def measurement_for_digest(
    measurements: Mapping[str, TaskMeasurement],
    compatibility_digest: str,
) -> TaskMeasurement:
    try:
        return measurements[compatibility_digest]
    except KeyError as error:
        raise ValueError(
            f"task-envelope admission lacks measurement {compatibility_digest!r}"
        ) from error


def task_memory_envelope(
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
    "dynamic_scratch_reserve_bytes",
    "measurement_for_digest",
    "task_memory_envelope",
]
