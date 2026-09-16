"""Task-envelope construction and cross-task physical admission."""

from __future__ import annotations

from collections.abc import Mapping

from shadowspill.pipeline.admission.selected import (
    SelectedAdmission,
    measurement_for_digest,
    task_memory_envelope,
)
from shadowspill.planner import ProgramPlanResult
from shadowspill.planner.admission.layout import (
    FixedLayoutAdmission,
)
from shadowspill.pytorch.profiling import (
    TaskMeasurement,
)
from shadowspill.runtime.plan import TaskMemoryEnvelope

from .bindings import TaskOutputBinding


def build_fixed_selected_admission(
    selected: ProgramPlanResult,
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
    selected: ProgramPlanResult,
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
            task_memory_envelope(
                measurement_for_digest(
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
