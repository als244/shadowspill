from __future__ import annotations

import pytest

from shadowspill.ir import MutationSpec, ResourceKind, ResourceSpec, TaskSpec
from shadowspill.planner import PressureFitOptions, pressurefit
from shadowspill.pytorch.profiling import TaskMeasurement
from shadowspill.pytorch.spatial_admission import (
    _task_workspace_extents,
    replay_selected_schedule,
)
from tests.planner._examples import (
    config,
    exact_capacity_program,
    exact_capacity_residency,
)


def test_selected_schedule_replays_object_generations_and_outputs() -> None:
    initial, final = exact_capacity_residency()
    selected = pressurefit(
        exact_capacity_program(),
        initial_residency=initial,
        final_residency=final,
        config=config(),
        options=PressureFitOptions(
            residency_strategies=("relaxed-stall",),
            prefetch_rules=("latest-safe",),
            evaluate_coalesced=False,
        ),
    )
    replay = replay_selected_schedule(
        selected,
        {
            "task_abi": TaskMeasurement(
                1_000,
                0,
                0,
                (),
                (1_000,),
                "unit-test",
            )
        },
        slab_bytes=122,
        alignment=1,
    )
    assert replay.peak_allocated_bytes == 122
    assert replay.final_allocated_bytes == 61


def test_selected_schedule_requires_every_profile_measurement() -> None:
    initial, final = exact_capacity_residency()
    selected = pressurefit(
        exact_capacity_program(),
        initial_residency=initial,
        final_residency=final,
        config=config(),
    )
    with pytest.raises(ValueError, match="lacks task measurement"):
        replay_selected_schedule(selected, {}, slab_bytes=122, alignment=1)


def test_accumulation_charge_retains_per_gradient_extent_geometry() -> None:
    task = TaskSpec(
        "accumulate",
        ResourceSpec("cuda_0", ResourceKind.COMPUTE),
        "backward_profile",
        mutations=(MutationSpec("first_gradient"), MutationSpec("second_gradient")),
    )
    measurement = TaskMeasurement(10, 4, 4, (4,), (10,), "unit-test")

    assert _task_workspace_extents(
        task,
        12,
        measurement,
        {"first_gradient": 3, "second_gradient": 5},
    ) == (4, 3, 5)


def test_unclassified_workspace_without_extent_geometry_is_rejected() -> None:
    task = TaskSpec(
        "opaque",
        ResourceSpec("cuda_0", ResourceKind.COMPUTE),
        "opaque_profile",
    )
    measurement = TaskMeasurement(10, 4, 4, (4,), (10,), "unit-test")

    with pytest.raises(ValueError, match="no complete physical extent"):
        _task_workspace_extents(task, 12, measurement, {})
