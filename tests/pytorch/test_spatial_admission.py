from __future__ import annotations

import pytest

from shadowspill.ir import MutationSpec, ResourceKind, ResourceSpec, TaskSpec
from shadowspill.planner import PressureFitOptions, pressurefit
from shadowspill.pytorch.profiling import (
    TaskAllocationEvent,
    TaskAllocationOperation,
    TaskMeasurement,
)
from shadowspill.pytorch.spatial_admission import (
    TaskOutputBinding,
    _task_allocation_events,
    _validate_profile_workspace,
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
        execution_pool_bytes=122,
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
        replay_selected_schedule(selected, {}, execution_pool_bytes=122, alignment=1)


def test_accumulation_charge_retains_per_gradient_extent_geometry() -> None:
    task = TaskSpec(
        "accumulate",
        ResourceSpec("cuda_0", ResourceKind.COMPUTE),
        "backward_profile",
        mutations=(MutationSpec("first_gradient"), MutationSpec("second_gradient")),
    )
    measurement = TaskMeasurement(10, 4, 4, (4,), (10,), "unit-test")

    assert (
        _validate_profile_workspace(
            task,
            12,
            measurement,
            {"first_gradient": 3, "second_gradient": 5},
        )
        is None
    )


def test_unclassified_workspace_without_extent_geometry_is_rejected() -> None:
    task = TaskSpec(
        "opaque",
        ResourceSpec("cuda_0", ResourceKind.COMPUTE),
        "opaque_profile",
    )
    measurement = TaskMeasurement(10, 4, 4, (4,), (10,), "unit-test")

    with pytest.raises(ValueError, match="no complete physical extent"):
        _validate_profile_workspace(task, 12, measurement, {})


def test_task_allocation_replay_preserves_callback_order_and_output_identity() -> None:
    task = TaskSpec(
        "backward",
        ResourceSpec("cuda_0", ResourceKind.COMPUTE),
        "backward_profile",
        outputs=("gradient",),
    )
    measurement = TaskMeasurement(
        10,
        7,
        7,
        (7,),
        (10,),
        "unit-test",
        (
            TaskAllocationEvent(0, TaskAllocationOperation.ALLOCATE, 7, 7),
            TaskAllocationEvent(0, TaskAllocationOperation.FREE, 7, 7),
            TaskAllocationEvent(
                1,
                TaskAllocationOperation.ALLOCATE,
                11,
                11,
                (3,),
            ),
        ),
    )

    events = _task_allocation_events(
        task,
        measurement,
        (TaskOutputBinding(3, "gradient_alias"),),
        {"gradient_alias": 11},
    )

    assert [(event.kind.name, event.identity) for event in events] == [
        ("TASK_ALLOCATION", "workspace:backward:0"),
        ("TASK_FREE", "workspace:backward:0"),
        ("TASK_ALLOCATION", "gradient_alias"),
    ]
    assert events[-1].alias_output


def test_task_allocation_replay_preserves_cached_extent_identity() -> None:
    task = TaskSpec(
        "forward",
        ResourceSpec("cuda_0", ResourceKind.COMPUTE),
        "forward_profile",
        outputs=("activation",),
    )
    measurement = TaskMeasurement(
        10,
        64,
        64,
        (64,),
        (10,),
        "unit-test",
        (
            TaskAllocationEvent(0, TaskAllocationOperation.ALLOCATE, 64, 64),
            TaskAllocationEvent(0, TaskAllocationOperation.FREE, 64, 64),
            TaskAllocationEvent(
                1,
                TaskAllocationOperation.ALLOCATE,
                48,
                64,
                (2,),
                reuses_ordinal=0,
            ),
        ),
    )

    events = _task_allocation_events(
        task,
        measurement,
        (TaskOutputBinding(2, "activation_alias"),),
        {"activation_alias": 64},
    )

    assert [(event.kind.name, event.identity) for event in events] == [
        ("TASK_ALLOCATION", "workspace:forward:0"),
        ("TASK_REUSE", "activation_alias"),
    ]
    assert events[1].source_identity == "workspace:forward:0"
