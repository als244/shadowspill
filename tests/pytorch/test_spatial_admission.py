from __future__ import annotations

import pytest

from shadowspill.ir import MutationSpec, ResourceKind, ResourceSpec, TaskSpec
from shadowspill.planner import PressureFitOptions, pressurefit
from shadowspill.pytorch.planning.admission.spatial import (
    TaskOutputBinding,
    _append_profiled_task_events,
    _SpatialEventKind,
    _SpatialTimeline,
    _task_allocation_events,
    _translate_spatial_timeline,
    _validate_profile_workspace,
    replay_selected_schedule,
)
from shadowspill.pytorch.profiling import (
    TaskAllocationEvent,
    TaskAllocationOperation,
    TaskMeasurement,
    TaskOutputInputBinding,
)
from shadowspill.runtime import plan_slab_layout
from shadowspill.simulator import TaskInterval
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
                (0,),
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


def test_persistent_output_placement_does_not_fragment_later_workspace() -> None:
    task = TaskSpec(
        "forward",
        ResourceSpec("cuda_0", ResourceKind.COMPUTE),
        "forward_profile",
        outputs=("activation",),
    )
    measurement = TaskMeasurement(
        10,
        160,
        160,
        (160,),
        (10,),
        "unit-test",
        (
            TaskAllocationEvent(0, TaskAllocationOperation.ALLOCATE, 100, 100),
            TaskAllocationEvent(
                1,
                TaskAllocationOperation.ALLOCATE,
                50,
                50,
                (2,),
                (0,),
            ),
            TaskAllocationEvent(0, TaskAllocationOperation.FREE, 100, 100),
            TaskAllocationEvent(2, TaskAllocationOperation.ALLOCATE, 160, 160),
            TaskAllocationEvent(2, TaskAllocationOperation.FREE, 160, 160),
        ),
    )
    allocations = _task_allocation_events(
        task,
        measurement,
        (TaskOutputBinding(2, "activation_alias"),),
        {"activation_alias": 50},
    )
    timeline = _SpatialTimeline()
    _append_profiled_task_events(
        task.task_id,
        TaskInterval(
            task.task_id,
            "cuda_0",
            ResourceKind.COMPUTE,
            0,
            0,
            0,
            10,
            160,
        ),
        allocations,
        timeline,
    )

    replay = plan_slab_layout(
        300,
        _translate_spatial_timeline(timeline.events, alignment=1).events,
    ).replay

    assert replay.peak_allocated_bytes == 210
    assert replay.final_allocated_bytes == 50


def test_initial_residency_overlaps_first_task_temporaries_at_time_zero() -> None:
    timeline = _SpatialTimeline()
    timeline.append(
        0,
        _SpatialEventKind.ALLOCATE_PREFETCH,
        "initial_input",
        100,
        planned=True,
    )
    timeline.append(
        0,
        _SpatialEventKind.TASK_ALLOCATION,
        "first_task_workspace",
        16,
        task_id="first_task",
        allocation_ordinal=0,
        requested_bytes=16,
    )
    timeline.append(
        0,
        _SpatialEventKind.TASK_FREE,
        "first_task_workspace",
        16,
        task_id="first_task",
        allocation_ordinal=0,
        requested_bytes=16,
    )

    layout = plan_slab_layout(
        116,
        _translate_spatial_timeline(timeline.events, alignment=1).events,
    )
    offsets = layout.offset_by_allocation()

    assert offsets["initial_input:0"] != offsets["first_task_workspace"]
    assert layout.layout_bytes == 116


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
                (0,),
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


def test_task_allocation_replay_models_input_lease_handoff_without_allocation() -> None:
    task = TaskSpec(
        "backward",
        ResourceSpec("cuda_0", ResourceKind.COMPUTE),
        "backward_profile",
        inputs=("residual",),
        outputs=("gradient",),
    )
    measurement = TaskMeasurement(
        10,
        0,
        0,
        (),
        (10,),
        "unit-test",
        output_input_bindings=(TaskOutputInputBinding(3, 0, 0),),
    )

    events = _task_allocation_events(
        task,
        measurement,
        (TaskOutputBinding(3, "gradient_alias", False, "residual_alias"),),
        {"gradient_alias": 64, "residual_alias": 64},
    )

    assert [(event.kind.name, event.identity) for event in events] == [
        ("HANDOFF_ALIAS", "gradient_alias")
    ]
    assert events[0].source_identity == "residual_alias"


def test_task_allocation_replay_skips_zero_size_output_extent() -> None:
    task = TaskSpec(
        "forward",
        ResourceSpec("cuda_0", ResourceKind.COMPUTE),
        "forward_profile",
        outputs=("empty",),
    )
    measurement = TaskMeasurement(10, 0, 0, (), (10,), "unit-test")

    events = _task_allocation_events(
        task,
        measurement,
        (TaskOutputBinding(2, "empty_alias"),),
        {"empty_alias": 0},
        {"empty_alias"},
    )

    assert events == ()
