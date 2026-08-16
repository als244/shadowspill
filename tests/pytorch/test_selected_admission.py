from __future__ import annotations

from dataclasses import replace

import pytest

from shadowspill.ir import (
    AliasGroupSpec,
    MutationSpec,
    ObjectSpec,
    Program,
    TaskProfile,
    TaskSpec,
)
from shadowspill.planner import PressureFitOptions, pressurefit
from shadowspill.pytorch.planning.admission.bindings import build_admission_topology
from shadowspill.pytorch.planning.admission.selection import (
    _task_memory_envelope,
    admit_selected_schedule,
    build_selected_admission,
)
from shadowspill.pytorch.profiling import (
    TaskAllocationABI,
    TaskAllocationEvent,
    TaskAllocationOperation,
    TaskMeasurement,
)
from tests.planner._examples import (
    COMPUTE,
    DEVICE,
    config,
    exact_capacity_program,
    exact_capacity_residency,
)


def _selected():  # type: ignore[no-untyped-def]
    initial, final = exact_capacity_residency()
    return pressurefit(
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


def test_selected_schedule_replays_only_task_boundary_state() -> None:
    replay = admit_selected_schedule(
        _selected(),
        execution_pool_bytes=122,
        alignment=1,
    )

    assert replay.pool.peak_allocated_bytes == 122
    assert replay.pool.final_allocated_bytes == 61
    assert replay.workspace_bytes_by_task == (
        ("task0", 0),
        ("task1", 0),
        ("task2", 0),
    )


def test_admission_topology_preserves_workspace_extent_multiset() -> None:
    program = exact_capacity_program()
    program = replace(
        program,
        profiles=(replace(program.profiles[0], workspace_bytes=96),),
    )

    topology = build_admission_topology(
        program,
        execution_pool_bytes=256,
        object_capacity_bytes=160,
        workspace_extents_by_compatibility={"task_abi": (32, 64)},
        alignment=1,
    )

    assert tuple(task.workspace_extents for task in topology.tasks) == (
        (32, 64),
        (32, 64),
        (32, 64),
    )


def test_admission_topology_preserves_gradient_contribution_extents() -> None:
    program = Program(
        devices=(DEVICE,),
        alias_groups=(
            AliasGroupSpec("gradient_a", "cuda_0", 32),
            AliasGroupSpec("gradient_b", "cuda_0", 64),
        ),
        objects=(
            ObjectSpec("gradient_a_object", "gradient_a", 0, 32),
            ObjectSpec("gradient_b_object", "gradient_b", 0, 64),
        ),
        profiles=(TaskProfile("backward_profile", 10, 112, "backward_abi"),),
        tasks=(
            TaskSpec(
                "backward",
                COMPUTE,
                "backward_profile",
                inputs=("gradient_a_object", "gradient_b_object"),
                mutations=(
                    MutationSpec("gradient_a_object"),
                    MutationSpec("gradient_b_object"),
                ),
                phase="backward",
            ),
        ),
    )

    topology = build_admission_topology(
        program,
        execution_pool_bytes=256,
        object_capacity_bytes=128,
        workspace_extents_by_compatibility={"backward_abi": (16,)},
        alignment=1,
    )

    assert topology.tasks[0].workspace_extents == (16, 32, 64)


def test_selected_admission_requires_every_task_envelope_measurement() -> None:
    with pytest.raises(ValueError, match="task-envelope admission lacks measurement"):
        build_selected_admission(
            _selected(),
            {},
            execution_pool_bytes=122,
            alignment=1,
        )


def test_selected_admission_replaces_prediction_without_changing_schedule() -> None:
    selected = _selected()
    measurement = TaskMeasurement(
        1_000,
        0,
        0,
        (),
        (1_000,),
        "unit-test",
    )
    admitted = build_selected_admission(
        selected,
        {"task_abi": measurement},
        execution_pool_bytes=122,
        alignment=1,
    )

    adjusted = admitted.apply_prediction(selected)

    assert adjusted.program is selected.program
    assert adjusted.schedule is selected.schedule
    assert adjusted.selections == selected.selections
    assert adjusted.simulation == admitted.simulation
    assert (
        adjusted.diagnostics.selected_makespan_ns
        == admitted.simulation.makespan_ns
    )


def test_task_envelope_counts_peak_live_bytes_not_allocation_volume() -> None:
    measurement = TaskMeasurement(
        10,
        96,
        96,
        (96,),
        (10,),
        "unit-test",
        (
            TaskAllocationEvent(0, TaskAllocationOperation.ALLOCATE, 96, 96),
            TaskAllocationEvent(0, TaskAllocationOperation.FREE, 96, 96),
            TaskAllocationEvent(1, TaskAllocationOperation.ALLOCATE, 64, 64),
            TaskAllocationEvent(1, TaskAllocationOperation.FREE, 64, 64),
        ),
    )

    envelope = _task_memory_envelope(measurement)

    assert envelope.maximum_requested_allocation_bytes == 96
    assert envelope.maximum_charged_allocation_bytes == 96
    assert envelope.live_requested_allocation_limit_bytes == 2 << 20
    assert envelope.live_charged_allocation_limit_bytes == 2 << 20


def test_manual_scratch_reserve_expands_runtime_envelope() -> None:
    measurement = TaskMeasurement(
        10,
        96,
        96,
        (96,),
        (10,),
        "unit-test",
        (
            TaskAllocationEvent(0, TaskAllocationOperation.ALLOCATE, 96, 96),
            TaskAllocationEvent(0, TaskAllocationOperation.FREE, 96, 96),
        ),
    )

    envelope = _task_memory_envelope(
        measurement,
        minimum_scratch_reserve_bytes=8 << 20,
    )

    assert envelope.dynamic_scratch_maximum_allocation_bytes == 8 << 20
    assert envelope.dynamic_scratch_live_limit_bytes == 10 << 20
    assert envelope.maximum_requested_allocation_bytes == 8 << 20
    assert envelope.maximum_charged_allocation_bytes == 8 << 20
    assert envelope.live_requested_allocation_limit_bytes == 12 << 20
    assert envelope.live_charged_allocation_limit_bytes == 12 << 20


def test_task_envelope_specializes_persistent_output_ownership() -> None:
    output = TaskAllocationEvent(
        0,
        TaskAllocationOperation.ALLOCATE,
        64,
        64,
        output_leaf_indices=(3,),
        output_view_offsets=(0,),
    )
    terminal_free = TaskAllocationEvent(
        0,
        TaskAllocationOperation.FREE,
        64,
        64,
    )
    measurement = TaskMeasurement(
        10,
        0,
        0,
        (),
        (10,),
        "unit-test",
        allocation_trace=(output,),
        allocation_abi=TaskAllocationABI.capture((output, terminal_free)),
    )

    discarded = _task_memory_envelope(measurement)
    retained = _task_memory_envelope(
        measurement,
        retained_output_leaves=(3,),
    )

    assert discarded.allocation_abi is not None
    assert retained.allocation_abi is not None
    assert len(discarded.allocation_abi.steps) == 2
    assert len(retained.allocation_abi.steps) == 1
    assert retained.allocation_abi.steps[0].persistent_after_task
