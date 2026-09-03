from __future__ import annotations

from dataclasses import replace

from reference.python.admission import replay_admission
from shadowspill.ir import (
    AliasGroupSpec,
    MutationSpec,
    ObjectSpec,
    Program,
    TaskProfile,
    TaskSpec,
)
from shadowspill.planner import (
    AdmissionFacts,
    PressureFitOptions,
    TaskAdmissionSpec,
    TaskAllocationStep,
    TaskAllocationStepKind,
    pressurefit,
)
from shadowspill.pytorch.planning.admission.bindings import (
    TaskOutputBinding,
    build_admission_facts,
)
from shadowspill.pytorch.planning.admission.physical import _runtime_record_reserve
from shadowspill.pytorch.planning.admission.selection import (
    _task_memory_envelope,
)
from shadowspill.pytorch.profiling import (
    TaskAllocationContract,
    TaskAllocationEvent,
    TaskAllocationOperation,
    TaskMeasurement,
)
from tests.shadowspill.planner._examples import (
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
            fetch_rules=("latest-safe",),
            evaluate_coalesced=False,
            minimum_object_bytes_evict_eligible=0,
        ),
    )


def _selected_facts() -> AdmissionFacts:
    return AdmissionFacts(
        "cuda_0",
        122,
        122,
        1,
        (
            TaskAdmissionSpec("task0"),
            TaskAdmissionSpec(
                "task1",
                fresh_output_aliases=("temporary",),
                allocation_steps=(
                    TaskAllocationStep(
                        0,
                        TaskAllocationStepKind.ALLOCATE,
                        61,
                        "temporary",
                    ),
                ),
            ),
            TaskAdmissionSpec("task2"),
        ),
    )


def _workspace_trace(extents: tuple[int, ...]) -> tuple[TaskAllocationEvent, ...]:
    allocations = tuple(
        TaskAllocationEvent(
            ordinal,
            TaskAllocationOperation.ALLOCATE,
            extent,
            extent,
            alignment_bytes=1,
        )
        for ordinal, extent in enumerate(extents)
    )
    releases = tuple(
        TaskAllocationEvent(
            ordinal,
            TaskAllocationOperation.FREE,
            extent,
            extent,
            alignment_bytes=1,
        )
        for ordinal, extent in enumerate(extents)
    )
    return (*allocations, *releases)


def test_selected_schedule_replays_only_task_boundary_state() -> None:
    selected = _selected()
    replay = replay_admission(
        selected.program,
        selected.schedule,
        selections=selected.selections,
        facts=_selected_facts(),
    )

    assert replay.pool.peak_allocated_bytes == 122
    assert replay.pool.final_allocated_bytes == 61
    assert replay.workspace_bytes_by_task == (
        ("task0", 0),
        ("task1", 0),
        ("task2", 0),
    )


def test_admission_facts_preserve_workspace_extent_multiset() -> None:
    program = exact_capacity_program()
    program = replace(
        program,
        profiles=(replace(program.profiles[0], workspace_bytes=96),),
        tasks=tuple(replace(task, outputs=()) for task in program.tasks),
    )

    facts = build_admission_facts(
        program,
        execution_pool_bytes=256,
        object_capacity_bytes=160,
        allocation_traces_by_compatibility={"task_abi": _workspace_trace((32, 64))},
        alignment=1,
    )

    assert tuple(task.workspace_extents for task in facts.tasks) == (
        (32, 64),
        (32, 64),
        (32, 64),
    )


def test_admission_facts_derive_gradient_contribution_extents_from_trace() -> None:
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

    facts = build_admission_facts(
        program,
        execution_pool_bytes=256,
        object_capacity_bytes=128,
        allocation_traces_by_compatibility={
            "backward_abi": _workspace_trace((16, 32, 64))
        },
        alignment=1,
    )

    assert facts.tasks[0].workspace_extents == (16, 32, 64)


def test_admission_uses_charged_bytes_for_replacement_transition() -> None:
    program = Program(
        devices=(DEVICE,),
        alias_groups=(AliasGroupSpec("state", "cuda_0", 4096),),
        objects=(ObjectSpec("state_object", "state", 0, 4096),),
        profiles=(TaskProfile("mutation_profile", 10, 8192, "mutation_contract"),),
        tasks=(
            TaskSpec(
                "mutation",
                COMPUTE,
                "mutation_profile",
                inputs=("state_object",),
                mutations=(MutationSpec("state_object"),),
            ),
        ),
    )
    output = TaskAllocationEvent(
        0,
        TaskAllocationOperation.ALLOCATE,
        4096,
        8192,
        output_leaf_indices=(0,),
        output_view_offsets=(0,),
    )

    facts = build_admission_facts(
        program,
        execution_pool_bytes=16_384,
        object_capacity_bytes=12_288,
        output_bindings={
            "mutation": (TaskOutputBinding(0, "state", replacement=True),)
        },
        allocation_traces_by_compatibility={"mutation_contract": (output,)},
        alignment=1,
    )

    assert facts.tasks[0].workspace_extents == ()
    assert facts.tasks[0].replacement_aliases == ("state",)


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


def test_runtime_record_reserve_covers_all_admitted_lifetimes() -> None:
    assert (
        _runtime_record_reserve(
            selected_task_count=135,
            initial_transfer_count=64,
            scheduled_transfer_count=700,
            event_pool_peak_in_use=32,
            fixed_lifetime_count=2450,
            dynamic_lifetime_count=2,
        )
        == 2516
    )


def test_runtime_record_reserve_preserves_event_inventory_floor() -> None:
    assert (
        _runtime_record_reserve(
            selected_task_count=135,
            initial_transfer_count=64,
            scheduled_transfer_count=700,
            event_pool_peak_in_use=32,
            fixed_lifetime_count=100,
            dynamic_lifetime_count=2,
        )
        == 1027
    )


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
        allocation_contract=TaskAllocationContract.capture((output, terminal_free)),
    )

    discarded = _task_memory_envelope(measurement)
    retained = _task_memory_envelope(
        measurement,
        retained_output_leaves=(3,),
    )

    assert discarded.allocation_contract is not None
    assert retained.allocation_contract is not None
    assert len(discarded.allocation_contract.steps) == 2
    assert len(retained.allocation_contract.steps) == 1
    assert retained.allocation_contract.steps[0].persistent_after_task
