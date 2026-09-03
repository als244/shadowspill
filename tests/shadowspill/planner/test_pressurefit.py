from __future__ import annotations

from dataclasses import replace

import pytest

from shadowspill.ir import (
    AliasGroupSpec,
    DeviceSpec,
    EntrypointSpec,
    MemoryActionKind,
    MemoryLocation,
    MutationSpec,
    ObjectSpec,
    Program,
    ResidencySpec,
    ResourceKind,
    ResourceSpec,
    TaskProfile,
    TaskSpec,
)
from shadowspill.planner import (
    InitialPlacement,
    PressureFitOptions,
    ResidentSlice,
    pressurefit,
)

from ._examples import (
    config,
    exact_capacity_program,
    exact_capacity_residency,
    mutation_program,
    recomputation_program,
)

FEW_CANDIDATES = PressureFitOptions(
    minimum_object_bytes_evict_eligible=0,
    residency_strategies=("relaxed-stall",),
    fetch_rules=("latest-safe",),
    evaluate_coalesced=False,
)

DEVICE = DeviceSpec("cuda_0", "process_0", "cuda", 0)
COMPUTE = ResourceSpec("cuda_0", ResourceKind.COMPUTE)


def test_default_repair_budget_covers_deep_monotonic_repairs() -> None:
    assert PressureFitOptions().max_repair_attempts == 256


def test_exact_capacity_schedule_uses_one_legal_round_trip() -> None:
    initial, final = exact_capacity_residency()
    result = pressurefit(
        exact_capacity_program(),
        initial_residency=initial,
        final_residency=final,
        config=config(),
        options=FEW_CANDIDATES,
    )

    assert tuple(
        (action.trigger_task_id, action.alias_group_id, action.kind)
        for action in result.schedule.actions
    ) == (
        ("task0", "later", MemoryActionKind.EVICT),
        ("task1", "temporary", MemoryActionKind.RELEASE),
        ("task1", "later", MemoryActionKind.FETCH),
        ("task2", "later", MemoryActionKind.RELEASE),
    )
    assert result.simulation.makespan_ns == 5_000
    assert result.simulation.device_peak("cuda_0").total_bytes == 122
    assert result.simulation.spill_peak_bytes == 61


def test_latest_safe_fetch_accounts_for_transfer_duration() -> None:
    program = Program(
        devices=(DEVICE,),
        alias_groups=(
            AliasGroupSpec("retained", "cuda_0", 61),
            AliasGroupSpec("later", "cuda_0", 61),
        ),
        objects=(
            ObjectSpec("retained_object", "retained", 0, 61),
            ObjectSpec("later_object", "later", 0, 61),
        ),
        profiles=(TaskProfile("profile", 1_000, 0, "task_abi"),),
        tasks=(
            TaskSpec("task0", COMPUTE, "profile", inputs=("retained_object",)),
            TaskSpec("task1", COMPUTE, "profile"),
            TaskSpec("task2", COMPUTE, "profile"),
            TaskSpec("task3", COMPUTE, "profile", inputs=("later_object",)),
        ),
    )
    result = pressurefit(
        program,
        initial_residency=(
            ResidencySpec("retained", MemoryLocation.DEVICE),
            ResidencySpec("later", MemoryLocation.SPILL),
        ),
        config=config(61),
        options=PressureFitOptions(
            initial_placement=InitialPlacement.REQUIRED,
            residency_strategies=("relaxed-stall",),
            fetch_rules=("latest-safe",),
            evaluate_coalesced=False,
            minimum_object_bytes_evict_eligible=0,
        ),
    )

    assert tuple(
        (action.trigger_task_id, action.alias_group_id, action.kind)
        for action in result.schedule.actions
    ) == (
        ("task0", "retained", MemoryActionKind.RELEASE),
        ("task1", "later", MemoryActionKind.FETCH),
        ("task3", "later", MemoryActionKind.RELEASE),
    )
    assert result.simulation.makespan_ns == 4_000


def test_demand_fetch_uses_final_legal_boundary() -> None:
    program = Program(
        devices=(DEVICE,),
        alias_groups=(
            AliasGroupSpec("retained", "cuda_0", 61),
            AliasGroupSpec("later", "cuda_0", 61),
        ),
        objects=(
            ObjectSpec("retained_object", "retained", 0, 61),
            ObjectSpec("later_object", "later", 0, 61),
        ),
        profiles=(TaskProfile("profile", 1_000, 0, "task_abi"),),
        tasks=(
            TaskSpec("task0", COMPUTE, "profile", inputs=("retained_object",)),
            TaskSpec("task1", COMPUTE, "profile"),
            TaskSpec("task2", COMPUTE, "profile"),
            TaskSpec("task3", COMPUTE, "profile", inputs=("later_object",)),
        ),
    )
    result = pressurefit(
        program,
        initial_residency=(
            ResidencySpec("retained", MemoryLocation.DEVICE),
            ResidencySpec("later", MemoryLocation.SPILL),
        ),
        config=config(61),
        options=PressureFitOptions(
            initial_placement=InitialPlacement.REQUIRED,
            residency_strategies=("relaxed-stall",),
            fetch_rules=("demand",),
            evaluate_coalesced=False,
            minimum_object_bytes_evict_eligible=0,
        ),
    )

    assert tuple(
        (action.trigger_task_id, action.alias_group_id, action.kind)
        for action in result.schedule.actions
    ) == (
        ("task0", "retained", MemoryActionKind.RELEASE),
        ("task2", "later", MemoryActionKind.FETCH),
        ("task3", "later", MemoryActionKind.RELEASE),
    )
    assert result.simulation.makespan_ns == 5_000


def test_zero_size_alias_is_omitted_from_physical_schedule() -> None:
    program = exact_capacity_program()
    program = replace(
        program,
        alias_groups=tuple(
            replace(item, size_bytes=0) if item.alias_group_id == "retained" else item
            for item in program.alias_groups
        ),
        objects=tuple(
            replace(item, size_bytes=0) if item.object_id == "retained_object" else item
            for item in program.objects
        ),
    )
    result = pressurefit(
        program,
        initial_residency=(ResidencySpec("later", MemoryLocation.DEVICE),),
        final_residency=(ResidencySpec("retained", MemoryLocation.DEVICE),),
        config=config(122),
        options=FEW_CANDIDATES,
    )

    assert all(
        item.alias_group_id != "retained"
        for item in (
            *result.schedule.initial_residency,
            *result.schedule.final_residency,
            *result.schedule.actions,
        )
    )


def test_dirty_mutation_requires_writeback_before_reuse() -> None:
    result = pressurefit(
        mutation_program(),
        initial_residency=(ResidencySpec("weight_storage", MemoryLocation.DEVICE),),
        config=config(61),
        options=FEW_CANDIDATES,
    )

    weight_actions = tuple(
        action.kind
        for action in result.schedule.actions
        if action.alias_group_id == "weight_storage"
    )
    assert weight_actions == (
        MemoryActionKind.EVICT,
        MemoryActionKind.FETCH,
        MemoryActionKind.RELEASE,
    )


def test_recomputation_competes_with_evict_among_the_same_candidates() -> None:
    result = pressurefit(
        recomputation_program(),
        initial_residency=(ResidencySpec("input_storage", MemoryLocation.DEVICE),),
        config=config(110),
        options=FEW_CANDIDATES,
    )

    assert tuple((item.group_id, item.option_id) for item in result.selections) == (
        ("activation_tradeoff", "recompute"),
    )
    assert result.diagnostics.selected_selection_id == ("activation_tradeoff=recompute")


def test_result_builds_the_canonical_execution_plan() -> None:
    program = exact_capacity_program()
    initial, final = exact_capacity_residency()
    result = pressurefit(
        program,
        initial_residency=initial,
        final_residency=final,
        config=config(),
        options=FEW_CANDIDATES,
    )
    entrypoints = tuple(
        EntrypointSpec(
            task.task_id,
            f"entrypoint.{task.task_id}",
            "test_executor",
            next(
                profile.compatibility_digest
                for profile in program.profiles
                if profile.profile_id == task.profile_id
            ),
        )
        for task in program.selected_tasks(result.selections)
        if task.requires_entrypoint
    )

    plan = result.to_execution_plan(entrypoints=entrypoints)

    assert plan.schedule is result.schedule
    assert plan.selections == result.selections
    assert plan.prediction.makespan_ns == result.simulation.makespan_ns
    assert plan.prediction.device_peak_bytes == 122


def small_object_program(later_bytes: int) -> Program:
    """The exact-capacity program with `later` shrunk to `later_bytes`."""

    program = exact_capacity_program()
    return replace(
        program,
        alias_groups=tuple(
            replace(group, size_bytes=later_bytes)
            if group.alias_group_id == "later"
            else group
            for group in program.alias_groups
        ),
        objects=tuple(
            replace(item, size_bytes=later_bytes)
            if item.alias_group_id == "later"
            else item
            for item in program.objects
        ),
    )


def test_a_mutated_resident_object_reserves_a_home_per_generation() -> None:
    # The task that mutates `small` in place holds both generations, so the
    # slice reserves two homes for what is one object.
    program = Program(
        devices=(DEVICE,),
        alias_groups=(AliasGroupSpec("small", "cuda_0", 8),),
        objects=(ObjectSpec("small_object", "small", 0, 8),),
        profiles=(TaskProfile("profile", 10, 0, "abi"),),
        tasks=(
            TaskSpec(
                "mutate",
                COMPUTE,
                "profile",
                inputs=("small_object",),
                mutations=(MutationSpec("small_object"),),
            ),
        ),
    )
    result = pressurefit(
        program,
        initial_residency=(ResidencySpec("small", MemoryLocation.DEVICE),),
        final_residency=(ResidencySpec("small", MemoryLocation.DEVICE),),
        config=config(64),
        options=replace(FEW_CANDIDATES, minimum_object_bytes_evict_eligible=9),
    )
    assert result.resident_slice == ResidentSlice(bytes=16, aliases=("small",))


def test_objects_under_the_eligibility_threshold_are_never_cut() -> None:
    # retained (61) is held for the final residency, later (8) is read by the
    # last task, temporary (61) is produced in between: one byte short of
    # holding all three, the cheapest cut is the small object's round trip.
    initial, final = exact_capacity_residency()
    program = small_object_program(8)
    unrestricted = pressurefit(
        program,
        initial_residency=initial,
        final_residency=final,
        config=config(129),
        options=FEW_CANDIDATES,
    )
    moved = {
        (action.alias_group_id, action.kind)
        for action in unrestricted.schedule.actions
        if action.kind != MemoryActionKind.RELEASE
    }
    assert ("later", MemoryActionKind.EVICT) in moved
    assert ("later", MemoryActionKind.FETCH) in moved

    restricted = pressurefit(
        program,
        initial_residency=initial,
        final_residency=final,
        config=config(129),
        options=replace(FEW_CANDIDATES, minimum_object_bytes_evict_eligible=9),
    )
    # The small object keeps its boundary contract, a release after its last
    # read, and nothing else; the cut moves to an eligible object.
    later_actions = [
        (action.trigger_task_id, action.kind)
        for action in restricted.schedule.actions
        if action.alias_group_id == "later"
    ]
    assert later_actions == [("task2", MemoryActionKind.RELEASE)]
    assert ("retained", MemoryActionKind.EVICT) in {
        (action.alias_group_id, action.kind) for action in restricted.schedule.actions
    }
    problem = restricted.diagnostics.recomputation_problems[0]
    assert problem.evict_ineligible_aliases == 1
    assert problem.evict_ineligible_bytes == 8
    # Given a static home of its own, in a slice the main layout never sees.
    assert restricted.resident_slice.bytes == 8
    assert restricted.resident_slice.aliases == ("later",)
    assert unrestricted.resident_slice.bytes == 0
    assert (
        unrestricted.diagnostics.recomputation_problems[0].evict_ineligible_aliases == 0
    )


def test_eligibility_threshold_is_validated() -> None:
    assert PressureFitOptions().minimum_object_bytes_evict_eligible == 1 << 20
    with pytest.raises(ValueError):
        PressureFitOptions(minimum_object_bytes_evict_eligible=-1)
