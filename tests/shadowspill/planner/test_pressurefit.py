from __future__ import annotations

from dataclasses import replace

from shadowspill.ir import (
    AliasGroupSpec,
    DeviceSpec,
    EntrypointSpec,
    MemoryActionKind,
    MemoryLocation,
    ObjectSpec,
    Program,
    ResidencySpec,
    ResourceKind,
    ResourceSpec,
    TaskProfile,
    TaskSpec,
)
from shadowspill.planner import (
    AdmissionFacts,
    InitialPlacement,
    PressureFitOptions,
    TaskAdmissionSpec,
    pressurefit,
)
from shadowspill.planner.pressurefit.refinement import (
    round_up_admission_reserve,
    scheduled_admission_refinement,
    with_object_capacity,
)

from ._examples import (
    config,
    exact_capacity_program,
    exact_capacity_residency,
    mutation_program,
    recomputation_program,
)

FEW_CANDIDATES = PressureFitOptions(
    residency_strategies=("relaxed-stall",),
    prefetch_rules=("latest-safe",),
    evaluate_coalesced=False,
)

DEVICE = DeviceSpec("cuda_0", "process_0", "cuda", 0)
COMPUTE = ResourceSpec("cuda_0", ResourceKind.COMPUTE)


def test_default_repair_budget_covers_deep_monotonic_repairs() -> None:
    assert PressureFitOptions().max_repair_attempts == 64


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
        ("task0", "later", MemoryActionKind.OFFLOAD),
        ("task1", "temporary", MemoryActionKind.RELEASE),
        ("task1", "later", MemoryActionKind.PREFETCH),
        ("task2", "later", MemoryActionKind.RELEASE),
    )
    assert result.simulation.makespan_ns == 5_000
    assert result.simulation.device_peak("cuda_0").total_bytes == 122
    assert result.simulation.spill_peak_bytes == 61


def test_latest_safe_prefetch_accounts_for_transfer_duration() -> None:
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
            prefetch_rules=("latest-safe",),
            evaluate_coalesced=False,
        ),
    )

    assert tuple(
        (action.trigger_task_id, action.alias_group_id, action.kind)
        for action in result.schedule.actions
    ) == (
        ("task0", "retained", MemoryActionKind.RELEASE),
        ("task1", "later", MemoryActionKind.PREFETCH),
        ("task3", "later", MemoryActionKind.RELEASE),
    )
    assert result.simulation.makespan_ns == 4_000


def test_demand_prefetch_uses_final_legal_boundary() -> None:
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
            prefetch_rules=("demand",),
            evaluate_coalesced=False,
        ),
    )

    assert tuple(
        (action.trigger_task_id, action.alias_group_id, action.kind)
        for action in result.schedule.actions
    ) == (
        ("task0", "retained", MemoryActionKind.RELEASE),
        ("task2", "later", MemoryActionKind.PREFETCH),
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
        MemoryActionKind.OFFLOAD,
        MemoryActionKind.PREFETCH,
        MemoryActionKind.RELEASE,
    )


def test_recomputation_competes_with_offload_among_the_same_candidates() -> None:
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


def test_admission_refinement_doubles_to_a_gibibyte_then_grows_by_512_mib() -> None:
    """The ladder is a pure function of the attempt, so ask it directly."""

    increments = tuple(scheduled_admission_refinement(attempt) for attempt in range(6))

    assert increments == (
        128 << 20,
        256 << 20,
        512 << 20,
        1 << 30,
        1536 << 20,
        2 << 30,
    )


def test_admission_reserve_rounds_up_to_its_granularity() -> None:
    granularity = 2 << 20

    assert round_up_admission_reserve(0) == 0
    assert round_up_admission_reserve(1) == granularity
    assert round_up_admission_reserve(granularity) == granularity
    assert round_up_admission_reserve(granularity + 1) == 2 * granularity


def test_object_capacity_is_reduced_on_every_device_and_in_the_facts() -> None:
    program = exact_capacity_program()
    one_gib = 1 << 30
    facts = AdmissionFacts(
        "cuda_0",
        9 * one_gib,
        8 * one_gib,
        1,
        tuple(TaskAdmissionSpec(task.task_id) for task in program.tasks),
    )

    reduced_config, reduced_facts = with_object_capacity(
        config(8 * one_gib),
        facts,
        4 * one_gib,
        shared_execution_bytes=0,
    )

    assert reduced_facts.object_capacity_bytes == 4 * one_gib
    assert {device.capacity_bytes for device in reduced_config.devices} == {4 * one_gib}
