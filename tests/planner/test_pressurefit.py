from __future__ import annotations

from shadowspill.ir import (
    EntrypointSpec,
    MemoryActionKind,
    MemoryLocation,
    ResidencySpec,
)
from shadowspill.planner import PressureFitOptions, pressurefit

from ._examples import (
    config,
    exact_capacity_program,
    exact_capacity_residency,
    mutation_program,
    recomputation_program,
)

SMALL_PORTFOLIO = PressureFitOptions(
    residency_strategies=("relaxed-stall",),
    prefetch_rules=("latest-safe",),
    evaluate_coalesced=False,
)


def test_exact_capacity_schedule_uses_one_legal_round_trip() -> None:
    initial, final = exact_capacity_residency()
    result = pressurefit(
        exact_capacity_program(),
        initial_residency=initial,
        final_residency=final,
        config=config(),
        options=SMALL_PORTFOLIO,
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
    assert result.simulation.host_peak_bytes == 61


def test_dirty_mutation_requires_writeback_before_reuse() -> None:
    result = pressurefit(
        mutation_program(),
        initial_residency=(ResidencySpec("weight_storage", MemoryLocation.DEVICE),),
        config=config(61),
        options=SMALL_PORTFOLIO,
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


def test_recomputation_competes_with_offload_in_the_same_portfolio() -> None:
    result = pressurefit(
        recomputation_program(),
        initial_residency=(ResidencySpec("input_storage", MemoryLocation.DEVICE),),
        config=config(110),
        options=SMALL_PORTFOLIO,
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
        options=SMALL_PORTFOLIO,
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
