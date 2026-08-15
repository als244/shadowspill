from __future__ import annotations

import importlib
from dataclasses import replace

import pytest

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
    AdmissionTopology,
    InitialPlacement,
    PressureFitInfeasibleError,
    PressureFitOptions,
    TaskAdmissionSpec,
    pressurefit,
)

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

DEVICE = DeviceSpec("cuda_0", "process_0", "cuda", 0)
COMPUTE = ResourceSpec("cuda_0", ResourceKind.COMPUTE)


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
            ResidencySpec("later", MemoryLocation.HOST),
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
            ResidencySpec("later", MemoryLocation.HOST),
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
        options=SMALL_PORTFOLIO,
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


def test_physical_admission_refinement_doubles_then_grows_by_512_mib(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    program = exact_capacity_program()
    initial, final = exact_capacity_residency()
    baseline = pressurefit(
        program,
        initial_residency=initial,
        final_residency=final,
        config=config(),
        options=SMALL_PORTFOLIO,
    )
    one_gib = 1 << 30
    eight_gib = 8 << 30
    topology = AdmissionTopology(
        "cuda_0",
        9 * one_gib,
        eight_gib,
        1,
        tuple(TaskAdmissionSpec(task.task_id) for task in program.tasks),
    )
    capacities: list[int] = []

    def fake_once(*args: object, **kwargs: object):
        admission = kwargs["admission"]
        assert isinstance(admission, AdmissionTopology)
        capacities.append(admission.object_capacity_bytes)
        if len(capacities) <= 6:
            raise PressureFitInfeasibleError(
                "synthetic physical admission failure",
                kind="physical_admission",
                required_bytes=1,
            )
        return baseline

    module = importlib.import_module("shadowspill.planner.pressurefit")
    monkeypatch.setattr(module, "_pressurefit_once", fake_once)
    result = pressurefit(
        program,
        initial_residency=initial,
        final_residency=final,
        config=config(eight_gib),
        options=SMALL_PORTFOLIO,
        admission=topology,
    )

    increments = (
        128 << 20,
        256 << 20,
        512 << 20,
        1 << 30,
        1536 << 20,
        2 << 30,
    )
    cumulative = 0
    expected_capacities = [eight_gib]
    for increment in increments:
        cumulative += increment
        expected_capacities.append(eight_gib - cumulative)
    assert capacities == expected_capacities
    assert tuple(
        item.reserve_increment_bytes
        for item in result.diagnostics.admission_refinements
    ) == increments
    assert result.diagnostics.effective_object_capacity_bytes == eight_gib - cumulative
