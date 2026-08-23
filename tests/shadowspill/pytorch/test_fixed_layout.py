from __future__ import annotations

from shadowspill.ir import (
    AliasGroupSpec,
    MemoryAction,
    MemoryActionKind,
    MemoryLocation,
    MemorySchedule,
    ObjectSpec,
    Program,
    ResidencySpec,
    TaskProfile,
    TaskSpec,
)
from shadowspill.planner import (
    AdmissionFacts,
    CandidateDiagnostic,
    PressureFitDiagnostics,
    PressureFitOptions,
    PressureFitResult,
    RecomputationProblemDiagnostics,
    TaskAdmissionSpec,
    TaskAllocationStep,
    TaskAllocationStepKind,
)
from shadowspill.pytorch.planning.admission import (
    DynamicTaskAllocationPolicy,
    build_fixed_layout_admission,
    project_runtime_fixed_layout,
)
from shadowspill.pytorch.runtime_adapter import RuntimePlacementKind
from shadowspill.simulator import SimulationConfig, simulate
from tests.shadowspill.planner._examples import COMPUTE, DEVICE


def _selected(
    program: Program,
    schedule: MemorySchedule,
    config: SimulationConfig,
) -> PressureFitResult:
    simulation = simulate(program, schedule, config=config)
    return PressureFitResult(
        program=program,
        options=PressureFitOptions(workers=1),
        initial_residency=schedule.initial_residency,
        final_residency=schedule.final_residency,
        simulation_config=config,
        schedule=schedule,
        selections=(),
        simulation=simulation,
        diagnostics=PressureFitDiagnostics(
            selected_candidate_id="fixture",
            selected_selection_id="fixture",
            selected_makespan_ns=simulation.makespan_ns,
            recomputation_problems=(
                RecomputationProblemDiagnostics(
                    selection_id="fixture",
                    choices=(),
                    selected_candidate_id="fixture",
                    selected_makespan_ns=simulation.makespan_ns,
                    candidate_evaluations=(
                        CandidateDiagnostic(
                            candidate_id="fixture",
                            selection_id="fixture",
                            status="valid",
                            makespan_ns=simulation.makespan_ns,
                        ),
                    ),
                ),
            ),
        ),
    )


def test_fixed_layout_reuses_completed_eviction_without_changing_makespan() -> None:
    program = Program(
        devices=(DEVICE,),
        alias_groups=(AliasGroupSpec("state", "cuda_0", 64, retain_spill_copy=True),),
        objects=(ObjectSpec("state_object", "state", 0, 64),),
        profiles=(TaskProfile("profile", 10, 0, "abi"),),
        tasks=(
            TaskSpec("produce", COMPUTE, "profile", inputs=("state_object",)),
            TaskSpec(
                "trigger",
                COMPUTE,
                "profile",
                dependencies=("produce",),
            ),
            TaskSpec(
                "consume",
                COMPUTE,
                "profile",
                dependencies=("trigger",),
                inputs=("state_object",),
            ),
        ),
    )
    schedule = MemorySchedule(
        initial_residency=(ResidencySpec("state", MemoryLocation.DEVICE),),
        actions=(
            MemoryAction("produce", "state", MemoryActionKind.OFFLOAD),
            MemoryAction("trigger", "state", MemoryActionKind.PREFETCH),
        ),
        final_residency=(ResidencySpec("state", MemoryLocation.DEVICE),),
    )
    config = SimulationConfig.single_device(
        "cuda_0",
        device_capacity_bytes=64,
        spill_capacity_bytes=128,
        fetch_bandwidth_bytes_per_second=64_000_000_000,
        evict_bandwidth_bytes_per_second=64_000_000_000,
    )
    facts = AdmissionFacts(
        "cuda_0",
        64,
        64,
        1,
        tuple(TaskAdmissionSpec(task.task_id) for task in program.tasks),
    )
    selected = _selected(program, schedule, config)

    admitted = build_fixed_layout_admission(selected, facts)

    assert admitted.layout.required_bytes == 64
    assert admitted.layout.slack_bytes == 0
    assert admitted.layout.initial_alias_leases == (("state", 0),)
    assert admitted.layout.action_destination_leases == ((1, 1),)
    assert len(admitted.layout.reuse_dependencies) == 1
    assert len(admitted.simulator_input.reuse_dependencies) == 1
    assert admitted.simulation.makespan_ns == selected.simulation.makespan_ns


def test_fixed_layout_maps_same_task_allocator_reuse_to_one_lease() -> None:
    program = Program(
        devices=(DEVICE,),
        alias_groups=(),
        objects=(),
        profiles=(TaskProfile("profile", 10, 32, "abi"),),
        tasks=(TaskSpec("task", COMPUTE, "profile"),),
    )
    schedule = MemorySchedule((), (), ())
    config = SimulationConfig.single_device(
        "cuda_0",
        device_capacity_bytes=32,
        spill_capacity_bytes=0,
        fetch_bandwidth_bytes_per_second=1,
        evict_bandwidth_bytes_per_second=1,
    )
    facts = AdmissionFacts(
        "cuda_0",
        32,
        32,
        1,
        (
            TaskAdmissionSpec(
                "task",
                workspace_extents=(32,),
                allocation_steps=(
                    TaskAllocationStep(
                        0,
                        TaskAllocationStepKind.ALLOCATE,
                        32,
                    ),
                    TaskAllocationStep(0, TaskAllocationStepKind.RELEASE),
                    TaskAllocationStep(
                        1,
                        TaskAllocationStepKind.ALLOCATE,
                        32,
                        reuses_allocation_ordinal=0,
                    ),
                    TaskAllocationStep(1, TaskAllocationStepKind.RELEASE),
                ),
            ),
        ),
    )

    admitted = build_fixed_layout_admission(
        _selected(program, schedule, config),
        facts,
    )

    assert admitted.layout.required_bytes == 32
    assert admitted.layout.task_allocation_leases == (
        ("task", 0, 0),
        ("task", 1, 0),
    )
    assert len(admitted.layout.placements) == 1
    assert admitted.layout.placements[0].offset == 0


def test_fixed_layout_keeps_caller_owned_output_outside_reusable_slice() -> None:
    program = Program(
        devices=(DEVICE,),
        alias_groups=(AliasGroupSpec("alias_000000", "cuda_0", 8),),
        objects=(ObjectSpec("object_000000", "alias_000000", 0, 8),),
        profiles=(TaskProfile("profile", 10, 0, "abi"),),
        tasks=(
            TaskSpec(
                "task_000000",
                COMPUTE,
                "profile",
                outputs=("object_000000",),
            ),
        ),
    )
    schedule = MemorySchedule(
        (),
        (),
        (ResidencySpec("alias_000000", MemoryLocation.DEVICE),),
    )
    config = SimulationConfig.single_device(
        "cuda_0",
        device_capacity_bytes=8,
        spill_capacity_bytes=0,
        fetch_bandwidth_bytes_per_second=1,
        evict_bandwidth_bytes_per_second=1,
    )
    facts = AdmissionFacts(
        "cuda_0",
        8,
        8,
        1,
        (
            TaskAdmissionSpec(
                "task_000000",
                fresh_output_aliases=("alias_000000",),
                allocation_steps=(
                    TaskAllocationStep(
                        0,
                        TaskAllocationStepKind.ALLOCATE,
                        8,
                        output_alias_group_id="alias_000000",
                    ),
                ),
            ),
        ),
    )

    admitted = build_fixed_layout_admission(
        _selected(program, schedule, config),
        facts,
        dynamic_alias_group_ids=frozenset({"alias_000000"}),
    )

    assert admitted.layout.fixed_slice_bytes == 0
    assert admitted.layout.dynamic_reserve_bytes == 8
    assert admitted.layout.required_bytes == 8
    assert admitted.layout.dynamic_lease_ids == (0,)
    assert admitted.layout.dynamic_lifetimes[0].bytes == 8
    assert admitted.layout.dynamic_lifetimes[0].task_id == "task_000000"
    assert admitted.layout.placements == ()
    runtime = project_runtime_fixed_layout(
        admitted.layout,
        program,
        schedule,
        initial_task_id=1 << 60,
        dynamic_task_allocations=(
            DynamicTaskAllocationPolicy("task_000000", 1, 32, 256),
        ),
    )
    assert runtime.slice_bytes == 0
    assert runtime.dependencies == ()
    assert len(runtime.placements) == 2
    assert all(
        item.kind is RuntimePlacementKind.DYNAMIC_TASK_ALLOCATION
        for item in runtime.placements
    )
    assert tuple(
        (item.task_id, item.ordinal, item.bytes) for item in runtime.placements
    ) == ((0, 0, 8), (0, 1, 32))


def test_fixed_layout_keeps_only_final_fetched_output_lease_dynamic() -> None:
    program = Program(
        devices=(DEVICE,),
        alias_groups=(
            AliasGroupSpec(
                "alias_000000",
                "cuda_0",
                8,
                retain_spill_copy=True,
            ),
        ),
        objects=(ObjectSpec("object_000000", "alias_000000", 0, 8),),
        profiles=(TaskProfile("profile", 10, 0, "abi"),),
        tasks=(
            TaskSpec(
                "task_000000",
                COMPUTE,
                "profile",
                outputs=("object_000000",),
            ),
            TaskSpec(
                "task_000001",
                COMPUTE,
                "profile",
                dependencies=("task_000000",),
            ),
        ),
    )
    schedule = MemorySchedule(
        (),
        (
            MemoryAction(
                "task_000000",
                "alias_000000",
                MemoryActionKind.OFFLOAD,
            ),
            MemoryAction(
                "task_000001",
                "alias_000000",
                MemoryActionKind.PREFETCH,
            ),
        ),
        (ResidencySpec("alias_000000", MemoryLocation.DEVICE),),
    )
    config = SimulationConfig.single_device(
        "cuda_0",
        device_capacity_bytes=16,
        spill_capacity_bytes=16,
        fetch_bandwidth_bytes_per_second=1_000_000_000,
        evict_bandwidth_bytes_per_second=1_000_000_000,
    )
    facts = AdmissionFacts(
        "cuda_0",
        16,
        16,
        1,
        (
            TaskAdmissionSpec(
                "task_000000",
                fresh_output_aliases=("alias_000000",),
                allocation_steps=(
                    TaskAllocationStep(
                        0,
                        TaskAllocationStepKind.ALLOCATE,
                        8,
                        output_alias_group_id="alias_000000",
                    ),
                ),
            ),
            TaskAdmissionSpec("task_000001"),
        ),
    )

    admitted = build_fixed_layout_admission(
        _selected(program, schedule, config),
        facts,
        dynamic_alias_group_ids=frozenset({"alias_000000"}),
    )

    assert admitted.layout.fixed_slice_bytes == 8
    assert admitted.layout.dynamic_reserve_bytes == 8
    assert admitted.layout.required_bytes == 16
    assert len(admitted.layout.dynamic_lifetimes) == 1
    dynamic = admitted.layout.dynamic_lifetimes[0]
    assert dynamic.purpose.value == "fetch_destination"
    assert dynamic.action_index == 1
    assert len(admitted.layout.placements) == 1
    assert admitted.layout.placements[0].purpose.value == "task_output"

    runtime = project_runtime_fixed_layout(
        admitted.layout,
        program,
        schedule,
        initial_task_id=1 << 60,
    )
    assert len(runtime.placements) == 2
    task_output = next(
        item
        for item in runtime.placements
        if item.kind is RuntimePlacementKind.TASK_ALLOCATION
    )
    fetched_output = next(
        item
        for item in runtime.placements
        if item.kind is RuntimePlacementKind.DYNAMIC_ACTION_DESTINATION
    )
    assert task_output.task_id == 0
    assert fetched_output.task_id == 1
    assert fetched_output.ordinal == 0
    assert fetched_output.object_id == 0
    assert fetched_output.bytes == 8


def test_fixed_layout_projects_eviction_reuse_to_indexed_runtime_ids() -> None:
    program = Program(
        devices=(DEVICE,),
        alias_groups=(
            AliasGroupSpec(
                "alias_000000",
                "cuda_0",
                64,
                retain_spill_copy=True,
            ),
        ),
        objects=(ObjectSpec("object_000000", "alias_000000", 0, 64),),
        profiles=(TaskProfile("profile", 10, 0, "abi"),),
        tasks=(
            TaskSpec(
                "task_000000",
                COMPUTE,
                "profile",
                inputs=("object_000000",),
            ),
            TaskSpec(
                "task_000001",
                COMPUTE,
                "profile",
                dependencies=("task_000000",),
            ),
            TaskSpec(
                "task_000002",
                COMPUTE,
                "profile",
                dependencies=("task_000001",),
                inputs=("object_000000",),
            ),
        ),
    )
    schedule = MemorySchedule(
        initial_residency=(ResidencySpec("alias_000000", MemoryLocation.DEVICE),),
        actions=(
            MemoryAction(
                "task_000000",
                "alias_000000",
                MemoryActionKind.OFFLOAD,
            ),
            MemoryAction(
                "task_000001",
                "alias_000000",
                MemoryActionKind.PREFETCH,
            ),
        ),
        final_residency=(ResidencySpec("alias_000000", MemoryLocation.DEVICE),),
    )
    config = SimulationConfig.single_device(
        "cuda_0",
        device_capacity_bytes=64,
        spill_capacity_bytes=128,
        fetch_bandwidth_bytes_per_second=64_000_000_000,
        evict_bandwidth_bytes_per_second=64_000_000_000,
    )
    facts = AdmissionFacts(
        "cuda_0",
        64,
        64,
        1,
        tuple(TaskAdmissionSpec(task.task_id) for task in program.tasks),
    )
    admitted = build_fixed_layout_admission(
        _selected(program, schedule, config), facts
    )

    runtime = project_runtime_fixed_layout(
        admitted.layout,
        program,
        schedule,
        initial_task_id=1 << 60,
    )

    assert len(runtime.placements) == 2
    initial = next(item for item in runtime.placements if item.task_id == 1 << 60)
    scheduled = next(item for item in runtime.placements if item.task_id == 1)
    assert initial.ordinal == 0
    assert initial.kind is RuntimePlacementKind.ACTION_DESTINATION
    assert scheduled.ordinal == 0
    assert len(runtime.dependencies) == 1
    dependency = runtime.dependencies[0]
    assert dependency.predecessor_task_id == 0
    assert dependency.predecessor_action_ordinal == 0
    assert dependency.successor_task_id == 1
    assert dependency.successor_ordinal == 0
    assert dependency.successor_kind is RuntimePlacementKind.ACTION_DESTINATION
