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
    AdmissionTopology,
    PressureFitDiagnostics,
    PressureFitOptions,
    PressureFitResult,
    TaskAdmissionSpec,
    TaskAllocationStep,
    TaskAllocationStepKind,
)
from shadowspill.pytorch.planning.admission import build_fixed_layout_admission
from shadowspill.simulator import SimulationConfig, simulate
from tests.planner._examples import COMPUTE, DEVICE


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
            candidate_count=1,
            valid_candidate_count=1,
            selected_makespan_ns=simulation.makespan_ns,
            candidates=(),
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
        host_capacity_bytes=128,
        fetch_bandwidth_bytes_per_second=64_000_000_000,
        evict_bandwidth_bytes_per_second=64_000_000_000,
    )
    topology = AdmissionTopology(
        "cuda_0",
        64,
        64,
        1,
        tuple(TaskAdmissionSpec(task.task_id) for task in program.tasks),
    )
    selected = _selected(program, schedule, config)

    admitted = build_fixed_layout_admission(selected, topology)

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
        host_capacity_bytes=0,
        fetch_bandwidth_bytes_per_second=1,
        evict_bandwidth_bytes_per_second=1,
    )
    topology = AdmissionTopology(
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
        topology,
    )

    assert admitted.layout.required_bytes == 32
    assert admitted.layout.task_allocation_leases == (
        ("task", 0, 0),
        ("task", 1, 0),
    )
    assert len(admitted.layout.placements) == 1
    assert admitted.layout.placements[0].offset == 0
