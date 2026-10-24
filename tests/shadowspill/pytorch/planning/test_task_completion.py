"""A fixed arena must retain communication-completion proofs across lanes."""

from shadowspill.ir import (
    DeviceSpec,
    MemorySchedule,
    ResourceKind,
    ResourceSpec,
    ShadowSpillProgram,
    TaskProfile,
    TaskSpec,
)
from shadowspill.pipeline.admission import project_runtime_fixed_layout
from shadowspill.planner import (
    AdmissionFacts,
    TaskAdmissionSpec,
    TaskAllocationStep,
    TaskAllocationStepKind,
)
from shadowspill.planner.admission.indexing import (
    encode_schedule,
    evaluate_schedule_admission,
    index_admission_facts,
)
from shadowspill.pytorch.planning.admission import build_fixed_layout_admission
from shadowspill.runtime.admission_capi import NO_ID
from shadowspill.runtime.fixed_layout import RuntimePlacementKind
from shadowspill.simulator import SimulationConfig
from shadowspill.simulator.indexing import index_simulation_template
from tests.shadowspill.pytorch.planning.test_fixed_layout import _selected


def test_fixed_arena_preserves_task_reuse_and_data_completion():
    compute = ResourceSpec("cuda_0", ResourceKind.COMPUTE)
    communication = ResourceSpec("cuda_0", ResourceKind.COMMUNICATION)
    program = ShadowSpillProgram(
        devices=(DeviceSpec("cuda_0", "worker", "cuda", 0),),
        alias_groups=(),
        objects=(),
        profiles=(
            TaskProfile("slow", 100, 64, "a"),
            TaskProfile("fast", 10, 64, "b"),
            TaskProfile("empty", 10, 0, "c"),
        ),
        tasks=(
            TaskSpec("task_000001", communication, "slow"),
            TaskSpec("task_000002", compute, "fast"),
            TaskSpec("task_000003", compute, "empty", dependencies=("task_000001",)),
        ),
    )
    schedule = MemorySchedule((), (), ())
    config = SimulationConfig.single_device(
        "cuda_0",
        device_capacity_bytes=64,
        spill_capacity_bytes=64,
        fetch_solo_bandwidth_bytes_per_second=1_000_000_000,
        fetch_concurrent_bandwidth_bytes_per_second=1_000_000_000,
        evict_solo_bandwidth_bytes_per_second=1_000_000_000,
        evict_concurrent_bandwidth_bytes_per_second=1_000_000_000,
    )
    contract = (
        TaskAllocationStep(0, TaskAllocationStepKind.ALLOCATE, 64),
        TaskAllocationStep(0, TaskAllocationStepKind.RELEASE),
    )
    facts = AdmissionFacts(
        "cuda_0",
        64,
        64,
        1,
        (
            TaskAdmissionSpec(
                "task_000001", workspace_extents=(64,), allocation_steps=contract
            ),
            TaskAdmissionSpec(
                "task_000002", workspace_extents=(64,), allocation_steps=contract
            ),
            TaskAdmissionSpec("task_000003"),
        ),
    )
    admitted = build_fixed_layout_admission(_selected(program, schedule, config), facts)
    # There are no transfer actions, so the initial zero-capacity edge arrays
    # must grow to return the task-completion dependency.
    template = index_simulation_template(program, (), config)
    compiled = evaluate_schedule_admission(
        template,
        index_admission_facts(facts, template),
        encode_schedule(schedule, template),
    )
    assert any(
        edge.predecessor_task_id == "task_000001"
        for edge in compiled.simulation_admission.reuse_dependencies
    )
    assert admitted.layout.fixed_slice_bytes == 64
    edges = admitted.simulator_input.reuse_dependencies
    assert any(
        edge.predecessor_task_id == "task_000001"
        and edge.successor_task_id == "task_000002"
        for edge in edges
    )
    assert admitted.simulation.makespan_ns == 120
    runtime = project_runtime_fixed_layout(admitted.layout, program, schedule)
    assert {
        (
            edge.predecessor_task_id,
            edge.predecessor_action_ordinal,
            edge.successor_task_id,
            edge.successor_kind,
        )
        for edge in runtime.dependencies
    } == {
        (1, NO_ID, 2, RuntimePlacementKind.TASK_ALLOCATION),
        (1, NO_ID, 3, RuntimePlacementKind.TASK_START),
    }
