"""Communication completion protects shared physical memory across lanes."""

from dataclasses import replace

import pytest

from reference.python.simulator import simulate_python as reference_simulate
from shadowspill.ir import (
    AliasGroupSpec,
    DeviceSpec,
    MemoryAction,
    MemoryActionKind,
    MemoryLocation,
    MemorySchedule,
    ObjectSpec,
    ResidencySpec,
    ResourceKind,
    ResourceSpec,
    ShadowSpillProgram,
    TaskProfile,
    TaskSpec,
)
from shadowspill.simulator import (
    MemoryReuseDependency,
    SimulationAdmission,
    SimulationConfig,
    simulate,
)


def _fixture(duration):
    compute = ResourceSpec("gpu", ResourceKind.COMPUTE)
    comm = ResourceSpec("gpu", ResourceKind.COMMUNICATION)
    program = ShadowSpillProgram(
        devices=(DeviceSpec("gpu", "worker", "cuda", 0),),
        alias_groups=(AliasGroupSpec("fetched", "gpu", 16),),
        objects=(ObjectSpec("input", "fetched", 0, 16),),
        profiles=(
            TaskProfile("slow", duration, 32, "a"),
            TaskProfile("fast", 10, 0, "b"),
        ),
        tasks=(
            TaskSpec("communication", comm, "slow"),
            TaskSpec("compute", compute, "fast"),
            TaskSpec(
                "consumer",
                compute,
                "fast",
                inputs=("input",),
                dependencies=("compute",),
            ),
        ),
    )
    schedule = MemorySchedule(
        initial_residency=(ResidencySpec("fetched", MemoryLocation.SPILL),),
        actions=(MemoryAction("compute", "fetched", MemoryActionKind.FETCH),),
        final_residency=(ResidencySpec("fetched", MemoryLocation.DEVICE),),
    )
    config = SimulationConfig.single_device(
        "gpu",
        device_capacity_bytes=64,
        spill_capacity_bytes=16,
        fetch_solo_bandwidth_bytes_per_second=1_000_000_000,
        fetch_concurrent_bandwidth_bytes_per_second=1_000_000_000,
        evict_solo_bandwidth_bytes_per_second=1_000_000_000,
        evict_concurrent_bandwidth_bytes_per_second=1_000_000_000,
    )
    return program, schedule, config


@pytest.mark.parametrize("duration", [100, 10_000])
@pytest.mark.parametrize("successor", ["task", "fetch"])
def test_task_completion_blocks_only_the_reused_allocation(duration, successor):
    program, schedule, config = _fixture(duration)
    edge = MemoryReuseDependency(
        predecessor_task_id="communication",
        successor_task_id="consumer" if successor == "task" else None,
        successor_action_index=0 if successor == "fetch" else None,
    )
    admission = SimulationAdmission(
        initial_physical_bytes=(("gpu", 64),),
        unlisted_delta_bytes=0,
        reuse_dependencies=(edge,),
    )
    for engine in (simulate, reference_simulate):
        result = engine(program, schedule, config=config, admission=admission)
        tasks = {task.task_id: task for task in result.task_intervals}
        assert tasks["compute"].start_ns == tasks["communication"].start_ns == 0
        assert tasks["compute"].end_ns == 10
        assert result.transfer_intervals[0].start_ns == (
            duration if successor == "fetch" else 10
        )
        assert tasks["consumer"].start_ns == duration + (
            16 if successor == "fetch" else 0
        )
        # Without the certificate these physical addresses would be reused
        # prematurely; independent computation must still overlap communication.
        unsafe = engine(
            program,
            schedule,
            config=config,
            admission=replace(admission, reuse_dependencies=()),
        )
        assert (
            next(
                task for task in unsafe.task_intervals if task.task_id == "consumer"
            ).start_ns
            == 26
        )


def test_task_reuse_validation():
    with pytest.raises(ValueError, match="predecessor"):
        MemoryReuseDependency(successor_task_id="consumer")
    with pytest.raises(ValueError, match="predecessor"):
        MemoryReuseDependency(
            0, predecessor_task_id="communication", successor_task_id="consumer"
        )
    with pytest.raises(ValueError, match="own completion"):
        MemoryReuseDependency(predecessor_task_id="same", successor_task_id="same")
    program, schedule, config = _fixture(100)
    admission = SimulationAdmission(
        initial_physical_bytes=(("gpu", 0),),
        reuse_dependencies=(
            MemoryReuseDependency(
                predecessor_task_id="missing", successor_task_id="consumer"
            ),
        ),
    )
    for engine in (simulate, reference_simulate):
        with pytest.raises(ValueError, match=r"predecessor task.*unknown"):
            engine(program, schedule, config=config, admission=admission)
