from __future__ import annotations

import pytest

from shadowspill.ir import (
    AliasGroupSpec,
    DeviceSpec,
    MemoryAction,
    MemoryActionKind,
    MemoryLocation,
    MemorySchedule,
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
    PressureFitOptions,
    TaskAdmissionSpec,
    pressurefit,
)
from shadowspill.planner._admission import (
    compile_admission_topology,
    encode_schedule,
    evaluate_schedule_admission,
)
from shadowspill.planner._capi import planner_library_path
from shadowspill.pytorch.planning.admission import (
    replay_admission,
    simulation_admission_from_replay,
)
from shadowspill.simulator import SimulationConfig
from shadowspill.simulator._compiled import compile_simulation_template
from tests.planner._examples import (
    training_chain_config,
    training_chain_initial,
    training_chain_program,
)
from tests.simulator.test_admission_accounting import (
    _config as causal_config,
)
from tests.simulator.test_admission_accounting import (
    _program as causal_program,
)
from tests.simulator.test_admission_accounting import (
    _schedule as causal_schedule,
)

pytestmark = pytest.mark.skipif(
    planner_library_path() is None,
    reason="compiled planner library is not installed",
)


def _causal_topology() -> AdmissionTopology:
    program = causal_program()
    return AdmissionTopology(
        "cuda_0",
        96,
        96,
        1,
        tuple(TaskAdmissionSpec(task.task_id) for task in program.tasks),
    )


def test_compiled_selected_admission_matches_python_oracle() -> None:
    program = causal_program()
    schedule = causal_schedule()
    topology = _causal_topology()
    template = compile_simulation_template(program, (), causal_config())

    compiled = evaluate_schedule_admission(
        template,
        compile_admission_topology(topology, template),
        encode_schedule(schedule, template),
    )
    replay = replay_admission(
        program,
        schedule,
        execution_pool_bytes=96,
        topology=topology,
    )
    reference = simulation_admission_from_replay(
        replay,
        program,
        schedule,
        device_capacity_bytes=96,
    )

    assert compiled.simulation_admission == reference
    assert compiled.decision_digest == replay.pool.decision_digest
    assert compiled.peak_allocated_bytes == replay.pool.peak_allocated_bytes
    assert compiled.peak_reserved_bytes == replay.pool.peak_reserved_bytes
    assert compiled.peak_fragmentation_bytes == replay.pool.peak_fragmentation_bytes


def test_pressurefit_publishes_the_same_admission_aware_selected_result() -> None:
    program = training_chain_program(2)
    object_alias = {
        item.object_id: item.alias_group_id for item in program.objects
    }
    profiles = {item.profile_id: item for item in program.profiles}
    topology = AdmissionTopology(
        "cuda_0",
        512,
        224,
        1,
        tuple(
            TaskAdmissionSpec(
                task_id=task.task_id,
                workspace_extents=(
                    (profiles[task.profile_id].workspace_bytes,)
                    if profiles[task.profile_id].workspace_bytes
                    else ()
                ),
                fresh_output_aliases=tuple(
                    dict.fromkeys(object_alias[item] for item in task.outputs)
                ),
            )
            for task in program.tasks
        ),
    )

    result = pressurefit(
        program,
        initial_residency=training_chain_initial(2),
        config=training_chain_config(224),
        admission=topology,
        options=PressureFitOptions(workers=1),
    )

    assert result.simulation.device_peak("cuda_0").total_bytes > 224
    assert result.simulation.device_peak("cuda_0").total_bytes <= 512
    assert result.diagnostics.selected_makespan_ns == result.simulation.makespan_ns


def test_compiled_admission_places_workspace_across_fragmented_ranges() -> None:
    compute = ResourceSpec("cuda_0", ResourceKind.COMPUTE)
    program = Program(
        devices=(DeviceSpec("cuda_0", "process_0", "cuda", 0),),
        alias_groups=tuple(
            AliasGroupSpec(alias, "cuda_0", 32) for alias in ("a", "b", "c")
        ),
        objects=tuple(ObjectSpec(alias, alias, 0, 32) for alias in ("a", "b", "c")),
        profiles=(TaskProfile("profile", 1, 64, "abi"),),
        tasks=(
            TaskSpec("release_middle", compute, "profile", inputs=("b",)),
            TaskSpec(
                "use_workspace",
                compute,
                "profile",
                dependencies=("release_middle",),
            ),
        ),
    )
    schedule = MemorySchedule(
        initial_residency=tuple(
            ResidencySpec(alias, MemoryLocation.DEVICE) for alias in ("a", "b", "c")
        ),
        actions=(MemoryAction("release_middle", "b", MemoryActionKind.RELEASE),),
        final_residency=(
            ResidencySpec("a", MemoryLocation.DEVICE),
            ResidencySpec("c", MemoryLocation.DEVICE),
        ),
    )
    config = SimulationConfig.single_device(
        "cuda_0",
        device_capacity_bytes=128,
        host_capacity_bytes=1,
        fetch_bandwidth_bytes_per_second=1,
        evict_bandwidth_bytes_per_second=1,
    )
    template = compile_simulation_template(program, (), config)

    def evaluate(extents: tuple[int, ...]):
        topology = AdmissionTopology(
            "cuda_0",
            128,
            128,
            1,
            (
                TaskAdmissionSpec("release_middle"),
                TaskAdmissionSpec("use_workspace", workspace_extents=extents),
            ),
        )
        return evaluate_schedule_admission(
            template,
            compile_admission_topology(topology, template),
            encode_schedule(schedule, template),
        )

    admitted = evaluate((32, 32))

    assert admitted.peak_allocated_bytes == 128
    with pytest.raises(ValueError, match="dynamic MemoryPool admission"):
        evaluate((64,))
