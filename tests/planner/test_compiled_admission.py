from __future__ import annotations

import pytest

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
        tuple(TaskAdmissionSpec(task.task_id, 0) for task in program.tasks),
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
                task.task_id,
                profiles[task.profile_id].workspace_bytes,
                tuple(dict.fromkeys(object_alias[item] for item in task.outputs)),
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
