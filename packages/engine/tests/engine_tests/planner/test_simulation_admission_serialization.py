"""Physical reuse dependencies retain task and transfer completion boundaries."""

import json

import pytest

from shadowspill.planner.serialization.simulation_admission import (
    _simulation_admission_from_value,
    simulation_admission_value,
)
from shadowspill.simulator import MemoryReuseDependency, SimulationAdmission


@pytest.mark.parametrize(
    "dependency",
    [
        MemoryReuseDependency(
            predecessor_task_id="backward", successor_task_id="update"
        ),
        MemoryReuseDependency(predecessor_task_id="backward", successor_action_index=1),
        MemoryReuseDependency(predecessor_action_index=0, successor_task_id="update"),
        MemoryReuseDependency(predecessor_action_index=0, successor_action_index=1),
    ],
)
def test_reuse_completion_round_trip(dependency):
    admission = SimulationAdmission(
        initial_physical_bytes=(("cuda_0", 0),),
        reuse_dependencies=(dependency,),
    )
    value = json.loads(json.dumps(simulation_admission_value(admission)))

    assert _simulation_admission_from_value(value, "admission") == admission


def test_missing_reuse_predecessor_is_rejected():
    admission = SimulationAdmission(
        initial_physical_bytes=(("cuda_0", 0),),
        reuse_dependencies=(
            MemoryReuseDependency(
                predecessor_task_id="backward", successor_task_id="update"
            ),
        ),
    )
    value = json.loads(json.dumps(simulation_admission_value(admission)))
    value["reuse_dependencies"][0].pop("predecessor_task_id")

    with pytest.raises(ValueError, match="exactly one memory-reuse predecessor"):
        _simulation_admission_from_value(value, "admission")
