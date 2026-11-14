"""Optimizer reduction/update/publication stay in one ordered compute sequence."""

from itertools import pairwise

import pytest

from shadowspill.ir import MutationSpec, ResourceKind, ResourceSpec, TaskSpec
from shadowspill.planning.training.communication import order_collectives
from shadowspill.planning.training.optimizer import place_optimizer
from shadowspill.planning.training.tasks import TrainingSequence


@pytest.mark.parametrize("ordering", ["tail", "stage_interleaved"])
def test_reductions_stay_with_update_and_do_not_overlap_compute(ordering):
    compute = ResourceSpec("cuda_0", ResourceKind.COMPUTE)
    sequence = TrainingSequence(
        tasks=[
            TaskSpec(
                "invocation_start",
                ResourceSpec("cuda_0", ResourceKind.CONTROL),
                "start",
                requires_entrypoint=False,
            ),
            TaskSpec(
                "backward.0",
                compute,
                "backward",
                inputs=("weight",),
                outputs=("local",),
            ),
            TaskSpec(
                "backward.1",
                compute,
                "backward",
                inputs=("weight", "local"),
                mutations=(MutationSpec("local"),),
            ),
            TaskSpec("unrelated", compute, "backward", outputs=("other",)),
        ]
    )
    template = {
        "name": "weight",
        "source": "local",
        "stored": "reduced",
        "padding": None,
        "phases": [
            {"profile_key": "scatter", "operation": "reduce_scatter"},
            {"profile_key": "reduce", "operation": "all_reduce"},
        ],
    }
    component = {
        "task_id": "update",
        "profile_key": "adam",
        "inputs": [
            {"object_id": "weight", "leaf_index": 0},
            {"object_id": "reduced", "leaf_index": 1},
        ],
        "mutations": ["weight"],
        "reductions": [template],
        "publications": [
            {
                "name": "weight",
                "profile_key": "gather",
                "source": "owned",
                "destination": "weight",
            }
        ],
        "host_inputs": [],
        "replacements": [],
    }
    result = place_optimizer(
        sequence,
        [component],
        ordering=ordering,
        device_id="cuda_0",
        aliases={
            "weight": "w",
            "owned": "w",
            "local": "g",
            "reduced": "r",
            "other": "o",
        },
    )
    order_collectives(result)
    names = [task.task_id for task in result.tasks]
    begin = names.index("gradient.weight.0.reduce_scatter")
    block = result.tasks[begin : begin + 4]
    assert [task.phase for task in block] == [
        "gradient_communication",
        "gradient_communication",
        "optimizer",
        "weight_communication",
    ]
    assert all(task.resource == compute for task in result.tasks[1:])
    assert {"backward.0", "backward.1"} <= set(block[0].dependencies)
    for before, after in pairwise(block):
        assert before.task_id in after.dependencies
    assert begin == (4 if ordering == "tail" else 3)
    assert block[0].outputs == (
        "reduced",
    )  # Do not modify spillable local contributions.
    assert {t.task_id for t in block} <= {e["task_id"] for e in result.entrypoints}
