"""Ready reductions overlap later backward work without including old gradients."""

import pytest

from shadowspill.ir import MutationSpec, ResourceKind, ResourceSpec, TaskSpec
from shadowspill.planning.training.gradients import complete_gradients
from shadowspill.planning.training.tasks import TrainingSequence


@pytest.mark.parametrize("accumulate", [False, True])
def test_reduce_after_last_contribution_not_parameter_declaration_order(accumulate):
    compute = ResourceSpec("cuda_0", ResourceKind.COMPUTE)
    sequence = TrainingSequence(
        tasks=[
            TaskSpec(
                "invocation_start",
                ResourceSpec("cuda_0", ResourceKind.CONTROL),
                "start",
                requires_entrypoint=False,
            ),
            TaskSpec("late.0", compute, "backward", outputs=("late.local",)),
            TaskSpec("early.0", compute, "backward", outputs=("early.local",)),
            TaskSpec(
                "late.1",
                compute,
                "backward",
                inputs=("late.local",),
                mutations=(MutationSpec("late.local"),),
            ),
            TaskSpec(
                "early.1",
                compute,
                "backward",
                inputs=("early.local",),
                mutations=(MutationSpec("early.local"),),
            ),
        ]
    )
    # Declaration order is opposite to readiness. Both parameters also get an
    # all-reduce between corresponding shard owners in repeated replicas.
    templates = [
        {
            "name": name,
            "source": name + ".local",
            "stored": name + ".stored",
            "new": name + ".new",
            "padding": None,
            "phases": [
                {"profile_key": "scatter", "operation": "reduce_scatter"},
                {"profile_key": "reduce", "operation": "all_reduce"},
            ],
            "accumulation_key": "add",
        }
        for name in ("early", "late")
    ]
    result = complete_gradients(
        sequence, templates, accumulate=accumulate, device_id="cuda_0"
    )
    tasks = {task.task_id: task for task in result.tasks}
    order = tuple(tasks)
    assert order.index("late.1") < order.index("gradient.late.0.reduce_scatter")
    assert order.index("gradient.late.1.all_reduce") < order.index("early.1")
    for name in ("early", "late"):
        scatter = tasks[f"gradient.{name}.0.reduce_scatter"]
        reduction = tasks[f"gradient.{name}.1.all_reduce"]
        assert {name + ".0", name + ".1"}.issubset(scatter.dependencies)
        assert scatter.inputs == (name + ".local",)
        destination = name + (".new" if accumulate else ".stored")
        assert scatter.outputs == reduction.inputs == (destination,)
        assert scatter.task_id in reduction.dependencies
        if accumulate:
            addition = tasks[f"gradient.{name}.accumulate"]
            assert addition.inputs == (name + ".stored", name + ".new")
            assert {scatter.task_id, reduction.task_id}.issubset(addition.dependencies)
        else:
            assert f"gradient.{name}.accumulate" not in tasks
    assert (
        "gradient.late.1.all_reduce"
        in tasks["gradient.early.0.reduce_scatter"].dependencies
    )
