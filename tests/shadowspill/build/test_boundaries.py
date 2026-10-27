"""Cross-stage storage dependencies survive compilation and saved-form binding."""

import pytest
import torch
from torch import nn

from shadowspill import build_forward, build_forward_backward
from shadowspill.ir import ObjectRole
from tests.shadowspill.build.test_forward import replay_for_test


class SkipNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.condition = nn.Linear(8, 8, bias=False)
        self.first = nn.Linear(8, 8, bias=False)
        self.second = nn.Linear(8, 8, bias=False)
        self.tied = self.first.weight

    def forward(self, x):
        shift = self.condition(x)
        x = torch.relu(self.first(x) + shift)
        return torch.relu(self.second(x) + shift)[:, 2:6]


@pytest.mark.cuda
def test_skip_inputs_have_producer_edges_and_tied_weights_share_storage():
    model = SkipNetwork()
    program = build_forward(model, example_args=(torch.empty(2, 8),))
    producers = {obj: task.task_id for task in program.tasks for obj in task.outputs}
    for task in program.tasks:
        assert all(
            producers[obj] in task.dependencies
            for obj in task.inputs
            if obj in producers
        )
    assert any(len(task.dependencies) > 1 for task in program.tasks)
    registered = {
        item["name"]: item["object_id"] for item in program.bindings["registrations"]
    }
    assert registered["tied"] == registered["first.weight"]
    parameter_aliases = {
        obj.alias_group_id
        for obj in program.objects
        if obj.role is ObjectRole.PARAMETER
    }
    assert not parameter_aliases.intersection(
        obj.alias_group_id for obj in program.objects if obj.object_id in producers
    )
    model = model.cuda()
    x = torch.randn(2, 8, device="cuda")
    with torch.no_grad():
        torch.testing.assert_close(replay_for_test(program, model, (x,), {}), model(x))


class PassThroughBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(8, 8)

    def forward(self, x, auxiliary):
        x = self.projection(x).relu()
        return x, auxiliary + x.square().mean()


class PassThroughNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(PassThroughBlock() for _ in range(3))

    def forward(self, x, auxiliary):
        for block in self.blocks:
            x, auxiliary = block(x, auxiliary)
        return x, auxiliary


@pytest.mark.cuda
def test_backward_handoffs_preserve_distinct_cotangent_storage():
    program = build_forward_backward(
        PassThroughNetwork(),
        objective=lambda m, x, aux: sum(v.sum() for v in m(x, aux)),
        example_args=(torch.empty(4, 8), torch.empty(())),
        num_examples=2,
    )
    aliases = {obj.object_id: obj.alias_group_id for obj in program.objects}
    tasks = {task.task_id: task for task in program.tasks}
    found = 0
    for entry in program.bindings["entrypoints"]:
        task = tasks[entry["task_id"]]
        if task.phase == "forward":
            assert all(tasks[dep].phase != "backward" for dep in task.dependencies)
        for handoff in entry["storage_handoffs"]:
            source, target = (
                handoff["source_object_id"],
                handoff["destination_object_id"],
            )
            assert source in task.inputs and target in task.outputs
            assert aliases[source] != aliases[target]
            found += 1
    assert found
