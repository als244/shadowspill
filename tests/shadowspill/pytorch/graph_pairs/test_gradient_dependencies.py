"""Only derivatives connected to the objective belong to task boundaries."""

import pytest
import torch
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx

from shadowspill.pytorch.capture.aot import capture_graph_pair
from shadowspill.pytorch.capture.fake import fake_device_inputs, fake_device_model
from shadowspill.pytorch.capture.objective import (
    capture_training,
    capture_training_objective,
)
from shadowspill.pytorch.graph_pairs import partition_training_capture


def test_selecting_roots_does_not_leave_zero_tangent_arguments():
    x = torch.randn(5, requires_grad=True)

    def function(x):
        return x.sin(), x.cos()

    pair = capture_graph_pair(
        make_fx(function)(x),
        (x,),
        original_output=function(x),
        root_output_positions=(0,),
    )
    assert len(pair.backward.example_arguments) - pair.saved_value_count == 1
    values = pair.forward.graph_module(x)
    got = pair.backward.graph_module(*values[2:], torch.ones_like(x))
    torch.testing.assert_close(got[0], x.cos())


def test_objective_root_uses_stage_local_position_after_buffer_mutation():
    class Stateful(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = nn.Linear(7, 5)
            self.last = nn.Linear(5, 3)
            self.register_buffer("count", torch.tensor(0))

        def forward(self, x):
            self.count.add_(1)
            return self.last(self.first(x).relu())

    capture = partition_training_capture(
        capture_training_objective(
            Stateful(), lambda model, x: model(x).square().mean(), [torch.randn(2, 7)]
        ),
        partition="auto",
    )
    assert len(capture.stages) > 1
    terminal = capture.stages[-1]
    local_loss = terminal.example.stage.user_output_indices[0]
    assert local_loss != capture.partitioned.user_output_indices[0]
    assert terminal.graph_pairs.root_output_indices == (local_loss,)


class _Branch(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(3, 3))

    def forward(self, x):
        y = x @ self.weight
        return y.sin(), y.cos()


class _Consumer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(3, 3))

    def forward(self, left, right):
        return (left + right.detach()) @ self.weight


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch = _Branch()
        self.consumer = _Consumer()

    def forward(self, x):
        return self.consumer(*self.branch(x))


class _Stages:
    def assign_stages(self, graph_module, module):
        result = {}
        current = 0
        for node in graph_module.graph.nodes:
            if node.op in {"placeholder", "get_attr", "output"}:
                continue
            paths = [
                entry[0] for entry in node.meta.get("nn_module_stack", {}).values()
            ]
            if any("consumer" in path for path in paths):
                current = 1
            result[node.name] = current
        return result


@pytest.mark.cuda
def test_detached_branch_does_not_require_an_unproduced_cotangent():
    mode = FakeTensorMode(allow_non_fake_inputs=True)
    model = fake_device_model(_Model(), mode)
    inputs = fake_device_inputs([torch.randn(2, 3)], mode)
    with mode:
        capture = partition_training_capture(
            capture_training(model, lambda m, x: m(x).sum(), inputs),
            partition=_Stages(),
        )
    assert len(capture.stages) == 2
    first = capture.stages[0]
    assert len(first.example.output) == 2
    assert len(first.differentiable_output_indices) == 1
    for variant in first.graph_pairs.variants:
        pair = variant.pair
        assert len(pair.backward.example_arguments) - pair.saved_value_count == 1
