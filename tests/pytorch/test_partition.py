from __future__ import annotations

import torch
import torch.nn as nn
from torch._subclasses.fake_tensor import FakeTensorMode

from shadowspill.pytorch.aot import capture_forward, capture_training
from shadowspill.pytorch.fake import fake_cuda_inputs, fake_cuda_model
from shadowspill.pytorch.partition import (
    capture_forward_stages,
    capture_training_stages,
    partition_export,
)


class _NestedRepeatedNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_Block() for _ in range(4)])

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            value = torch.relu(block(value))
        return value


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.experts = nn.ModuleList([nn.Linear(8, 8) for _ in range(2)])

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.experts[0](value) + self.experts[1](value)


def test_auto_partition_uses_outer_repeated_blocks_not_nested_experts() -> None:
    model = _NestedRepeatedNetwork()
    mode = FakeTensorMode(allow_non_fake_inputs=True)
    replica = fake_cuda_model(model, mode)
    inputs = fake_cuda_inputs([torch.randn(2, 8)], mode)
    with mode, torch.no_grad():
        exported = capture_forward(replica, inputs)
        partitioned = partition_export(exported, replica)
        artifacts = capture_forward_stages(partitioned)
    assert partitioned.repeated_groups == ("blocks",)
    assert len(partitioned.stages) == 4
    assert len(artifacts) == 4
    assert all(artifact.operator_targets for artifact in artifacts)
    assert len({artifact.compatibility_digest for artifact in artifacts}) == 1


def test_each_training_stage_has_save_and_recompute_vjp() -> None:
    model = _NestedRepeatedNetwork()
    mode = FakeTensorMode(allow_non_fake_inputs=True)
    replica = fake_cuda_model(model, mode)
    inputs = fake_cuda_inputs([torch.randn(2, 8), torch.randn(2, 8)], mode)

    def objective(
        current: nn.Module, value: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return torch.nn.functional.mse_loss(current(value), target)

    with mode:
        capture = capture_training(replica, objective, inputs)
        partitioned = partition_export(capture.exported, capture.capture_module)
        stages = capture_training_stages(partitioned)
    assert partitioned.repeated_groups == ("model.blocks",)
    assert len(stages) == 4
    assert all(stage.save_pair.backward.operator_targets for stage in stages)
    assert all(stage.recompute_pair.forward.operator_targets for stage in stages)
    assert (
        stages[1].save_pair.forward.compatibility_digest
        == stages[2].save_pair.forward.compatibility_digest
    )


def test_whole_partition_is_one_stage() -> None:
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    mode = FakeTensorMode(allow_non_fake_inputs=True)
    replica = fake_cuda_model(model, mode)
    inputs = fake_cuda_inputs([torch.randn(2, 4)], mode)
    with mode, torch.no_grad():
        partitioned = partition_export(
            capture_forward(replica, inputs), replica, partition="whole"
        )
    assert partitioned.repeated_groups == ()
    assert len(partitioned.stages) == 1
