from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch._subclasses.fake_tensor import FakeTensorMode

from shadowspill.ir import MemoryLocation, ObjectRole
from shadowspill.planner import pressurefit
from shadowspill.pytorch.aot import capture_forward
from shadowspill.pytorch.contracts import CaptureError
from shadowspill.pytorch.fake import fake_cuda_inputs, fake_cuda_model
from shadowspill.pytorch.lowering import lower_forward_program
from shadowspill.pytorch.partition import capture_forward_stages, partition_export
from shadowspill.pytorch.profiling import TaskMeasurement
from shadowspill.simulator import SimulationConfig


class _ForwardNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(8, 8, bias=False) for _ in range(3)])
        self.tied = self.layers[0].weight

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            value = torch.relu(layer(value))
        return value[:, 2:6]


def _lowered() -> object:
    mode = FakeTensorMode(allow_non_fake_inputs=True)
    model = fake_cuda_model(_ForwardNetwork(), mode)
    inputs = fake_cuda_inputs([torch.randn(2, 8)], mode)
    with mode, torch.no_grad():
        partitioned = partition_export(capture_forward(model, inputs), model)
        artifacts = capture_forward_stages(partitioned)
    measurements = tuple(
        TaskMeasurement(1_000, 256, 256, (256,), (1_000,), "unit-test")
        for _ in artifacts
    )
    return lower_forward_program(model, partitioned, artifacts, measurements)


def test_forward_lowering_is_dense_alias_aware_and_plannable() -> None:
    lowered = _lowered()
    assert len(lowered.program.tasks) == 3
    assert len(lowered.program.profiles) == 2
    assert tuple(task.task_id for task in lowered.program.tasks) == (
        "task_000000",
        "task_000001",
        "task_000002",
    )
    assert lowered.program.tasks[1].dependencies == ("task_000000",)
    registrations = {item.name: item.object_id for item in lowered.registrations}
    assert registrations["tied"] == registrations["layers.0.weight"]
    output_objects = [
        item for item in lowered.program.objects if item.role is ObjectRole.OUTPUT
    ]
    assert len(output_objects) == 1
    output_alias = output_objects[0].alias_group_id
    assert lowered.final_residency[-1].alias_group_id == output_alias
    assert lowered.final_residency[-1].location is MemoryLocation.DEVICE
    assert lowered.program.to_json() == _lowered().program.to_json()

    config = SimulationConfig.single_device(
        "cuda_0",
        device_capacity_bytes=1 << 20,
        host_capacity_bytes=1 << 20,
        h2d_bandwidth_bytes_per_second=10 << 30,
        d2h_bandwidth_bytes_per_second=10 << 30,
    )
    planned = pressurefit(
        lowered.program,
        initial_residency=lowered.initial_residency,
        final_residency=lowered.final_residency,
        config=config,
    )
    assert planned.simulation.makespan_ns >= 3_000


def test_forward_lowering_rejects_incomplete_profile_scatter() -> None:
    mode = FakeTensorMode(allow_non_fake_inputs=True)
    model = fake_cuda_model(_ForwardNetwork(), mode)
    inputs = fake_cuda_inputs([torch.randn(2, 8)], mode)
    with mode, torch.no_grad():
        partitioned = partition_export(capture_forward(model, inputs), model)
        artifacts = capture_forward_stages(partitioned)
    with pytest.raises(CaptureError, match="counts"):
        lower_forward_program(model, partitioned, artifacts, ())
