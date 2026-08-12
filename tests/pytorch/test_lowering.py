from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch._subclasses.fake_tensor import FakeTensorMode

from shadowspill.ir import MemoryLocation, ObjectRole, Persistence
from shadowspill.planner import pressurefit
from shadowspill.pytorch.aot import capture_forward
from shadowspill.pytorch.contracts import CaptureError
from shadowspill.pytorch.fake import fake_cuda_inputs, fake_cuda_model
from shadowspill.pytorch.lowering import (
    _CompiledOutputAllocation,
    _TensorInventory,
    lower_forward_program,
)
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
        fetch_bandwidth_bytes_per_second=10 << 30,
        evict_bandwidth_bytes_per_second=10 << 30,
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


def test_isolated_compiled_output_uses_measured_physical_extent() -> None:
    inventory = _TensorInventory(device_id="cuda_0")
    output = torch.empty(20, dtype=torch.bfloat16)[10:]
    object_id = inventory.add_compiled_output(
        output,
        role=ObjectRole.ACTIVATION,
        persistence=Persistence.STEP,
        allocation_scope=inventory.compiled_output_scope(),
        physical_allocation=_CompiledOutputAllocation(0, 24),
    )

    object_spec = next(
        item for item in inventory.objects() if item.object_id == object_id
    )
    alias_spec = next(
        item
        for item in inventory.alias_groups()
        if item.alias_group_id == object_spec.alias_group_id
    )
    assert object_spec.offset_bytes == 0
    assert object_spec.size_bytes == 20
    assert alias_spec.size_bytes == 24


def test_compiled_output_views_keep_shared_storage_bundle() -> None:
    inventory = _TensorInventory(device_id="cuda_0")
    storage = torch.empty(20, dtype=torch.bfloat16)
    left, right = storage[:10], storage[10:]
    scope = inventory.compiled_output_scope()
    allocation = _CompiledOutputAllocation(0, 40)
    left_id = inventory.add_compiled_output(
        left,
        role=ObjectRole.ACTIVATION,
        persistence=Persistence.STEP,
        allocation_scope=scope,
        physical_allocation=allocation,
    )
    right_id = inventory.add_compiled_output(
        right,
        role=ObjectRole.ACTIVATION,
        persistence=Persistence.STEP,
        allocation_scope=scope,
        physical_allocation=allocation,
    )

    objects = {item.object_id: item for item in inventory.objects()}
    assert objects[left_id].alias_group_id == objects[right_id].alias_group_id
    assert objects[left_id].offset_bytes == 0
    assert objects[right_id].offset_bytes == 0
    alias = next(
        item
        for item in inventory.alias_groups()
        if item.alias_group_id == objects[left_id].alias_group_id
    )
    assert alias.size_bytes == 40


def test_compiled_output_allocations_split_one_fake_storage() -> None:
    inventory = _TensorInventory(device_id="cuda_0")
    storage = torch.empty(40, dtype=torch.bfloat16)
    left, right = storage[:10], storage[10:20]
    scope = inventory.compiled_output_scope()
    left_id = inventory.add_compiled_output(
        left,
        role=ObjectRole.ACTIVATION,
        persistence=Persistence.STEP,
        allocation_scope=scope,
        physical_allocation=_CompiledOutputAllocation(3, 20),
    )
    right_id = inventory.add_compiled_output(
        right,
        role=ObjectRole.ACTIVATION,
        persistence=Persistence.STEP,
        allocation_scope=scope,
        physical_allocation=_CompiledOutputAllocation(4, 20),
    )

    objects = {item.object_id: item for item in inventory.objects()}
    assert objects[left_id].alias_group_id != objects[right_id].alias_group_id
