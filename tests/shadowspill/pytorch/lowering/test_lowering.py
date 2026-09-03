from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch._subclasses.fake_tensor import FakeTensorMode

from shadowspill.errors import CaptureError
from shadowspill.ir import (
    MemoryLocation,
    ObjectRole,
    Persistence,
    SharedResidencyPolicy,
)
from shadowspill.planner import pressurefit
from shadowspill.pytorch.capture.aot import capture_forward
from shadowspill.pytorch.capture.artifacts import capture_forward_stage_artifacts
from shadowspill.pytorch.capture.fake import fake_device_inputs, fake_device_model
from shadowspill.pytorch.lowering.catalog import ObjectCatalog
from shadowspill.pytorch.lowering.forward import (
    lower_partitioned_forward_program,
)
from shadowspill.pytorch.partition import partition_export
from shadowspill.pytorch.profiling import (
    TaskAllocationEvent,
    TaskAllocationOperation,
    TaskMeasurement,
)
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
    model = fake_device_model(_ForwardNetwork(), mode)
    inputs = fake_device_inputs([torch.randn(2, 8)], mode)
    with mode, torch.no_grad():
        partitioned = partition_export(capture_forward(model, inputs), model)
        artifacts = capture_forward_stage_artifacts(partitioned)
    measurements = tuple(_measurement(artifact) for artifact in artifacts)
    return lower_partitioned_forward_program(
        model, partitioned, artifacts, measurements
    )


def _measurement(artifact: object) -> TaskMeasurement:
    contract = artifact.storage_contract
    events = []
    for root in contract.roots:
        if root.kind.value != "fresh" or root.minimum_span_bytes == 0:
            continue
        views = tuple(
            view for view in contract.output_views if view.root_id == root.root_id
        )
        events.append(
            TaskAllocationEvent(
                len(events),
                TaskAllocationOperation.ALLOCATE,
                root.minimum_span_bytes,
                root.minimum_span_bytes,
                tuple(view.leaf_index for view in views),
                tuple(view.offset_bytes for view in views),
            )
        )
    return TaskMeasurement(
        1_000,
        256,
        256,
        (256,),
        (1_000,),
        "unit-test",
        tuple(events),
    )


def test_forward_lowering_is_indexed_alias_aware_and_plannable() -> None:
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
        spill_capacity_bytes=1 << 20,
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
    model = fake_device_model(_ForwardNetwork(), mode)
    inputs = fake_device_inputs([torch.randn(2, 8)], mode)
    with mode, torch.no_grad():
        partitioned = partition_export(capture_forward(model, inputs), model)
        artifacts = capture_forward_stage_artifacts(partitioned)
    with pytest.raises(CaptureError, match="counts"):
        lower_partitioned_forward_program(model, partitioned, artifacts, ())


def test_forward_lowering_uses_export_mutation_as_canonical_object_write() -> None:
    class Stateful(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("running", torch.zeros(8))

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            self.running.add_(value.sum(0))
            return self.running[2:] * 0.5

    mode = FakeTensorMode(allow_non_fake_inputs=True)
    model = fake_device_model(Stateful(), mode)
    inputs = fake_device_inputs([torch.randn(2, 8)], mode)
    with mode, torch.no_grad():
        partitioned = partition_export(
            capture_forward(model, inputs), model, partition="whole"
        )
        artifacts = capture_forward_stage_artifacts(partitioned)
    lowered = lower_partitioned_forward_program(
        model,
        partitioned,
        artifacts,
        tuple(_measurement(artifact) for artifact in artifacts),
    )

    buffer_object = next(
        item.object_id for item in lowered.registrations if item.name == "running"
    )
    task = lowered.program.tasks[0]
    assert tuple(item.object_id for item in task.mutations) == (buffer_object,)
    assert buffer_object in task.inputs
    assert buffer_object not in task.outputs
    buffer_alias = next(
        item.alias_group_id
        for item in lowered.program.objects
        if item.object_id == buffer_object
    )
    assert (
        next(
            item.location
            for item in lowered.final_residency
            if item.alias_group_id == buffer_alias
        )
        is MemoryLocation.SPILL
    )


def test_shared_residency_is_read_only_unless_emitted_tasks_write_it() -> None:
    catalog = ObjectCatalog(device_id="device_0")
    first = catalog.add(
        torch.ones(4),
        role=ObjectRole.INPUT,
        persistence=Persistence.STEP,
    )
    second = catalog.add(
        torch.ones(5),
        role=ObjectRole.INPUT,
        persistence=Persistence.STEP,
    )
    catalog.mark_shared_residency(
        first,
        SharedResidencyPolicy.SHARED_WRITABLE_CAUSAL,
        retain_spill_copy=False,
    )
    catalog.mark_shared_residency(
        second,
        SharedResidencyPolicy.SHARED_WRITABLE_UNORDERED,
        retain_spill_copy=False,
    )

    catalog.finalize_shared_writes((second,))

    policies = {
        item.alias_group_id: item.shared_residency for item in catalog.alias_groups()
    }
    assert policies[catalog.alias_id(first)] is (SharedResidencyPolicy.SHARED_READ_ONLY)
    assert policies[catalog.alias_id(second)] is (
        SharedResidencyPolicy.SHARED_WRITABLE_UNORDERED
    )
