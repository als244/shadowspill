from __future__ import annotations

import torch
import torch.nn as nn
from torch._subclasses.fake_tensor import FakeTensorMode

from shadowspill.ir import RecomputationSelection
from shadowspill.planner import pressurefit
from shadowspill.pytorch.aot import capture_training
from shadowspill.pytorch.fake import fake_cuda_inputs, fake_cuda_model
from shadowspill.pytorch.optimizer import capture_optimizer
from shadowspill.pytorch.profiling import TaskMeasurement
from shadowspill.pytorch.training_lowering import (
    LoweredTrainingProgram,
    lower_training_program,
)
from shadowspill.simulator import SimulationConfig


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, 2)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.projection(value)


class _MultiLinearModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Linear(3, 8)
        self.second = nn.Linear(8, 2)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.second(torch.relu(self.first(value)))


def _objective(
    model: nn.Module, value: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    return torch.nn.functional.mse_loss(model(value), target)


def _lowered() -> LoweredTrainingProgram:
    real_model = _Model()
    optimizer = torch.optim.SGD(real_model.parameters(), lr=0.1, foreach=False)
    for parameter in real_model.parameters():
        parameter.grad = torch.zeros_like(parameter)
    optimizer_capture = capture_optimizer(
        dict(real_model.named_parameters()), optimizer
    )
    assert optimizer_capture.recurrent is not None
    mode = FakeTensorMode(allow_non_fake_inputs=True)
    model = fake_cuda_model(real_model, mode)
    examples = (
        [torch.randn(4, 3), torch.randn(4, 2)],
        [torch.randn(5, 3), torch.randn(5, 2)],
    )
    with mode:
        captures = tuple(
            capture_training(model, _objective, fake_cuda_inputs(values, mode))
            for values in examples
        )
    artifacts = (
        *(
            artifact
            for capture in captures
            for pair in (capture.save_pair, capture.recompute_pair)
            for artifact in (pair.forward, pair.backward)
        ),
        optimizer_capture.recurrent,
    )
    measurements = {
        artifact.compatibility_digest: TaskMeasurement(
            100, 10, 10, (10,), (100,), "unit-test"
        )
        for artifact in artifacts
    }
    return lower_training_program(model, captures, measurements, optimizer_capture)


def test_training_lowering_composes_accumulation_and_recomputation() -> None:
    lowered = _lowered()
    assert len(lowered.program.recomputation_groups) == 2
    assert len(lowered.program.tasks) == 9
    assert len(lowered.gradients) == 2
    assert lowered.program.tasks[-1].phase == "optimizer"
    selections = tuple(
        RecomputationSelection(group.group_id, "save")
        for group in lowered.program.recomputation_groups
    )
    selected = lowered.program.selected_tasks(selections)
    assert [task.phase for task in selected] == [
        "forward",
        "backward",
        "forward",
        "backward",
        "optimizer",
    ]
    assert selected[3].mutations
    assert selected[-1].mutations
    for entrypoint in lowered.entrypoints:
        if entrypoint.phase == "backward":
            assert tuple(slot.leaf_index for slot in entrypoint.input_slots) == tuple(
                range(len(entrypoint.input_slots))
            )

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
    assert len(planned.selections) == 2
    assert planned.simulation.makespan_ns > 0


def test_training_lowering_is_deterministic() -> None:
    assert _lowered().program.to_json() == _lowered().program.to_json()


def test_saved_parameter_views_are_not_declared_as_outputs() -> None:
    real_model = _MultiLinearModel()
    optimizer = torch.optim.SGD(real_model.parameters(), lr=0.1, foreach=False)
    for parameter in real_model.parameters():
        parameter.grad = torch.zeros_like(parameter)
    optimizer_capture = capture_optimizer(
        dict(real_model.named_parameters()), optimizer
    )
    assert optimizer_capture.recurrent is not None
    mode = FakeTensorMode(allow_non_fake_inputs=True)
    model = fake_cuda_model(real_model, mode)
    with mode:
        captures = tuple(
            capture_training(
                model,
                _objective,
                fake_cuda_inputs([torch.randn(rows, 3), torch.randn(rows, 2)], mode),
            )
            for rows in (4, 5)
        )
    artifacts = (
        *(
            artifact
            for capture in captures
            for pair in (capture.save_pair, capture.recompute_pair)
            for artifact in (pair.forward, pair.backward)
        ),
        optimizer_capture.recurrent,
    )
    measurements = {
        artifact.compatibility_digest: TaskMeasurement(
            100, 10, 10, (10,), (100,), "unit-test"
        )
        for artifact in artifacts
    }
    lowered = lower_training_program(model, captures, measurements, optimizer_capture)
    parameter_aliases = {
        next(
            item.alias_group_id
            for item in lowered.program.objects
            if item.object_id == binding.parameter_object_id
        )
        for binding in lowered.gradients
    }
    produced_aliases = {
        next(
            item.alias_group_id
            for item in lowered.program.objects
            if item.object_id == object_id
        )
        for task in lowered.program.tasks
        if task.phase == "forward"
        for object_id in task.outputs
    }
    assert parameter_aliases.isdisjoint(produced_aliases)
