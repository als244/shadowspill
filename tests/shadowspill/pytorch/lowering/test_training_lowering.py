from __future__ import annotations

from dataclasses import replace

import pytest
import torch
import torch.nn as nn
from torch._subclasses.fake_tensor import FakeTensorMode

from shadowspill.errors import CaptureError
from shadowspill.ir import RecomputationSelection
from shadowspill.planner import pressurefit
from shadowspill.pytorch.capture.aot import capture_training
from shadowspill.pytorch.capture.artifacts import GraphArtifact
from shadowspill.pytorch.capture.fake import fake_cuda_inputs, fake_cuda_model
from shadowspill.pytorch.graph_pairs import GraphPairVariant, partition_training_capture
from shadowspill.pytorch.lowering.training import (
    LoweredTrainingProgram,
    lower_partitioned_training_program,
    lower_training_storage_layout,
)
from shadowspill.pytorch.optimizer import capture_optimizer
from shadowspill.pytorch.profiling import (
    TaskAllocationEvent,
    TaskAllocationOperation,
    TaskMeasurement,
    TaskOutputInputBinding,
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


class _LongLivedBoundaryModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Linear(3, 8)
        self.second = nn.Linear(8, 8)
        self.third = nn.Linear(8, 2)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        early = torch.relu(self.first(value))
        middle = torch.relu(self.second(early))
        return self.third(middle + early)


class _AuxiliaryPassThroughBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(4, 4)

    def forward(
        self, value: torch.Tensor, auxiliary: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        projected = torch.relu(self.projection(value))
        return projected, auxiliary + projected.square().mean()


class _AuxiliaryPassThroughModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_AuxiliaryPassThroughBlock() for _ in range(3)])

    def forward(
        self, value: torch.Tensor, auxiliary: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for block in self.blocks:
            value, auxiliary = block(value, auxiliary)
        return value, auxiliary


class _StatefulTrainingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(8))
        self.register_buffer("running", torch.zeros(8))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.running.add_(value.mean(0))
        return value * self.weight + self.running


def _objective(
    model: nn.Module, value: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    return torch.nn.functional.mse_loss(model(value), target)


def _auxiliary_objective(
    model: nn.Module,
    value: torch.Tensor,
    target: torch.Tensor,
    auxiliary: torch.Tensor,
) -> torch.Tensor:
    output, passed = model(value, auxiliary)
    return torch.nn.functional.mse_loss(output, target) + passed


def _stateful_objective(
    model: nn.Module, value: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    return torch.nn.functional.mse_loss(model(value), target)


def _measurement(artifact: object) -> TaskMeasurement:
    events: list[TaskAllocationEvent] = []
    input_bindings: list[TaskOutputInputBinding] = []
    if isinstance(artifact, GraphArtifact):
        contract = artifact.storage_contract
        for root in contract.roots:
            views = tuple(
                view for view in contract.output_views if view.root_id == root.root_id
            )
            if root.kind.value == "input":
                assert root.source_input is not None
                input_bindings.extend(
                    TaskOutputInputBinding(
                        view.leaf_index,
                        root.source_input,
                        view.offset_bytes,
                    )
                    for view in views
                    if view.span_bytes > 0
                )
                continue
            if root.minimum_span_bytes == 0:
                continue
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
        100,
        10,
        10,
        (10,),
        (100,),
        "unit-test",
        tuple(events),
        tuple(input_bindings),
    )


def _lowered(*, include_intermediate_variant: bool = False) -> LoweredTrainingProgram:
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
            partition_training_capture(
                capture_training(model, _objective, fake_cuda_inputs(values, mode))
            )
            for values in examples
        )
    if include_intermediate_variant:
        captures = tuple(
            replace(
                capture,
                stages=tuple(
                    replace(
                        stage,
                        graph_pairs=replace(
                            stage.graph_pairs,
                            variants=(
                                stage.graph_pairs.variant("save"),
                                GraphPairVariant(
                                    "recompute_050",
                                    0.5,
                                    stage.graph_pairs.variant("recompute").pair,
                                ),
                                stage.graph_pairs.variant("recompute"),
                            ),
                        ),
                    )
                    for stage in capture.stages
                ),
            )
            for capture in captures
        )
    artifacts = (
        *(
            artifact
            for capture in captures
            for stage in capture.stages
            for option in stage.graph_pairs.variants
            for pair in (option.pair,)
            for artifact in (pair.forward, pair.backward)
        ),
        optimizer_capture.recurrent,
        *(task.artifact for task in optimizer_capture.recurrent_tasks),
    )
    measurements = {
        artifact.compatibility_digest: _measurement(artifact) for artifact in artifacts
    }
    return lower_partitioned_training_program(
        model,
        captures,
        measurements,
        optimizer_capture,
    )


def test_training_lowering_composes_accumulation_and_recomputation() -> None:
    lowered = _lowered()
    assert len(lowered.program.recomputation_groups) == 2
    assert len(lowered.program.tasks) == 8 + len(lowered.optimizer_task_ids)
    assert len(lowered.gradients) == 2
    assert lowered.fixed_tensors == ()
    assert lowered.program.tasks[-1].phase == "optimizer"
    assert all(
        not next(
            option for option in group.options if option.option_id == "recompute"
        ).retained_alias_group_ids
        for group in lowered.program.recomputation_groups
    )
    assert any(
        next(
            option for option in group.options if option.option_id == "save"
        ).retained_alias_group_ids
        for group in lowered.program.recomputation_groups
    )
    selections = tuple(
        RecomputationSelection(group.group_id, "save")
        for group in lowered.program.recomputation_groups
    )
    selected = lowered.program.selected_tasks(selections)
    assert [task.phase for task in selected[:4]] == [
        "forward",
        "backward",
        "forward",
        "backward",
    ]
    assert all(task.phase == "optimizer" for task in selected[4:])


def test_training_lowering_accepts_arbitrary_graph_pairs() -> None:
    lowered = _lowered(include_intermediate_variant=True)
    assert len(lowered.program.recomputation_groups) == 2
    assert all(
        tuple(option.option_id for option in group.options)
        == ("save", "recompute_050", "recompute")
        for group in lowered.program.recomputation_groups
    )
    selections = tuple(
        RecomputationSelection(group.group_id, "recompute_050")
        for group in lowered.program.recomputation_groups
    )
    selected = lowered.program.selected_tasks(selections)
    assert [task.phase for task in selected[:4]] == [
        "forward",
        "backward",
        "forward",
        "backward",
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
            partition_training_capture(
                capture_training(
                    model,
                    _objective,
                    fake_cuda_inputs(
                        [torch.randn(rows, 3), torch.randn(rows, 2)], mode
                    ),
                )
            )
            for rows in (4, 5)
        )
    artifacts = (
        *(
            artifact
            for capture in captures
            for stage in capture.stages
            for option in stage.graph_pairs.variants
            for pair in (option.pair,)
            for artifact in (pair.forward, pair.backward)
        ),
        optimizer_capture.recurrent,
        *(task.artifact for task in optimizer_capture.recurrent_tasks),
    )
    measurements = {
        artifact.compatibility_digest: _measurement(artifact) for artifact in artifacts
    }
    lowered = lower_partitioned_training_program(
        model,
        captures,
        measurements,
        optimizer_capture,
    )
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


def test_preinitialized_optimizer_uses_one_recurrent_state_flow() -> None:
    real_model = _Model()
    optimizer = torch.optim.AdamW(real_model.parameters(), lr=0.01, foreach=False)
    for parameter in real_model.parameters():
        parameter.grad = torch.zeros_like(parameter)
    optimizer_capture = capture_optimizer(
        dict(real_model.named_parameters()), optimizer
    )
    assert not optimizer_capture.first_step_is_opaque
    assert optimizer_capture.preinitialized_state_names
    assert optimizer_capture.recurrent is not None
    mode = FakeTensorMode(allow_non_fake_inputs=True)
    model = fake_cuda_model(real_model, mode)
    with mode:
        captures = (
            partition_training_capture(
                capture_training(
                    model,
                    _objective,
                    fake_cuda_inputs([torch.randn(4, 3), torch.randn(4, 2)], mode),
                )
            ),
        )
    artifacts = (
        *(
            artifact
            for capture in captures
            for stage in capture.stages
            for option in stage.graph_pairs.variants
            for pair in (option.pair,)
            for artifact in (pair.forward, pair.backward)
        ),
        optimizer_capture.recurrent,
        *(task.artifact for task in optimizer_capture.recurrent_tasks),
    )
    measurements = {
        artifact.compatibility_digest: _measurement(artifact) for artifact in artifacts
    }
    initial = lower_partitioned_training_program(
        model,
        captures,
        measurements,
        optimizer_capture,
        optimizer_phase="initial",
    )
    recurrent = lower_partitioned_training_program(
        model,
        captures,
        measurements,
        optimizer_capture,
        optimizer_phase="recurrent",
    )
    assert initial.optimizer_objects
    assert initial.program.objects == recurrent.program.objects
    initial_tasks = tuple(
        task
        for task in initial.program.tasks
        if task.task_id in initial.optimizer_task_ids
    )
    recurrent_tasks = tuple(
        task
        for task in recurrent.program.tasks
        if task.task_id in recurrent.optimizer_task_ids
    )
    assert not any(item.created_on_first_step for item in initial.optimizer_objects)
    assert initial_tasks == recurrent_tasks


def test_partitioned_lowering_preserves_boundary_residual_aliases() -> None:
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
        capture = partition_training_capture(
            capture_training(
                model,
                _objective,
                fake_cuda_inputs([torch.randn(4, 3), torch.randn(4, 2)], mode),
            )
        )
    artifacts = (
        *(
            artifact
            for stage in capture.stages
            for option in stage.graph_pairs.variants
            for pair in (option.pair,)
            for artifact in (pair.forward, pair.backward)
        ),
        optimizer_capture.recurrent,
        *(task.artifact for task in optimizer_capture.recurrent_tasks),
    )
    measurements = {
        artifact.compatibility_digest: _measurement(artifact) for artifact in artifacts
    }
    lowered = lower_partitioned_training_program(
        model, (capture,), measurements, optimizer_capture
    )
    assert len(lowered.program.recomputation_groups) == len(capture.stages)
    alias_by_object = {
        item.object_id: item.alias_group_id for item in lowered.program.objects
    }
    parameter_aliases = {
        alias_by_object[binding.parameter_object_id] for binding in lowered.gradients
    }
    produced_aliases = {
        alias_by_object[object_id]
        for task in lowered.program.tasks
        for object_id in task.outputs
    }
    initial_non_parameter_aliases = {
        item.alias_group_id for item in lowered.initial_residency
    } - parameter_aliases
    assert initial_non_parameter_aliases.isdisjoint(produced_aliases)
    for task in lowered.program.tasks:
        if task.phase != "forward":
            continue
        aliases = tuple(alias_by_object[object_id] for object_id in task.outputs)
        assert len(aliases) == len(set(aliases))
    first_forward = next(
        item
        for item in lowered.entrypoints
        if item.phase == "forward" and item.variant == "save"
    )
    boundary = first_forward.output_slots[0].object_id
    assert any(slot.object_id == boundary for slot in first_forward.output_slots[1:])

    with pytest.raises(CaptureError, match="profile scatter"):
        lower_partitioned_training_program(model, (capture,), {}, optimizer_capture)
    with pytest.raises(CaptureError, match="unknown optimizer phase"):
        lower_partitioned_training_program(
            model,
            (capture,),
            measurements,
            optimizer_capture,
            optimizer_phase="unknown",  # type: ignore[arg-type]
        )
    with pytest.raises(CaptureError, match="bounded optimizer task"):
        lower_partitioned_training_program(
            model,
            (capture,),
            measurements,
            replace(optimizer_capture, recurrent=None),
        )


def test_partitioned_forward_dependencies_cover_long_lived_boundaries() -> None:
    real_model = _LongLivedBoundaryModel()
    optimizer = torch.optim.SGD(real_model.parameters(), lr=0.1, foreach=False)
    for parameter in real_model.parameters():
        parameter.grad = torch.zeros_like(parameter)
    optimizer_capture = capture_optimizer(
        dict(real_model.named_parameters()), optimizer
    )
    mode = FakeTensorMode(allow_non_fake_inputs=True)
    model = fake_cuda_model(real_model, mode)
    with mode:
        capture = partition_training_capture(
            capture_training(
                model,
                _objective,
                fake_cuda_inputs([torch.randn(4, 3), torch.randn(4, 2)], mode),
            )
        )
    artifacts = (
        *(
            artifact
            for stage in capture.stages
            for option in stage.graph_pairs.variants
            for pair in (option.pair,)
            for artifact in (pair.forward, pair.backward)
        ),
        optimizer_capture.recurrent,
        *(task.artifact for task in optimizer_capture.recurrent_tasks),
    )
    measurements = {
        artifact.compatibility_digest: _measurement(artifact)
        for artifact in artifacts
        if artifact is not None
    }

    lowered = lower_partitioned_training_program(
        model, (capture,), measurements, optimizer_capture
    )
    producers: dict[str, list[str]] = {}
    for task in lowered.program.tasks:
        for object_id in task.inputs:
            candidates = producers.get(object_id, ())
            if candidates:
                assert any(candidate in task.dependencies for candidate in candidates)
        for object_id in task.outputs:
            producers.setdefault(object_id, []).append(task.task_id)


def test_partitioned_backward_uses_task_local_cotangent_handoff() -> None:
    real_model = _AuxiliaryPassThroughModel()
    optimizer = torch.optim.SGD(real_model.parameters(), lr=0.1, foreach=False)
    for parameter in real_model.parameters():
        parameter.grad = torch.zeros_like(parameter)
    optimizer_capture = capture_optimizer(
        dict(real_model.named_parameters()), optimizer
    )
    mode = FakeTensorMode(allow_non_fake_inputs=True)
    model = fake_cuda_model(real_model, mode)
    with mode:
        capture = partition_training_capture(
            capture_training(
                model,
                _auxiliary_objective,
                fake_cuda_inputs(
                    [torch.randn(2, 4), torch.randn(2, 4), torch.randn(())], mode
                ),
            )
        )
    artifacts = (
        *(
            artifact
            for stage in capture.stages
            for option in stage.graph_pairs.variants
            for pair in (option.pair,)
            for artifact in (pair.forward, pair.backward)
        ),
        optimizer_capture.recurrent,
        *(task.artifact for task in optimizer_capture.recurrent_tasks),
    )
    measurements = {
        artifact.compatibility_digest: _measurement(artifact)
        for artifact in artifacts
        if artifact is not None
    }
    lowered = lower_partitioned_training_program(
        model, (capture,), measurements, optimizer_capture
    )
    task_by_id = {task.task_id: task for task in lowered.program.tasks}
    alias_by_object = {
        item.object_id: item.alias_group_id for item in lowered.program.objects
    }
    handoffs = []
    for entrypoint in lowered.entrypoints:
        if entrypoint.phase != "backward":
            continue
        task = task_by_id[entrypoint.task_id]
        for handoff in entrypoint.storage_handoffs:
            assert handoff.source_object_id in task.inputs
            assert handoff.destination_object_id in task.outputs
            assert (
                alias_by_object[handoff.source_object_id]
                != alias_by_object[handoff.destination_object_id]
            )
            handoffs.append(handoff)
    assert handoffs


def test_functional_buffer_mutation_does_not_displace_objective_output() -> None:
    real_model = _StatefulTrainingModel()
    optimizer = torch.optim.SGD(real_model.parameters(), lr=0.1, foreach=False)
    for parameter in real_model.parameters():
        parameter.grad = torch.zeros_like(parameter)
    optimizer_capture = capture_optimizer(
        dict(real_model.named_parameters()), optimizer
    )
    mode = FakeTensorMode(allow_non_fake_inputs=True)
    model = fake_cuda_model(real_model, mode)
    with mode:
        capture = partition_training_capture(
            capture_training(
                model,
                _stateful_objective,
                fake_cuda_inputs([torch.randn(2, 8), torch.randn(2, 8)], mode),
            )
        )
    artifacts = (
        *(
            artifact
            for stage in capture.stages
            for option in stage.graph_pairs.variants
            for pair in (option.pair,)
            for artifact in (pair.forward, pair.backward)
        ),
        optimizer_capture.recurrent,
        *(task.artifact for task in optimizer_capture.recurrent_tasks),
    )
    measurements = {
        artifact.compatibility_digest: _measurement(artifact)
        for artifact in artifacts
        if artifact is not None
    }
    lowered = lower_partitioned_training_program(
        model, (capture,), measurements, optimizer_capture
    )
    forward_entries = tuple(
        entrypoint
        for entrypoint in lowered.entrypoints
        if entrypoint.phase == "forward"
    )
    assert forward_entries
    assert all(entrypoint.public_output_count == 1 for entrypoint in forward_entries)
    assert all(entrypoint.public_output_leaves for entrypoint in forward_entries)
    assert any(entrypoint.replacement_output_leaves for entrypoint in forward_entries)
    assert all(
        set(entrypoint.public_output_leaves).isdisjoint(
            entrypoint.replacement_output_leaves
        )
        for entrypoint in forward_entries
    )


def test_training_lowering_rejects_empty_templates() -> None:
    model = _Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, foreach=False)
    for parameter in model.parameters():
        parameter.grad = torch.zeros_like(parameter)
    optimizer_capture = capture_optimizer(dict(model.named_parameters()), optimizer)
    with pytest.raises(CaptureError, match="storage layout"):
        lower_training_storage_layout(model, ())
    with pytest.raises(CaptureError, match="requires a microbatch"):
        lower_partitioned_training_program(model, (), {}, optimizer_capture)
