from __future__ import annotations

from dataclasses import replace

import pytest
import torch
import torch.nn as nn
from torch._subclasses.fake_tensor import FakeTensorMode

from shadowspill.errors import CaptureError
from shadowspill.ir import TaskAlternativeChoice
from shadowspill.planner import PressureFitOptions, StepDataOrdering, pressurefit
from shadowspill.pytorch.capture.aot import capture_training
from shadowspill.pytorch.capture.artifacts import GraphArtifact
from shadowspill.pytorch.capture.fake import fake_device_inputs, fake_device_model
from shadowspill.pytorch.graph_pairs import (
    GraphPairVariant,
    TaskGraphPairs,
    partition_training_capture,
)
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


def _with_both_forms(
    plain: tuple[GraphPairVariant, ...], *, accumulating: bool
) -> tuple[GraphPairVariant, ...]:
    """Substituted variants need their accumulating forms like captured ones."""

    if not accumulating:
        return plain
    return (*plain, *(item.accumulating() for item in plain))


def _both_forms(graph_pairs: TaskGraphPairs) -> tuple[GraphPairVariant, ...]:
    """Every option in both forms, so a fixture can profile whichever is used."""

    return (
        *graph_pairs.options(accumulates=False),
        *(graph_pairs.options(accumulates=True) or graph_pairs.accumulating_variants()),
    )


def _lowered(
    *,
    include_intermediate_variant: bool = False,
    microbatches: int = 2,
    model_factory: type[nn.Module] = _Model,
    data_ordering: StepDataOrdering | None = None,
) -> LoweredTrainingProgram:
    real_model = model_factory()
    optimizer = torch.optim.SGD(real_model.parameters(), lr=0.1, foreach=False)
    for parameter in real_model.parameters():
        parameter.grad = torch.zeros_like(parameter)
    optimizer_capture = capture_optimizer(
        dict(real_model.named_parameters()), optimizer
    )
    assert optimizer_capture.recurrent is not None
    mode = FakeTensorMode(allow_non_fake_inputs=True)
    model = fake_device_model(real_model, mode)
    examples = tuple(
        [torch.randn(4 + position, 3), torch.randn(4 + position, 2)]
        for position in range(microbatches)
    )
    with mode:
        captures = tuple(
            partition_training_capture(
                capture_training(model, _objective, fake_device_inputs(values, mode)),
                accumulating=microbatches > 1,
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
                            variants=_with_both_forms(
                                (
                                    stage.graph_pairs.variant("save"),
                                    GraphPairVariant(
                                        "recompute_050",
                                        0.5,
                                        stage.graph_pairs.variant("recompute").pair,
                                    ),
                                    stage.graph_pairs.variant("recompute"),
                                ),
                                accumulating=len(capture.stages) and microbatches > 1,
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
            for option in _both_forms(stage.graph_pairs)
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
        data_ordering=data_ordering,
    )


def test_training_lowering_composes_accumulation_and_recomputation() -> None:
    lowered = _lowered()
    assert len(lowered.program.task_alternative_groups) == 2
    assert len(lowered.program.tasks) == 8 + len(lowered.optimizer_task_ids)
    assert len(lowered.gradients) == 2
    assert lowered.fixed_tensors == ()
    assert lowered.program.tasks[-1].phase == "optimizer"
    assert all(
        not next(
            option for option in group.options if option.option_id == "recompute"
        ).retained_alias_group_ids
        for group in lowered.program.task_alternative_groups
    )
    assert any(
        next(
            option for option in group.options if option.option_id == "save"
        ).retained_alias_group_ids
        for group in lowered.program.task_alternative_groups
    )
    selections = tuple(
        TaskAlternativeChoice(group.group_id, "save")
        for group in lowered.program.task_alternative_groups
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
    assert len(lowered.program.task_alternative_groups) == 2
    assert all(
        tuple(option.option_id for option in group.options)
        == ("save", "recompute_050", "recompute")
        for group in lowered.program.task_alternative_groups
    )
    selections = tuple(
        TaskAlternativeChoice(group.group_id, "recompute_050")
        for group in lowered.program.task_alternative_groups
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
        options=PressureFitOptions(minimum_object_bytes_evict_eligible=0),
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
    model = fake_device_model(real_model, mode)
    with mode:
        captures = tuple(
            partition_training_capture(
                capture_training(
                    model,
                    _objective,
                    fake_device_inputs(
                        [torch.randn(rows, 3), torch.randn(rows, 2)], mode
                    ),
                ),
                accumulating=position > 0,
            )
            for position, rows in enumerate((4, 5))
        )
    artifacts = (
        *(
            artifact
            for capture in captures
            for stage in capture.stages
            for option in _both_forms(stage.graph_pairs)
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


def test_state_installed_before_capture_uses_one_recurrent_state_flow() -> None:
    """State that exists before capture needs no initial step to create it.

    Planning installs declared state in the pool before capturing; this is the
    same shape without a runtime, so the capture sees state present exactly as
    it does in a plan.
    """

    real_model = _Model()
    optimizer = torch.optim.AdamW(real_model.parameters(), lr=0.01, foreach=False)
    for parameter in real_model.parameters():
        parameter.grad = torch.zeros_like(parameter)
        optimizer.state[parameter] = {
            "step": torch.zeros(()),
            "exp_avg": torch.zeros_like(parameter),
            "exp_avg_sq": torch.zeros_like(parameter),
        }
    optimizer_capture = capture_optimizer(
        dict(real_model.named_parameters()), optimizer
    )
    assert not optimizer_capture.first_step_is_opaque
    assert optimizer_capture.created_state_names == ()
    assert optimizer_capture.recurrent is not None
    mode = FakeTensorMode(allow_non_fake_inputs=True)
    model = fake_device_model(real_model, mode)
    with mode:
        captures = (
            partition_training_capture(
                capture_training(
                    model,
                    _objective,
                    fake_device_inputs([torch.randn(4, 3), torch.randn(4, 2)], mode),
                )
            ),
        )
    artifacts = (
        *(
            artifact
            for capture in captures
            for stage in capture.stages
            for option in _both_forms(stage.graph_pairs)
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
    model = fake_device_model(real_model, mode)
    with mode:
        capture = partition_training_capture(
            capture_training(
                model,
                _objective,
                fake_device_inputs([torch.randn(4, 3), torch.randn(4, 2)], mode),
            )
        )
    artifacts = (
        *(
            artifact
            for stage in capture.stages
            for option in _both_forms(stage.graph_pairs)
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
    assert len(lowered.program.task_alternative_groups) == len(capture.stages)
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
    model = fake_device_model(real_model, mode)
    with mode:
        capture = partition_training_capture(
            capture_training(
                model,
                _objective,
                fake_device_inputs([torch.randn(4, 3), torch.randn(4, 2)], mode),
            )
        )
    artifacts = (
        *(
            artifact
            for stage in capture.stages
            for option in _both_forms(stage.graph_pairs)
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


def test_forward_tasks_depend_only_on_data() -> None:
    """Order is the schedule; dependencies are data.

    The first forward stage of the second microbatch used to depend on the
    first microbatch's last backward, across which no value flows. That wrote
    the microbatch-major order into the graph and made every other order
    illegal.
    """

    lowered = _lowered()
    backward_ids = {
        task.task_id for task in lowered.program.tasks if task.phase == "backward"
    }
    for task in lowered.program.tasks:
        if task.phase == "forward":
            assert not backward_ids.intersection(task.dependencies), task.task_id


def test_partitioned_backward_uses_task_local_cotangent_handoff() -> None:
    real_model = _AuxiliaryPassThroughModel()
    optimizer = torch.optim.SGD(real_model.parameters(), lr=0.1, foreach=False)
    for parameter in real_model.parameters():
        parameter.grad = torch.zeros_like(parameter)
    optimizer_capture = capture_optimizer(
        dict(real_model.named_parameters()), optimizer
    )
    mode = FakeTensorMode(allow_non_fake_inputs=True)
    model = fake_device_model(real_model, mode)
    with mode:
        capture = partition_training_capture(
            capture_training(
                model,
                _auxiliary_objective,
                fake_device_inputs(
                    [torch.randn(2, 4), torch.randn(2, 4), torch.randn(())], mode
                ),
            )
        )
    artifacts = (
        *(
            artifact
            for stage in capture.stages
            for option in _both_forms(stage.graph_pairs)
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
    model = fake_device_model(real_model, mode)
    with mode:
        capture = partition_training_capture(
            capture_training(
                model,
                _stateful_objective,
                fake_device_inputs([torch.randn(2, 8), torch.randn(2, 8)], mode),
            )
        )
    artifacts = (
        *(
            artifact
            for stage in capture.stages
            for option in _both_forms(stage.graph_pairs)
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


def test_later_microbatches_add_onto_the_gradients_they_inherit() -> None:
    """The backward itself performs the accumulation, so the task declares it."""

    lowered = _lowered()
    backwards = tuple(
        task for task in lowered.program.tasks if task.phase == "backward"
    )
    assert backwards
    assert any(task.mutations for task in backwards)
    assert all(
        object_id in task.inputs
        for task in backwards
        for object_id in (item.object_id for item in task.mutations)
    )


def test_one_microbatch_has_nothing_to_accumulate_onto() -> None:
    """A single microbatch creates its gradients, so it runs the captured backward.

    Nothing about accumulation reaches it: the accumulating form is derived
    per microbatch position, so it is never built, compiled, or profiled here.
    """

    lowered = _lowered(microbatches=1)
    backwards = tuple(
        task for task in lowered.program.tasks if task.phase == "backward"
    )
    assert backwards
    assert all(not task.mutations for task in backwards)


def _walk(lowered: LoweredTrainingProgram) -> list[tuple[str, int, int]]:
    """The emitted order as (phase, microbatch, stage), variants collapsed."""
    walk: list[tuple[str, int, int]] = []
    for entrypoint in lowered.entrypoints:
        if entrypoint.phase not in ("forward", "backward"):
            continue
        assert entrypoint.microbatch is not None
        assert entrypoint.stage_index is not None
        item = (
            entrypoint.phase[0].upper(),
            entrypoint.microbatch,
            entrypoint.stage_index,
        )
        if not walk or walk[-1] != item:
            walk.append(item)
    return walk


def _assert_topological(lowered: LoweredTrainingProgram) -> None:
    seen: set[str] = set()
    for task in lowered.program.tasks:
        missing = [item for item in task.dependencies if item not in seen]
        assert not missing, f"{task.task_id} runs before {missing}"
        seen.add(task.task_id)


def _depth_first_walk(microbatches: int, head: int) -> list[tuple[str, int, int]]:
    walk: list[tuple[str, int, int]] = []
    for position in range(microbatches):
        walk += [("F", position, stage) for stage in range(head + 1)]
        walk += [("B", position, stage) for stage in range(head, -1, -1)]
    return walk


def test_default_ordering_is_depth_first_to_the_byte() -> None:
    implicit = _lowered(model_factory=_MultiLinearModel, microbatches=2)
    explicit = _lowered(
        model_factory=_MultiLinearModel,
        microbatches=2,
        data_ordering=StepDataOrdering.depth_first(2),
    )
    assert implicit.program.digest == explicit.program.digest
    walk = _walk(implicit)
    head = max(stage for _, _, stage in walk)
    assert walk == _depth_first_walk(2, head)


def test_breadth_first_runs_every_microbatch_through_a_stage_before_the_next() -> None:
    lowered = _lowered(
        model_factory=_MultiLinearModel,
        microbatches=4,
        data_ordering=StepDataOrdering(1, 4, reverse_breadth=False, pair_loss=False),
    )
    walk = _walk(lowered)
    head = max(stage for _, _, stage in walk)
    assert walk == [("F", p, s) for s in range(head + 1) for p in range(4)] + [
        ("B", p, s) for s in range(head, -1, -1) for p in range(4)
    ]
    _assert_topological(lowered)


def test_paired_loss_and_reversed_walk_place_stages_and_creators_as_told() -> None:
    lowered = _lowered(
        model_factory=_MultiLinearModel,
        microbatches=4,
        data_ordering=StepDataOrdering(2, 2),
    )
    walk = _walk(lowered)
    head = max(stage for _, _, stage in walk)
    assert head >= 1, "the ordering tests need a model that partitions into stages"
    expected: list[tuple[str, int, int]] = []
    for pass_index in range(2):
        positions = [2 * pass_index, 2 * pass_index + 1]
        expected += [("F", p, s) for s in range(head) for p in positions]
        for position in positions:
            expected += [("F", position, head), ("B", position, head)]
        expected += [
            ("B", p, s) for s in range(head - 1, -1, -1) for p in reversed(positions)
        ]
    assert walk == expected
    _assert_topological(lowered)

    gradient_ids = {item.gradient_object_id for item in lowered.gradients}
    tasks = {task.task_id: task for task in lowered.program.tasks}
    backward = {
        (entrypoint.microbatch, entrypoint.stage_index): tasks[entrypoint.task_id]
        for entrypoint in lowered.entrypoints
        if entrypoint.phase == "backward"
    }

    def creates(position: int, stage: int) -> bool:
        task = backward[(position, stage)]
        produced = set(task.outputs) & gradient_ids
        added = {item.object_id for item in task.mutations} & gradient_ids
        assert bool(produced) != bool(added), (position, stage)
        return bool(produced)

    for stage in range(head):
        # the pass's last microbatch goes first in backward, so it creates
        assert [creates(p, stage) for p in range(4)] == [False, True, False, False]
    # the paired last stage walks forward, so its first microbatch creates
    assert [creates(p, head) for p in range(4)] == [True, False, False, False]


def test_interleaved_optimizer_follows_the_stages_last_backward() -> None:
    lowered = _lowered(
        model_factory=_MultiLinearModel,
        microbatches=4,
        data_ordering=StepDataOrdering(1, 4, reverse_breadth=False, pair_loss=False),
    )
    order = [task.task_id for task in lowered.program.tasks]
    tasks = {task.task_id: task for task in lowered.program.tasks}
    for task_id in lowered.optimizer_task_ids:
        # the nearest task before it that is not optimizer work is a backward
        # it depends on: the stage's last, whichever microbatch ran it
        index = order.index(task_id) - 1
        while tasks[order[index]].phase == "optimizer":
            index -= 1
        previous = tasks[order[index]]
        assert previous.phase == "backward"
        assert previous.task_id in tasks[task_id].dependencies


def test_ordering_must_cover_the_step() -> None:
    with pytest.raises(CaptureError, match="covers 4 microbatches, but the step has 2"):
        _lowered(microbatches=2, data_ordering=StepDataOrdering(1, 4))
