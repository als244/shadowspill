from __future__ import annotations

import copy
from collections.abc import Iterable
from functools import partial
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from shadowspill.errors import (
    InputGuardError,
)
from shadowspill.pytorch import (
    ObjectiveResult,
    export_model_state,
    import_model_state,
    plan_step,
    read_model_state,
    read_optimizer_state,
)
from shadowspill.pytorch.execution.training.optimizer_state import OptimizerState
from shadowspill.pytorch.materialization.training import TrainingMaterializedState
from shadowspill.pytorch.optimizer import trace as optimizer_trace
from shadowspill.pytorch.state.storage import persistent_state
from shadowspill.runtime import RuntimeConfigurationError
from shadowspill.runtime.abi import runtime_library
from shadowspill.runtime.configuration import adapter_path
from shadowspill.runtime.occupancy import live_allocations
from shadowspill.runtime.plan.lifecycle import wait_plan_idle

from ..runtime_test_support import public_test_runtime


class _TrainingNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Linear(6, 10)
        self.second = nn.Linear(10, 3)
        self.register_buffer("runtime_scale", torch.tensor(1.0), persistent=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.second(torch.relu(self.first(value))) * self.runtime_scale


class _OpaqueSgd(torch.optim.Optimizer):
    def __init__(self, parameters: object, *, lr: float) -> None:
        super().__init__(parameters, {"lr": lr})

    @torch.no_grad()
    def step(self, closure: object = None) -> None:
        del closure
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    state = self.state[parameter]
                    if not state:
                        state["momentum"] = torch.zeros_like(parameter)
                    state["momentum"].mul_(0.9).add_(parameter.grad)
                    parameter.add_(state["momentum"], alpha=-group["lr"])


def _training_objective(
    model: nn.Module, value: torch.Tensor, target: torch.Tensor, tag: str
) -> ObjectiveResult:
    error = model(value) - target
    return ObjectiveResult(
        error.square().mean(), {"mean": error.detach().mean(), "tag": tag}
    )


def _require_adapter() -> None:
    if torch.cuda.is_initialized():
        pytest.skip("public allocator installation requires a fresh process")
    try:
        adapter_path(None)
    except RuntimeError:
        pytest.skip("the built PyTorch adapter is not installed")


@pytest.mark.cuda
@pytest.mark.fresh_process
def test_public_training_accumulates_replays_and_restores(tmp_path: object) -> None:
    _require_adapter()
    torch.manual_seed(41)
    model = _TrainingNetwork()
    reference = _TrainingNetwork()
    reference.load_state_dict(model.state_dict())
    examples = [
        [torch.randn(2, 6), torch.randn(2, 3), "left"],
        [torch.randn(4, 6), torch.randn(4, 3), "right"],
    ]
    steps = [
        [
            [torch.randn(2, 6), torch.randn(2, 3), "left"],
            [torch.randn(4, 6), torch.randn(4, 3), "right"],
        ],
        [
            [torch.randn(2, 6), torch.randn(2, 3), "left"],
            [torch.randn(4, 6), torch.randn(4, 3), "right"],
        ],
    ]
    reference_optimizer = torch.optim.SGD(
        reference.parameters(), lr=0.02, foreach=False
    )
    expected_losses: list[tuple[torch.Tensor, ...]] = []
    for microbatches in steps:
        reference_optimizer.zero_grad(set_to_none=True)
        losses: list[torch.Tensor] = []
        for value, target, tag in microbatches:
            result = _training_objective(reference, value, target, tag)
            result.loss.backward()
            losses.append(result.loss.detach())
        reference_optimizer.step()
        expected_losses.append(tuple(losses))

    runtime = public_test_runtime()
    model = import_model_state(
        model,
        runtime=runtime,
        pool="spill",
        release_source=True,
    )
    parameter_ids = tuple(id(parameter) for parameter in model.parameters())
    training = plan_step(
        model,
        objective=_training_objective,
        optimizer=partial(torch.optim.SGD, lr=0.02, foreach=False),
        example_inputs=examples,
        runtime=runtime,
        execution="execution",
        spill="spill",
        artifact_store=tmp_path,
        profiling_metadata=(
            {"batch_size": 2, "tag": "left"},
            {"batch_size": 4, "tag": "right"},
        ),
    )
    assert training.plan_report.mode == "training"
    assert training.plan_report.captured_stage_count == 4
    assert training.plan_report.aot_unique_stage_contracts == 4
    assert training.plan_report.aot_graph_pair_cache_hits == 0
    assert training.plan_report.aot_graph_pair_cache_misses == 4
    assert training.plan_report.program is training.plan_report.execution_plan.program
    assert training.plan_report.search_result.program == training.plan_report.program
    assert training.plan_report.diagnostics.cache_artifacts
    assert len(training.plan_report.diagnostics.profiling_metadata) == 2
    layouts = training.plan_report.diagnostics.physical_layouts
    assert tuple(item.plan_role for item in layouts) == ("step",)
    assert all(item.strategy == "fixed" for item in layouts)
    assert all(item.required_bytes <= item.pool_capacity_bytes for item in layouts)
    assert all(item.attempts[-1].accepted for item in layouts)
    assert all(
        attempt.search_wall_time_ns > 0 and attempt.physical_admission_wall_time_ns > 0
        for layout in layouts
        for attempt in layout.attempts
    )
    assert all(parameter.device.type == "cuda" for parameter in model.parameters())
    with pytest.raises(InputGuardError):
        training([[*steps[0][0][:-1], "changed"], steps[0][1]])

    first = training(steps[0])
    assert first.step_number == 1
    assert first.diagnostics is None
    assert tuple(metric["tag"] for metric in first.metrics) == ("left", "right")
    first_objective_pointers = tuple(
        int(objective.untyped_storage().data_ptr()) for objective in first.objectives
    )
    assert len(set(first_objective_pointers)) == len(first.objectives)
    first_objective_values = tuple(objective.cpu() for objective in first.objectives)
    for actual, expected in zip(first.objectives, expected_losses[0], strict=True):
        torch.testing.assert_close(actual.cpu(), expected, rtol=2e-5, atol=2e-6)

    checkpoint = training.state_dict()
    assert "runtime_scale" not in checkpoint["model"]
    with pytest.raises(RuntimeError, match="keys differ"):
        training.load_state_dict({})
    with pytest.raises(TypeError, match="mappings"):
        training.load_state_dict(
            {"model": 1, "optimizer": checkpoint["optimizer"], "step": 1}
        )
    with pytest.raises(TypeError, match="non-negative"):
        training.load_state_dict(
            {
                "model": checkpoint["model"],
                "optimizer": checkpoint["optimizer"],
                "step": True,
            }
        )
    with pytest.raises(RuntimeError, match="model state_dict keys differ"):
        training.load_state_dict(
            {"model": {}, "optimizer": checkpoint["optimizer"], "step": 1}
        )
    second = training(steps[1])
    second_objective_pointers = tuple(
        int(objective.untyped_storage().data_ptr()) for objective in second.objectives
    )
    assert len(set(second_objective_pointers)) == len(second.objectives)
    assert set(first_objective_pointers).isdisjoint(second_objective_pointers)
    for actual, retained in zip(first.objectives, first_objective_values, strict=True):
        torch.testing.assert_close(actual.cpu(), retained, rtol=0, atol=0)
    for actual, expected in zip(second.objectives, expected_losses[1], strict=True):
        torch.testing.assert_close(actual.cpu(), expected, rtol=2e-5, atol=2e-6)
    uninterrupted = {
        name: tensor.clone() for name, tensor in training.state_dict()["model"].items()
    }
    training.load_state_dict(checkpoint)
    replay = training(steps[1])
    assert replay.step_number == 2
    replayed = training.state_dict()["model"]
    assert all(
        torch.equal(uninterrupted[name], replayed[name]) for name in uninterrupted
    )

    training.close()
    training.close()
    export_model_state(model, runtime=runtime, release_runtime=True)
    assert tuple(id(parameter) for parameter in model.parameters()) == parameter_ids
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())
    for actual, expected in zip(
        model.parameters(), reference.parameters(), strict=True
    ):
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    # Closing releases optimizer state with the plan, so the checkpoint
    # taken above is the only one there will be.
    with pytest.raises(RuntimeError, match="take the checkpoint before close"):
        training.state_dict()
    with pytest.raises(RuntimeError, match="closed"):
        training(steps[0])
    with pytest.raises(RuntimeError, match="closed"):
        training.__enter__()


@pytest.mark.cuda
@pytest.mark.fresh_process
def test_public_training_breadth_first_matches_the_eager_reference(
    tmp_path: object,
) -> None:
    """A stage-major walk trains the same weights as the eager, microbatch-major one.

    Under ``breadth=2`` with the flags at their defaults the pass's second
    microbatch creates every gradient but the paired last stage's, and the
    first adds into them, so this is the test that the creator moving with
    the walk changes nothing about what the step computes.
    """
    _require_adapter()
    torch.manual_seed(43)
    model = _TrainingNetwork()
    reference = _TrainingNetwork()
    reference.load_state_dict(model.state_dict())
    examples = [
        [torch.randn(2, 6), torch.randn(2, 3), "left"],
        [torch.randn(4, 6), torch.randn(4, 3), "right"],
    ]
    steps = [
        [
            [torch.randn(2, 6), torch.randn(2, 3), "left"],
            [torch.randn(4, 6), torch.randn(4, 3), "right"],
        ]
        for _ in range(3)
    ]
    reference_optimizer = torch.optim.SGD(
        reference.parameters(), lr=0.02, foreach=False
    )
    expected_losses: list[tuple[torch.Tensor, ...]] = []
    for microbatches in steps:
        reference_optimizer.zero_grad(set_to_none=True)
        losses: list[torch.Tensor] = []
        for value, target, tag in microbatches:
            result = _training_objective(reference, value, target, tag)
            result.loss.backward()
            losses.append(result.loss.detach())
        reference_optimizer.step()
        expected_losses.append(tuple(losses))
    runtime = public_test_runtime()
    model = import_model_state(
        model,
        runtime=runtime,
        pool="spill",
        release_source=True,
    )
    training = plan_step(
        model,
        objective=_training_objective,
        optimizer=partial(torch.optim.SGD, lr=0.02, foreach=False),
        example_inputs=examples,
        runtime=runtime,
        execution="execution",
        spill="spill",
        depth=1,
        breadth=2,
        artifact_store=tmp_path,
        profiling_metadata=(
            {"batch_size": 2, "tag": "left"},
            {"batch_size": 4, "tag": "right"},
        ),
    )
    report = training.plan_report
    assert report.data_ordering is not None
    assert report.data_ordering.label == "1x2rp"
    for microbatches, expected in zip(steps, expected_losses, strict=True):
        outcome = training(microbatches)
        for actual, loss in zip(outcome.objectives, expected, strict=True):
            torch.testing.assert_close(actual.cpu(), loss, rtol=2e-5, atol=2e-6)
    training.close()
    export_model_state(model, runtime=runtime, release_runtime=True)
    for actual, expected in zip(
        model.parameters(), reference.parameters(), strict=True
    ):
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


@pytest.mark.cuda
@pytest.mark.fresh_process
def test_public_training_declared_adamw_state_replays(tmp_path: object) -> None:
    _require_adapter()
    torch.manual_seed(73)
    model = _TrainingNetwork()
    examples = [
        [torch.randn(2, 6), torch.randn(2, 3), "left"],
        [torch.randn(4, 6), torch.randn(4, 3), "right"],
    ]
    torch.manual_seed(74)
    first_inputs = [
        [torch.randn(2, 6), torch.randn(2, 3), "left"],
        [torch.randn(4, 6), torch.randn(4, 3), "right"],
    ]
    torch.manual_seed(75)
    second_inputs = [
        [torch.randn(2, 6), torch.randn(2, 3), "left"],
        [torch.randn(4, 6), torch.randn(4, 3), "right"],
    ]
    runtime = public_test_runtime()
    model = import_model_state(
        model,
        runtime=runtime,
        pool="spill",
        release_source=True,
    )
    built: list[torch.optim.Optimizer] = []

    def build_optimizer(
        parameters: Iterable[torch.nn.Parameter],
    ) -> torch.optim.Optimizer:
        optimizer = torch.optim.AdamW(parameters, lr=0.003, foreach=False)
        built.append(optimizer)
        return optimizer

    training = plan_step(
        model,
        objective=_training_objective,
        optimizer=build_optimizer,
        example_inputs=examples,
        runtime=runtime,
        execution="execution",
        spill="spill",
        artifact_store=tmp_path,
    )
    optimizer_owner = persistent_state(runtime, built[0])
    assert optimizer_owner is not None
    assert optimizer_owner.storages
    # Adopting a persistent storage into a plan rekeys it, and the plan being
    # idle restores it. Planning alone leaves nothing adopted, so a storage
    # still carrying a rekeyed identity here is one that leaked out of a plan.
    assert all(
        item.current_object_id == item.persistent_object_id
        for item in optimizer_owner.storages
    )
    # The point of importing the optimizer's state is that it lives in spill.
    assert {item.pool_id for item in optimizer_owner.storages} == {1}
    initial_state = training.state_dict()
    assert initial_state["optimizer"]["state"]
    training.load_state_dict(initial_state)

    training(first_inputs)
    checkpoint = training.state_dict()
    optimizer = checkpoint["optimizer"]
    assert isinstance(optimizer, dict)
    assert all(
        not isinstance(value, torch.Tensor) or value.device.type == "cpu"
        for parameter_state in optimizer["state"].values()
        for value in parameter_state.values()
    )
    training(second_inputs)
    uninterrupted = training.state_dict()
    training.load_state_dict(checkpoint)
    training(second_inputs)
    replayed = training.state_dict()
    assert all(
        torch.equal(value, replayed["model"][name])
        for name, value in uninterrupted["model"].items()
    )
    for parameter_id, parameter_state in uninterrupted["optimizer"]["state"].items():
        for name, value in parameter_state.items():
            other = replayed["optimizer"]["state"][parameter_id][name]
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, other)

    training.close()
    # Closing released the optimizer's runtime state along with the plan.
    assert persistent_state(runtime, built[0]) is None
    export_model_state(model, runtime=runtime, release_runtime=True)
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())


class _StartingOptimizer(torch.optim.Optimizer):
    """Keeps a moment that starts at a quarter and a float64 master of each
    weight, both made on the first step."""

    def __init__(self, parameters: Iterable[torch.nn.Parameter], *, lr: float) -> None:
        super().__init__(parameters, {"lr": lr})

    @torch.no_grad()
    def step(self, closure: object = None) -> None:
        del closure
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                state = self.state[parameter]
                if not state:
                    state["moment"] = torch.full_like(parameter, 0.25)
                    state["master"] = parameter.detach().to(torch.float64).clone()
                state["moment"].mul_(0.5).add_(parameter.grad)
                state["master"].sub_(group["lr"] * state["moment"])
                parameter.copy_(state["master"])


@pytest.mark.cuda
@pytest.mark.fresh_process
def test_public_training_fills_declared_state_in_the_spill_pool(
    tmp_path: object,
) -> None:
    """Each state entry is made in the spill pool, at what the optimizer's own
    first step starts it at.

    The host never holds the state beside the pool: every entry is imported
    before it holds anything and given its start there -- a constant, or its
    weight at the entry's own precision.
    """

    _require_adapter()
    torch.manual_seed(77)
    model = _TrainingNetwork()
    examples = [[torch.randn(2, 6), torch.randn(2, 3), "left"]]
    runtime = public_test_runtime()
    model = import_model_state(model, runtime=runtime, pool="spill")
    weights = read_model_state(model, runtime=runtime)
    built: list[torch.optim.Optimizer] = []

    def build_optimizer(
        parameters: Iterable[torch.nn.Parameter],
    ) -> torch.optim.Optimizer:
        optimizer = _StartingOptimizer(parameters, lr=0.003)
        built.append(optimizer)
        return optimizer

    training = plan_step(
        model,
        objective=_training_objective,
        optimizer=build_optimizer,
        example_inputs=examples,
        runtime=runtime,
        execution="execution",
        spill="spill",
        artifact_store=tmp_path,
    )
    # The pool owns the state the plan created; what it holds is read back
    # from there.
    assert persistent_state(runtime, built[0]) is not None
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    # State is read back as `state.<ordinal>.<entry>`, in the order of the
    # optimizer's own state, which is keyed by parameter.
    owners = [names[id(parameter)] for parameter in built[0].state]
    values = read_optimizer_state(built[0], runtime=runtime)
    moments = [value for key, value in values.items() if key.endswith(".moment")]
    masters = {key: value for key, value in values.items() if key.endswith(".master")}
    assert moments and all(torch.all(value == 0.25) for value in moments)
    assert len(masters) == len(names)
    for key, value in masters.items():
        weight = owners[int(key.split(".")[1])]
        assert value.dtype == torch.float64
        assert torch.equal(value, weights[weight].to(torch.float64))
    training.close()


@pytest.mark.cuda
@pytest.mark.fresh_process
def test_public_training_refuses_state_with_no_value_before_its_first_step(
    tmp_path: object,
) -> None:
    """State an optimizer starts from the gradient is refused, not guessed.

    SGD's momentum buffer begins as the first gradient, which does not exist
    before the first step; any value planning chose would be one the optimizer
    never starts at.
    """

    _require_adapter()
    torch.manual_seed(78)
    model = _TrainingNetwork()
    examples = [[torch.randn(2, 6), torch.randn(2, 3), "left"]]
    runtime = public_test_runtime()
    model = import_model_state(model, runtime=runtime, pool="spill")
    built: list[torch.optim.Optimizer] = []

    def build_optimizer(
        parameters: Iterable[torch.nn.Parameter],
    ) -> torch.optim.Optimizer:
        optimizer = torch.optim.SGD(parameters, lr=0.003, momentum=0.9)
        built.append(optimizer)
        return optimizer

    with pytest.raises(RuntimeError, match="no value before its first step"):
        plan_step(
            model,
            objective=_training_objective,
            optimizer=build_optimizer,
            example_inputs=examples,
            runtime=runtime,
            execution="execution",
            spill="spill",
            artifact_store=tmp_path,
        )
    # The refused plan's state went with it; the caller's model stays imported.
    assert persistent_state(runtime, built[0]) is None
    assert persistent_state(runtime, model) is not None


def _refuse_copy(*args: object, **kwargs: object) -> None:
    raise AssertionError("the state was copied out of the pool")


def _same_state(first: dict[str, object], second: dict[str, object]) -> bool:
    models = first["model"], second["model"]
    assert isinstance(models[0], dict) and isinstance(models[1], dict)
    return set(models[0]) == set(models[1]) and all(
        torch.equal(value, models[1][name]) for name, value in models[0].items()
    )


@pytest.mark.cuda
@pytest.mark.fresh_process
def test_public_training_saves_straight_from_the_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A checkpoint is written from the pool itself, and resumes exactly."""

    _require_adapter()
    torch.manual_seed(81)
    runtime = public_test_runtime()
    model = import_model_state(_TrainingNetwork(), runtime=runtime, pool="spill")
    batches = [[[torch.randn(2, 6), torch.randn(2, 3), "left"]] for _ in range(3)]
    training = plan_step(
        model,
        objective=_training_objective,
        optimizer=partial(torch.optim.AdamW, lr=0.003, foreach=False),
        example_inputs=batches[0],
        runtime=runtime,
        execution="execution",
        spill="spill",
        artifact_store=tmp_path,
    )
    training(batches[1])
    path = tmp_path / "checkpoint.pt"
    with monkeypatch.context() as patch:
        patch.setattr(OptimizerState, "_copied_alias_buffer", _refuse_copy)
        patch.setattr(TrainingMaterializedState, "_read_model_aliases", _refuse_copy)
        training.save(path)
    saved = torch.load(path, mmap=True, weights_only=True)
    expected = training.state_dict()
    assert _same_state(saved, expected)
    for index, entries in expected["optimizer"]["state"].items():
        for key, value in entries.items():
            assert torch.equal(saved["optimizer"]["state"][index][key], value)
    assert saved["step"] == 1 and not saved["model_from_optimizer"]

    training(batches[2])
    uninterrupted = training.state_dict()
    training.load_state_dict(torch.load(path, mmap=True, weights_only=True))
    training(batches[2])
    assert _same_state(training.state_dict(), uninterrupted)
    training.close()


@pytest.mark.cuda
@pytest.mark.fresh_process
def test_public_training_saves_a_master_copy_once(tmp_path: Path) -> None:
    """A weight the optimizer keeps a higher-precision master of is its master
    cast, bit for bit, after every update, so a checkpoint writes it once."""

    mlops = pytest.importorskip("mlops")
    _require_adapter()
    torch.manual_seed(82)
    runtime = public_test_runtime()
    model = import_model_state(
        _TrainingNetwork().to(torch.bfloat16), runtime=runtime, pool="spill"
    )
    batches = [
        [
            [
                torch.randn(2, 6, dtype=torch.bfloat16),
                torch.randn(2, 3, dtype=torch.bfloat16),
                "left",
            ]
        ]
        for _ in range(3)
    ]

    training = plan_step(
        model,
        objective=_training_objective,
        optimizer=partial(
            mlops.optim.AdamW, lr=0.003, master_parameter_dtype=torch.float32
        ),
        example_inputs=batches[0],
        runtime=runtime,
        execution="execution",
        spill="spill",
        artifact_store=tmp_path,
    )
    training(batches[1])
    path = tmp_path / "checkpoint.pt"
    training.save(path)
    saved = torch.load(path, mmap=True, weights_only=True)
    weights = {name for name, _parameter in model.named_parameters()}
    assert set(saved["model_from_optimizer"]) == weights
    assert {key for _index, key in saved["model_from_optimizer"].values()} == {
        "master_parameter"
    }
    assert not weights & set(saved["model"])

    training(batches[2])
    uninterrupted = training.state_dict()
    training.load_state_dict(torch.load(path, mmap=True, weights_only=True))
    training(batches[2])
    assert _same_state(training.state_dict(), uninterrupted)
    training.close()


@pytest.mark.cuda
@pytest.mark.fresh_process
def test_public_training_owns_the_model_state_it_imported(tmp_path: object) -> None:
    _require_adapter()
    torch.manual_seed(76)
    model = _TrainingNetwork()
    examples = [[torch.randn(2, 6), torch.randn(2, 3), "left"]]
    runtime = public_test_runtime()

    # No import_model_state: planning imports the state, so the plan owns it.
    training = plan_step(
        model,
        objective=_training_objective,
        optimizer=partial(torch.optim.SGD, lr=0.02, foreach=False),
        example_inputs=examples,
        runtime=runtime,
        execution="execution",
        spill="spill",
        artifact_store=tmp_path,
    )
    owned = persistent_state(runtime, model)
    assert owned is not None
    assert owned.owning_plan is not None
    weights = training.state_dict()["model"]["first.weight"].clone()

    # Reading answers while the plan holds the model, which exporting cannot.
    name = "first.weight"
    assert torch.equal(read_model_state(model, runtime=runtime)[name], weights)
    with pytest.raises(RuntimeConfigurationError):
        export_model_state(model, runtime=runtime)

    result = training(examples)
    del result
    training.close()

    # The plan created the state, so closing released it and emptied the
    # parameters that viewed it. What was read before the close survives.
    assert persistent_state(runtime, model) is None
    assert all(parameter.numel() == 0 for parameter in model.parameters())
    assert torch.isfinite(weights).all()


@pytest.mark.cuda
@pytest.mark.fresh_process
def test_public_training_profiles_bounded_opaque_optimizer(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    _require_adapter()

    def reject_graph(_optimizer: torch.optim.Optimizer) -> torch.fx.GraphModule:
        raise RuntimeError("optimizer graph intentionally unavailable")

    monkeypatch.setattr(optimizer_trace, "_export_optimizer_graph", reject_graph)
    torch.manual_seed(81)
    model = _TrainingNetwork()
    reference = _TrainingNetwork()
    reference.load_state_dict(model.state_dict())
    examples = [[torch.randn(2, 6), torch.randn(2, 3), "opaque"]]
    values = [[torch.randn(2, 6), torch.randn(2, 3), "opaque"]]
    reference_optimizer = _OpaqueSgd(reference.parameters(), lr=0.02)
    expected = _training_objective(reference, *values[0])
    expected.loss.backward()
    reference_optimizer.step()

    runtime = public_test_runtime()
    model = import_model_state(
        model,
        runtime=runtime,
        pool="spill",
        release_source=True,
    )
    training = plan_step(
        model,
        objective=_training_objective,
        optimizer=partial(_OpaqueSgd, lr=0.02),
        example_inputs=examples,
        runtime=runtime,
        execution="execution",
        spill="spill",
        artifact_store=tmp_path,
    )
    actual = training(values)
    torch.testing.assert_close(actual.objectives[0].cpu(), expected.loss.detach())
    training.close()
    export_model_state(model, runtime=runtime, release_runtime=True)
    for planned, eager in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(planned, eager)


@pytest.mark.cuda
@pytest.mark.fresh_process
def test_public_training_partitions_device_only_optimizer_and_replays(
    tmp_path: object,
) -> None:
    mlops = pytest.importorskip("mlops")
    _require_adapter()

    class Network(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = nn.Linear(8, 12, bias=False, dtype=torch.bfloat16)
            self.second = nn.Linear(12, 4, bias=False, dtype=torch.bfloat16)

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return self.second(torch.relu(self.first(value)))

    def objective(
        model: nn.Module, value: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return torch.nn.functional.mse_loss(model(value).float(), target.float())

    def inputs(seed: int) -> list[list[torch.Tensor]]:
        torch.manual_seed(seed)
        return [
            [
                torch.randn(2, 8, dtype=torch.bfloat16),
                torch.randn(2, 4, dtype=torch.bfloat16),
            ],
            [
                torch.randn(3, 8, dtype=torch.bfloat16),
                torch.randn(3, 4, dtype=torch.bfloat16),
            ],
        ]

    torch.manual_seed(91)
    model = Network()
    runtime = public_test_runtime()
    model = import_model_state(
        model,
        runtime=runtime,
        pool="spill",
        release_source=True,
    )
    training = plan_step(
        model,
        objective=objective,
        optimizer=partial(
            mlops.optim.AdamW,
            lr=3e-3,
            state_dtype=torch.bfloat16,
            master_parameter_dtype=torch.bfloat16,
        ),
        example_inputs=inputs(92),
        runtime=runtime,
        execution="execution",
        spill="spill",
        artifact_store=tmp_path,
    )
    optimizer_tasks = tuple(
        task
        for task in training.plan_report.execution_plan.program.tasks
        if task.phase == "optimizer"
    )
    assert len(optimizer_tasks) == 2
    assert training.state_dict()["optimizer"]["state"]

    training(inputs(93))
    checkpoint = copy.deepcopy(training.state_dict())
    training(inputs(94))
    uninterrupted = copy.deepcopy(training.state_dict())
    training.load_state_dict(checkpoint)
    training(inputs(94))
    replayed = training.state_dict()

    for name, value in uninterrupted["model"].items():
        assert torch.equal(value, replayed["model"][name])
    for parameter_id, parameter_state in uninterrupted["optimizer"]["state"].items():
        for name, value in parameter_state.items():
            other = replayed["optimizer"]["state"][parameter_id][name]
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, other)
            else:
                assert value == other
    training.close()
    export_model_state(model, runtime=runtime, release_runtime=True)


@pytest.mark.cuda
@pytest.mark.fresh_process
def test_public_training_follows_a_learning_rate_schedule() -> None:
    """A rate named at planning and set per step reproduces eager training.

    This is the whole claim behind ``hyperparams``: one plan, one capture, and
    a value the step reads rather than one the capture folded in. Comparing
    against an eager run of the same schedule is what distinguishes that from
    a step that quietly uses the rate it was planned with.
    """

    _require_adapter()
    schedule = (1e-3, 5e-4, 2.5e-4, 1.25e-4, 8e-4, 1e-5)

    def network() -> nn.Module:
        torch.manual_seed(91)
        return nn.Sequential(
            nn.Linear(8, 12, bias=False),
            nn.ReLU(),
            nn.Linear(12, 4, bias=False),
        )

    def batches(seed: int) -> list[list[torch.Tensor]]:
        torch.manual_seed(seed)
        return [[torch.randn(2, 8), torch.randn(2, 4)]]

    def objective(model: nn.Module, value: torch.Tensor, target: torch.Tensor):
        return torch.nn.functional.mse_loss(model(value), target)

    build = partial(torch.optim.AdamW, foreach=False)
    runtime = public_test_runtime()
    model = import_model_state(
        network(), runtime=runtime, pool="spill", release_source=True
    )
    training = plan_step(
        model,
        objective=objective,
        optimizer=build,
        hyperparams=("lr",),
        example_inputs=batches(92),
        runtime=runtime,
        execution="execution",
        spill="spill",
    )
    for index, rate in enumerate(schedule):
        training(batches(100 + index), hyperparams={"lr": rate})
    planned = {
        name: value.detach().float().cpu()
        for name, value in training.state_dict()["model"].items()
    }
    training.close()

    reference = network().cuda()
    optimizer = build(reference.parameters())
    for index, rate in enumerate(schedule):
        for group in optimizer.param_groups:
            group["lr"] = rate
        value, target = (item.cuda() for item in batches(100 + index)[0])
        optimizer.zero_grad()
        objective(reference, value, target).backward()
        optimizer.step()

    for name, expected in reference.state_dict().items():
        torch.testing.assert_close(
            planned[name], expected.detach().float().cpu(), rtol=1e-5, atol=1e-6
        )


@pytest.mark.cuda
@pytest.mark.fresh_process
def test_public_training_keeps_no_output_the_caller_dropped() -> None:
    """A step's outputs are the caller's alone once handed over.

    The losses a step returns live in the execution pool for as long as the
    caller keeps them. When the caller lets its result go and the step has
    finished, none of them may still be held there.
    """

    _require_adapter()

    def objective(model: nn.Module, value: torch.Tensor, target: torch.Tensor):
        return torch.nn.functional.mse_loss(model(value), target)

    def batches() -> list[list[torch.Tensor]]:
        return [[torch.randn(2, 8), torch.randn(2, 4)] for _ in range(2)]

    runtime = public_test_runtime()
    torch.manual_seed(5)
    model = import_model_state(
        nn.Linear(8, 4), runtime=runtime, pool="spill", release_source=True
    )
    training = plan_step(
        model,
        objective=objective,
        optimizer=partial(torch.optim.SGD, lr=0.1),
        example_inputs=batches(),
        runtime=runtime,
        execution="execution",
        spill="spill",
    )
    plan_id = int(runtime_library().shadowspill_plan_id(training._plan_handle))

    def handed_over() -> list[object]:
        return [
            item
            for item in live_allocations(runtime, "execution")
            if item.origin_plan_id == plan_id
            and item.ever_plan_owned
            and not item.logical_freed
        ]

    result = training(batches())
    assert len(result.objectives) == 2
    wait_plan_idle(training._plan_handle)
    assert len(handed_over()) == 2
    del result
    assert handed_over() == []
    training.close()
