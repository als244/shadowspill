from __future__ import annotations

import copy
from functools import partial

import pytest
import torch
import torch.nn as nn

from shadowspill.pytorch import (
    InputGuardError,
    ObjectiveResult,
    externalize_model_state,
    plan_step,
    relocate_model_state,
)
from shadowspill.pytorch.optimizer import capture as optimizer_module
from shadowspill.pytorch.runtime_adapter.runtime import _adapter_path

from .runtime_test_support import public_test_runtime


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
        _adapter_path(None)
    except RuntimeError:
        pytest.skip("the built PyTorch adapter is not installed")


@pytest.mark.cuda
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
    model = relocate_model_state(
        model,
        runtime=runtime,
        pool="spill",
        release_source=True,
    )
    parameter_ids = tuple(id(parameter) for parameter in model.parameters())
    training = plan_step(
        model,
        objective=_training_objective,
        opt=partial(torch.optim.SGD, lr=0.02, foreach=False),
        example_inputs=examples,
        runtime=runtime,
        execution="execution",
        spill="spill",
        planning_cachedir=tmp_path,
        profiling_metadata=(
            {"batch_size": 2, "tag": "left"},
            {"batch_size": 4, "tag": "right"},
        ),
    )
    assert training.plan_report.mode == "training"
    assert training.plan_report.captured_stage_count == 4
    assert training.plan_report.aot_unique_stage_abis == 4
    assert training.plan_report.aot_graph_pair_cache_hits == 0
    assert training.plan_report.aot_graph_pair_cache_misses == 4
    assert training.plan_report.program is training.plan_report.execution_plan.program
    assert (
        training.plan_report.pressurefit_result.program == training.plan_report.program
    )
    assert training.plan_report.diagnostics.cache_artifacts
    assert len(training.plan_report.diagnostics.profiling_metadata) == 2
    assert all(parameter.device.type == "cuda" for parameter in model.parameters())
    with pytest.raises(InputGuardError):
        training([[*steps[0][0][:-1], "changed"], steps[0][1]])

    first = training(steps[0])
    assert first.step_number == 1
    assert first.diagnostics is None
    assert tuple(metric["tag"] for metric in first.metrics) == ("left", "right")
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
    externalize_model_state(model, runtime=runtime, release_runtime=True)
    assert tuple(id(parameter) for parameter in model.parameters()) == parameter_ids
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())
    for actual, expected in zip(
        model.parameters(), reference.parameters(), strict=True
    ):
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    assert set(training.state_dict()) == {"model", "optimizer", "step"}
    with pytest.raises(RuntimeError, match="closed"):
        training(steps[0])
    with pytest.raises(RuntimeError, match="closed"):
        training.__enter__()


@pytest.mark.cuda
def test_public_training_lazy_adamw_state_replays(tmp_path: object) -> None:
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
    model = relocate_model_state(
        model,
        runtime=runtime,
        pool="spill",
        release_source=True,
    )
    training = plan_step(
        model,
        objective=_training_objective,
        opt=partial(torch.optim.AdamW, lr=0.003, foreach=False),
        example_inputs=examples,
        runtime=runtime,
        execution="execution",
        spill="spill",
        planning_cachedir=tmp_path,
    )
    assert training.plan_report.initial_execution_plan is None
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
    externalize_model_state(model, runtime=runtime, release_runtime=True)
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())


@pytest.mark.cuda
def test_public_training_profiles_bounded_opaque_optimizer(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    _require_adapter()

    def reject_graph(_optimizer: torch.optim.Optimizer) -> torch.fx.GraphModule:
        raise RuntimeError("optimizer graph intentionally unavailable")

    monkeypatch.setattr(optimizer_module, "_export_optimizer_graph", reject_graph)
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
    model = relocate_model_state(
        model,
        runtime=runtime,
        pool="spill",
        release_source=True,
    )
    training = plan_step(
        model,
        objective=_training_objective,
        opt=partial(_OpaqueSgd, lr=0.02),
        example_inputs=examples,
        runtime=runtime,
        execution="execution",
        spill="spill",
        planning_cachedir=tmp_path,
    )
    actual = training(values)
    torch.testing.assert_close(actual.objectives[0].cpu(), expected.loss.detach())
    training.close()
    externalize_model_state(model, runtime=runtime, release_runtime=True)
    for planned, eager in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(planned, eager)


@pytest.mark.cuda
def test_public_training_partitions_cuda_only_optimizer_and_replays(
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
    model = relocate_model_state(
        model,
        runtime=runtime,
        pool="spill",
        release_source=True,
    )
    training = plan_step(
        model,
        objective=objective,
        opt=partial(
            mlops.optim.AdamW,
            lr=3e-3,
            state_dtype=torch.bfloat16,
            master_parameter_dtype=torch.bfloat16,
        ),
        example_inputs=inputs(92),
        runtime=runtime,
        execution="execution",
        spill="spill",
        planning_cachedir=tmp_path,
    )
    assert training.plan_report.initial_execution_plan is None
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
    externalize_model_state(model, runtime=runtime, release_runtime=True)
