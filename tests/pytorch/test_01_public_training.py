from __future__ import annotations

import os
from functools import partial

import pytest
import torch
import torch.nn as nn

from shadowspill.pytorch import InputGuardError, ObjectiveResult, plan


class _TrainingNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Linear(6, 10)
        self.second = nn.Linear(10, 3)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.second(torch.relu(self.first(value)))


def _training_objective(
    model: nn.Module, value: torch.Tensor, target: torch.Tensor, tag: str
) -> ObjectiveResult:
    error = model(value) - target
    return ObjectiveResult(
        error.square().mean(), {"mean": error.detach().mean(), "tag": tag}
    )


@pytest.mark.cuda
def test_public_training_accumulates_replays_and_restores(tmp_path: object) -> None:
    if "SHADOWSPILL_PYTORCH_LIBRARY" not in os.environ:
        pytest.skip("the built PyTorch adapter was not provided")
    os.environ["SHADOWSPILL_PROFILE_CACHE"] = str(tmp_path)
    torch.manual_seed(41)
    model = _TrainingNetwork()
    reference = _TrainingNetwork()
    reference.load_state_dict(model.state_dict())
    parameter_ids = tuple(id(parameter) for parameter in model.parameters())
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

    training = plan(
        model,
        objective=_training_objective,
        opt=partial(torch.optim.SGD, lr=0.02, foreach=False),
        example_inputs=examples,
        device_budget=2 << 30,
        host_budget=1 << 30,
    )
    assert training.plan_report.mode == "training"
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
