from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.export.graph_signature import InputKind

from shadowspill.pytorch import ObjectiveResult
from shadowspill.pytorch.aot import TrainingCapture, capture_training
from shadowspill.pytorch.contracts import CaptureError
from shadowspill.pytorch.fake import fake_cuda_inputs, fake_cuda_model
from shadowspill.pytorch.training import ObjectivePairExecutor


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, 2)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.projection(value)


def _objective(
    model: nn.Module, value: torch.Tensor, target: torch.Tensor
) -> ObjectiveResult:
    output = model(value)
    return ObjectiveResult(
        torch.nn.functional.mse_loss(output, target),
        {"mean": output.mean(), "label": "training"},
    )


def _capture(model: nn.Module, values: list[torch.Tensor]) -> TrainingCapture:
    mode = FakeTensorMode(allow_non_fake_inputs=True)
    fake_model = fake_cuda_model(model, mode)
    fake_values = fake_cuda_inputs(values, mode)
    with mode:
        return capture_training(fake_model, _objective, fake_values)


def _arguments(
    captured: TrainingCapture,
    model: nn.Module,
    values: list[torch.Tensor],
) -> tuple[object, ...]:
    flatten = captured.exported.exported_program._graph_module_flat_inputs
    arguments = list(flatten(tuple(values), {}))
    for index, spec in enumerate(
        captured.exported.exported_program.graph_signature.input_specs
    ):
        if spec.kind is InputKind.PARAMETER:
            assert spec.target is not None
            arguments[index] = model.get_parameter(spec.target.removeprefix("model."))
        elif spec.kind is InputKind.BUFFER:
            assert spec.target is not None
            arguments[index] = model.get_buffer(spec.target.removeprefix("model."))
    return tuple(arguments)


@pytest.mark.parametrize("variant", ["save_pair", "recompute_pair"])
def test_explicit_pair_matches_autograd(variant: str) -> None:
    torch.manual_seed(3)
    model = _Model()
    reference = _Model()
    reference.load_state_dict(model.state_dict())
    values = [torch.randn(4, 3), torch.randn(4, 2)]
    captured = _capture(model, values)
    pair = getattr(captured, variant)
    result = ObjectivePairExecutor(pair, captured.objective_schema)(
        _arguments(captured, model, values)
    )
    reference_result = _objective(reference, *values)
    reference_result.loss.backward()
    torch.testing.assert_close(result.loss, reference_result.loss)
    torch.testing.assert_close(result.metrics["mean"], reference_result.metrics["mean"])
    assert result.metrics["label"] == "training"
    parameter_gradients = [
        value for value in result.gradients if isinstance(value, torch.Tensor)
    ][:2]
    for actual, parameter in zip(
        parameter_gradients, reference.parameters(), strict=True
    ):
        torch.testing.assert_close(actual, parameter.grad)


def test_pair_executor_rejects_changed_abi() -> None:
    model = _Model()
    values = [torch.randn(2, 3), torch.randn(2, 2)]
    captured = _capture(model, values)
    executor = ObjectivePairExecutor(captured.save_pair, captured.objective_schema)
    with pytest.raises(CaptureError, match="argument count"):
        executor(())
