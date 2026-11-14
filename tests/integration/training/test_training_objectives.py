"""Initialization and normalization do not add hidden model state."""

import pytest
import torch
from torch import nn

from shadowspill.training._inputs import ScaledObjective
from shadowspill.training._model import initialize_model


def test_scaled_tuple_objective_is_accepted_by_capture():
    from shadowspill.pytorch.capture.objective import capture_training_objective

    model = nn.Linear(3, 2)

    def objective(model, inputs):
        output = model(inputs["features"])
        return output.square().mean(), {"sum": output.detach().sum()}

    capture = capture_training_objective(
        model,
        ScaledObjective(objective),
        ({"features": torch.randn(5, 3)}, torch.tensor(0.25)),
    )
    assert capture.objective_schema.tensor_metric_positions == (0,)


@pytest.mark.parametrize("buffer_only", [False, True])
def test_meta_state_materializes_and_scale_is_an_input(buffer_only):
    model = nn.Module()
    value = torch.empty(2, device="meta", dtype=torch.float16)
    if buffer_only:
        model.register_buffer("weight", value)
    else:
        model.register_parameter("weight", nn.Parameter(value))
    initialize_model(model, initialize=lambda m: m.weight.data.fill_(4))
    assert model.weight.device.type == "cpu"
    assert model.weight.dtype == torch.float16
    assert set(model.state_dict()) == {"weight"}
    objective = ScaledObjective(lambda m, _: m.weight.float().sum())
    result, _ = objective(model, None, torch.tensor(0.25))
    assert result.item() == 2
    assert set(model.state_dict()) == {"weight"}
