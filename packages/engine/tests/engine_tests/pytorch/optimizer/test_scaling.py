"""Final scaling preserves stored gradients and independent optimizer frontiers."""

import pytest
import torch

from shadowspill.pytorch.optimizer import capture_optimizer
from shadowspill.pytorch.optimizer.scaling import with_gradient_scale


@pytest.mark.parametrize("gradient_dtype", [torch.float32, torch.bfloat16])
def test_runtime_scale_matches_summed_gradient_without_changing_the_capture(
    gradient_dtype,
):
    parameters = {
        name: torch.nn.Parameter(torch.arange(4.0) + index)
        for index, name in enumerate(("first", "second"))
    }
    for index, parameter in enumerate(parameters.values()):
        parameter.grad = torch.arange(4.0) / 3 + index
    optimizer = torch.optim.SGD(parameters.values(), lr=0.125, foreach=False)
    owners = {name: (index,) for index, name in enumerate(parameters)}
    source = capture_optimizer(
        parameters,
        optimizer,
        gradient_dtype=gradient_dtype,
        parameter_stage_owners=owners,
    )
    original = source.update.graph_module.code
    scaled, name = with_gradient_scale(source, parameter_stage_owners=owners)
    assert source.update.graph_module.code == original
    assert len(scaled.update_tasks) == len(source.update_tasks) == 2
    assert {task.completion_stage_index for task in scaled.update_tasks} == {0, 1}
    assert all(name in task.binding_names for task in scaled.update_tasks)
    for factor in (0.0, -0.75, 1.0, 1.0 / 3):
        values = {
            **{key: p.detach().clone() for key, p in parameters.items()},
            **{
                f"gradient.{key}": p.grad.to(gradient_dtype).clone()
                for key, p in parameters.items()
            },
            name: torch.tensor(factor),
        }
        before = {key: value.clone() for key, value in values.items()}
        with torch.no_grad():
            scaled.update.graph_module(
                *(values[binding.name] for binding in scaled.bindings)
            )
        for key in parameters:
            gradient = before[f"gradient.{key}"]
            torch.testing.assert_close(
                values[key], before[key] - 0.125 * factor * gradient.float()
            )
            torch.testing.assert_close(
                values[f"gradient.{key}"], gradient, rtol=0, atol=0
            )
    assert len(source.bindings) + 1 == len(scaled.bindings)


def test_control_name_cannot_shadow_a_model_parameter():
    parameter = torch.nn.Parameter(torch.ones(2))
    parameter.grad = torch.ones_like(parameter)
    source = capture_optimizer(
        {"grad_scale": parameter}, torch.optim.SGD([parameter], lr=0.1)
    )
    scaled, name = with_gradient_scale(source)
    assert name != "grad_scale"
    assert len({binding.name for binding in scaled.bindings}) == len(scaled.bindings)
