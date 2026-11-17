"""Metric semantics: unequal microbatches and final accumulated gradients."""

import pytest
import torch

from qualification.reference.training.weights import Weights as _Masters
from shadowspill.training.observations import (
    StepObservations,
    parameter_norms,
    parameter_scalars,
)


def test_norms_read_compute_weights_and_accumulated_gradients_before_master_update():
    model = torch.nn.Linear(2, 1, bias=False, dtype=torch.bfloat16)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[3, 4]]))
    masters = _Masters(model, torch.float32, torch.float32)
    for gradient in (torch.tensor([[1, 2]]), torch.tensor([[2, 2]])):
        model.weight.grad = gradient.to(torch.bfloat16)
        masters.accumulate()
    observed = masters.observe(parameter_norms)
    assert all(isinstance(value, torch.Tensor) for value in observed["weight"].values())
    # The norm of the sum (5), not the sum of the per-microbatch norms.
    assert observed["weight"]["grad_norm"].item() == 5
    assert observed["weight"]["param_norm"].item() == 5
    optimizer = torch.optim.SGD(masters.parameters(), lr=1)
    masters.step(optimizer)
    assert torch.count_nonzero(model.weight) == 0
    host = StepObservations.collect((torch.tensor(1.5),), parameter_metrics=observed)
    assert tuple(host) == (1.5,)
    scalars = parameter_scalars(host.parameter_metrics, {"weight": 2})
    assert scalars["grad_norm/global/l2"] == 5
    assert scalars["param_norm/global/l2"] == 5


def test_derived_norms_use_sizes_and_sum_squared_gradients_through_module_tree():
    observed = {
        "blocks.0.attn.weight": {
            "grad_norm": torch.tensor(3.0),
            "param_norm": torch.tensor(6.0),
        },
        "blocks.0.moe.weight": {
            "grad_norm": torch.tensor(4.0),
            "param_norm": torch.tensor(8.0),
        },
        "blocks.1.weight": {
            "grad_norm": torch.tensor(12.0),
            "param_norm": torch.tensor(0.0),
        },
    }
    sizes = {"blocks.0.attn.weight": 9, "blocks.0.moe.weight": 4, "blocks.1.weight": 9}
    scalars = parameter_scalars(observed, sizes)
    assert scalars["grad_rms/blocks.00.attn/weight"] == 1
    assert scalars["param_rms/blocks.00.attn/weight"] == 2
    assert scalars["grad_rms/blocks.00.moe/weight"] == 2
    assert scalars["param_rms/blocks.00.moe/weight"] == 4
    assert scalars["grad_weight_ratio/blocks.00.attn/weight"] == pytest.approx(0.5)
    assert scalars["grad_weight_ratio/blocks.01/weight"] == pytest.approx(12e12)
    assert scalars["grad_norm/global/l2"] == 13
    assert scalars["grad_squared_share/blocks.00"] == pytest.approx(25 / 169)
    assert scalars["grad_squared_share/blocks.01"] == pytest.approx(144 / 169)
    assert scalars["grad_squared_share/blocks"] == 1
    assert scalars["grad_squared_share/blocks.00.attn"] == pytest.approx(9 / 169)
    assert scalars["grad_squared_share/blocks.00.moe"] == pytest.approx(16 / 169)


def test_zero_gradients_and_empty_parameters_do_not_divide_by_zero():
    observed = {
        "weight": {"grad_norm": torch.tensor(0.0), "param_norm": torch.tensor(0.0)},
        "empty": {"grad_norm": torch.tensor(0.0), "param_norm": torch.tensor(0.0)},
    }
    scalars = parameter_scalars(observed, {"weight": 4, "empty": 0})
    assert scalars["grad_rms/parameters/weight"] == 0
    assert scalars["param_rms/parameters/weight"] == 0
    assert scalars["grad_weight_ratio/parameters/weight"] == 0
    assert scalars["grad_squared_share/parameters"] == 0
    assert "grad_rms/parameters/empty" not in scalars
