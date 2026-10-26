"""Inline host tensors survive saved training with unequal example shapes."""

import copy

import pytest
import torch
from torch import nn

from shadowspill import build_step
from tests.shadowspill.execution.helpers import run_worker, save_plan


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([0.5, -0.25]))

    def forward(self, x):
        bounds = torch.tensor([0, x.shape[0]], device="cpu", dtype=torch.int64)
        return bounds[1:].clone(), x * self.weight


def objective(model, x):
    bounds, output = model(x)
    return output.square().sum(), {"bounds": bounds}


@pytest.mark.cuda
def test_host_metrics_and_mixed_shapes_in_saved_training(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    model = Network()
    initial = copy.deepcopy(model.state_dict())
    calls = [((torch.ones(size, 2),), {}) for size in (3, 5)]
    program = build_step(
        model,
        objective=objective,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01, foreach=False),
        example_calls=calls,
    )
    save_plan(program, tmp_path)
    reference = Network()
    optimizer = torch.optim.SGD(reference.parameters(), lr=0.01, foreach=False)
    expected = []
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        outputs = []
        for args, kwargs in calls:
            loss, metrics = objective(reference, *args, **kwargs)
            loss.backward()
            outputs.append((loss.detach(), metrics))
        optimizer.step()
        expected.append((outputs, copy.deepcopy(reference.state_dict())))
    torch.save(
        dict(
            initial=initial,
            examples=calls,
            expected=expected,
            hyperparams=[{}] * 3,
            accumulate=[False] * 3,
            restore=False,
            host_metrics=True,
        ),
        tmp_path / "oracle.pt",
    )
    run_worker(tmp_path)
