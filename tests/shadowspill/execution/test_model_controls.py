"""Per-call model buffers survive saved phases, real updates and checkpoint replay."""

import copy

import pytest
import torch
from torch import nn

from shadowspill import build_forward, build_step
from tests.shadowspill.execution._oracle import optimizer_values
from tests.shadowspill.execution.helpers import run_worker, save_plan


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(8, 12), nn.ReLU(), nn.Linear(12, 4))
        self.register_buffer("scale", torch.ones(()))
        self.register_buffer("temporary", torch.ones(()), persistent=False)

    def forward(self, x):
        return self.layers(x) * self.scale * self.temporary


def objective(model, x):
    return model(x).square().mean()


def snapshot(model):
    return {
        name: value.detach().clone()
        for name, value in (*model.named_parameters(), *model.named_buffers())
    }


@pytest.mark.cuda
@pytest.mark.parametrize("training", [False, True], ids=["forward", "step"])
def test_model_buffer_controls_in_saved_workers(tmp_path, training):
    torch.manual_seed(91)
    model, x = Network(), torch.randn(2, 8)
    initial = snapshot(model)
    template = copy.deepcopy(model).to("meta")
    options = dict(example_args=(x,))
    if training:
        program = build_step(
            template,
            optimizer=torch.optim.AdamW(template.parameters(), foreach=False),
            objective=objective,
            num_examples=2,
            **options,
        )
    else:
        program = build_forward(template, **options)
    registrations = {r["name"]: r for r in program.bindings["registrations"]}
    assert registrations["scale"]["persistent"]
    assert not registrations["temporary"]["persistent"]
    save_plan(program, tmp_path)
    optimizer = torch.optim.AdamW(model.parameters(), foreach=False)
    expected = []
    calls = [{"scale": 2.0}, {"scale": 0.5}, {}]
    for controls in calls:
        if "scale" in controls:
            model.scale.fill_(controls["scale"])
        if training:
            optimizer.zero_grad(set_to_none=True)
            losses = [objective(model, x) for _ in range(2)]
            for loss in losses:
                loss.backward()
            optimizer.step()
            output = [(loss.detach(), None) for loss in losses]
        else:
            output = model(x).detach()
        state = snapshot(model)
        if training:
            state.update(optimizer_values(model, optimizer))
        expected.append((output, state))
    torch.save(
        dict(
            initial=initial,
            examples=[((x,), {})] * (2 if training else 1),
            expected=expected,
            hyperparams=calls,
            accumulate=[False] * 3,
            invalid_hyperparams=[{"temporary": 2}, {"layers.0.weight": 2}],
            restore=False,
        ),
        tmp_path / "oracle.pt",
    )
    run_worker(tmp_path)
