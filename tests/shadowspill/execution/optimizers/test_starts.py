"""Saved initializer descriptions create optimizer state directly in spill storage."""

import copy

import pytest
import torch
from torch import nn

from shadowspill import build_step
from tests.shadowspill.execution._oracle import optimizer_values
from tests.shadowspill.execution.optimizers.common import run_worker, save_plan


class StartingOptimizer(torch.optim.Optimizer):
    def __init__(self, parameters):
        super().__init__(parameters, {"lr": 0.003})

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                state = self.state[parameter]
                if not state:
                    state["moment"] = torch.full_like(parameter, 0.25)
                    state["master"] = parameter.detach().double().clone()
                state["moment"].mul_(0.5).add_(parameter.grad)
                state["master"].sub_(group["lr"] * state["moment"])
                parameter.copy_(state["master"])


def objective(model, features):
    return model(features).square().mean()


@pytest.mark.cuda
def test_constant_and_parameter_initializers_survive_cold_execution(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.manual_seed(77)
    model = nn.Linear(6, 3)
    initial = copy.deepcopy(model.state_dict())
    examples = [((torch.randn(2, 6),), {})]
    template = copy.deepcopy(model).to("meta")
    update = StartingOptimizer(template.parameters())
    program = build_step(
        template,
        optimizer=update,
        objective=objective,
        example_args=examples[0][0],
    )
    assert not update.state and all(p.is_meta for p in template.parameters())
    bindings = {b["name"]: b for b in program.bindings["optimizer_bindings"]}
    initial_state = dict(initial)
    for name, value in initial.items():
        assert bindings[f"optimizer.{name}.moment"]["initialization"] == {
            "kind": "constant",
            "value": 0.25,
        }
        assert bindings[f"optimizer.{name}.master"]["initialization"] == {
            "kind": "parameter",
            "name": name,
        }
        initial_state[f"optimizer.{name}.moment"] = torch.full_like(value, 0.25)
        initial_state[f"optimizer.{name}.master"] = value.double()
    save_plan(program, tmp_path)
    optimizer = StartingOptimizer(model.parameters())
    expected = []
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = objective(model, *examples[0][0])
        loss.backward()
        optimizer.step()
        expected.append(
            (
                [(loss.detach(), None)],
                {
                    **copy.deepcopy(model.state_dict()),
                    **optimizer_values(model, optimizer),
                },
            )
        )
    torch.save(
        dict(
            initial=initial,
            initial_state=initial_state,
            examples=examples,
            expected=expected,
            hyperparams=[{}] * 3,
            accumulate=[False] * 3,
            restore=False,
        ),
        tmp_path / "oracle.pt",
    )
    run_worker(tmp_path)
