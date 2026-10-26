"""Python optimizer fallback still awaiting a saved-Plan migration decision.

Ordinary training, initialization, precision, checkpoints and lifecycle coverage
live under tests/shadowspill/{execution,training}.
"""

from __future__ import annotations

from functools import partial

import pytest
import torch
from torch import nn

from qualification.profiling import CORRECTNESS_PROFILING
from shadowspill.pytorch import (
    ObjectiveResult,
    export_model_state,
    import_model_state,
    plan_step,
)
from shadowspill.pytorch.optimizer import trace as optimizer_trace

from ..runtime_test_support import public_test_runtime, require_adapter


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


@pytest.mark.cuda
@pytest.mark.fresh_process
def test_public_training_profiles_bounded_opaque_optimizer(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    require_adapter()

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
        profiling_options=CORRECTNESS_PROFILING,
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
