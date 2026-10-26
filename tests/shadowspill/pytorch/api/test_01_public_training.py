"""Legacy behaviors still awaiting saved-Plan equivalents.

Ordinary training, initialization, precision, checkpoints and lifecycle coverage
live under tests/shadowspill/{execution,training}. These two cases remain until
Python optimizer fallbacks and per-call model-buffer controls migrate.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

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


@pytest.mark.cuda
@pytest.mark.fresh_process
def test_public_training_sets_a_model_buffer_each_step(tmp_path: Path) -> None:
    """A buffer named at planning and set per step is read by the step.

    The objective divides by the buffer, so the value the step read shows in
    every objective it returns and in the weights an eager run of the same
    totals reaches. A value folded in at capture, or written where the step
    does not look, is off by the total. The module's handle on the buffer is
    the plan's, so the write has to reach the pool the state lives in.
    """

    require_adapter()
    totals = (4.0, 16.0, 8.0, 2.0)
    rate = 1e-3

    class Network(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            torch.manual_seed(91)
            self.body = nn.Sequential(
                nn.Linear(8, 12, bias=False), nn.ReLU(), nn.Linear(12, 4, bias=False)
            )
            self.register_buffer("trained_total", torch.ones(()))

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return self.body(value)

    def objective(model: nn.Module, value: torch.Tensor, target: torch.Tensor):
        summed = torch.nn.functional.mse_loss(model(value), target, reduction="sum")
        return summed / model.trained_total

    def batches(seed: int) -> list[list[torch.Tensor]]:
        torch.manual_seed(seed)
        return [[torch.randn(2, 8), torch.randn(2, 4)] for _ in range(2)]

    build = partial(torch.optim.AdamW, foreach=False)
    runtime = public_test_runtime()
    model = import_model_state(
        Network(), runtime=runtime, pool="spill", release_source=True
    )
    training = plan_step(
        model,
        profiling_options=CORRECTNESS_PROFILING,
        objective=objective,
        optimizer=build,
        hyperparams=("lr", "trained_total"),
        artifact_store=tmp_path,
        example_inputs=batches(92),
        runtime=runtime,
        execution="execution",
        spill="spill",
    )
    objectives = []
    for index, total in enumerate(totals):
        result = training(
            batches(100 + index), hyperparams={"lr": rate, "trained_total": total}
        )
        objectives.append([value.cpu() for value in result.objectives])
    planned = {
        name: value.detach().float().cpu()
        for name, value in training.state_dict()["model"].items()
    }
    training.close()
    export_model_state(model, runtime=runtime, release_runtime=True)
    assert planned["trained_total"].item() == totals[-1]
    assert model.trained_total.item() == totals[-1]

    reference = Network().cuda()
    optimizer = build(reference.parameters(), lr=rate)
    for index, total in enumerate(totals):
        with torch.no_grad():
            reference.trained_total.fill_(total)
        optimizer.zero_grad()
        for (value, target), actual in zip(
            batches(100 + index), objectives[index], strict=True
        ):
            loss = objective(reference, value.cuda(), target.cuda())
            loss.backward()
            torch.testing.assert_close(
                actual, loss.detach().cpu(), rtol=2e-5, atol=2e-6
            )
        optimizer.step()
    for name, value in reference.named_parameters():
        torch.testing.assert_close(
            planned[name], value.detach().float().cpu(), rtol=1e-4, atol=1e-6
        )
