"""Fresh-process accumulated-training parity and lifecycle canary."""

from __future__ import annotations

import ctypes
import os
import sys
import tempfile
from collections.abc import Iterable
from pathlib import Path

import torch
import torch.nn as nn

from shadowspill.pytorch import ObjectiveResult, plan
from shadowspill.pytorch._abi import AdapterStatistics
from shadowspill.pytorch._allocator import installed_allocator


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(1024, 1024, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.projection(value)


def _objective(
    model: nn.Module, value: torch.Tensor, target: torch.Tensor, label: str
) -> ObjectiveResult:
    error = model(value) - target
    loss = error.square().mean()
    return ObjectiveResult(loss, {"mean": error.detach().mean(), "label": label})


def _clone_model_state(state: object) -> dict[str, torch.Tensor]:
    if not isinstance(state, dict) or not isinstance(state.get("model"), dict):
        raise AssertionError("training checkpoint has an invalid model payload")
    return {
        name: value.clone()
        for name, value in state["model"].items()
        if isinstance(name, str) and isinstance(value, torch.Tensor)
    }


def _assert_bitwise(
    left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]
) -> None:
    if set(left) != set(right):
        raise AssertionError("checkpoint replay changed model keys")
    for name in left:
        if not torch.equal(left[name], right[name]):
            raise AssertionError(f"checkpoint replay changed {name!r}")


def _statistics() -> AdapterStatistics:
    installed = installed_allocator()
    if installed is None:
        raise AssertionError("training did not install the allocator")
    result = AdapterStatistics()
    status = int(
        installed.library.shadowspill_pytorch_allocator_statistics(ctypes.byref(result))
    )
    if status != 0:
        raise AssertionError(f"statistics failed with status {status}")
    return result


def main(arguments: Iterable[str] | None = None) -> int:
    values = tuple(sys.argv[1:] if arguments is None else arguments)
    adapter = Path(values[0]).resolve()
    os.environ["SHADOWSPILL_PYTORCH_LIBRARY"] = str(adapter)
    with tempfile.TemporaryDirectory() as cache:
        os.environ["SHADOWSPILL_PROFILE_CACHE"] = cache
        torch.manual_seed(127)
        model = _Model()
        reference = _Model()
        reference.load_state_dict(model.state_dict())
        parameter_ids = tuple(id(parameter) for parameter in model.parameters())
        example_inputs = [
            [torch.randn(3, 1024), torch.randn(3, 1024), "short"],
            [torch.randn(5, 1024), torch.randn(5, 1024), "long"],
        ]
        steps: list[list[list[object]]] = []
        for step in range(5):
            torch.manual_seed(1000 + step)
            steps.append(
                [
                    [torch.randn(3, 1024), torch.randn(3, 1024), "short"],
                    [torch.randn(5, 1024), torch.randn(5, 1024), "long"],
                ]
            )

        reference_optimizer = torch.optim.AdamW(
            reference.parameters(), lr=0.003, foreach=False
        )

        constructed: list[torch.optim.AdamW] = []
        optimizer_calls: list[int] = []

        def optimizer_factory(
            parameters: Iterable[torch.nn.Parameter],
        ) -> torch.optim.AdamW:
            optimizer = torch.optim.AdamW(parameters, lr=0.003, foreach=False)
            constructed.append(optimizer)

            def count_actual_step(
                stepped: torch.optim.Optimizer,
                _args: tuple[object, ...],
                _kwargs: dict[str, object],
            ) -> None:
                if stepped is optimizer:
                    optimizer_calls.append(1)

            optimizer.register_step_post_hook(count_actual_step)
            return optimizer

        planned = plan(
            model,
            objective=_objective,
            opt=optimizer_factory,
            example_inputs=example_inputs,
            device_budget=2 << 30,
            host_budget=1 << 30,
        )
        if len(constructed) != 1:
            raise AssertionError("optimizer factory was not invoked exactly once")
        if planned.plan_report.initial_execution_plan is None:
            raise AssertionError("lazy AdamW state has no initial execution plan")
        active = planned.plan_report.execution_plan.program.selected_tasks(
            planned.plan_report.execution_plan.selections
        )
        if tuple(task.phase for task in active) != (
            "forward",
            "backward",
            "forward",
            "backward",
            "optimizer",
        ):
            raise AssertionError("training plan has the wrong accumulated task order")

        checkpoint: dict[str, object] | None = None
        for step, microbatches in enumerate(steps):
            reference_optimizer.zero_grad(set_to_none=True)
            reference_losses: list[torch.Tensor] = []
            for value, target, label in microbatches:
                result = _objective(reference, value, target, label)
                result.loss.backward()
                reference_losses.append(result.loss.detach())
            reference_optimizer.step()
            actual = planned(microbatches)
            if actual.step_number != step + 1 or len(actual.objectives) != 2:
                raise AssertionError("StepResult has the wrong logical step")
            for loss, expected in zip(actual.objectives, reference_losses, strict=True):
                torch.testing.assert_close(loss.cpu(), expected, rtol=2e-5, atol=2e-6)
            if tuple(metric["label"] for metric in actual.metrics) != (
                "short",
                "long",
            ):
                raise AssertionError("static objective metrics changed")
            if step == 2:
                checkpoint = planned.state_dict()
        if len(optimizer_calls) != 5 or checkpoint is None:
            raise AssertionError("optimizer mutation count differs from step count")
        uninterrupted = _clone_model_state(planned.state_dict())
        planned.load_state_dict(checkpoint)
        for microbatches in steps[3:]:
            planned(microbatches)
        replayed = _clone_model_state(planned.state_dict())
        _assert_bitwise(uninterrupted, replayed)
        if len(optimizer_calls) != 7:
            raise AssertionError("checkpoint replay did not run one update per call")
        optimizer_state = planned.state_dict()["optimizer"]
        if not isinstance(optimizer_state, dict):
            raise AssertionError("optimizer checkpoint is not a mapping")
        for parameter_state in optimizer_state["state"].values():
            for value in parameter_state.values():
                if isinstance(value, torch.Tensor) and value.device.type != "cpu":
                    raise AssertionError("optimizer checkpoint retained CUDA storage")

        planned.close()
        planned.close()
        if tuple(id(parameter) for parameter in model.parameters()) != parameter_ids:
            raise AssertionError("training replaced a Parameter object")
        if any(parameter.device.type != "cpu" for parameter in model.parameters()):
            raise AssertionError("training close did not restore CPU state")
        closed_optimizer = planned.state_dict()["optimizer"]
        for parameter_state in closed_optimizer["state"].values():
            for value in parameter_state.values():
                if isinstance(value, torch.Tensor) and value.device.type != "cpu":
                    raise AssertionError("close did not restore optimizer state to CPU")
        for actual, expected in zip(
            model.parameters(), reference.parameters(), strict=True
        ):
            torch.testing.assert_close(actual, expected, rtol=1e-3, atol=5e-5)
        statistics = _statistics()
        if statistics.callback_failures != 0 or statistics.pointer_lookup_failures != 0:
            raise AssertionError("training produced allocator callback failures")
        if statistics.cuda.device_allocations != 1:
            raise AssertionError("training grew the CUDA slab")
        if statistics.cuda.pinned_host_allocations != 2:
            raise AssertionError("training did not reconcile pinned host admission")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
