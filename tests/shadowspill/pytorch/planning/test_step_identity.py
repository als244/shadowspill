"""The step key answers for what a capture depends on, before it runs."""

from __future__ import annotations

import torch
import torch.nn as nn

from shadowspill.pytorch.planning.identity import step_identity, step_key
from shadowspill.step import StepDataOrdering

MACHINE = {"execution_budget_bytes": 1 << 30, "spill_budget_bytes": 4 << 30}
ENVIRONMENT = {"torch_version": torch.__version__, "device_name": "test"}


def _objective(model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    return model(inputs).sum()


def _other_objective(model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    return model(inputs).mean()


def _identity(model: nn.Module, **overrides: object) -> dict[str, object]:
    named: dict[str, object] = dict(
        objective=_objective,
        build_optimizer=lambda parameters: torch.optim.SGD(parameters, lr=0.1),
        hyperparams=("lr",),
        example_inputs=[[torch.zeros(2, 4)], [torch.zeros(2, 4)]],
        partition="auto",
        profiling_metadata=None,
        optimizer_ordering="stage_interleaved",
        allocation_probe_seeds=1,
        allocation_probe_repetitions=2,
        export_bypass_key="rev-1",
        machine=MACHINE,
        environment=ENVIRONMENT,
    )
    named.update(overrides)
    return step_identity(model, **named)  # type: ignore[arg-type]


def test_the_same_request_has_the_same_key() -> None:
    torch.manual_seed(0)
    first, second = nn.Linear(4, 3), nn.Linear(4, 3)
    ordering = StepDataOrdering(2, 1)
    assert step_key(_identity(first), ordering) == step_key(_identity(second), ordering)


def test_every_fact_the_capture_depends_on_moves_the_key() -> None:
    model = nn.Linear(4, 3)
    ordering = StepDataOrdering(2, 1)
    base = step_key(_identity(model), ordering)
    assert step_key(_identity(model), StepDataOrdering(1, 2)) != base
    assert step_key(_identity(model, export_bypass_key="rev-2"), ordering) != base
    assert step_key(_identity(model, objective=_other_objective), ordering) != base
    assert step_key(_identity(nn.Linear(4, 5)), ordering) != base
    wider = [[torch.zeros(3, 4)], [torch.zeros(3, 4)]]
    assert step_key(_identity(model, example_inputs=wider), ordering) != base
    heavier = _identity(
        model, build_optimizer=lambda parameters: torch.optim.SGD(parameters, lr=0.2)
    )
    assert step_key(heavier, ordering) != base
    assert (
        step_key(
            _identity(model, machine={**MACHINE, "spill_budget_bytes": 1}), ordering
        )
        != base
    )
    assert (
        step_key(_identity(model, environment={"device_name": "other"}), ordering)
        != base
    )
    assert step_key(_identity(model, master_dtype=torch.float32), ordering) != base
    assert step_key(_identity(model, grad_dtype=torch.float32), ordering) != base
    assert step_key(_identity(model, round_accumulation_once=True), ordering) != base


def test_the_identity_is_readable() -> None:
    identity = _identity(nn.Linear(4, 3))
    assert identity["export_bypass_key"] == "rev-1"
    assert identity["optimizer"]["type"].endswith("SGD")  # type: ignore[index]
    assert len(identity["inputs"]) == 2  # type: ignore[arg-type]
    assert identity["model"]["parameters"][0][0] == "weight"  # type: ignore[index]
