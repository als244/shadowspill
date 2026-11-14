"""What each optimizer-state entry starts at, read from how the optimizer makes it."""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn

from shadowspill.pytorch.optimizer.capture import (
    DeclaredStateEntry,
    declare_optimizer_state,
)
from shadowspill.pytorch.optimizer.starts import (
    ConstantStart,
    HeldStart,
    NoStart,
    ParameterStart,
    ValueStart,
)


class _Keeper(torch.optim.Optimizer):
    """Makes one entry of each kind a step can start from, on its first step."""

    def __init__(self, parameters: Iterable[nn.Parameter]) -> None:
        super().__init__(parameters, {"lr": 0.1})

    @torch.no_grad()
    def step(self, closure: object = None) -> None:
        del closure
        for group in self.param_groups:
            for parameter in group["params"]:
                state = self.state[parameter]
                if not state:
                    state["quarter"] = torch.full_like(parameter, 0.25)
                    state["master"] = parameter.detach().to(torch.float64).clone()
                    state["written"] = torch.empty_like(parameter)
                    state["written"].copy_(parameter)
                    state["counter"] = torch.zeros((), dtype=torch.int64)
                state["quarter"].add_(parameter.grad)
                state["master"].sub_(state["quarter"])
                state["counter"].add_(1)
                parameter.copy_(state["master"])


def _declared(make: type | object, model: nn.Module) -> dict[str, DeclaredStateEntry]:
    optimizer = make(model.parameters())  # type: ignore[operator]
    entries = declare_optimizer_state(dict(model.named_parameters()), optimizer)
    return {f"{item.parameter_name}.{item.entry_name}": item for item in entries}


def test_adamw_starts_every_entry_at_zero() -> None:
    declared = _declared(
        lambda parameters: torch.optim.AdamW(parameters, lr=1e-3), nn.Linear(3, 2)
    )

    assert declared["weight.exp_avg"].start == ConstantStart(0)
    assert declared["weight.exp_avg_sq"].start == ConstantStart(0)
    # A counter built from a number keeps that number exactly.
    step = declared["weight.step"].start
    assert isinstance(step, ValueStart) and step.value.item() == 0


def test_each_way_of_making_an_entry_says_where_it_starts() -> None:
    declared = _declared(_Keeper, nn.Linear(3, 2, bias=False))

    assert declared["weight.quarter"].start == ConstantStart(0.25)
    assert declared["weight.master"].start == ParameterStart()
    assert declared["weight.master"].dtype == torch.float64
    assert declared["weight.written"].start == ParameterStart()
    assert declared["weight.counter"].start == ConstantStart(0)


def test_momentum_made_from_the_gradient_has_no_start() -> None:
    declared = _declared(
        lambda parameters: torch.optim.SGD(parameters, lr=0.1, momentum=0.9),
        nn.Linear(3, 2),
    )

    start = declared["weight.momentum_buffer"].start
    assert isinstance(start, NoStart) and "gradient" in start.reason


def test_state_the_optimizer_holds_starts_at_what_it_holds() -> None:
    model = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model(torch.randn(4, 3)).sum().backward()
    optimizer.step()
    entries = declare_optimizer_state(dict(model.named_parameters()), optimizer)

    assert entries and all(item.start == HeldStart() for item in entries)
