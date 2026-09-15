"""Setting a value per step: what resolves, what is written, what is refused."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from shadowspill.pytorch.callables import PlannedTrainStep
from shadowspill.pytorch.state.optimizer import declare_varying_hyperparams


def _apply(model: nn.Module, groups: list[dict], values: dict) -> None:
    """Drive the resolution with a stand-in for the planned callable.

    The rule under test is which named value a step reaches and what happens
    when it cannot, which needs an optimizer and a model and nothing else.
    """

    step = object.__new__(PlannedTrainStep)
    step._model = model
    step._executor = SimpleNamespace(optimizer=SimpleNamespace(param_groups=groups))
    PlannedTrainStep._apply_hyperparams(step, values)


class _Model(nn.Module):
    def __init__(self, buffers: dict[str, torch.Tensor] | None = None) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(4))
        for name, value in (buffers or {}).items():
            self.register_buffer(name, value)


def test_a_value_held_in_a_tensor_is_written() -> None:
    rate = torch.tensor(1.0)
    _apply(_Model(), [{"lr": rate}], {"lr": 3.0e-4})
    assert rate.item() == pytest.approx(3.0e-4)


def test_every_group_carrying_the_name_is_written() -> None:
    first, second = torch.tensor(1.0), torch.tensor(2.0)
    _apply(_Model(), [{"lr": first}, {"lr": second}], {"lr": 5.0e-4})
    assert first.item() == pytest.approx(5.0e-4)
    assert second.item() == pytest.approx(5.0e-4)


def test_a_group_without_the_name_is_left_alone() -> None:
    held, other = torch.tensor(1.0), torch.tensor(9.0)
    _apply(_Model(), [{"lr": held}, {"weight_decay": other}], {"lr": 2.0e-4})
    assert held.item() == pytest.approx(2.0e-4)
    assert other.item() == pytest.approx(9.0)


def test_an_entry_holding_several_values_is_written_element_wise() -> None:
    betas = (torch.tensor(0.0), torch.tensor(0.0))
    _apply(_Model(), [{"betas": betas}], {"betas": (0.9, 0.95)})
    assert betas[0].item() == pytest.approx(0.9)
    assert betas[1].item() == pytest.approx(0.95)

    one = (torch.tensor(0.0), torch.tensor(0.0))
    _apply(_Model(), [{"betas": one}], {"betas": 0.8})
    assert [value.item() for value in one] == pytest.approx([0.8, 0.8])


def test_a_model_buffer_is_reached_by_the_same_name() -> None:
    model = _Model({"temperature": torch.tensor(1.0)})
    _apply(model, [{"lr": torch.tensor(1.0)}], {"temperature": 0.7})
    assert model.temperature.item() == pytest.approx(0.7)


def test_a_plain_number_is_refused_and_the_message_names_the_fix() -> None:
    with pytest.raises(TypeError, match="name it when the step"):
        _apply(_Model(), [{"lr": 1.0e-3}], {"lr": 3.0e-4})


def test_an_unknown_name_is_refused() -> None:
    with pytest.raises(KeyError, match="no optimizer value or model buffer"):
        _apply(_Model(), [{"lr": torch.tensor(1.0)}], {"momentum": 0.9})


def test_a_name_in_both_registries_is_refused_rather_than_guessed() -> None:
    model = _Model({"lr": torch.tensor(1.0)})
    with pytest.raises(KeyError, match="ambiguous"):
        _apply(model, [{"lr": torch.tensor(1.0)}], {"lr": 3.0e-4})


def test_a_sequence_of_the_wrong_length_is_refused() -> None:
    betas = (torch.tensor(0.0), torch.tensor(0.0))
    with pytest.raises(ValueError, match="holds 2 values"):
        _apply(_Model(), [{"betas": betas}], {"betas": (0.9, 0.95, 0.99)})


def test_nothing_asked_for_writes_nothing() -> None:
    rate = torch.tensor(1.0)
    _apply(_Model(), [{"lr": rate}], {})
    assert rate.item() == pytest.approx(1.0)


# --- the declaration that makes the above possible -------------------------


def _optimizer(**settings: object) -> torch.optim.Optimizer:
    parameter = torch.nn.Parameter(torch.zeros(4))
    return torch.optim.SGD([parameter], lr=0.1, **settings)


def test_a_named_value_is_held_in_a_float64_scalar_where_the_update_runs() -> None:
    model, optimizer = _Model(), _optimizer()
    assert declare_varying_hyperparams(model, optimizer, ("lr",)) == ("lr",)
    held = optimizer.param_groups[0]["lr"]
    assert isinstance(held, torch.Tensor)
    # a Python float is a float64, so the promotion loses nothing
    assert held.dtype is torch.float64
    assert held.item() == pytest.approx(0.1)
    assert held.device == optimizer.param_groups[0]["params"][0].device


def test_only_the_named_values_are_touched() -> None:
    model, optimizer = _Model(), _optimizer(momentum=0.9, weight_decay=0.01)
    declare_varying_hyperparams(model, optimizer, ("lr",))
    group = optimizer.param_groups[0]
    assert isinstance(group["lr"], torch.Tensor)
    assert group["momentum"] == 0.9
    assert group["weight_decay"] == 0.01


def test_an_entry_holding_several_values_has_each_of_them_held() -> None:
    parameter = torch.nn.Parameter(torch.zeros(4))
    optimizer = torch.optim.AdamW([parameter], betas=(0.9, 0.95))
    declare_varying_hyperparams(_Model(), optimizer, ("betas",))
    betas = optimizer.param_groups[0]["betas"]
    assert all(isinstance(value, torch.Tensor) for value in betas)
    assert [value.item() for value in betas] == pytest.approx([0.9, 0.95])


def test_declaring_the_same_name_twice_is_harmless() -> None:
    optimizer = _optimizer()
    assert declare_varying_hyperparams(_Model(), optimizer, ("lr", "lr")) == ("lr",)
    declare_varying_hyperparams(_Model(), optimizer, ("lr",))
    assert optimizer.param_groups[0]["lr"].item() == pytest.approx(0.1)


def test_declaring_a_model_buffer_leaves_the_tensor_it_already_is() -> None:
    model = _Model({"temperature": torch.tensor(0.7)})
    declare_varying_hyperparams(model, _optimizer(), ("temperature",))
    assert model.temperature.item() == pytest.approx(0.7)


def test_an_unknown_name_is_refused_at_planning_time() -> None:
    with pytest.raises(ValueError, match="nothing for a step to set"):
        declare_varying_hyperparams(_Model(), _optimizer(), ("nonesuch",))


def test_an_integer_value_is_held_without_losing_its_kind() -> None:
    parameter = torch.nn.Parameter(torch.zeros(4))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    optimizer.param_groups[0]["warmup"] = 100
    declare_varying_hyperparams(_Model(), optimizer, ("warmup",))
    held = optimizer.param_groups[0]["warmup"]
    assert isinstance(held, torch.Tensor)
    # a Python int is an int64, so holding it loses nothing either
    assert held.dtype is torch.int64
    assert held.item() == 100


def test_a_bool_is_refused_because_it_selects_what_the_update_does() -> None:
    with pytest.raises(TypeError, match="selects what the update does"):
        declare_varying_hyperparams(_Model(), _optimizer(), ("maximize",))


def test_a_value_that_is_not_a_number_is_refused() -> None:
    optimizer = _optimizer()
    optimizer.param_groups[0]["schedule"] = "cosine"
    with pytest.raises(TypeError, match="only a number"):
        declare_varying_hyperparams(_Model(), optimizer, ("schedule",))


def test_a_name_in_both_registries_is_refused() -> None:
    model = _Model({"lr": torch.tensor(1.0)})
    with pytest.raises(ValueError, match="ambiguous"):
        declare_varying_hyperparams(model, _optimizer(), ("lr",))
