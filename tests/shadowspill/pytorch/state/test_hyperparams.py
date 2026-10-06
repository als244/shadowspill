"""Setting a value per step: what resolves, what is written, what is refused."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from shadowspill.pytorch.callables import PlannedForward, PlannedTrainStep
from shadowspill.pytorch.materialization import replacement as replacement_module
from shadowspill.pytorch.materialization.replacement import MaterializedState
from shadowspill.pytorch.state.optimizer import declare_varying_hyperparams


def _apply(
    model: nn.Module, groups: list[dict], values: dict
) -> dict[str, torch.Tensor]:
    """Drive the resolution with a stand-in for the planned step.

    The rule under test is which named value a step reaches and what happens
    when it cannot, which needs an optimizer, a model and a record of what
    would go into the pool, and nothing else. Returns that record.
    """

    step = object.__new__(PlannedTrainStep)
    step._model = model
    step._executor = SimpleNamespace(
        optimizer_state=SimpleNamespace(optimizer=SimpleNamespace(param_groups=groups))
    )
    written: dict[str, torch.Tensor] = {}
    step._state = SimpleNamespace(write_model_entries=written.update)
    step._closed = False
    PlannedTrainStep._apply_hyperparams(step, values)
    return written


def _apply_forward(model: nn.Module, values: dict) -> dict[str, torch.Tensor]:
    """The same, for a planned forward, which has no optimizer."""

    forward = object.__new__(PlannedForward)
    forward._model = model
    written: dict[str, torch.Tensor] = {}
    forward._state = SimpleNamespace(write_model_entries=written.update)
    forward._closed = False
    PlannedForward._apply_hyperparams(forward, values)
    return written


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


def test_a_model_buffer_is_written_into_the_pool_by_the_same_name() -> None:
    model = _Model({"temperature": torch.tensor(1.0)})
    written = _apply(model, [{"lr": torch.tensor(1.0)}], {"temperature": 0.7})
    assert list(written) == ["temperature"]
    assert written["temperature"].shape == ()
    assert written["temperature"].dtype is torch.float32
    assert written["temperature"].item() == pytest.approx(0.7)
    # The module's own tensor is the plan's handle, not where the value lives.
    assert model.temperature.item() == pytest.approx(1.0)


def test_a_buffer_of_any_shape_is_filled_with_the_one_number() -> None:
    model = _Model({"scale": torch.ones(3, dtype=torch.float64)})
    written = _apply(model, [], {"scale": 2.5})
    assert written["scale"].tolist() == [2.5, 2.5, 2.5]
    assert written["scale"].dtype is torch.float64
    with pytest.raises(TypeError, match="takes one number"):
        _apply(model, [], {"scale": (2.5, 3.5, 4.5)})


def test_a_forward_sets_a_model_buffer_and_nothing_else() -> None:
    model = _Model({"temperature": torch.tensor(1.0)})
    written = _apply_forward(model, {"temperature": 0.7})
    assert written["temperature"].item() == pytest.approx(0.7)
    assert _apply_forward(model, {}) == {}
    with pytest.raises(KeyError, match="no model buffer named 'lr'"):
        _apply_forward(model, {"lr": 3.0e-4})


# --- the write that puts a buffer's value where the state lives -------------


def _template(shape: tuple[int, ...], offset: int) -> torch.Tensor:
    """A tensor with the geometry of a registered entry: shape, contiguous
    strides and a storage offset, over a storage large enough to hold it."""

    return torch.empty(64).as_strided(shape, tuple(torch.empty(shape).stride()), offset)


class _PooledState(MaterializedState):
    """A materialized state over bytes standing in for the spill pool: two
    aliases, one holding a weight and one holding two buffers side by side."""

    def __init__(self) -> None:
        self.pool = {
            "a": torch.zeros(16, dtype=torch.uint8),
            "b": torch.zeros(8, dtype=torch.uint8),
        }
        self.reads: list[str] = []
        self.writes: list[str] = []
        self._state_names = ("weight", "temperature", "other")
        self.bridge = SimpleNamespace(
            wait_runtime_idle=lambda: None,
            objects=SimpleNamespace(alias_for_object=lambda object_id: object_id[0]),
        )
        self.object_store = {}
        self._entries = (
            ("weight", "a-weight", (4,), 0),
            ("temperature", "b-temperature", (), 0),
            ("other", "b-other", (), 1),
        )
        self.model = nn.Module()
        for name, _object_id, shape, offset in self._entries:
            self.model.register_buffer(name, _template(shape, offset))
        with torch.no_grad():
            self.view("a", 0, (4,)).copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
            self.view("b", 0, ()).fill_(1.0)
            self.view("b", 1, ()).fill_(5.0)

    def view(self, alias_id: str, offset: int, shape: tuple[int, ...]) -> torch.Tensor:
        return self._cpu_view(self.pool[alias_id], _template(shape, offset))

    def _empty_model_aliases(
        self, *, aliases: set[str] | None = None
    ) -> dict[str, torch.Tensor]:
        selected = set(self.pool) if aliases is None else aliases
        return {
            alias: torch.empty(self.pool[alias].numel(), dtype=torch.uint8)
            for alias in selected
        }

    def _registrations(self) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(
                binding=SimpleNamespace(name=name, object_id=object_id),
                tensor=getattr(self.model, name),
            )
            for name, object_id, shape, offset in self._entries
        ]


@pytest.fixture
def pooled(monkeypatch: pytest.MonkeyPatch) -> _PooledState:
    state = _PooledState()

    def read(_objects: object, alias_id: str, owner: torch.Tensor) -> None:
        state.reads.append(alias_id)
        owner.copy_(state.pool[alias_id])

    def write(_objects: object, alias_id: str, owner: torch.Tensor) -> None:
        state.writes.append(alias_id)
        state.pool[alias_id].copy_(owner)

    monkeypatch.setattr(replacement_module, "read_spill_tensor", read)
    monkeypatch.setattr(replacement_module, "write_spill_tensor", write)
    return state


def test_a_written_entry_reaches_the_pool_and_its_neighbours_stay(
    pooled: _PooledState,
) -> None:
    pooled.write_model_entries({"temperature": torch.tensor(0.7)})
    assert pooled.view("b", 0, ()).item() == pytest.approx(0.7)
    assert pooled.view("b", 1, ()).item() == pytest.approx(5.0)
    assert pooled.view("a", 0, (4,)).tolist() == [1.0, 2.0, 3.0, 4.0]
    # Only the alias holding the entry crossed the pool's edge.
    assert pooled.reads == ["b"]
    assert pooled.writes == ["b"]


def test_an_entry_of_another_geometry_is_refused_before_anything_is_written(
    pooled: _PooledState,
) -> None:
    with pytest.raises(RuntimeError, match="incompatible geometry"):
        pooled.write_model_entries({"temperature": torch.tensor([0.7, 0.8])})
    with pytest.raises(RuntimeError, match="incompatible geometry"):
        pooled.write_model_entries(
            {"temperature": torch.tensor(0.7, dtype=torch.float64)}
        )
    assert pooled.writes == []


def test_an_entry_that_is_not_state_is_refused(pooled: _PooledState) -> None:
    with pytest.raises(KeyError, match="persistent"):
        pooled.write_model_entries({"runtime_scale": torch.tensor(2.0)})
    assert pooled.reads == []


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
