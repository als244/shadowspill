"""CPU checks for shared counters, independent controls and canonical exports."""

from types import SimpleNamespace

import pytest
import torch

from engine_tests.execution.test_initialization import storage
from shadowspill.execution.controls import Controls
from shadowspill.execution.resources import plan_inventory
from shadowspill.execution.tensors import tensor_descriptors, visit_tensors
from shadowspill.execution.views import StorageViews


def control_storage():
    bindings = [
        {
            "name": name,
            "role": role,
            "object_id": None,
            "device": "cpu",
            "shape": (),
            "stride": (),
            "dtype": "torch.float32",
            "initialization": {"kind": "constant", "value": value},
        }
        for name, role, value in (
            ("optimizer.weight.step", "state", 0),
            ("optimizer_group.0.lr", "hyperparameter", 0.01),
        )
    ]
    return SimpleNamespace(
        plan=SimpleNamespace(
            program=SimpleNamespace(bindings={"optimizer_bindings": bindings})
        )
    )


def test_counters_are_shared_without_reinitializing_or_exporting_twice():
    first = Controls(control_storage())
    counter, lr = "optimizer.weight.step", "optimizer_group.0.lr"
    first.initialize(lambda *_: pytest.fail("unexpected loader"))
    first.host[counter].add_(3)
    second = Controls(control_storage(), shared_host={counter: first.host[counter]})
    second.initialize(lambda *_: pytest.fail("unexpected loader"), skip_shared=True)
    assert second.host[counter] is first.host[counter]
    assert first.host[counter].item() == 3
    second.apply({lr: 0.25})
    assert first.host[lr].item() == pytest.approx(0.01)
    second.apply({})
    assert second.host[lr].item() == pytest.approx(0.01)
    second.host[counter].add_(1)
    assert first.host[counter].item() == 4
    exported = {}
    second.export(
        lambda request, value: exported.update({request.name: value.clone()}),
        skip_shared=True,
        prefix="plans.second.",
    )
    assert set(exported) == {"plans.second." + lr}
    assert {d.name for d in first.descriptors(prefix="plans.first.")} == {
        counter,
        "plans.first." + lr,
    }
    restored = []
    second.initialize(
        lambda request, value: restored.append(request.name),
        restore=True,
        skip_shared=True,
        prefix="plans.second.",
    )
    assert restored == ["plans.second." + lr]


@pytest.mark.parametrize(
    "values",
    [
        {"missing": torch.tensor(0.0)},
        {"optimizer_group.0.lr": torch.tensor(0.0)},
        {"optimizer.weight.step": torch.tensor(0, dtype=torch.int64)},
        {"optimizer.weight.step": torch.zeros(2)},
    ],
)
def test_shared_host_state_rejects_controls_unknown_names_and_layout_changes(values):
    with pytest.raises(ValueError, match="shared host state"):
        Controls(control_storage(), shared_host=values)


def test_views_route_to_multiple_original_owners_and_clear_only_local_values():
    model, optimizer = StorageViews(), StorageViews()
    model["weight"] = torch.ones(2)
    borrowed = StorageViews({"w": (model, "weight"), "m": (optimizer, "moment")})
    assert set(borrowed) == {"w"}
    borrowed["m"] = torch.zeros(2)
    borrowed["input"] = torch.full((2,), 3)
    assert borrowed["w"] is model["weight"]
    assert borrowed["m"] is optimizer["moment"]
    assert len(borrowed) == 3
    borrowed.local.clear()
    assert set(borrowed) == {"w", "m"}
    del borrowed["m"]
    assert "moment" not in optimizer


def test_later_owned_state_keeps_canonical_names_while_constants_remain_private():
    values = {"w": torch.arange(4.0), "m": torch.zeros(4), "c": torch.ones(1)}
    state = storage(
        {
            "kind": "step",
            "registrations": [{"name": "weight", "object_id": "w"}],
            "optimizer_bindings": [
                {"name": "optimizer.weight.moment", "role": "state", "object_id": "m"}
            ],
            "root_inputs": [[{"leaf_index": 0, "object_id": "c"}]],
            "root_sources": [{"kind": "CONSTANT_TENSOR", "name": "constant"}],
        },
        {
            name: {
                "shape": tuple(value.shape),
                "stride": (1,),
                "dtype": str(value.dtype),
            }
            for name, value in values.items()
        },
        values,
    )
    state.shared_aliases = {"w"}
    exported = []
    visit_tensors(
        state,
        lambda request, value: exported.append(request.name),
        16,
        initialize=False,
        skip_shared=True,
        prefix="plans.train.",
    )
    assert exported == [
        "optimizer.weight.moment",
        "plans.train.constant.example_0.constant",
    ]
    assert [
        d.name
        for d in tensor_descriptors(state, skip_shared=True, prefix="plans.train.")
    ] == exported


def test_resource_inventory_includes_all_named_accumulation_paths():
    accumulated = SimpleNamespace(_accumulation_plan=None)
    primary = SimpleNamespace(_accumulation_plan=None)
    second = SimpleNamespace(_accumulation_plan=accumulated)
    assert plan_inventory(primary, {"backward": second}) == {
        "default": primary,
        "named.backward": second,
        "named.backward.accumulate": accumulated,
    }
