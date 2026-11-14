"""Canonical state can be owned by any earlier operation, including host state."""

from dataclasses import replace

import pytest

from engine_tests.offline.test_shared_plans import registered
from engine_tests.offline.test_state_inventory import training
from shadowspill.planning.shared_state import shared_state


def test_state_first_introduced_by_backward_or_optimizer_has_one_owner():
    resolved = shared_state(
        {
            "": registered(),
            "update": training(prefix="update-"),
            "second": training(prefix="second-"),
            "eval": registered(prefix="eval-"),
        }
    )
    assert resolved.aliases[""] == {}
    assert resolved.aliases["update"] == {"update-0": ("", "0")}
    assert resolved.aliases["second"] == {
        "second-0": ("", "0"),
        "second-master": ("update", "update-master"),
        "second-moment": ("update", "update-moment"),
        "second-gradient": ("update", "update-gradient"),
    }
    assert resolved.aliases["eval"] == {"eval-0": ("", "0")}
    counter = "optimizer.linear.weight.step"
    assert resolved.host["second"] == {counter: ("update", counter)}
    assert resolved.host["update"] == {}


def test_borrowing_maps_are_immutable_and_resolve_directly_to_original_owner():
    resolved = shared_state(
        {
            "a": training(prefix="a-"),
            "b": training(prefix="b-"),
            "c": training(prefix="c-"),
        }
    )
    assert resolved.aliases["c"]["c-master"] == ("a", "a-master")
    with pytest.raises(TypeError):
        resolved.aliases["b"]["b-master"] = ("elsewhere", "other")


@pytest.mark.parametrize("tied", [True, False])
def test_changed_ties_are_rejected_before_worker_binding(tied):
    with pytest.raises(ValueError, match="aliasing differs"):
        shared_state(
            {"a": registered(tied=tied), "b": registered(prefix="b-", tied=not tied)}
        )


def test_changed_model_and_optimizer_are_not_silently_combined():
    source = registered()
    changed = replace(
        source,
        bindings={
            **source.bindings,
            "registrations": source.bindings["registrations"][:1],
        },
    )
    with pytest.raises(ValueError, match="same model inventory"):
        shared_state({"a": source, "b": changed})
    source = training()
    adam = replace(source, bindings={**source.bindings, "optimizer_type": "Adam"})
    sgd = replace(source, bindings={**source.bindings, "optimizer_type": "SGD"})
    with pytest.raises(ValueError, match="same optimizer family"):
        shared_state({"a": adam, "b": sgd})
