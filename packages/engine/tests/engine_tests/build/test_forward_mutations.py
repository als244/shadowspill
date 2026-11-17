"""Distributed forward admits only worker-owned buffer mutations."""

from types import SimpleNamespace

import pytest

from shadowspill.build.forward import _validate_distributed_mutations
from shadowspill.training.preparation.capture_checks import _validate_mutations


def program(target="buffer_view"):
    return SimpleNamespace(
        objects=[
            SimpleNamespace(object_id=name, alias_group_id=alias)
            for name, alias in (
                ("parameter", "p"),
                ("buffer", "b"),
                ("buffer_view", "b"),
                ("input", "x"),
            )
        ],
        tasks=[
            SimpleNamespace(
                phase="forward", mutations=[SimpleNamespace(object_id=target)]
            )
        ],
        bindings={
            "registrations": [
                {"name": "weight", "object_id": "parameter", "parameter": True},
                {"name": "calls", "object_id": "buffer", "parameter": False},
            ]
        },
    )


def test_rank_local_buffer_views_keep_their_mutation_contract():
    value = program()
    _validate_distributed_mutations(value)
    binding = SimpleNamespace(
        model=SimpleNamespace(definition=SimpleNamespace(mutable_buffers=("calls",))),
        contract=SimpleNamespace(key=SimpleNamespace(mode="training")),
    )
    _validate_mutations(value, binding, {"x"})
    binding.contract.key.mode = "evaluation"
    with pytest.raises(ValueError, match="must preserve"):
        _validate_mutations(value, binding, {"x"})


@pytest.mark.parametrize("target", ["parameter", "input"])
def test_distributed_forward_still_rejects_unowned_or_parameter_writes(target):
    with pytest.raises(NotImplementedError, match="rank-local model buffers"):
        _validate_distributed_mutations(program(target))


def test_a_registered_buffer_cannot_make_parameter_storage_mutable():
    value = program("buffer_view")
    value.bindings["registrations"].append(
        {"name": "tied_weight", "object_id": "buffer", "parameter": True}
    )
    with pytest.raises(NotImplementedError, match="rank-local model buffers"):
        _validate_distributed_mutations(value)
