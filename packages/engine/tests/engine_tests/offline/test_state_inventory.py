"""Logical state keys preserve roles, geometry, ownership and physical aliases."""

import json
from dataclasses import replace

import pytest

from engine_tests.offline.test_shared_plans import registered
from shadowspill.artifacts.state import StateId, StateInventory
from shadowspill.artifacts.state_inventory import program_state
from shadowspill.ir import AliasGroupSpec, ObjectSpec
from shadowspill.parallelism import Parallelism
from shadowspill.parallelism.ownership import ParameterOwnership, StorageOwnership


def training(*, prefix=""):
    source = registered(prefix=prefix)
    aliases = [
        AliasGroupSpec(prefix + name, "rank0", 64)
        for name in ("master", "moment", "gradient")
    ]
    objects = [ObjectSpec(a.alias_group_id, a.alias_group_id, 0, 64) for a in aliases]
    layout = {"shape": [4, 4], "stride": [4, 1], "dtype": "torch.float32"}
    parameter = "linear.weight"
    bindings = [
        {"name": parameter, "role": "parameter", "object_id": prefix + "master"},
        {
            "name": "compute." + parameter,
            "role": "compute_copy",
            "object_id": prefix + parameter,
        },
        {
            "name": "optimizer." + parameter + ".exp_avg",
            "role": "state",
            "object_id": prefix + "moment",
        },
        {
            "name": "optimizer." + parameter + ".step",
            "role": "state",
            "object_id": None,
            "shape": [],
            "stride": [],
            "dtype": "torch.int64",
        },
        {
            "name": "optimizer_group.0.lr",
            "role": "hyperparameter",
            "object_id": None,
            "shape": [],
            "stride": [],
        },
    ]
    return replace(
        source,
        alias_groups=(*source.alias_groups, *aliases),
        objects=(*source.objects, *objects),
        bindings={
            **source.bindings,
            "optimizer_bindings": [
                {**layout, "parameter_name": parameter, **b} for b in bindings
            ],
            "persistent_gradients": [
                {
                    "parameter_name": parameter,
                    "gradient_object_id": prefix + "gradient",
                    "shape": [16],
                    "dtype": "torch.float32",
                }
            ],
        },
    )


def test_inventory_uses_state_roles_instead_of_program_local_ids():
    left, right = program_state(training()), program_state(training(prefix="next-"))
    assert {t.identity.role for t in left.tensors} == {
        "parameter",
        "master",
        "gradient",
        "optimizer",
    }
    assert len(left.tensors) == 6
    assert len([b for b in left.bindings if b.host_binding is not None]) == 1
    assert left.shared_aliases(right) == {
        "next-0": "0",
        "next-master": "master",
        "next-moment": "moment",
        "next-gradient": "gradient",
    }
    restored = StateInventory.from_dict(json.loads(json.dumps(left.to_dict())))
    assert restored == left
    assert all(t.checkpoint is (t.identity.role != "gradient") for t in left.tensors)


def test_state_keys_do_not_collide_by_role_components_or_dotted_names():
    keys = [
        StateId("parameter", "encoder.weight.scale"),
        StateId("parameter", "encoder.weight", ("scale",)),
        StateId("master", "encoder.weight", ("scale",)),
        StateId("parameter", "encoder", ("weight", "scale")),
    ]
    assert len({key.key for key in keys}) == 4
    assert all(StateId.from_dict(key.to_dict()) == key for key in keys)


@pytest.mark.parametrize("left_tied,right_tied", [(True, False), (False, True)])
def test_shared_state_preserves_tied_storage(left_tied, right_tied):
    left = program_state(registered(tied=left_tied))
    right = program_state(registered(prefix="eval-", tied=right_tied))
    with pytest.raises(ValueError, match="aliasing differs"):
        left.shared_aliases(right)


@pytest.mark.parametrize(
    "key,value",
    [
        ("dtype", "torch.float64"),
        ("shape", [2, 8]),
        ("stride", [1, 4]),
        ("storage_offset", 1),
    ],
)
def test_layout_changes_require_conversion_instead_of_direct_sharing(key, value):
    left = program_state(registered())
    changed = replace(
        left.tensors[0], geometry={**left.tensors[0].geometry, key: value}
    )
    right = replace(left, tensors=(changed, *left.tensors[1:]))
    with pytest.raises(ValueError, match="layout or ownership differs"):
        left.shared_aliases(right)


def test_physical_padding_and_host_storage_are_not_inferred_from_equal_names():
    left = program_state(training())
    binding = left.bindings[0]
    changed = replace(binding, storage={**binding.storage, "size_bytes": 128})
    with pytest.raises(ValueError, match="physical storage differs"):
        left.shared_aliases(replace(left, bindings=(changed, *left.bindings[1:])))
    with pytest.raises(ValueError, match="exactly one binding"):
        replace(left, bindings=left.bindings[:-1])
    with pytest.raises(ValueError, match="duplicate state"):
        replace(left, tensors=(*left.tensors, left.tensors[0]))


def test_existing_generic_ownership_is_preserved_and_replicas_never_alias():
    source = registered()
    storage = StorageOwnership(16, ((0, 1),))
    ownership = ParameterOwnership(
        "linear.weight",
        ("linear.weight", "head.weight"),
        (4, 4),
        (4, 4),
        True,
        ((0, 1),),
        storage,
        storage,
        storage,
    )
    source = replace(
        source,
        parallelism=Parallelism.dp(ranks_per_replica=2, min_shard_bytes=0),
        bindings={**source.bindings, "parameter_ownership": [ownership.to_dict()]},
    )
    first, second = program_state(source, 0), program_state(source, 1)
    assert first.tensors[0].ownership["optimizer"]["groups"] == ((0, 1),)
    with pytest.raises(TypeError):
        first.tensors[0].ownership["optimizer"]["elements"] = 1
    with pytest.raises(ValueError, match="different logical workers"):
        first.shared_aliases(second)
    with pytest.raises(ValueError, match="outside"):
        program_state(source, 2)


def test_unknown_inventory_versions_are_rejected():
    encoded = program_state(registered()).to_dict()
    with pytest.raises(ValueError, match="schema"):
        StateInventory.from_dict({**encoded, "version": 2})
    with pytest.raises(ValueError, match="schema"):
        StateInventory.from_dict({**encoded, "version": True})


def test_optimizer_family_is_part_of_state_compatibility():
    source = training()
    first = program_state(
        replace(source, bindings={**source.bindings, "optimizer_type": "external.Adam"})
    )
    second = program_state(
        replace(
            source,
            bindings={**source.bindings, "optimizer_type": "external.OtherOptimizer"},
        )
    )
    assert all(
        t.optimizer_type == "external.Adam"
        for t in first.tensors
        if t.identity.role == "optimizer"
    )
    with pytest.raises(ValueError, match="layout or ownership differs"):
        first.shared_aliases(second)


def test_unsharded_backward_uses_original_gradient_binding_as_persistent_state():
    source = training()
    bindings = dict(source.bindings)
    bindings["kind"] = "forward_backward"
    bindings["gradients"] = bindings.pop("persistent_gradients")
    bindings["gradient_geometry"] = {
        "gradient": {"dtype": "torch.float32", "shape": [16], "stride": [1]}
    }
    inventory = program_state(replace(source, bindings=bindings))
    assert any(t.identity.role == "gradient" for t in inventory.tensors)
    bindings["kind"] = "step"
    assert not any(
        t.identity.role == "gradient"
        for t in program_state(replace(source, bindings=bindings)).tensors
    )
