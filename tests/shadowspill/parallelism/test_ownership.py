"""Parameter identity, not parallelism labels, determines reductions and storage."""

import json
from dataclasses import replace

import pytest

from shadowspill import Parallelism, ParameterStorage, Replicated, Sharded
from shadowspill.parallelism.resolve import ParameterGeometry, resolve_parameters


def resolve(spec, names=("weight",), shape=(11,), trainable=True):
    return resolve_parameters(spec, (ParameterGeometry(names, shape, trainable),))[0]


def expert_spec(num_replicas=1):
    return Parallelism(
        ranks_per_replica=16,
        num_replicas=num_replicas,
        groups={
            "all_ranks": [tuple(range(16))],
            "exchange": [tuple(range(8)), tuple(range(8, 16))],
            "same_experts": [(index, index + 8) for index in range(8)],
        },
        parameter_placements={"layers.*.experts.*": Sharded("exchange", dim=0)},
        parameter_storage={
            "layers.*.experts.*": ParameterStorage(group="same_experts")
        },
    )


def test_dp_replicas_repeat_storage_but_reduce_all_copies():
    spec = Parallelism.dp(ranks_per_replica=4, num_replicas=2)
    parameter = resolve(spec)
    assert spec.world_size == 8
    assert spec.group("all_ranks").groups == ((0, 1, 2, 3), (4, 5, 6, 7))
    assert spec.group("all_ranks").size == 4
    assert parameter.copies(0) == tuple(range(8))
    for component in ("weights", "gradients", "optimizer"):
        owned = parameter.storage(component, 0)
        assert owned.capacity == 3
        assert owned.region(0) == owned.region(4) == (0, 3)
        assert owned.region(3) == owned.region(7) == (9, 2)
        assert owned.matching_owners(2) == (2, 6)
    assert sum(parameter.weights.region(rank)[1] for rank in range(4)) == 11


def test_sixteen_rank_experts_and_attention_have_different_ownership():
    spec = expert_spec(num_replicas=2)
    attention, expert = resolve_parameters(
        spec,
        (
            ParameterGeometry(("layers.0.attention.weight",), (4096, 4096), True),
            ParameterGeometry(("layers.0.experts.w13",), (32, 4096, 1024), True),
        ),
    )
    assert attention.global_shape == attention.local_shape
    assert attention.copies(9) == tuple(range(32))
    assert attention.weights.capacity * 16 == 4096 * 4096
    assert attention.storage("optimizer", 9).matching_owners(9) == (9, 25)
    assert expert.global_shape == (256, 4096, 1024)
    assert expert.copies(1) == (1, 9, 17, 25)
    assert expert.storage("weights", 1).groups == ((1, 9), (17, 25))
    assert expert.weights.capacity * 2 == 32 * 4096 * 1024
    assert expert.storage("gradients", 1).matching_owners(1) == (1, 17)
    assert expert.storage("optimizer", 9).matching_owners(9) == (9, 25)
    with pytest.raises(ValueError, match="different logical parameter shards"):
        resolve_parameters(
            replace(spec, parameter_storage={}),
            (ParameterGeometry(("layers.0.experts.w13",), (32, 4096, 1024), True),),
        )


def test_group_order_determines_slices_not_numeric_rank_order():
    spec = Parallelism(
        ranks_per_replica=4,
        groups={"reverse": [(3, 2, 1, 0)]},
        default_parameter_placement=Replicated("reverse"),
        default_parameter_storage=ParameterStorage(
            group="reverse", weights="sharded", gradients="sharded", optimizer="sharded"
        ),
    )
    parameter = resolve(spec, shape=(16,))
    assert parameter.weights.region(3) == (0, 4)
    assert parameter.weights.region(0) == (12, 4)


def test_storage_components_can_use_different_groups_and_modes():
    spec = Parallelism(
        ranks_per_replica=4,
        groups={"all_ranks": [(0, 1, 2, 3)], "pairs": [(0, 1), (2, 3)]},
        parameter_storage={
            "weight": ParameterStorage(
                weights="replicated",
                gradient_group="pairs",
            )
        },
    )
    parameter = resolve(spec, shape=(8,))
    assert parameter.weights.capacity == 8
    assert parameter.gradients.capacity == 4
    assert parameter.optimizer.capacity == 2
    assert parameter.storage("gradients", 0).matching_owners(1) == (1, 3)


def test_tied_names_exact_priority_and_conflicting_wildcards():
    names = ("embedding.weight", "lm_head.weight")
    spec = Parallelism.dp(ranks_per_replica=4)
    spec = replace(
        spec,
        parameter_storage={
            "*.weight": ParameterStorage(weights="replicated"),
            "embedding.*": ParameterStorage(optimizer="replicated"),
        },
    )
    with pytest.raises(ValueError, match="conflicting declarations"):
        resolve(spec, names)
    # The exact entry inherits only the default, not either wildcard.
    spec = replace(
        spec,
        parameter_storage={
            **spec.parameter_storage,
            "lm_head.weight": ParameterStorage(),
        },
    )
    parameter = resolve(spec, names)
    assert parameter.names == names
    assert parameter.weights.capacity == parameter.optimizer.capacity == 3
    spec = replace(
        spec,
        parameter_storage={
            **spec.parameter_storage,
            "embedding.weight": ParameterStorage(weights="replicated"),
        },
    )
    with pytest.raises(ValueError, match="conflicting declarations"):
        resolve(spec, names)


def test_agreeing_patterns_compare_effective_values_and_typo_is_reported():
    spec = replace(
        Parallelism.dp(ranks_per_replica=4),
        parameter_storage={
            "*": ParameterStorage(),
            "weight*": ParameterStorage(weights="sharded"),
        },
    )
    assert resolve(spec).weights.capacity == 3
    with pytest.raises(ValueError, match="matches no registered tensor"):
        resolve(replace(spec, parameter_storage={"weigth": ParameterStorage()}))


def test_serializable_immutable_declarations_and_ordered_group_refs():
    spec = expert_spec()
    saved = json.loads(json.dumps(spec.to_dict()))
    loaded = Parallelism.from_dict(saved)
    assert loaded.to_dict() == spec.to_dict()
    assert loaded.group("exchange").members(9) == tuple(range(8, 16))
    with pytest.raises(TypeError):
        loaded.parameter_storage["new"] = ParameterStorage()
    saved["groups"]["exchange"][0][0] = 99
    assert loaded.group("exchange").members(0) == tuple(range(8))
    with pytest.raises(ValueError, match="unknown logical group"):
        loaded.group("missing")


@pytest.mark.parametrize(
    "groups, message",
    [
        ({"all_ranks": [(0, 1), (1, 2)]}, "more than one group"),
        ({"all_ranks": [(0, 1), (2,)]}, "equal size"),
        ({"all_ranks": [(0, 1)]}, "exactly one replica"),
        ({"all_ranks": [(0, 1, 2, True)]}, "nonnegative integers"),
    ],
)
def test_invalid_groups_are_rejected(groups, message):
    with pytest.raises(ValueError, match=message):
        Parallelism(ranks_per_replica=4, groups=groups)
