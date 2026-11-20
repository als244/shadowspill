"""The client reference's parallelism declaration builds and means what it says."""

import ast
import textwrap
from pathlib import Path
from typing import Any

import pytest
from torch import nn

from shadowspill import Parallelism
from shadowspill.build.parallelism import model_parallelism
from shadowspill.parallelism.reductions import gradient_reduction
from shadowspill_client.parallelism import data_parallel, expert_parallel

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "docs" / "reference" / "client" / "parallelism.rst"
MARKER = ":name: parallelism-declaration-example"


def _example() -> dict[str, Any]:
    """The dict of the page's example code block, ``parallelism = {...}``."""
    lines = PAGE.read_text(encoding="utf-8").splitlines()
    starts = [index for index, line in enumerate(lines) if line.strip() == MARKER]
    assert len(starts) == 1, f"{PAGE} needs one block named by {MARKER!r}"
    body = []
    for line in lines[starts[0] + 1 :]:
        if line and not line.startswith(" "):
            break
        body.append(line)
    (statement,) = ast.parse(textwrap.dedent("\n".join(body))).body
    assert isinstance(statement, ast.Assign)
    return ast.literal_eval(statement.value)


def test_the_example_is_a_declaration_in_to_dict_form():
    declared = _example()
    spec = Parallelism.from_dict(declared)
    assert (spec.ranks_per_replica, spec.num_replicas, spec.world_size) == (4, 2, 8)
    assert Parallelism.from_dict(spec.to_dict()) == spec
    # to_dict writes every storage key, the unset ones as None.
    written = spec.to_dict()
    storage = written["default_parameter_storage"]
    written["default_parameter_storage"] = {
        key: value for key, value in storage.items() if value is not None
    }
    assert written == declared
    # Every rank trains on its own part of each batch, and its loss counts.
    assert spec.inputs is not None and spec.inputs.partition_count == 8
    assert all(spec.inputs.contributes_loss(rank) for rank in range(8))


def test_the_example_shards_within_a_replica_and_copies_across_replicas():
    # A 4 MiB weight, at least the 2 MiB floor, and a 4 KiB bias under it.
    model = nn.Linear(1024, 1024, device="meta")
    _, (weight, bias) = model_parallelism(model, Parallelism.from_dict(_example()))
    # Every rank computes with the whole weight.
    assert weight.local_shape == weight.global_shape == (1024, 1024)
    assert weight.copies(0) == tuple(range(8))
    for component in (weight.weights, weight.gradients, weight.optimizer):
        assert component.groups == ((0, 1, 2, 3), (4, 5, 6, 7))
        assert component.capacity == 1024 * 1024 // 4
        for rank in range(4):
            assert component.matching_owners(rank) == (rank, rank + 4)
            assert component.region(rank) == component.region(rank + 4)
    phases = gradient_reduction(weight).phases
    assert [(phase.operation, phase.groups) for phase in phases] == [
        ("reduce_scatter", ((0, 1, 2, 3), (4, 5, 6, 7))),
        ("all_reduce", ((0, 4), (1, 5), (2, 6), (3, 7))),
    ]
    for component in (bias.weights, bias.gradients, bias.optimizer):
        assert component.groups == tuple((rank,) for rank in range(8))
        assert component.capacity == 1024


def test_the_helpers_return_declarations_of_the_same_form():
    for declared in (
        data_parallel(4),
        data_parallel(4, fields=[("inputs", "tokens")]),
        expert_parallel(4, experts=["layers.*.experts.*"]),
    ):
        spec = Parallelism.from_dict(declared)
        assert spec.world_size == 4
        assert Parallelism.from_dict(spec.to_dict()) == spec


def _four_ranks(**changes: Any) -> dict[str, Any]:
    """A one-replica declaration of four ranks, two collections, no floor."""
    declared = data_parallel(4, min_shard_bytes=0)
    declared["groups"]["pairs"] = [[0, 1], [2, 3]]
    declared["groups"]["local"] = [[0], [1], [2], [3]]
    declared.update(changes)
    return declared


def test_a_storage_records_group_covers_every_component_it_does_not_set():
    model = nn.Linear(8, 8, bias=False, device="meta")
    declared = _four_ranks(
        default_parameter_storage={
            "group": "all_ranks",
            "weight_group": "local",
            "weights": "sharded",
            "gradients": "sharded",
            "optimizer": "sharded",
        },
        parameter_storage={"weight": {"group": "pairs", "optimizer_group": "local"}},
    )
    _, (weight,) = model_parallelism(model, Parallelism.from_dict(declared))
    # The record's group outranks the default's weight_group; its own
    # optimizer_group outranks its group.
    assert weight.weights.groups == ((0, 1), (2, 3))
    assert weight.gradients.groups == ((0, 1), (2, 3))
    assert weight.optimizer.groups == ((0,), (1,), (2,), (3,))
    # Without a record, the default's weight_group outranks its group.
    del declared["parameter_storage"]["weight"]
    _, (weight,) = model_parallelism(model, Parallelism.from_dict(declared))
    assert weight.weights.groups == ((0,), (1,), (2,), (3,))
    assert weight.gradients.groups == ((0, 1, 2, 3),)


def test_only_a_sharded_component_needs_ranks_of_one_slice():
    model = nn.Linear(8, 8, bias=False, device="meta")
    sharded = {"kind": "Sharded", "group": "pairs", "dim": 0}
    # all_ranks holds ranks computing with both slices of the weight.
    declared = _four_ranks(
        parameter_placements={"weight": sharded},
        parameter_storage={
            "weight": {
                "weights": "replicated",
                "gradients": "replicated",
                "optimizer": "replicated",
            }
        },
    )
    _, (weight,) = model_parallelism(model, Parallelism.from_dict(declared))
    assert weight.weights.groups == ((0,), (1,), (2,), (3,))
    declared["parameter_storage"]["weight"]["optimizer"] = "sharded"
    with pytest.raises(ValueError, match="weight optimizer: storage group combines"):
        model_parallelism(model, Parallelism.from_dict(declared))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda d: d.pop("parameter_storage"), "missing key 'parameter_storage'"),
        (lambda d: d.update(replicas=2), "unknown key 'replicas'"),
        (
            lambda d: d.update(default_parameter_placement={}),
            "default_parameter_placement: missing key 'kind'",
        ),
        (
            lambda d: d.update(
                default_parameter_placement={"kind": "Replicated", "group": "local"}
            ),
            "default_parameter_placement: unknown key 'group'",
        ),
        (
            lambda d: d.update(
                parameter_placements={"weight": {"kind": "Sharded", "group": "local"}}
            ),
            "parameter_placements\\['weight'\\]: missing key 'dim'",
        ),
        (
            lambda d: d.update(parameter_storage={"weight": {"grup": "local"}}),
            "parameter_storage\\['weight'\\]: unknown key 'grup'",
        ),
        (
            lambda d: d.update(
                default_parameter_storage={
                    **d["default_parameter_storage"],
                    "weight_groups": "local",
                }
            ),
            "default_parameter_storage: unknown key 'weight_groups'",
        ),
    ],
)
def test_a_missing_or_unknown_key_is_refused_by_name(change, message):
    declared = _four_ranks()
    change(declared)
    with pytest.raises(ValueError, match=message):
        Parallelism.from_dict(declared)


def test_conflicting_selectors_name_the_parameter_they_disagree_on():
    model = nn.Linear(8, 8, bias=False, device="meta")
    declared = _four_ranks(
        parameter_storage={"w*": {"weights": "replicated"}, "*t": {}}
    )
    with pytest.raises(ValueError) as refused:
        model_parallelism(model, Parallelism.from_dict(declared))
    assert str(refused.value) == "conflicting declarations for 'weight'"
