"""Saved logical identities address canonical exported state across layouts."""

from dataclasses import replace
from math import prod

import numpy as np
import pytest
import torch

from engine_tests.checkpoint.test_state import plan
from engine_tests.execution.test_shard_initialization import fixture
from engine_tests.offline.test_shared_plans import registered
from engine_tests.offline.test_state_inventory import training
from shadowspill.artifacts.state import StateId
from shadowspill.checkpoint.mapping import LogicalSpan, intersections
from shadowspill.checkpoint.state import PlanState
from shadowspill.ir import TransferPhase, TransferSpec
from shadowspill.parallelism import Parallelism, ParameterStorage, Sharded
from shadowspill.parallelism.resolve import ParameterGeometry, resolve_parameters


def model_spec(ranks, dim):
    return Parallelism(
        ranks_per_replica=len(ranks),
        groups={"model": (ranks,), "storage": tuple((rank,) for rank in ranks)},
        default_parameter_placement=Sharded("model", dim),
        default_parameter_storage=ParameterStorage(
            group="storage",
            weights="replicated",
            gradients="replicated",
            optimizer="replicated",
        ),
    )


def shaped(shape, spec):
    source = training()
    count = prod(shape)
    stride = tuple(prod(shape[i + 1 :]) for i in range(len(shape)))
    geometry = {"shape": shape, "stride": stride, "dtype": "torch.float32"}
    owner = resolve_parameters(
        spec,
        (ParameterGeometry(("linear.weight", "head.weight"), shape, True, count * 4),),
    )[0]
    return replace(
        source,
        parallelism=spec,
        alias_groups=tuple(
            replace(a, size_bytes=count * 4) for a in source.alias_groups
        ),
        objects=tuple(replace(o, size_bytes=count * 4) for o in source.objects),
        bindings={
            **source.bindings,
            "parameter_ownership": [owner.to_dict()],
            "model_tensor_geometry": {
                name: geometry for name in source.bindings["model_tensor_geometry"]
            },
            "optimizer_bindings": [
                {**binding, **(geometry if binding["shape"] else {})}
                for binding in source.bindings["optimizer_bindings"]
            ],
            "persistent_gradients": [],
        },
    )


def test_logical_inventory_uses_shared_export_owners_and_keeps_ties(monkeypatch):
    state = PlanState(
        plan(registered(), kind="forward"),
        {name: plan(training(prefix=name)) for name in ("first", "second")},
        logical_rank=0,
    )
    monkeypatch.setattr(torch, "empty", lambda *a, **k: pytest.fail("allocated tensor"))
    entries = {item.tensor.identity: item for item in state.logical_tensors()}
    assert len(entries) == 5
    weight = entries[StateId("parameter", "linear.weight")]
    tied = entries[StateId("parameter", "head.weight")]
    assert weight.archive == tied.archive
    assert weight.archive.name == "linear.weight"
    assert weight.view == tied.view
    assert entries[StateId("master", "linear.weight")].archive.name == (
        "master.linear.weight"
    )
    counter = entries[StateId("optimizer", "optimizer.linear.weight.step")]
    assert counter.view.shape == () and counter.view.region_elements == 1
    assert counter.archive.name == "optimizer.linear.weight.step"
    descriptors = set(state.tensor_chunks())
    assert all(entry.archive in descriptors for entry in entries.values())
    assert all(entry.tensor.identity.role != "gradient" for entry in entries.values())
    assert all(
        "optimizer_group" not in entry.archive.name for entry in entries.values()
    )


def test_tied_strided_views_retain_distinct_coordinates_over_one_archive():
    source = registered()
    changed = {
        **source.bindings["model_tensor_geometry"],
        "head.weight": {
            "dtype": "torch.float32",
            "shape": (4, 4),
            "stride": (1, 4),
        },
    }
    state = PlanState(
        plan(
            replace(
                source, bindings={**source.bindings, "model_tensor_geometry": changed}
            )
        ),
        {},
        logical_rank=0,
    )
    weight, tied = state.logical_tensors()
    assert weight.archive == tied.archive
    assert weight.view.stride == (4, 1)
    assert tied.view.stride == (1, 4)
    assert list(tied.view.spans())[:2] == [LogicalSpan(0, 0, 1), LogicalSpan(1, 4, 1)]


def test_real_ownership_descriptions_map_rows_to_columns_in_declared_rank_order():
    source = shaped((3, 8), model_spec((1, 0), 0))
    target = shaped((6, 2), model_spec((3, 1, 0, 2), -1))
    sources = [
        {
            item.tensor.identity: item
            for item in PlanState(plan(source), {}, logical_rank=rank).logical_tensors()
        }
        for rank in range(2)
    ]
    targets = [
        {
            item.tensor.identity: item
            for item in PlanState(plan(target), {}, logical_rank=rank).logical_tensors()
        }
        for rank in range(4)
    ]
    original = np.arange(48).reshape(6, 8)
    for role, name in (
        ("parameter", "linear.weight"),
        ("parameter", "head.weight"),
        ("master", "linear.weight"),
        ("optimizer", "optimizer.linear.weight.exp_avg"),
    ):
        identity = StateId(role, name)
        for rank, inventory in enumerate(targets):
            destination = inventory[identity]
            expected_column = 2 * (3, 1, 0, 2).index(rank)
            assert destination.view.origin == (0, expected_column)
            output = np.full(12, -1)
            writes = np.zeros(12, dtype=int)
            for source_rank, source_inventory in enumerate(sources):
                entry = source_inventory[identity]
                start_row = 3 * (1, 0).index(source_rank)
                assert entry.view.origin == (start_row, 0)
                payload = original[start_row : start_row + 3].reshape(-1)
                for part in intersections(entry.view, destination.view):
                    target_slice = slice(
                        part.target_offset, part.target_offset + part.elements
                    )
                    output[target_slice] = payload[
                        part.source_offset : part.source_offset + part.elements
                    ]
                    writes[target_slice] += 1
            np.testing.assert_array_equal(
                output.reshape(6, 2), original[:, expected_column : expected_column + 2]
            )
            assert (writes == 1).all()
    for inventory in (*sources, *targets):
        assert (
            inventory[StateId("optimizer", "optimizer.linear.weight.step")].view.shape
            == ()
        )


@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("replicated_compute", [False, True])
def test_flat_owned_regions_reuse_existing_resolver_and_exclude_padding(
    rank, replicated_compute
):
    source = fixture(elements=15)
    source = replace(
        source,
        bindings={
            **source.bindings,
            "registrations": [
                {**item, "parameter": True} for item in source.bindings["registrations"]
            ],
            "parameter_ownership": [
                {**source.bindings["parameter_ownership"][0], "global_shape": (15,)}
            ],
            **({"weight_shards": {}} if replicated_compute else {}),
        },
    )
    target = plan(source)
    if not replicated_compute:
        target._costed_program = replace(
            source,
            alias_groups=tuple(
                replace(
                    a,
                    transfer=TransferSpec(
                        16,
                        (TransferPhase("copy", bytes=16),),
                        (TransferPhase("copy", bytes=16),),
                    ),
                )
                if a.alias_group_id == "weight"
                else a
                for a in source.alias_groups
            ),
        )
    state = PlanState(target, {}, logical_rank=rank)
    entries = list(state.logical_tensors())
    assert len(entries) == 3
    offset, length = (8, 7) if rank % 2 == 0 else (0, 8)
    for entry in entries:
        expected = (
            (0, 15)
            if replicated_compute and entry.tensor.identity.role == "parameter"
            else (offset, length)
        )
        assert entry.view.shape == (15,)
        assert (entry.view.region_offset, entry.view.region_elements) == expected
        assert list(entry.view.spans()) == [LogicalSpan(expected[0], 0, expected[1])]
        assert entry.archive in list(state.tensor_chunks())


def test_factored_optimizer_state_uses_its_own_shape_and_needs_sharded_mapping():
    source = shaped((4, 4), Parallelism.dp(ranks_per_replica=2, optimizer="replicated"))
    moment = "optimizer.linear.weight.exp_avg"
    source = replace(
        source,
        bindings={
            **source.bindings,
            "optimizer_bindings": [
                {**b, "shape": (4,), "stride": (1,)} if b["name"] == moment else b
                for b in source.bindings["optimizer_bindings"]
            ],
        },
    )
    entries = {
        item.tensor.identity.name: item
        for item in PlanState(plan(source), {}, logical_rank=0).logical_tensors()
    }
    assert entries[moment].view.shape == (4,)
    sharded = shaped((4, 4), model_spec((1, 0), 0))
    sharded = replace(
        sharded,
        bindings={
            **sharded.bindings,
            "optimizer_bindings": source.bindings["optimizer_bindings"],
        },
    )
    with pytest.raises(ValueError, match="own logical placement"):
        list(PlanState(plan(sharded), {}, logical_rank=0).logical_tensors())


def test_checkpointable_registered_state_without_export_binding_is_not_dropped():
    source = registered()
    source = replace(
        source,
        bindings={
            **source.bindings,
            "registered_state": [
                {
                    "identity": StateId("rng", "training-stream").to_dict(),
                    "host_binding": "training-stream",
                    "geometry": {"shape": (8,), "stride": (1,), "dtype": "torch.uint8"},
                }
            ],
        },
    )
    with pytest.raises(ValueError, match="no owned export binding"):
        list(PlanState(plan(source), {}, logical_rank=0).logical_tensors())


def test_empty_logical_state_is_retained_without_inventing_an_archive():
    source = shaped((0, 4), Parallelism())
    entries = list(PlanState(plan(source), {}, logical_rank=0).logical_tensors())
    assert len(entries) == 5
    for entry in entries:
        if entry.view.shape:
            assert entry.archive is None and not list(entry.view.spans())


@pytest.mark.parametrize(
    "change,error",
    [
        ("missing", "no saved logical ownership"),
        ("global", "disagrees with global ownership"),
        ("encoded", "explicit logical map"),
    ],
)
def test_undefined_or_inconsistent_logical_coordinates_are_rejected(change, error):
    source = shaped((3, 8), model_spec((1, 0), 0))
    bindings = dict(source.bindings)
    if change == "missing":
        bindings["parameter_ownership"] = []
    elif change == "global":
        bindings["parameter_ownership"] = [
            {**bindings["parameter_ownership"][0], "global_shape": (3, 8)}
        ]
    else:
        bindings["registrations"] = [
            {**b, "component_path": ("payload",)} for b in bindings["registrations"]
        ]
    with pytest.raises(ValueError, match=error):
        list(
            PlanState(
                plan(replace(source, bindings=bindings)), {}, logical_rank=0
            ).logical_tensors()
        )
