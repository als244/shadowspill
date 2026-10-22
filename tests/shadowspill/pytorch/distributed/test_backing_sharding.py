from types import SimpleNamespace

import pytest
import torch
from torch import nn

from shadowspill.pytorch.distributed._ownership import Parameter
from shadowspill.pytorch.distributed._shards import (
    optimizer_parameters,
    parameter_layouts,
)


@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("local_shard_size", [2, 4])
@pytest.mark.parametrize("shard_optimizer", [True, False])
def test_backing_group_does_not_change_optimizer_or_master_ownership(
    monkeypatch, rank, local_shard_size, shard_optimizer
):
    """Local spill groups must not change global optimizer ownership."""
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: rank)
    parameter = nn.Parameter(torch.zeros(17, dtype=torch.bfloat16))
    replica_group = SimpleNamespace(group_name="full-dp-group")
    bound = SimpleNamespace(
        control=SimpleNamespace(
            exchange=lambda *_: [["weight"] for _ in range(4)],
            members=(0, 1, 2, 3),
        ),
        parameters=(
            Parameter(
                "weight",
                ("weight",),
                (17,),
                (1,),
                "torch.bfloat16",
                True,
                (0, 1, 2, 3),
                (0, 1, 2, 3),
            ),
        ),
        parameter_groups={"weight": (replica_group, replica_group)},
        groups=SimpleNamespace(source_names={"full-dp-group": "global"}),
        shard_optimizer=shard_optimizer,
        transfer_sharding={
            "weight": SimpleNamespace(
                group=SimpleNamespace(size=local_shard_size),
            )
        },
    )
    weights = {"weight": parameter}
    layouts = parameter_layouts(bound, weights, {"weight"}, master_dtype=torch.float32)
    layout = layouts["weight"]
    assert layout.group_size == (4 if shard_optimizer else 1)
    assert layout.group_rank == (rank if shard_optimizer else 0)
    assert layout.gradient_group_size == 4
    assert layout.gradient_group_name == "global"
    masters = optimizer_parameters(weights, layouts, master_dtype=torch.float32)
    assert masters["weight"].shape == ((5,) if shard_optimizer else (17,))
    assert masters["weight"].dtype == torch.float32


def test_compressed_data_scales_and_tied_names_resolve_to_parameter_ownership():
    from shadowspill.pytorch.distributed._ownership import storage_parameter_names

    wrapped = Parameter(
        "weight",
        ("weight", "tied"),
        (128, 128),
        (128, 1),
        "torch.bfloat16",
        False,
        (0, 1),
        (0, 1),
        components=(
            (("_data",), (128, 128), "torch.float8_e4m3fn"),
            (("_scale",), (1, 1), "torch.float32"),
        ),
    )
    names = storage_parameter_names((wrapped,))
    assert names == {
        "weight._data": "weight",
        "weight._scale": "weight",
        "tied._data": "weight",
        "tied._scale": "weight",
    }


@pytest.mark.parametrize("replicas", [1, 4])
@pytest.mark.parametrize("shard_optimizer", [False, True])
def test_single_participant_gradient_group_needs_no_collective(
    monkeypatch, replicas, shard_optimizer
):
    from shadowspill.pytorch.distributed._optimizer import _gradient

    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    weight = nn.Parameter(torch.arange(16, dtype=torch.float32))
    members = tuple(range(replicas))
    group = SimpleNamespace(group_name="replicas")
    singleton = SimpleNamespace(group_name="one-contributor")
    bound = SimpleNamespace(
        control=SimpleNamespace(
            exchange=lambda *_: [["weight"] for _ in members], members=members
        ),
        parameters=(
            Parameter(
                "weight", ("weight",), (16,), (1,), "torch.float32", True, members, (0,)
            ),
        ),
        parameter_groups={"weight": (group, singleton)},
        groups=SimpleNamespace(
            source_names={"replicas": "replicas", "one-contributor": "one-contributor"}
        ),
        shard_optimizer=shard_optimizer,
    )
    layout = parameter_layouts(
        bound, {"weight": weight}, {"weight"}, master_dtype=None
    )["weight"]
    assert layout.gradient_group_name is None
    assert layout.gradient_group_size == 1
    assert layout.group_size == (replicas if shard_optimizer else 1)
    gradient = torch.arange(16, dtype=torch.float32)
    # No process group is registered: an attempted collective fails here.
    actual = _gradient(gradient, layout)
    expected = gradient[: 16 // replicas] if shard_optimizer else gradient
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("group_members", [(0, 1, 2, 3), (0, 1)])
def test_initial_sync_skips_reconstructing_only_a_full_replica_shard_group(
    monkeypatch, group_members
):
    from shadowspill.pytorch.distributed._backing_storage import synchronize_state
    from shadowspill.pytorch.state import storage

    parameter = Parameter(
        "weight",
        ("weight",),
        (8,),
        (1,),
        "torch.float32",
        True,
        (0, 1, 2, 3),
        (0, 1, 2, 3),
    )
    host_group, control_group = object(), object()
    item = SimpleNamespace(
        sharding=SimpleNamespace(host_group=host_group),
        views=(SimpleNamespace(name="weight"),),
    )
    bound = SimpleNamespace(
        parameters=(parameter,),
        control=SimpleNamespace(
            members=(0, 1, 2, 3),
            group=control_group,
            agree=lambda *_: None,
        ),
    )
    calls = []
    monkeypatch.setattr(
        torch.distributed, "get_process_group_ranks", lambda _: list(group_members)
    )
    monkeypatch.setattr(
        storage, "persistent_state", lambda *_: SimpleNamespace(storages=(item,))
    )
    value = torch.arange(8)

    def read(*args, **kwargs):
        calls.append("read")
        return value

    def broadcast(tensor, *, src, group):
        assert tensor is value and src == 0 and group is control_group
        calls.append("broadcast")

    monkeypatch.setattr(storage, "_storage_bytes", read)
    monkeypatch.setattr(torch.distributed, "broadcast", broadcast)
    monkeypatch.setattr(
        storage, "_write_storage_bytes", lambda *_: calls.append("write")
    )
    assert synchronize_state(bound, object(), object()) == frozenset({"weight"})
    assert calls == ([] if len(group_members) == 4 else ["read", "broadcast", "write"])
