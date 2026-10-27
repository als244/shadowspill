"""Initialization and checkpoint import put values directly in pool storage."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from shadowspill.pytorch.state.initialization import initialize_storage_


def test_storage_initializer_preserves_untouched_bytes_of_a_strided_view():
    value = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    expected = value.clone()
    source = torch.full((2, 3), 17.0)
    expected[1:3, ::2] = source
    destination = value[1:3, ::2]
    result = initialize_storage_(destination, lambda scratch: scratch.copy_(source))
    assert result is destination
    torch.testing.assert_close(value, expected, rtol=0, atol=0)


@pytest.mark.parametrize("leader", [True, False])
@pytest.mark.parametrize("fails", [True, False])
def test_storage_initializer_executes_only_the_shard_leader(monkeypatch, leader, fails):
    from types import SimpleNamespace

    from shadowspill.pytorch.state import initialization

    destination = torch.zeros(4)
    identity = destination.untyped_storage()._cdata
    item = SimpleNamespace(
        unbacked=True, size_bytes=16, sharding=SimpleNamespace(host_group="local")
    )
    registry = SimpleNamespace(storage=lambda key: item if key == identity else None)
    monkeypatch.setattr(initialization, "registry_for", lambda runtime: registry)
    monkeypatch.setattr(
        initialization.dist, "get_process_group_ranks", lambda group: [2, 3]
    )
    monkeypatch.setattr(initialization.dist, "get_rank", lambda: 2 if leader else 3)
    calls, writes = [], []

    def fill(value):
        calls.append(True)
        if fails:
            raise OSError("checkpoint read failed")
        value.copy_(torch.arange(4))

    def broadcast(status, *, src, group):
        assert src == 2 and group == "local"
        if not leader and fails:
            status[0] = "OSError: checkpoint read failed"

    monkeypatch.setattr(initialization.dist, "broadcast_object_list", broadcast)
    monkeypatch.setattr(
        initialization,
        "_write_storage_bytes",
        lambda runtime, record, value, **kwargs: writes.append(value.clone()),
    )
    with initialization.pool_values(object()):
        if fails:
            with pytest.raises(RuntimeError, match="checkpoint read failed"):
                initialize_storage_(destination, fill)
        else:
            initialize_storage_(destination, fill)
    assert calls == ([True] if leader else [])
    assert len(writes) == int(not fails)
    if not fails:
        if leader:
            torch.testing.assert_close(
                writes[0].view(torch.float32), torch.arange(4).float()
            )
        else:
            assert writes[0].numel() == 0
    assert initialization._active.get() is None


class _Initializer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(32, 16))
        self.tied = self.weight
        self.register_buffer("window", self.weight.detach()[2:5].T)
        self.register_buffer("count", torch.empty((), dtype=torch.int64))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.weight.uniform_(-1, 1)
            self.weight.mul_(2)
            self.window.add_(3)
            self.count.fill_(7)


def test_persistent_state_indexes_owners_once(monkeypatch):
    from shadowspill.pytorch.state.records import (
        PersistentState,
        PersistentStorage,
        TensorView,
    )

    calls = []
    original = PersistentStorage.storage_identity.fget

    def counted(storage):
        calls.append(storage)
        return original(storage)

    monkeypatch.setattr(PersistentStorage, "storage_identity", property(counted))
    owners = tuple(torch.empty(8) for _ in range(4))
    storages = tuple(
        PersistentStorage(
            persistent_object_id=i,
            current_object_id=i,
            pool_id=0,
            size_bytes=32,
            pool_pointer=0,
            anchor=owner,
            views=(TensorView(owner.view(2, 4), (2, 4), (4, 1), 0, False),),
            frontend_storage_is_separate=False,
        )
        for i, owner in enumerate(owners)
    )
    state = PersistentState(object(), "spill", storages, None)
    for _ in range(3):
        for owner, storage in zip(owners, storages, strict=True):
            view = owner.view(2, 4)
            assert state.by_storage_identity()[view.untyped_storage()._cdata] is storage
    assert len(calls) == len(storages)


def test_effect_wrapper_stages_pool_values(monkeypatch):
    """Effectful operations must see authoritative bytes, not metadata payloads."""
    from types import SimpleNamespace

    from torch._higher_order_ops.effects import with_effects

    from shadowspill.pytorch.state import initialization

    declaration = torch.zeros(4)
    authoritative = torch.arange(4, dtype=torch.float32)
    identity = declaration.untyped_storage()._cdata
    item = SimpleNamespace(unbacked=True, size_bytes=16)
    registry = SimpleNamespace(storage=lambda key: item if key == identity else None)
    monkeypatch.setattr(initialization, "registry_for", lambda runtime: registry)

    def read_bytes(record, *, runtime, owner):
        owner.copy_(authoritative.view(torch.uint8))
        return owner

    monkeypatch.setattr(initialization, "_storage_bytes", read_bytes)
    with initialization.pool_values(object()):
        token, result = with_effects(
            torch.empty(0), torch.ops.aten.sin.default, declaration
        )
    torch.testing.assert_close(result, authoritative.sin())
    assert token.numel() == 0
    assert torch.count_nonzero(declaration) == 0


@pytest.mark.parametrize(
    "operation",
    ["normal_", "uniform_", "zero_", "fill_", "add_", "copy_", "copy_alias"],
)
@pytest.mark.parametrize("view", ["whole", "prefix", "offset", "strided"])
def test_pool_initializers_skip_only_reads_of_fully_overwritten_storage(
    monkeypatch, operation, view
):
    from types import SimpleNamespace

    from shadowspill.pytorch.state import initialization

    declaration = torch.zeros(4, 6)
    authoritative = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    expected = authoritative.clone()
    identity = declaration.untyped_storage()._cdata
    item = SimpleNamespace(unbacked=True, size_bytes=96, anchor=declaration)
    registry = SimpleNamespace(storage=lambda key: item if key == identity else None)
    monkeypatch.setattr(initialization, "registry_for", lambda runtime: registry)
    reads = []

    def read_bytes(record, *, runtime, owner):
        reads.append(record)
        owner.copy_(authoritative.view(torch.uint8).flatten())
        return owner

    def write_bytes(runtime, record, owner, *, synchronize):
        assert record is item and synchronize
        authoritative.view(torch.uint8).flatten().copy_(owner)

    monkeypatch.setattr(initialization, "_storage_bytes", read_bytes)
    monkeypatch.setattr(initialization, "_write_storage_bytes", write_bytes)

    def selected(tensor):
        return {
            "whole": lambda: tensor,
            "prefix": lambda: tensor[:2],
            "offset": lambda: tensor[1:3],
            "strided": lambda: tensor[:, ::2],
        }[view]()

    def initialize(tensor):
        torch.manual_seed(37)
        if operation == "copy_":
            return tensor.copy_(torch.randn(tensor.shape))
        if operation == "copy_alias":
            # Broadcasting a row from the same storage still needs the old
            # authoritative bytes, even when the destination covers the root.
            return tensor.copy_(tensor[:1].expand_as(tensor))
        args = (7,) if operation in {"fill_", "add_"} else ()
        return getattr(tensor, operation)(*args)

    initialize(selected(expected))
    with initialization.pool_values(object()):
        result = initialize(selected(declaration))
    torch.testing.assert_close(authoritative, expected, rtol=0, atol=0)
    assert len(reads) == int(view != "whole" or operation in {"add_", "copy_alias"})
    assert result.untyped_storage()._cdata == identity


def test_whole_copy_reads_the_pool_source_without_reading_the_destination(monkeypatch):
    from types import SimpleNamespace

    from shadowspill.pytorch.state import initialization

    declarations = [torch.zeros(4) for _ in range(2)]
    values = [torch.full((4,), -7.0), torch.arange(4, dtype=torch.float32)]
    records = [
        SimpleNamespace(unbacked=True, size_bytes=16, anchor=tensor, index=index)
        for index, tensor in enumerate(declarations)
    ]
    registry = SimpleNamespace(
        storage={
            tensor.untyped_storage()._cdata: record
            for tensor, record in zip(declarations, records, strict=True)
        }.get
    )
    monkeypatch.setattr(initialization, "registry_for", lambda runtime: registry)
    reads = []

    def read_bytes(record, *, runtime, owner):
        reads.append(record.index)
        owner.copy_(values[record.index].view(torch.uint8))
        return owner

    def write_bytes(runtime, record, owner, *, synchronize):
        assert synchronize
        values[record.index].view(torch.uint8).copy_(owner)

    monkeypatch.setattr(initialization, "_storage_bytes", read_bytes)
    monkeypatch.setattr(initialization, "_write_storage_bytes", write_bytes)
    with initialization.pool_values(object()):
        declarations[0].copy_(declarations[1])
    assert reads == [1]
    torch.testing.assert_close(values[0], values[1], rtol=0, atol=0)


def test_pool_copy_releases_temporary_storages_without_cyclic_gc(monkeypatch):
    """A slice must not keep its entire source bank alive after publication."""
    import gc
    import weakref
    from types import SimpleNamespace

    from shadowspill.pytorch.state import initialization

    registry = SimpleNamespace(storage=lambda key: None)
    monkeypatch.setattr(initialization, "registry_for", lambda runtime: registry)
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        with initialization.pool_values(object()):
            for _ in range(4):
                bank = torch.empty(1024, dtype=torch.uint8)
                source = torch.ones(16, dtype=torch.uint8)
                bank_ref, source_ref = weakref.ref(bank), weakref.ref(source)
                bank[:16].copy_(source)
                del bank, source
                assert bank_ref() is None
                assert source_ref() is None
    finally:
        if was_enabled:
            gc.enable()
