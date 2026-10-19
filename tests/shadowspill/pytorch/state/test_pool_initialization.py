"""Initialization and checkpoint import put values directly in pool storage."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from shadowspill.memory import device, transfer_route
from shadowspill.pytorch import (
    Runtime,
    import_model_state,
    import_model_state_from_file,
    initialize_storage_,
    read_model_state,
    release_model_state,
)
from shadowspill.pytorch.state.storage import persistent_state
from tests.spill_pool import spill_pool


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
        initialization, "_write_storage_bytes",
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


@pytest.mark.cuda
@pytest.mark.fresh_process
def test_meta_initialization_and_checkpoint_import_preserve_values_and_ties(
    tmp_path: Path,
) -> None:
    torch.manual_seed(29)
    reference = _Initializer()
    checkpoint = tmp_path / "weights.pt"
    torch.save(dict(reference.state_dict()), checkpoint)
    with Runtime(
        pools={
            "execution": device(physical_capacity=2 << 30),
            "spill": spill_pool(512 << 20),
        },
        routes={
            "fetch": transfer_route(source="spill", destination="execution"),
            "evict": transfer_route(source="execution", destination="spill"),
        },
        calibrate=False,
    ) as runtime:
        for from_file in (False, True):
            with torch.device("meta"):
                model = _Initializer()
            if from_file:
                import_model_state_from_file(
                    model, checkpoint, runtime=runtime, pool="spill"
                )
            else:
                torch.manual_seed(29)
                model = import_model_state(model, runtime=runtime, pool="spill")
            try:
                state = persistent_state(runtime, model)
                assert state is not None
                assert len(state.storages) == 2
                assert model.weight is model.tied
                assert model.window.untyped_storage()._cdata == (
                    model.weight.untyped_storage()._cdata
                )
                values = read_model_state(model, runtime=runtime)
                for name, expected in reference.state_dict().items():
                    torch.testing.assert_close(values[name], expected, rtol=0, atol=0)
                del values
                if runtime.pools["spill"].addressable:
                    roots = {item.pool_pointer for item in state.storages}
                    assert model.weight.untyped_storage().data_ptr() in roots
                else:
                    with pytest.raises(RuntimeError, match="non-addressable"):
                        model.weight.sum()
            finally:
                release_model_state(model, runtime=runtime)
            assert runtime.pool_statistics("spill").allocated_bytes == 0

        class Broken(_Initializer):
            def reset_parameters(self) -> None:
                super().reset_parameters()
                if not self.weight.is_meta:
                    raise ValueError("initializer failed")

        with torch.device("meta"):
            broken = Broken()
        with pytest.raises(ValueError, match="initializer failed"):
            import_model_state(broken, runtime=runtime, pool="spill")
        assert persistent_state(runtime, broken) is None
        assert runtime.pool_statistics("spill").allocated_bytes == 0


@pytest.mark.cuda
@pytest.mark.fresh_process
def test_a_wrapped_weight_in_an_unreadable_pool_is_read_component_by_component(
    tmp_path: Path,
) -> None:
    """A wrapper owns no storage of its own; its components are what a pool
    holds. Reading its value for profiling, from a pool the host cannot read,
    reads each component through the runtime and rebuilds the wrapper."""

    from shadowspill.pytorch.representations import detached_representation
    from shadowspill.pytorch.state.storage import read_tensor_value
    from shadowspill.ssd import ssd

    from .representations import ScaledWeight, model

    reference = model()
    with Runtime(
        pools={
            "execution": device(physical_capacity=2 << 30),
            "spill": ssd(capacity=512 << 20, directory=tmp_path),
        },
        routes={
            "fetch": transfer_route(source="spill", destination="execution"),
            "evict": transfer_route(source="execution", destination="spill"),
        },
        calibrate=False,
    ) as runtime:
        assert not runtime.pools["spill"].addressable
        imported = import_model_state(model(), runtime=runtime, pool="spill")
        try:
            with pytest.raises(RuntimeError, match="non-addressable"):
                imported.weight.payload.sum()
            value = read_tensor_value(runtime, detached_representation(imported.weight))
            assert isinstance(value, ScaledWeight)
            torch.testing.assert_close(
                value.dense(), reference.weight.dense(), rtol=0, atol=0
            )
            del value
        finally:
            release_model_state(imported, runtime=runtime)
        assert runtime.pool_statistics("spill").allocated_bytes == 0


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
