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
    read_model_state,
    release_model_state,
)
from shadowspill.pytorch.state.storage import persistent_state
from tests.spill_pool import spill_pool


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
