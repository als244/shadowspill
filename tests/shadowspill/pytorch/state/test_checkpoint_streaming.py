"""Pool checkpoint files restore values and aliases without reading them early."""

from types import SimpleNamespace

import pytest
import torch

from shadowspill.pytorch.state.checkpoint import PoolCheckpoint


def test_deferred_checkpoint_preserves_views_and_control_values(tmp_path, monkeypatch):
    data = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    calls = []
    bridge = SimpleNamespace(objects=object(), wait_runtime_idle=lambda: None)

    def read(objects, name, target):
        assert objects is bridge.objects
        calls.append(name)
        target.copy_(data.view(torch.uint8).flatten())

    monkeypatch.setattr("shadowspill.pytorch.state.checkpoint.read_spill_tensor", read)
    writer = PoolCheckpoint(bridge)
    owner = writer.alias("weight", data.untyped_storage().nbytes())
    payload = {
        "weight": writer.view(owner, data),
        "transpose": writer.view(owner, data.T),
        "slice": writer.view(owner, data[2:, 1:]),
        "step": writer.value(torch.tensor(7)),
    }
    assert calls == []
    path = tmp_path / "checkpoint.pt"
    writer.save(payload, path)
    assert calls == ["weight"]
    loaded = torch.load(path, mmap=True, weights_only=True)
    torch.testing.assert_close(loaded["weight"], data, rtol=0, atol=0)
    torch.testing.assert_close(loaded["transpose"], data.T, rtol=0, atol=0)
    torch.testing.assert_close(loaded["slice"], data[2:, 1:], rtol=0, atol=0)
    assert loaded["step"].item() == 7
    assert (
        loaded["weight"].untyped_storage().data_ptr()
        == loaded["slice"].untyped_storage().data_ptr()
    )


def test_failed_checkpoint_preserves_previous_file(tmp_path, monkeypatch):
    def fail(*args):
        raise OSError("pool read failed")

    monkeypatch.setattr("shadowspill.pytorch.state.checkpoint.read_spill_tensor", fail)
    writer = PoolCheckpoint(
        SimpleNamespace(objects=object(), wait_runtime_idle=lambda: None)
    )
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"previous checkpoint")
    payload = {"weight": writer.alias("weight", 256)}
    with pytest.raises(OSError, match="pool read failed"):
        writer.save(payload, path)
    assert path.read_bytes() == b"previous checkpoint"
    assert list(tmp_path.iterdir()) == [path]
