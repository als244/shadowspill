"""Streaming and failure behavior independent of a GPU/runtime instance."""

import json
import zipfile
from types import SimpleNamespace

import pytest
import torch

from shadowspill import load_checkpoint, save_checkpoint
from shadowspill.execution.tensors import TensorChunk


class MemoryExecutor:
    def __init__(self):
        self.plan = SimpleNamespace(program=SimpleNamespace(digest="example"))
        self.plans = {}
        self.logical_rank = 0
        self.values = {
            "bf16": torch.arange(45, dtype=torch.bfloat16).reshape(9, 5).t(),
            "step": torch.tensor(3, dtype=torch.int64),
            "fp8": torch.arange(37).to(torch.float8_e4m3fn),
        }
        self.initialized = True
        self.chunks = []

    def export_tensors(self, callback, *, chunk_bytes):
        for name, tensor in self.values.items():
            count = tensor.untyped_storage().nbytes() // tensor.element_size()
            raw = tensor.as_strided((count,), (1,), 0)
            width = max(1, chunk_bytes // tensor.element_size())
            for offset in range(0, count, width):
                chunk = raw[offset : offset + width]
                self.chunks.append(chunk.nbytes)
                callback(
                    TensorChunk(
                        name,
                        tuple(tensor.shape),
                        tensor.dtype,
                        offset,
                        count,
                        tuple(tensor.stride()),
                        tensor.storage_offset(),
                    ),
                    chunk,
                )

    def initialize(self, callback, *, chunk_bytes, restore):
        assert restore
        self.initialized = False
        self.export_tensors(callback, chunk_bytes=chunk_bytes)
        self.initialized = True

    def tensor_chunks(self):
        for name, tensor in self.values.items():
            count = tensor.untyped_storage().nbytes() // tensor.element_size()
            yield TensorChunk(
                name,
                tuple(tensor.shape),
                tensor.dtype,
                0,
                count,
                tuple(tensor.stride()),
                tensor.storage_offset(),
            )


def test_streams_strided_and_mixed_dtype_storage_with_different_chunk_sizes(tmp_path):
    executor = MemoryExecutor()
    expected = {name: tensor.clone() for name, tensor in executor.values.items()}
    path = tmp_path / "checkpoint.ssp"
    save_checkpoint(executor, path, metadata={"updates": 7}, chunk_bytes=16)
    assert max(executor.chunks) <= 16
    with zipfile.ZipFile(path) as archive:
        assert len(archive.namelist()) == 4
    for tensor in executor.values.values():
        tensor.zero_()
    executor.chunks.clear()
    assert load_checkpoint(executor, path, chunk_bytes=24) == {"updates": 7}
    assert max(executor.chunks) <= 24
    for name, tensor in executor.values.items():
        torch.testing.assert_close(tensor.float(), expected[name].float())
    assert executor.initialized


def test_partial_write_preserves_existing_checkpoint(tmp_path):
    path = tmp_path / "checkpoint.ssp"
    path.write_bytes(b"existing checkpoint")
    executor = MemoryExecutor()
    original = executor.export_tensors

    def fail(callback, **kwargs):
        def write(request, chunk):
            callback(request, chunk)
            raise RuntimeError("injected export failure")

        original(write, **kwargs)

    executor.export_tensors = fail
    with pytest.raises(RuntimeError, match="injected export failure"):
        save_checkpoint(executor, path, chunk_bytes=16)
    assert path.read_bytes() == b"existing checkpoint"
    assert list(tmp_path.iterdir()) == [path]


def test_corrupted_tensor_cannot_be_used_after_restore(tmp_path):
    executor = MemoryExecutor()
    path = tmp_path / "checkpoint.ssp"
    save_checkpoint(executor, path, chunk_bytes=16)
    with zipfile.ZipFile(path) as archive:
        contents = {name: archive.read(name) for name in archive.namelist()}
    # Keep ZIP CRC valid, but invalidate the tensor's content hash.
    data = contents["tensors/0"]
    contents["tensors/0"] = bytes([data[0] ^ 1]) + data[1:]
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in contents.items():
            archive.writestr(name, data)
    with pytest.raises(ValueError, match="checksum"):
        load_checkpoint(executor, path, chunk_bytes=16)
    assert not executor.initialized


def test_another_program_of_the_same_state_restores(tmp_path):
    """A checkpoint binds to the tensors, not to the Program that saved it: a
    rebuild with other kernels or another geometry has the same state."""
    executor = MemoryExecutor()
    path = tmp_path / "checkpoint.ssp"
    save_checkpoint(executor, path, metadata={"updates": 3})
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    assert manifest["program"] == "example" and len(manifest["state"]) == 64
    executor.plan.program.digest = "rebuilt"
    for tensor in executor.values.values():
        tensor.zero_()
    assert load_checkpoint(executor, path) == {"updates": 3}
    assert executor.values["step"].item() == 3


def test_refuses_a_different_state_before_touching_executor(tmp_path):
    executor = MemoryExecutor()
    path = tmp_path / "checkpoint.ssp"
    save_checkpoint(executor, path)
    other = MemoryExecutor()
    other.values["fp8"] = other.values["fp8"][:30].clone()
    other.values["extra"] = torch.zeros(2)
    del other.values["step"]
    with pytest.raises(ValueError, match="state identity differs") as error:
        load_checkpoint(other, path)
    message = str(error.value)
    assert "only in the checkpoint: step" in message
    assert "only in this executor: extra" in message
    assert "geometry differs: fp8" in message
    assert other.initialized
