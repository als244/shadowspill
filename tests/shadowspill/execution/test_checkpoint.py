"""Streaming and failure behavior independent of a GPU/runtime instance."""

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


def test_refuses_different_program_before_touching_executor(tmp_path):
    executor = MemoryExecutor()
    path = tmp_path / "checkpoint.ssp"
    save_checkpoint(executor, path)
    executor.plan.program.digest = "different"
    with pytest.raises(ValueError, match="Program identity"):
        load_checkpoint(executor, path)
    assert executor.initialized
