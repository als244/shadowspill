"""Checkpoint archives contain only owned regions and identify logical ranks."""

import json
import zipfile

import pytest
import torch

from shadowspill import load_checkpoint, save_checkpoint
from shadowspill.execution.tensors import TensorChunk
from tests.shadowspill.execution.test_checkpoint import MemoryExecutor


class ShardedExecutor(MemoryExecutor):
    def __init__(self, rank):
        super().__init__()
        self.logical_rank = rank
        self.offset = rank * 8
        self.shard = torch.arange(self.offset, min(15, self.offset + 8)).float()

    def export_tensors(self, callback, *, chunk_bytes):
        width = max(1, chunk_bytes // 4)
        for start in range(0, self.shard.numel(), width):
            callback(
                TensorChunk(
                    "weight",
                    (3, 5),
                    torch.float32,
                    self.offset + start,
                    15,
                    (5, 1),
                    region_offset=self.offset,
                    region_elements=self.shard.numel(),
                ),
                self.shard[start : start + width],
            )


@pytest.mark.parametrize("rank", [0, 1])
def test_checkpoint_roundtrip_owns_only_declared_region(tmp_path, rank):
    executor = ShardedExecutor(rank)
    expected = executor.shard.clone()
    path = tmp_path / "checkpoint.ssp"
    save_checkpoint(executor, path, chunk_bytes=12)
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        entry = manifest["tensors"][0]
        assert entry["bytes"] == executor.shard.nbytes
        assert entry["region_offset"] == rank * 8
        assert entry["storage_elements"] == 15
    executor.shard.zero_()
    load_checkpoint(executor, path, chunk_bytes=20)
    torch.testing.assert_close(executor.shard, expected)
    with pytest.raises(ValueError, match="Program identity differs"):
        load_checkpoint(ShardedExecutor(1 - rank), path)
