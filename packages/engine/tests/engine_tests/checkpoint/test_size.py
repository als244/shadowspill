"""Exact resource admission uses descriptors, never an initial tensor export."""

import io

import pytest
import torch

from engine_tests.checkpoint.test_weights import mixed_executor
from engine_tests.execution.test_checkpoint import MemoryExecutor
from engine_tests.execution.test_partial_checkpoint import executor as partial_executor
from engine_tests.execution.test_sharded_checkpoint import ShardedExecutor
from shadowspill.artifacts.tensors import TensorChunk
from shadowspill.checkpoint import checkpoint_size, write_checkpoint


@pytest.mark.parametrize(
    "kind", ["plain", "master", "shard0", "shard1", "partial", "empty"]
)
@pytest.mark.parametrize("weights", ["master", "compute"])
def test_envelope_matches_real_framing_and_selection_without_reading_state(
    kind, weights, monkeypatch
):
    options = {
        "weights": weights,
        "metadata": {"updates": 17, "cursor": {"index": [1, 2]}},
    }
    if kind == "master":
        executor = mixed_executor()
    elif kind.startswith("shard"):
        executor = ShardedExecutor(int(kind[-1]))
    elif kind == "partial":
        executor = partial_executor()
        options["frozen_state_id"] = "base-v1"
    else:
        executor = MemoryExecutor()
        if kind == "empty":
            executor.values["empty"] = torch.empty(0)
    with monkeypatch.context() as guard:
        guard.setattr(
            executor,
            "export_tensors",
            lambda *args, **kwargs: pytest.fail("read tensor payload for admission"),
        )
        size = checkpoint_size(executor, **options)
    output = io.BytesIO()
    manifest = write_checkpoint(executor, output, chunk_bytes=11, **options)
    assert output.tell() == size
    assert size % 512 == 0
    if kind == "empty":
        assert "empty" not in {entry["name"] for entry in manifest["tensors"]}


def test_gnu_large_tensor_envelope_does_not_allocate_or_export_payload(monkeypatch):
    executor = MemoryExecutor()
    elements = ((9 << 30) + 8) // 4
    chunk = TensorChunk("giant", (elements,), torch.float32, 0, elements)
    monkeypatch.setattr(executor, "tensor_chunks", lambda: iter([chunk]))
    monkeypatch.setattr(
        executor,
        "export_tensors",
        lambda *args, **kwargs: pytest.fail("allocated giant payload"),
    )
    size = checkpoint_size(executor)
    assert 9 << 30 < size < (9 << 30) + (16 << 10)
    assert size % 512 == 0


def test_invalid_or_oversized_metadata_fails_without_export(monkeypatch):
    executor = MemoryExecutor()
    monkeypatch.setattr(
        executor,
        "export_tensors",
        lambda *args, **kwargs: pytest.fail("exported invalid checkpoint"),
    )
    for metadata in ({"invalid": float("nan")}, {"oversized": "x" * (16 << 20)}):
        with pytest.raises(ValueError):
            checkpoint_size(executor, metadata=metadata)
