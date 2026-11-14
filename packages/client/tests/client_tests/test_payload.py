"""Submission snapshots own data and preserve binary identity under bounded I/O."""

import hashlib
import importlib.util
import io
import json
import struct
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import numpy as np
import pytest

from shadowspill_client import Datum
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.codec import MAX_METADATA_BYTES, ProtocolError
from shadowspill_client.protocol.payload import MAGIC, MEDIA_TYPE, snapshot_batch
from shadowspill_client.protocol.payload_io import read_payload
from shadowspill_client.protocol.requests import ArtifactRef
from shadowspill_client.protocol.tensors import TensorSource


def reference(data):
    return ArtifactRef(
        "payload", hashlib.sha256(data).hexdigest(), len(data), MEDIA_TYPE
    )


def test_owned_nested_snapshot_survives_caller_mutation_and_binary_roundtrip():
    source = np.arange(24, dtype=np.float32).reshape(4, 6)[:, ::-2]
    nested = {"image": source, "metadata": {"enabled": True}, "empty": {}}
    targets = [[1, 2], [3, 4]]
    budget = BufferBudget(1 << 20)
    with snapshot_batch([Datum(nested, {"labels": targets})], budget) as snapshot:
        expected = source.copy()
        source.fill(-10)
        targets[0][0] = -100
        nested["metadata"]["enabled"] = False
        values = snapshot.datums[0]
        np.testing.assert_array_equal(values.inputs["image"].array(), expected)
        np.testing.assert_array_equal(
            values.loss_inputs["labels"].array(), [[1, 2], [3, 4]]
        )
        assert values.inputs["metadata"]["enabled"].array().item()
        assert not values.inputs["empty"]
        with pytest.raises(TypeError):
            values.inputs["metadata"]["enabled"] = False
        with pytest.raises(ValueError):
            values.inputs["image"].array().setflags(write=True)
        chunks = list(snapshot.chunks(7))
        assert all(len(chunk) <= 7 for chunk in chunks)
        wire = b"".join(chunks)
        assert len(wire) == snapshot.size_bytes
        assert hashlib.sha256(wire).hexdigest() == snapshot.sha256
        with read_payload(
            io.BytesIO(wire), reference(wire), BufferBudget(1 << 20), chunk_bytes=3
        ) as decoded:
            assert decoded.sha256 == snapshot.sha256
            assert b"".join(decoded.chunks()) == wire
            for a, b in zip(decoded.tensors, snapshot.tensors, strict=True):
                assert a == b
    assert budget.used == (0, 0)
    snapshot.close()
    with pytest.raises(RuntimeError, match="closed"):
        list(snapshot.chunks())


@pytest.mark.parametrize(
    "value",
    [
        np.array(1.25, dtype=np.float32),
        np.empty((0, 3), dtype=np.int32),
        np.arange(5, dtype=">i4"),
        [],
        [[True, False], [False, True]],
        [1, 2.5],
        -4,
    ],
)
def test_scalar_empty_endian_and_list_tensor_values(value):
    with snapshot_batch([Datum({"x": value})], BufferBudget(1 << 20)) as snapshot:
        tensor = snapshot.datums[0].inputs["x"]
        np.testing.assert_array_equal(tensor.array(), np.asarray(value))
        assert tensor.array().dtype.byteorder != ">"
        wire = b"".join(snapshot.chunks())
        with read_payload(
            io.BytesIO(wire), reference(wire), BufferBudget(1 << 20)
        ) as decoded:
            assert decoded.tensors == snapshot.tensors


def test_digest_is_canonical_by_tree_keys_and_covers_shape_and_dtype():
    budget = BufferBudget(1 << 20, max_requests=3)
    with (
        snapshot_batch([Datum({"b": [2], "a": [1]})], budget) as a,
        snapshot_batch([Datum({"a": [1], "b": [2]})], budget) as b,
        snapshot_batch([Datum({"a": [[1]], "b": [2]})], budget) as changed,
    ):
        assert a.sha256 == b.sha256
        assert a.sha256 != changed.sha256


def test_oversized_snapshot_is_rejected_before_copy_and_copy_failures_release(
    monkeypatch,
):
    copied = []
    monkeypatch.setattr(TensorSource, "copy", lambda self: copied.append(self))
    with pytest.raises(ValueError, match="staging bytes"):
        snapshot_batch([Datum({"x": np.zeros(1000)})], BufferBudget(32))
    assert not copied
    budget = BufferBudget(1 << 20)

    def fail(self):
        raise RuntimeError("copy failed")

    monkeypatch.setattr(TensorSource, "copy", fail)
    with pytest.raises(RuntimeError, match="copy failed"):
        snapshot_batch([Datum({"x": [1, 2]})], budget)
    assert budget.used == (0, 0)


def test_backpressure_waits_for_both_bytes_and_request_slots_and_close_wakes():
    budget = BufferBudget(100, max_requests=1)
    first = budget.reserve(80)
    started = Event()

    def reserve():
        started.set()
        return budget.reserve(20, timeout=2)

    with ThreadPoolExecutor(1) as pool:
        pending = pool.submit(reserve)
        assert started.wait(1)
        assert not pending.done()
        first.shrink(1)
        assert not pending.done()  # The request slot is still owned.
        first.close()
        with pending.result(timeout=2):
            assert budget.used == (20, 1)
        held = budget.reserve(100)
        started.clear()
        waiting = pool.submit(reserve)
        assert started.wait(1)
        budget.close()
        with pytest.raises(RuntimeError, match="closed"):
            waiting.result(timeout=2)
        held.close()
    assert budget.used == (0, 0)
    budget = BufferBudget(100, max_requests=2)
    with budget.reserve(90), pytest.raises(TimeoutError):
        budget.reserve(20, timeout=0)


@pytest.mark.parametrize(
    "value",
    [
        [[1], [2, 3]],
        np.array([object()]),
        np.array([1 + 2j]),
        1 << 80,
        {"bad": {1: [1]}},
    ],
)
def test_unsupported_input_fails_without_leaking_staging(value):
    budget = BufferBudget(1 << 20)
    with pytest.raises((ValueError, TypeError)):
        snapshot_batch([Datum({"x": value})], budget)
    assert budget.used == (0, 0)


def test_checksum_truncation_and_bad_layout_never_publish_partial_batches():
    with snapshot_batch([Datum({"x": [1, 2, 3]})], BufferBudget(1 << 20)) as snapshot:
        wire = b"".join(snapshot.chunks())
    budget = BufferBudget(1 << 20)
    for corrupted in (wire[:-1], wire[:-1] + bytes([wire[-1] ^ 1])):
        with pytest.raises(ProtocolError, match=r"truncated|checksum"):
            read_payload(io.BytesIO(corrupted), reference(wire), budget, chunk_bytes=2)
        assert budget.used == (0, 0)
    header = json.loads(wire[16 : 16 + struct.unpack("<Q", wire[8:16])[0]])
    for invalid in (
        {**header, "version": True},
        {**header, "unknown": 1},
        {**header, "tensors": [{"dtype": "int64", "shape": [True]}]},
        {**header, "tensors": [{"dtype": "int64", "shape": [1 << 50]}]},
        {**header, "datums": [{"inputs": {"x": 0, "y": 0}, "loss_inputs": {}}]},
    ):
        data = json.dumps(invalid).encode()
        malformed = MAGIC + struct.pack("<Q", len(data)) + data
        with pytest.raises(ProtocolError):
            read_payload(io.BytesIO(malformed), reference(malformed), budget)
        assert budget.used == (0, 0)
    too_large = MAGIC + struct.pack("<Q", MAX_METADATA_BYTES + 1)
    with pytest.raises(ProtocolError, match="manifest exceeds"):
        read_payload(io.BytesIO(too_large), reference(too_large), budget)


def test_snapshot_and_binary_codec_import_without_engine_or_torch():
    code = """
import io, sys
for name in ('torch', 'shadowspill', 'shadowspill_workloads', 'mlops', 'triton'):
    sys.modules[name] = None
from shadowspill_client import Datum
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.payload import snapshot_batch, MEDIA_TYPE
from shadowspill_client.protocol.payload_io import read_payload
from shadowspill_client.protocol.requests import ArtifactRef
with snapshot_batch([Datum({'pixels': [[0.1, 0.2]]})], BufferBudget(1048576)) as s:
    wire = b''.join(s.chunks())
    ref = ArtifactRef('x', s.sha256, len(wire), MEDIA_TYPE)
    with read_payload(io.BytesIO(wire), ref, BufferBudget(1048576)) as d:
        assert d.datums[0].inputs['pixels'].array().shape == (1, 2)
"""
    run = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
    )
    assert run.returncode == 0, run.stdout + run.stderr


def test_optional_cpu_torch_and_bfloat16_preserve_values_in_separate_process():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("optional CPU Tensor interoperability requires installed Torch")
    code = """
import torch
from shadowspill_client import Datum
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.payload import snapshot_batch
for dtype in (torch.float32, torch.bfloat16):
    original = torch.arange(12, dtype=dtype).reshape(3, 4).T.requires_grad_(True)
    before = original.detach().clone()
    with snapshot_batch([Datum({'x': original})], BufferBudget(1048576)) as s:
        with torch.no_grad(): original.fill_(-1)
        t = s.datums[0].inputs['x']
        actual = torch.from_numpy(t.array().copy())
        if dtype == torch.bfloat16: actual = actual.view(dtype)
        torch.testing.assert_close(actual, before)
"""
    run = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
    )
    assert run.returncode == 0, run.stdout + run.stderr
