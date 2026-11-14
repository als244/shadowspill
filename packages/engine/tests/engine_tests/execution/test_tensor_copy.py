"""Strided and broadcast output copies use a single bounded host scratch block."""

import io
from itertools import islice

import numpy as np
import pytest
import torch

from shadowspill.execution.tensor_copy import host_chunks, storage_runs
from shadowspill.training.inputs.frames import FRAME_CHUNK_BYTES, InputFrame
from shadowspill_client.buffers import BufferBudget


def copied(tensor, chunk_bytes=32):
    stream = io.BytesIO()
    for block in host_chunks(tensor, chunk_bytes=chunk_bytes):
        assert 0 < len(block) <= chunk_bytes
        stream.write(block)
    return stream.getvalue()


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.int64, torch.bfloat16, torch.bool]
)
@pytest.mark.parametrize(
    "view", ["dense", "transpose", "slice", "broadcast", "scalar", "empty"]
)
def test_cpu_strides_offsets_scalars_empty_and_bf16_have_exact_c_order(dtype, view):
    base = torch.arange(120).reshape(5, 4, 6).to(dtype)
    value = {
        "dense": base,
        "transpose": base.permute(2, 0, 1),
        "slice": base[1:4, 1:, ::2],
        "broadcast": base[0, 0, :1].expand(9, 5),
        "scalar": base[2, 1, 3],
        "empty": base[:0],
    }[view]
    expected = value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    assert copied(value) == expected


def test_large_noncontiguous_coordinates_do_not_materialize_prefix_indexes():
    assert list(islice(storage_runs((10**12, 2), (3, 1), max_elements=4), 3)) == [
        (0, 2),
        (3, 2),
        (6, 2),
    ]
    assert list(storage_runs((), (), max_elements=4)) == [(0, 1)]
    assert not list(storage_runs((0, 5), (5, 1), max_elements=4))


def test_tensor_copy_allocates_only_its_fixed_cpu_chunk(monkeypatch):
    original = torch.empty
    allocations = []

    def empty(size, **kwargs):
        allocations.append((size, kwargs))
        assert size * kwargs["dtype"].itemsize <= 32 and kwargs["device"] == "cpu"
        return original(size, **kwargs)

    value = torch.arange(10000, dtype=torch.float32).reshape(100, 100).T
    monkeypatch.setattr(torch, "empty", empty)
    assert copied(value) == value.contiguous().view(torch.uint8).numpy().tobytes()
    assert len(allocations) == 1


def test_byte_producer_and_numpy_writer_use_the_identical_existing_frame():
    value = np.arange(30000, dtype=np.float32).reshape(100, 300).T
    field = ("inputs", "output")
    frame = InputFrame({field: value.shape}, {field: "float32"})
    budget = BufferBudget(frame.write_bytes, max_requests=1)
    direct, chunks = io.BytesIO(), io.BytesIO()
    expected = frame.write(direct, {field: value}, budget, "artifact")
    actual = frame.write_chunks(
        chunks,
        lambda path: host_chunks(
            torch.from_numpy(value), chunk_bytes=FRAME_CHUNK_BYTES
        ),
        budget,
        "artifact",
    )
    assert direct.getvalue() == chunks.getvalue() and actual == expected
    assert budget.used == (0, 0)


def test_frame_reserves_scratch_before_starting_a_producer():
    field = ("inputs", "output")
    frame = InputFrame({field: (4,)}, {field: "float32"})
    budget = BufferBudget(frame.write_bytes, max_requests=1)

    def forbidden(path):
        pytest.fail("unadmitted producer started")

    with budget.reserve(frame.write_bytes), pytest.raises(TimeoutError):
        frame.write_chunks(io.BytesIO(), forbidden, budget, "artifact")
    assert budget.used == (0, 0)


@pytest.mark.parametrize("failure", ["short", "long", "empty", "strided", "write"])
def test_frame_producer_failure_closes_scratch_before_releasing_budget(failure):
    field = ("inputs", "output")
    frame = InputFrame({field: (4,)}, {field: "float32"})
    budget = BufferBudget(frame.write_bytes, max_requests=1)
    closed = []

    def chunks(path):
        try:
            assert budget.used[0] == frame.write_bytes
            yield {
                "short": memoryview(b"0" * 4),
                "long": memoryview(b"0" * 17),
                "empty": memoryview(b""),
                "strided": memoryview(b"0" * 32)[::2],
                "write": memoryview(b"0" * 16),
            }[failure]
        finally:
            assert budget.used[0] == frame.write_bytes
            closed.append(True)

    class Broken(io.BytesIO):
        def write(self, value):
            if self.tell() >= 16 + len(frame.header):
                raise OSError("output disk full")
            return super().write(value)

    stream = Broken() if failure == "write" else io.BytesIO()
    with pytest.raises(OSError if failure == "write" else ValueError):
        frame.write_chunks(stream, chunks, budget, "artifact")
    assert closed == [True] and budget.used == (0, 0)


@pytest.mark.cuda
def test_cuda_copy_makes_no_contiguous_device_temporary():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    base = torch.arange(30000, device="cuda", dtype=torch.float32).reshape(100, 300)
    for value in (base, base.T, base[1::3, 2::5], base[0, :1].expand(10, 3)):
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        actual = copied(value, FRAME_CHUNK_BYTES)
        assert torch.cuda.max_memory_allocated() == before
        expected = value.cpu().contiguous().view(torch.uint8).numpy().tobytes()
        assert actual == expected
