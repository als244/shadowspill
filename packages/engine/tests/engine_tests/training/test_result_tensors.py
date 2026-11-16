"""Streamed caller-region writes and explicit two-stage output reductions."""

import io

import numpy as np
import pytest

from engine_tests.training.test_frame_reader import encoded
from shadowspill.training.inputs.frame_reader import FrameReader
from shadowspill.training.inputs.frames import FRAME_CHUNK_BYTES
from shadowspill.training.result_tensors import (
    REDUCTION_SCRATCH_BYTES,
    write_tensor_region,
)
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.inputs.values import _convert


def pieces(value):
    raw = memoryview(value).cast("B")
    try:
        for start in range(0, raw.nbytes, FRAME_CHUNK_BYTES):
            yield raw[start : start + FRAME_CHUNK_BYTES]
    finally:
        raw.release()


def write(stream, shape, value, memory, *, dtype="float32", axes=None, reduction=None):
    write_tensor_region(
        stream,
        offset=7,
        shape=shape,
        dtype=dtype,
        axes=tuple(slice(None) for _ in shape) if axes is None else axes,
        chunks=pieces(value),
        budget=memory,
        reduction=reduction,
    )


class Partial(io.BytesIO):
    def read(self, size=-1):
        assert 0 < size <= FRAME_CHUNK_BYTES
        return super().read(min(811, size))

    def write(self, data):
        assert 0 < len(data) <= FRAME_CHUNK_BYTES
        return super().write(data[:919])


@pytest.mark.parametrize("reduction", [None, "sum", "min", "max", "identical"])
def test_strided_source_and_destination_preserve_guard_bytes_under_partial_io(
    reduction,
):
    source = np.arange(2 * 100000, dtype=np.float32).reshape(2, 100000)
    frame, wire = encoded(source)
    expected = np.full((2, 120000), 123, dtype=np.float32)
    selected = source[:, 9:99999]
    target = expected[:, 3:99993]
    if reduction in (None, "identical"):
        target[:] = selected
    elif reduction == "sum":
        target += selected
    elif reduction == "min":
        np.minimum(target, selected, out=target)
    else:
        np.maximum(target, selected, out=target)
    initial = np.full_like(expected, 123)
    if reduction == "identical":
        initial[:, 3:99993] = selected
    destination = Partial(b"guard!!" + initial.tobytes() + b"untouched")
    memory = BufferBudget(4 << 20, 4)
    with FrameReader(io.BytesIO(wire), frame, memory) as reader:
        write_tensor_region(
            destination,
            offset=7,
            shape=initial.shape,
            dtype="float32",
            axes=(slice(None), slice(3, 99993)),
            chunks=reader.chunks(("inputs", "values"), (slice(None), slice(9, 99999))),
            budget=memory,
            reduction=reduction,
        )
        assert not reader._active
    assert memory.used == (0, 0)
    assert destination.getvalue() == b"guard!!" + expected.tobytes() + b"untouched"


def test_worker_sum_is_completed_before_global_buffer_maximum():
    memory = BufferBudget(4 << 20, 4)
    global_output = io.BytesIO(b"guard!!" + bytes(4))
    for index, (left, right) in enumerate(((10, 40), (30, 30))):
        buffer = io.BytesIO(b"guard!!" + bytes(4))
        write(buffer, (), np.asarray(left, np.float32), memory)
        write(buffer, (), np.asarray(right, np.float32), memory, reduction="sum")
        combined = np.frombuffer(buffer.getvalue(), dtype=np.float32, offset=7)
        write(global_output, (), combined, memory, reduction="max" if index else None)
    assert np.frombuffer(global_output.getvalue(), np.float32, offset=7).item() == 60
    assert memory.used == (0, 0)


@pytest.mark.parametrize(
    "dtype", ["float16", "float32", "float64", "int32", "int64", "bool", "bfloat16"]
)
@pytest.mark.parametrize("reduction", ["sum", "min", "max"])
def test_reductions_preserve_declared_storage_dtype(dtype, reduction):
    a = np.asarray([1.5, 3.0, -1.0], np.float32)
    b = np.asarray([2.0, -1.0, 4.0], np.float32)
    left, right = _convert(a, dtype), _convert(b, dtype)
    memory = BufferBudget(4 << 20, 4)
    destination = io.BytesIO(b"guard!!" + left.tobytes())
    write(destination, (3,), right, memory, dtype=dtype, reduction=reduction)
    operation = {"sum": np.add, "min": np.minimum, "max": np.maximum}[reduction]
    expected = (
        _convert(operation(a, b), dtype)
        if dtype == "bfloat16"
        else operation(left, right)
    )
    assert destination.getvalue() == b"guard!!" + expected.tobytes()
    assert memory.used == (0, 0)


def test_identical_requires_exact_encoded_values_and_failure_releases_active_reader():
    frame, wire = encoded(np.asarray([1, 2], np.int64))
    memory = BufferBudget(4 << 20, 4)
    with FrameReader(io.BytesIO(wire), frame, memory) as reader:
        before = memory.used
        destination = io.BytesIO(b"guard!!" + np.asarray([1, 3], np.int64).tobytes())
        with pytest.raises(ValueError, match="disagree"):
            write_tensor_region(
                destination,
                offset=7,
                shape=(2,),
                dtype="int64",
                axes=(slice(None),),
                chunks=reader.chunks(("inputs", "values")),
                budget=memory,
                reduction="identical",
            )
        assert memory.used == before and not reader._active
    assert memory.used == (0, 0)


@pytest.mark.parametrize(
    "issue", ["short", "excess", "oversized", "unaligned", "no_space", "small_file"]
)
def test_bad_producers_and_capacity_errors_never_leave_an_active_borrow(issue):
    closed = []

    def source():
        try:
            size = {
                "short": 4,
                "excess": 12,
                "oversized": FRAME_CHUNK_BYTES + 4,
                "unaligned": 7,
                "no_space": 8,
                "small_file": 8,
            }[issue]
            yield memoryview(bytes(size))
        finally:
            closed.append(True)

    memory = BufferBudget(
        REDUCTION_SCRATCH_BYTES - 1 if issue == "no_space" else 4 << 20
    )
    target = io.BytesIO(b"guard!!" + bytes(1 if issue == "small_file" else 8))
    producer = source()
    with pytest.raises(ValueError):
        write_tensor_region(
            target,
            offset=7,
            shape=(2,),
            dtype="float32",
            axes=(slice(None),),
            chunks=producer,
            budget=memory,
        )
    assert memory.used == (0, 0)
    assert producer.gi_frame is None
    assert closed == ([] if issue in ("no_space", "small_file") else [True])


def test_empty_region_requires_no_source_elements_and_keeps_other_bytes():
    target = io.BytesIO(b"guard!!" + np.arange(6, dtype=np.float32).tobytes())
    before = target.getvalue()
    memory = BufferBudget(4 << 20)
    write(
        target,
        (2, 3),
        np.empty((0,), np.float32),
        memory,
        axes=(slice(1, 1), slice(None)),
    )
    assert target.getvalue() == before and memory.used == (0, 0)
