"""Prepared output slices reuse the existing spans and bounded binary format."""

import io
from math import prod

import numpy as np
import pytest

from shadowspill.training.inputs.frame_reader import FrameReader, FrameRegion
from shadowspill.training.inputs.frames import FRAME_CHUNK_BYTES, InputFrame
from shadowspill_client.buffers import BufferBudget


def encoded(array):
    fields = {
        ("inputs", "a"): np.asarray(3.5, dtype=np.float64),
        ("inputs", "values"): array,
        ("loss_inputs", "z"): np.asarray(False),
    }
    frame = InputFrame(
        {k: v.shape for k, v in fields.items()},
        {k: v.dtype.name for k, v in fields.items()},
    )
    stream = io.BytesIO()
    frame.write(stream, fields, BufferBudget(2 << 20), "frame")
    return frame, stream.getvalue()


@pytest.mark.parametrize(
    "shape,axes",
    [
        ((), ()),
        ((3, 5), (slice(None), slice(1, 4))),
        ((3, 5), (1, slice(None))),
        ((3, 5), (slice(None), 2)),
        ((2, 3, 5), (slice(None), 1, slice(2, 5))),
        ((3, 5), (slice(2, 2), slice(None))),
        ((0, 5), (slice(None), slice(None))),
        ((2, 100000), (slice(None), slice(7, 99997))),
    ],
)
def test_region_stream_matches_numpy_without_full_tensor_decoding(shape, axes):
    array = np.arange(prod(shape), dtype=np.int64).reshape(shape)
    frame, raw = encoded(array)
    seen = []

    class Partial(io.BytesIO):
        def readinto(self, view):
            assert len(view) <= FRAME_CHUNK_BYTES
            seen.append(len(view))
            return super().readinto(view[:997])

    memory = BufferBudget(2 << 20, 2)
    with FrameReader(Partial(raw), frame, memory) as reader:
        before = memory.used
        chunks = reader.chunks(("inputs", "values"), axes)
        actual = b"".join(bytes(chunk) for chunk in chunks)
        assert actual == array[axes].tobytes(order="C")
        assert memory.used == before and before[0] < 100000
        scalar = b"".join(reader.chunks(("inputs", "a")))
        assert np.frombuffer(scalar, dtype=np.float64).item() == 3.5
    assert memory.used == (0, 0) and seen


@pytest.mark.parametrize(
    "axes",
    [
        (slice(None),),
        (-1, 0),
        (0, 5),
        (True, 0),
        (slice(None, None, 2), 0),
        (slice(-1, 3), 0),
        (slice(0, 4), 0),
        (slice(2, 1), 0),
    ],
)
def test_noncanonical_or_outside_selections_are_rejected(axes):
    with pytest.raises((TypeError, ValueError)):
        FrameRegion.select((3, 5), axes)


def test_active_iterator_owns_scratch_and_refuses_concurrent_reads_or_close():
    frame, raw = encoded(np.arange(100000, dtype=np.int64))
    memory = BufferBudget(2 << 20, 2)
    with FrameReader(io.BytesIO(raw), frame, memory) as reader:
        chunks = reader.chunks(("inputs", "values"))
        view = next(chunks)
        assert view.readonly and len(view) == FRAME_CHUNK_BYTES
        with pytest.raises(RuntimeError, match="borrow"):
            next(reader.chunks(("inputs", "a")))
        with pytest.raises(RuntimeError, match="borrows"):
            reader.close()
        del view
        chunks.close()
        assert memory.used[0] > 0
    assert memory.used == (0, 0)


@pytest.mark.parametrize("damage", ["length", "header", "late_truncation"])
def test_prepared_abi_and_truncated_verified_stream_fail_with_owned_cleanup(damage):
    frame, raw = encoded(np.arange(3, dtype=np.int64))
    if damage == "length":
        raw = raw[:-1]
    elif damage == "header":
        raw = b"INVALID!" + raw[8:]
    memory = BufferBudget(2 << 20, 2)
    stream = io.BytesIO(raw)
    with (
        pytest.raises((ValueError, OSError)),
        FrameReader(stream, frame, memory) as reader,
    ):
        assert damage == "late_truncation"
        stream.truncate(len(raw) - 10)
        b"".join(reader.chunks(("inputs", "values")))
    assert memory.used == (0, 0) and not stream.closed


def test_scratch_is_admitted_before_touching_the_stream():
    frame, raw = encoded(np.arange(3, dtype=np.int64))

    class NeverRead(io.BytesIO):
        def seek(self, *args):
            pytest.fail("unadmitted output read")

    memory = BufferBudget(FRAME_CHUNK_BYTES - 1)
    with pytest.raises(ValueError, match="staging bytes"):
        FrameReader(NeverRead(raw), frame, memory)
    assert memory.used == (0, 0)


def test_bfloat16_selections_preserve_bits_without_a_framework_or_numeric_cast():
    path = ("inputs", "values")
    bits = np.asarray([[0x3F80, 0x7FC1, 0x8000], [0xFF80, 0x0001, 0x7F80]], np.uint16)
    frame = InputFrame({path: bits.shape}, {path: "bfloat16"})
    stream = io.BytesIO()
    frame.write(stream, {path: bits}, BufferBudget(2 << 20), "bf16")
    memory = BufferBudget(2 << 20)
    with FrameReader(stream, frame, memory) as reader:
        actual = b"".join(
            bytes(chunk) for chunk in reader.chunks(path, (slice(None), slice(1, 3)))
        )
        assert actual == bits[:, 1:].tobytes()
    assert memory.used == (0, 0)
