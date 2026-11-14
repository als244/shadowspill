"""Serial bounded views into an already verified, fixed binary tensor frame."""

from __future__ import annotations

import struct
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from math import prod
from threading import RLock
from typing import BinaryIO, Self, cast

from shadowspill.artifacts.tensor_spans import spans
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.payload import MAGIC
from shadowspill_client.protocol.payload_io import BufferReader
from shadowspill_client.protocol.tensors import storage_dtype, tensor_bytes

from .frames import FRAME_CHUNK_BYTES, InputFrame, Path


@dataclass(frozen=True)
class FrameRegion:
    """A unit-step selection in a C-ordered file, in logical element coordinates."""

    shape: tuple[int, ...]
    stride: tuple[int, ...]
    offset: int

    @classmethod
    def select(cls, shape: tuple[int, ...], axes: Sequence[int | slice]) -> FrameRegion:
        if len(axes) != len(shape):
            raise ValueError("frame selection must describe every physical axis")
        selected, strides, offset = [], [], 0
        for axis, (size, choice) in enumerate(zip(shape, axes, strict=True)):
            stride = prod(shape[axis + 1 :])
            if type(choice) is int:
                if not 0 <= choice < size:
                    raise ValueError("frame integer selection is outside its axis")
                offset += choice * stride
            elif isinstance(choice, slice):
                start = 0 if choice.start is None else choice.start
                stop = size if choice.stop is None else choice.stop
                if (
                    type(start) is not int
                    or type(stop) is not int
                    or not 0 <= start <= stop <= size
                    or (choice.step is not None and type(choice.step) is not int)
                    or choice.step not in (None, 1)
                ):
                    raise ValueError("frame slices require bounded unit-step axes")
                offset += start * stride
                selected.append(stop - start)
                strides.append(stride)
            else:
                raise TypeError("frame axes require integers or slices")
        return cls(tuple(selected), tuple(strides), offset)

    def spans(self) -> Iterator[tuple[int, int, int]]:
        return spans(self.shape, self.stride, self.offset)


class FrameReader:
    """Read saved tensor regions without decoding a complete worker output.

    The caller owns the seekable stream and must verify its artifact checksum
    before constructing this reader. WorkerResults.open enforces that boundary.
    This reader additionally checks the exact prepared header and file length.
    One scratch block serves serial reads; yielded read-only views are borrowed
    only until advancing or closing their iterator and must not be retained.
    """

    def __init__(self, stream: BinaryIO, frame: InputFrame, budget: BufferBudget):
        self._lease = budget.reserve(
            4096 + 8 * frame.size.header + frame.size.tree + FRAME_CHUNK_BYTES,
            timeout=0,
        )
        self._stream, self.frame = stream, frame
        self._mutex = RLock()
        self._active = self.closed = False
        self._offsets: dict[Path, int] = {}
        self._scratch = bytearray()
        try:
            if not callable(getattr(stream, "readinto", None)):
                raise TypeError("frame streams require bounded readinto support")
            self._scratch = bytearray(FRAME_CHUNK_BYTES)
            if stream.seek(0, 2) != frame.size.wire:
                raise ValueError("frame file length differs from its prepared ABI")
            stream.seek(0)
            for expected in (
                MAGIC + struct.pack("<Q", len(frame.header)),
                frame.header,
            ):
                view = memoryview(expected)
                for start in range(0, len(view), FRAME_CHUNK_BYTES):
                    piece = view[start : start + FRAME_CHUNK_BYTES]
                    if self._read(len(piece)) != piece:
                        raise ValueError("frame header differs from its prepared ABI")
            offset = 16 + len(frame.header)
            for path, shape in frame.shapes.items():
                self._offsets[path] = offset
                offset += tensor_bytes(frame.dtypes[path], shape)
        except BaseException:
            self._scratch = bytearray()
            self._offsets.clear()
            self._lease.close()
            raise

    def _read(self, size: int) -> memoryview:
        view = memoryview(self._scratch)[:size]
        offset = 0
        try:
            while offset < size:
                target = view[offset:]
                try:
                    count = cast(BufferReader, self._stream).readinto(target)
                finally:
                    target.release()
                if type(count) is not int or not 0 < count <= size - offset:
                    raise OSError("verified frame became truncated during reading")
                offset += count
            return view.toreadonly()
        finally:
            # A retained I/O traceback must not retain our scratch after close.
            view.release()

    def chunks(
        self, path: Path, axes: Sequence[int | slice] | None = None
    ) -> Iterator[memoryview]:
        """Borrow byte chunks of a selected tensor, in its logical C order."""
        with self._mutex:
            if self.closed or self._active:
                raise RuntimeError("frame reader is closed or already borrowed")
            shape = self.frame.shapes[path]
            region = FrameRegion.select(
                shape, tuple(slice(None) for _ in shape) if axes is None else axes
            )
            itemsize = storage_dtype(self.frame.dtypes[path]).itemsize
            start = self._offsets[path]
            self._active = True
        try:
            for _, physical, count in region.spans():
                self._stream.seek(start + physical * itemsize)
                remaining = count * itemsize
                while remaining:
                    size = min(remaining, FRAME_CHUNK_BYTES)
                    yield self._read(size)
                    remaining -= size
        finally:
            with self._mutex:
                self._active = False

    def close(self) -> None:
        with self._mutex:
            if self.closed:
                return
            if self._active:
                raise RuntimeError("frame iterator still borrows its scratch")
            self._scratch = bytearray()
            self._offsets.clear()
            self._lease.close()
            self.closed = True

    def __enter__(self) -> Self:
        if self.closed:
            raise RuntimeError("frame reader is closed")
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
