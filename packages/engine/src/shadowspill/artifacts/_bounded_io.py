"""Seekable archive writes constrained by an already reserved disk allowance."""

import io
from collections.abc import Buffer
from typing import BinaryIO


class _BoundedWriter(io.BufferedIOBase):
    def __init__(self, stream: BinaryIO, limit: int) -> None:
        self._stream, self._limit = stream, limit

    def write(self, buffer: Buffer, /) -> int:
        data = memoryview(buffer)
        if self.tell() + data.nbytes > self._limit:
            raise ValueError("archive exceeds its reserved disk envelope")
        return self._stream.write(data)

    def tell(self) -> int:
        return self._stream.tell()

    def seek(self, offset: int, whence: int = 0) -> int:
        return self._stream.seek(offset, whence)

    def flush(self) -> None:
        self._stream.flush()
