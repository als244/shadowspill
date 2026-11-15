"""Bounded binary output inside one already released private worker invocation."""

from collections.abc import Buffer
from hashlib import sha256
from typing import TYPE_CHECKING, BinaryIO, Protocol, Self, cast

from shadowspill_client._records import integer
from shadowspill_client.protocol.codec import ProtocolError
from shadowspill_client.protocol.requests import ArtifactRef

from .channel import _CHUNK, _PREFIX, _read, _write

if TYPE_CHECKING:
    from shadowspill_client.protocol.payload_io import BufferReader

_BINARY = 1 << 31


class BinarySink(Protocol):
    def write(self, data: memoryview, /) -> int: ...


class WorkerStream:
    """Frame borrowed encoder bytes; no tensor format or complete host snapshot.

    Binary frames have a distinct prefix bit and at most 64 KiB of payload. A zero
    frame terminates the stream, not the invocation: its normal ticketed reply
    must still confirm completion. The stream borrows and never closes the channel.
    """

    def __init__(self, stream: BinaryIO, *, size_bytes: int) -> None:
        self.size_bytes = integer(size_bytes, "worker output bytes")
        self._stream, self._written = stream, 0
        self._hash = sha256()
        self.closed = self._failed = False

    def write(self, data: Buffer, /) -> int:
        if self.closed or self._failed:
            raise RuntimeError("worker output stream is closed or failed")
        with memoryview(data).cast("B") as view:
            if len(view) > self.size_bytes - self._written:
                raise ValueError("worker output exceeds its admitted byte count")
            try:
                for offset in range(0, len(view), _CHUNK):
                    with view[offset : offset + _CHUNK] as part:
                        _write(self._stream, _PREFIX.pack(_BINARY | len(part)))
                        _write(self._stream, part)
                        self._hash.update(part)
                        self._written += len(part)
            except BaseException:
                self._failed = True
                raise
            return len(view)

    def finish(self) -> None:
        self._finish(successful=True)

    def _finish(self, *, successful: bool) -> None:
        if self.closed:
            if self._failed and successful:
                raise RuntimeError("worker output stream failed")
            return
        if self._failed:
            if successful:
                raise RuntimeError("worker output stream failed")
            return  # An incomplete frame cannot be repaired by appending a marker.
        try:
            _write(self._stream, _PREFIX.pack(_BINARY))
            self._stream.flush()
        except BaseException:
            self._failed = True
            raise
        self.closed = True
        self._failed = not successful or self._written != self.size_bytes
        if successful and self._failed:
            raise ValueError("worker output ended before its admitted byte count")

    def reference(self, artifact_id: str, media_type: str) -> ArtifactRef:
        if not self.closed or self._failed:
            raise RuntimeError("worker output has not completed successfully")
        return ArtifactRef(
            artifact_id, self._hash.hexdigest(), self.size_bytes, media_type
        )

    def __enter__(self) -> Self:
        if self.closed or self._failed:
            raise RuntimeError("worker output stream is closed or failed")
        return self

    def __exit__(self, *exc: object) -> None:
        # A failed encoder may have emitted complete frames. Termination still
        # cannot establish success: the receiver requires exact bytes and reply.
        self._finish(successful=not exc or exc[0] is None)


def receive_stream(
    stream: BinaryIO, destination: BinarySink, *, size_bytes: int
) -> None:
    """Copy exactly the admitted bytes; the artifact owner verifies the digest."""
    remaining = integer(size_bytes, "worker output bytes")
    buffer: bytearray | None = None
    while True:
        prefix = _read(stream, _PREFIX.size)
        assert prefix is not None
        (frame,) = _PREFIX.unpack(prefix)
        size = frame & ~_BINARY
        if not frame & _BINARY or size > min(_CHUNK, remaining):
            raise ProtocolError(
                "worker binary frame exceeds its admitted bound or kind"
            )
        if not size:
            if remaining:
                raise ProtocolError(
                    "worker binary stream ended before its declared size"
                )
            return
        if buffer is None:
            buffer = bytearray(min(_CHUNK, remaining))
        with memoryview(buffer)[:size] as view:
            offset = 0
            while offset < size:
                with view[offset:] as part:
                    # BinaryIO's typing stub omits readinto; private pipe/socket
                    # streams provide the same buffer API as input transfer.
                    count = cast("BufferReader", stream).readinto(part)
                if type(count) is not int or not 0 < count <= size - offset:
                    raise ProtocolError(
                        "worker binary frame read made no valid progress"
                    )
                offset += count
            offset = 0
            while offset < size:
                with view[offset:] as part:
                    count = destination.write(part)
                if type(count) is not int or not 0 < count <= size - offset:
                    raise ProtocolError(
                        "worker binary destination made no write progress"
                    )
                offset += count
        remaining -= size
