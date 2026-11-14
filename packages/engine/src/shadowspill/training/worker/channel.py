"""Bounded private control framing over inherited pipes or authenticated streams."""

import struct
from typing import BinaryIO

from shadowspill_client._records import Record
from shadowspill_client.protocol.codec import MAX_METADATA_BYTES, ProtocolError

from .protocol import CODEC

_PREFIX = struct.Struct("!I")
_CHUNK = 64 << 10


def _read(stream: BinaryIO, size: int, *, eof: bool = False) -> bytes | None:
    result = bytearray(size)
    offset = 0
    while offset < size:
        block = stream.read(min(size - offset, _CHUNK))
        if not isinstance(block, bytes) or len(block) > min(size - offset, _CHUNK):
            raise ProtocolError("worker channel returned an invalid read")
        if not block:
            if eof and not offset:
                return None
            raise ProtocolError("worker channel ended inside a control record")
        result[offset : offset + len(block)] = block
        offset += len(block)
    return bytes(result)


def read_message(stream: BinaryIO) -> Record | None:
    prefix = _read(stream, _PREFIX.size, eof=True)
    if prefix is None:
        return None
    (size,) = _PREFIX.unpack(prefix)
    if not 0 < size <= MAX_METADATA_BYTES:
        raise ProtocolError("worker metadata length exceeds the control record bound")
    payload = _read(stream, size)
    assert payload is not None
    return CODEC.decode(payload)


def write_message(stream: BinaryIO, value: Record) -> None:
    encoded = CODEC.encode(value)
    for value_bytes in (_PREFIX.pack(len(encoded)), encoded):
        view = memoryview(value_bytes)
        while view:
            written = stream.write(view[:_CHUNK])
            if type(written) is not int or not 0 < written <= min(len(view), _CHUNK):
                raise ProtocolError("worker channel made no valid write progress")
            view = view[written:]
    stream.flush()
