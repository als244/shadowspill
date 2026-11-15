"""Immutable local grants for a trusted CPU preparation process."""

import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO

from shadowspill_client.protocol.requests import ArtifactRef


def file_digest(stream: BinaryIO) -> tuple[str, int]:
    digest, size = hashlib.sha256(), 0
    while block := stream.read(64 << 10):
        digest.update(block)
        size += len(block)
    return digest.hexdigest(), size


@contextmanager
def verified_file(reference: ArtifactRef, path: Path) -> Iterator[BinaryIO]:
    from .reading import _signature

    with path.open("rb") as stream:
        before = _signature(stream)
        if file_digest(stream) != (reference.sha256, reference.size_bytes):
            raise ValueError(
                "checkpoint dependency differs from its immutable checksum"
            )
        if _signature(stream) != before:
            raise ValueError("checkpoint dependency changed during verification")
        stream.seek(0)
        yield stream
        if _signature(stream) != before:
            raise ValueError("checkpoint dependency changed during conversion")
