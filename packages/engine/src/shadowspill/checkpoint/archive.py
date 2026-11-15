"""Uncompressed TAR framing: payload integrity belongs to one content hash."""

from __future__ import annotations

import json
import tarfile
from collections.abc import Buffer
from typing import Any, Protocol

from blake3 import blake3

BLOCK = 512
MANIFEST_LIMIT = 16 << 20


class WriteSink(Protocol):
    def write(self, data: Buffer, /) -> int: ...


class WriteAll:
    """A borrowed blocking sink; partial writes never drop checkpoint bytes."""

    def __init__(self, stream: WriteSink):
        self.stream = stream

    def write(self, data: Buffer, /) -> int:
        with memoryview(data).cast("B") as view:
            offset = 0
            while offset < len(view):
                with view[offset:] as remaining:
                    count = self.stream.write(remaining)
                    if type(count) is not int or not 0 < count <= len(remaining):
                        raise OSError("checkpoint sink made no valid write progress")
                    offset += count
            return offset


def header(stream: WriteSink, name: str, size: int) -> None:
    info = tarfile.TarInfo(name)
    info.size = size
    # GNU numeric fields accommodate individual tensors larger than 8 GiB.
    stream.write(info.tobuf(format=tarfile.GNU_FORMAT))


def padding(stream: WriteSink, size: int) -> None:
    stream.write(b"\0" * (-size % BLOCK))


def finish(stream: WriteSink, manifest: bytes) -> None:
    if len(manifest) > MANIFEST_LIMIT:
        raise ValueError("checkpoint manifest exceeds 16 MiB")
    for name, data in (
        ("manifest.json", manifest),
        ("manifest.blake3", blake3(manifest).hexdigest().encode("ascii")),
    ):
        header(stream, name, len(data))
        stream.write(data)
        padding(stream, len(data))
    stream.write(b"\0" * (2 * BLOCK))


def read_manifest(archive: tarfile.TarFile) -> dict[str, Any]:
    members = archive.getmembers()
    if len({info.name for info in members}) != len(members) or any(
        not info.isfile() or info.issparse() for info in members
    ):
        raise ValueError("invalid checkpoint archive members")
    info = archive.getmember("manifest.json")
    checksum = archive.getmember("manifest.blake3")
    if not 0 <= info.size <= MANIFEST_LIMIT:
        raise ValueError("checkpoint manifest exceeds 16 MiB")
    if checksum.size != 64:
        raise ValueError("invalid checkpoint manifest checksum")
    manifest_file, checksum_file = (
        archive.extractfile(info),
        archive.extractfile(checksum),
    )
    assert manifest_file is not None and checksum_file is not None
    with manifest_file, checksum_file:
        data = manifest_file.read()
        if blake3(data).hexdigest().encode("ascii") != checksum_file.read():
            raise ValueError("checkpoint manifest checksum differs")
    result = json.loads(data)
    if not isinstance(result, dict):
        raise ValueError("invalid checkpoint manifest")
    return result


class ContentHash:
    """Reuse BLAKE3's worker pool across tensors, without parallel tiny updates."""

    def __init__(self) -> None:
        self.small = blake3()
        self.large: blake3 | None = None
        self.current = self.small

    def reset(self, size: int) -> None:
        if size >= 1 << 20:
            if self.large is None:
                self.large = blake3(max_threads=4)
            self.current = self.large
        else:
            self.current = self.small
        self.current.reset()

    def update(self, data: memoryview) -> None:
        self.current.update(data)

    def hexdigest(self) -> str:
        return self.current.hexdigest()
