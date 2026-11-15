"""Bound archive structure before stdlib TAR/JSON parsers allocate metadata."""

from __future__ import annotations

import os
import re
import tarfile
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from typing import BinaryIO

BLOCK = 512
MANIFEST_LIMIT = 16 << 20
MEMBER_LIMIT = 65538  # Tensor members plus manifest and its checksum.
MANIFEST_NODES = 1 << 20
MANIFEST_DEPTH = 64


def check_headers(
    stream: BinaryIO, *, max_manifest_bytes: int = MANIFEST_LIMIT
) -> None:
    """Inspect fixed headers only, without loading tensor payloads or extensions.

    The rank writer emits ordinary uncompressed GNU TAR members. Long-name,
    sparse and PAX extension records are unnecessary and rejected before TarFile
    can expand them. GNU base-256 sizes remain supported for large tensors.
    """
    if type(max_manifest_bytes) is not int or max_manifest_bytes < 0:
        raise ValueError("checkpoint manifest byte limit must be nonnegative")
    original = stream.tell()
    try:
        size = stream.seek(0, os.SEEK_END)
        offset = 0
        names: set[str] = set()
        while True:
            stream.seek(offset)
            block = stream.read(BLOCK)
            if len(block) != BLOCK:
                raise ValueError("checkpoint archive is truncated")
            if block == bytes(BLOCK):
                remaining = size - offset
                if not 2 * BLOCK <= remaining <= tarfile.RECORDSIZE:
                    raise ValueError("invalid checkpoint archive end padding")
                tail = stream.read(remaining - BLOCK)
                if len(tail) != remaining - BLOCK or any(tail):
                    raise ValueError("invalid checkpoint archive end padding")
                break
            if len(names) >= MEMBER_LIMIT:
                raise ValueError("checkpoint archive members exceed their count bound")
            try:
                info = tarfile.TarInfo.frombuf(block, "utf-8", "strict")
            except (tarfile.HeaderError, UnicodeError) as error:
                raise ValueError("invalid checkpoint archive members") from error
            if (
                info.type not in (tarfile.REGTYPE, tarfile.AREGTYPE)
                or info.name in names
                or info.size < 0
                or (
                    info.name not in ("manifest.json", "manifest.blake3")
                    and re.fullmatch(r"tensors/[0-9]+", info.name) is None
                )
            ):
                raise ValueError("invalid checkpoint archive members")
            if info.name == "manifest.json" and info.size > min(
                MANIFEST_LIMIT, max_manifest_bytes
            ):
                raise ValueError(
                    "checkpoint manifest exceeds 16 MiB or configured byte limit"
                )
            if info.name == "manifest.blake3" and info.size != 64:
                raise ValueError("invalid checkpoint manifest checksum")
            names.add(info.name)
            offset += BLOCK + info.size + (-info.size % BLOCK)
            if offset > size - 2 * BLOCK:
                raise ValueError("checkpoint archive payload is truncated")
        if not {"manifest.json", "manifest.blake3"} <= names:
            raise ValueError("checkpoint archive is missing its manifest")
    finally:
        stream.seek(original)


def check_json(data: bytes) -> None:
    """Count JSON nodes/depth before decoding; the JSON parser validates syntax."""
    nodes = depth = 0
    quoted = escaped = primitive = False
    for char in data:
        if quoted:
            if escaped:
                escaped = False
            elif char == 92:
                escaped = True
            elif char == 34:
                quoted = False
            continue
        if char == 34:
            nodes += 1
            quoted, primitive = True, False
        elif char in (91, 123):
            nodes += 1
            depth += 1
            primitive = False
            if depth > MANIFEST_DEPTH:
                raise ValueError("checkpoint manifest nesting exceeds its depth bound")
        elif char in (93, 125):
            depth -= 1
            primitive = False
        elif char in (9, 10, 13, 32, 44, 58):
            primitive = False
        elif not primitive:
            nodes += 1
            primitive = True
        if nodes > MANIFEST_NODES:
            raise ValueError("checkpoint manifest exceeds its JSON node bound")


@contextmanager
def open_checkpoint(
    source: str | os.PathLike[str] | BinaryIO,
    *,
    max_manifest_bytes: int = MANIFEST_LIMIT,
) -> Iterator[tarfile.TarFile]:
    """Validate rank framing; supplied seekable streams remain caller-owned."""
    with (
        open(source, "rb")
        if isinstance(source, (str, os.PathLike))
        else nullcontext(source)
    ) as stream:
        original = stream.tell()
        check_headers(stream, max_manifest_bytes=max_manifest_bytes)
        try:
            stream.seek(0)
            with tarfile.open(fileobj=stream, mode="r:") as archive:
                yield archive
        finally:
            stream.seek(original)
