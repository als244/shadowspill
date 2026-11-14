"""Bounded binary Datum decoding; no pickle, dynamic imports or tensor JSON."""

from __future__ import annotations

import hashlib
import hmac
import json
import struct
from collections.abc import Callable
from typing import Any, Protocol

from ..buffers import BufferBudget
from .codec import MAX_METADATA_BYTES, ProtocolError, _constant, _depth, _object
from .payload import MAGIC, MEDIA_TYPE, BatchSnapshot, owned_datums
from .requests import ArtifactRef
from .tensors import TensorPayload, tensor_bytes


class BufferReader(Protocol):
    def readinto(self, buffer: memoryview, /) -> int | None: ...


def _manifest(
    data: bytes,
) -> tuple[list[Any], list[tuple[str, tuple[int, ...], int]], int]:
    try:
        header = json.loads(data, object_pairs_hook=_object, parse_constant=_constant)
        _depth(header)
        if not isinstance(header, dict) or set(header) != {
            "version",
            "kind",
            "datums",
            "tensors",
        }:
            raise ValueError("invalid tensor manifest fields")
        if (
            type(header["version"]) is not int
            or header["version"] != 1
            or header["kind"] != "datum.tensors"
        ):
            raise ValueError("unsupported tensor manifest version/kind")
        if not isinstance(header["datums"], list) or not isinstance(
            header["tensors"], list
        ):
            raise ValueError("tensor manifest datums/tensors must be lists")
        layouts = []
        for item in header["tensors"]:
            if (
                not isinstance(item, dict)
                or set(item) != {"dtype", "shape"}
                or not isinstance(item["shape"], list)
                or not isinstance(item["dtype"], str)
            ):
                raise ValueError("invalid tensor layout")
            shape = tuple(item["shape"])
            layouts.append((item["dtype"], shape, tensor_bytes(item["dtype"], shape)))
        indices, metadata_size = [], 0

        def tree(value: Any) -> None:
            nonlocal metadata_size
            metadata_size += 64
            if isinstance(value, dict):
                if any(not key for key in value):
                    raise ValueError("empty tensor tree key")
                for key, child in sorted(value.items()):
                    metadata_size += len(key.encode("utf-8"))
                    tree(child)
            elif type(value) is int and 0 <= value < len(layouts):
                indices.append(value)
            else:
                raise ValueError("invalid tensor tree reference")

        for datum in header["datums"]:
            if not isinstance(datum, dict) or set(datum) != {"inputs", "loss_inputs"}:
                raise ValueError("invalid Datum trees")
            for name in ("inputs", "loss_inputs"):
                if not isinstance(datum[name], dict):
                    raise ValueError("Datum roots must be mappings")
                tree(datum[name])
        if indices != list(range(len(layouts))):
            raise ValueError(
                "tensor references must cover each payload exactly once in order"
            )
        if metadata_size > MAX_METADATA_BYTES:
            raise ValueError("input tree metadata exceeds its byte limit")
        return header["datums"], layouts, metadata_size
    except (TypeError, ValueError, RecursionError, UnicodeError) as error:
        raise ProtocolError(f"invalid tensor manifest: {error}") from error


def _read_into(
    stream: BufferReader,
    buffer: bytearray,
    chunk_bytes: int,
    update: Callable[[memoryview], object] | None = None,
) -> None:
    view, offset = memoryview(buffer), 0
    while offset < len(buffer):
        target = view[offset : offset + chunk_bytes]
        size = stream.readinto(target)
        if not size or not 0 < size <= len(target):
            raise ProtocolError("truncated binary tensor payload")
        if update is not None:
            update(target[:size])
        offset += size


def read_payload(
    stream: BufferReader,
    reference: ArtifactRef,
    budget: BufferBudget,
    *,
    chunk_bytes: int = 1 << 20,
    timeout: float | None = None,
) -> BatchSnapshot:
    """Read exactly the declared artifact frame, hashing each incoming chunk.

    The transport bounds the stream by Content-Length/artifact size. Validate
    dimensions and total length before reserving/allocating any tensor storage.
    Rejected or corrupt input never exposes a partially decoded Datum batch.
    """
    if type(chunk_bytes) is not int or chunk_bytes < 1:
        raise ValueError("chunk_bytes must be a positive integer")
    if reference.media_type != MEDIA_TYPE or reference.size_bytes < 16:
        raise ProtocolError("expected a versioned binary Datum artifact")
    prefix = bytearray(16)
    _read_into(stream, prefix, chunk_bytes)
    if prefix[:8] != MAGIC:
        raise ProtocolError("unsupported binary Datum framing")
    length = struct.unpack("<Q", prefix[8:])[0]
    if not 0 < length <= min(MAX_METADATA_BYTES, reference.size_bytes - 16):
        raise ProtocolError("tensor manifest exceeds its byte limit")
    header_buffer = bytearray(length)
    _read_into(stream, header_buffer, chunk_bytes)
    header = bytes(header_buffer)
    del header_buffer
    trees, layouts, metadata_size = _manifest(header)
    total = sum(item[2] for item in layouts)
    if 16 + length + total != reference.size_bytes:
        raise ProtocolError("tensor artifact size differs from declared layouts")
    retained = total + 8 * len(header) + metadata_size
    scratch = max((item[2] for item in layouts), default=0)
    lease = budget.reserve(retained + scratch, timeout=timeout)
    try:
        digest = hashlib.sha256(prefix)
        digest.update(header)
        tensors = []
        for dtype, shape, size in layouts:
            buffer = bytearray(size)
            _read_into(stream, buffer, chunk_bytes, digest.update)
            tensors.append(TensorPayload(dtype, shape, bytes(buffer)))
            del buffer
        if not hmac.compare_digest(digest.hexdigest(), reference.sha256):
            raise ProtocolError("binary tensor artifact checksum mismatch")
        result = BatchSnapshot(
            header, tensors, owned_datums(trees, tensors), reference.sha256, lease
        )
        lease.shrink(retained)
        return result
    except BaseException:
        lease.close()
        raise
