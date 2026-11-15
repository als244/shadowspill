"""Verified, bounded source reads for offline checkpoint transformations."""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import replace
from types import MappingProxyType
from typing import Any, BinaryIO

import torch

from shadowspill.artifacts._values import freeze, nonnegative_int
from shadowspill.artifacts.tensors import TensorChunk

from .frozen import validate_restore
from .inspection import open_checkpoint
from .load import _manifest, _Reader
from .state import CheckpointState
from .tensors import byte_view, geometry


class RankReader:
    """Read saved representations after complete geometry and payload validation.

    The owner keeps the source immutable and publishes derived artifacts only
    after the enclosing ``read_rank`` context exits successfully. This object
    borrows its file; it owns no full-tensor payload or destination buffer and
    performs no weight reconstruction, dtype cast or replica selection.
    """

    def __init__(
        self,
        reader: _Reader,
        chunks: Mapping[str, TensorChunk],
        manifest: Mapping[str, Any],
        chunk_bytes: int,
    ) -> None:
        self._reader = reader
        self._chunks = MappingProxyType(dict(chunks))
        self._manifest: Mapping[str, Any] = freeze(manifest)
        self.chunk_bytes = chunk_bytes
        self.closed = False

    @property
    def manifest(self) -> Mapping[str, Any]:
        return self._manifest

    @property
    def chunks(self) -> Mapping[str, TensorChunk]:
        """Only stored representations, excluding derived and omitted tensors."""
        return self._chunks

    def readinto(self, request: TensorChunk, destination: torch.Tensor) -> None:
        """Fill one bounded physical subregion, in any order, without casting."""
        if self.closed:
            raise ValueError("checkpoint source reader is closed")
        expected = self._chunks.get(request.name)
        if expected is None or geometry(expected) != geometry(request):
            raise ValueError("checkpoint source request geometry differs")
        nonnegative_int(request.offset, "checkpoint source offset")
        view = byte_view(destination)
        entry = self._reader.entries[request.name]
        begin = (request.offset - request.region_offset) * request.dtype.itemsize
        if (
            destination.dtype != request.dtype
            or view.nbytes > self.chunk_bytes
            or begin < 0
            or begin + view.nbytes > entry["bytes"]
        ):
            raise ValueError("checkpoint source request exceeds its region or budget")
        self._reader.stream.seek(
            self._reader.archive.getmember(entry["member"]).offset_data + begin
        )
        if self._reader.stream.readinto(view) != view.nbytes:
            raise ValueError("checkpoint source tensor is truncated")


def _signature(stream: Any) -> tuple[int, ...]:
    value = os.fstat(stream.fileno())
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


@contextmanager
def read_rank(
    state: CheckpointState,
    path: str | os.PathLike[str] | BinaryIO,
    *,
    chunk_bytes: int,
    frozen_state_id: str | None = None,
) -> Iterator[RankReader]:
    """Verify every stored byte before allowing random subregion reads.

    The existing archive parser, geometry checks and streaming content hashes
    remain authoritative. Verification owns at most ``chunk_bytes`` of payload
    scratch, freed before yielding; subsequent reads borrow bounded destinations.
    Metadata retains the archive's existing count, byte and JSON bounds. The
    caller separately admits this metadata and any destination staging.

    Local file changes are detected again on successful exit. Borrowed streams
    require a real file descriptor for the same immutability check and remain
    open at their original position. Source immutability is an owner requirement;
    this is not a locking or publication service.
    """
    if type(chunk_bytes) is not int or chunk_bytes < 1:
        raise ValueError("checkpoint source chunk_bytes must be a positive integer")
    with open_checkpoint(path) as archive:
        initial = _signature(archive.fileobj)
        manifest = _manifest(archive, state)
        omitted = validate_restore(state, manifest, frozen_state_id)
        chunks = {chunk.name: chunk for chunk in state.tensor_chunks()}
        reader = _Reader(archive, manifest, chunks, omitted)
        stored = {name: chunks[name] for name in reader.entries}
        if any(chunk.dtype.itemsize > chunk_bytes for chunk in stored.values()):
            raise ValueError("checkpoint source chunk budget cannot hold one element")
        width = min(
            chunk_bytes, max((e["bytes"] for e in reader.entries.values()), default=0)
        )
        scratch = torch.empty(width, dtype=torch.uint8, device="cpu")
        for name, chunk in stored.items():
            entry = reader.entries[name]
            step = chunk_bytes // chunk.dtype.itemsize
            elements = entry["bytes"] // chunk.dtype.itemsize
            for start in range(0, elements, step):
                count = min(step, elements - start)
                part = scratch[: count * chunk.dtype.itemsize].view(chunk.dtype)
                reader.fill(replace(chunk, offset=chunk.region_offset + start), part)
                del part
        del scratch
        if _signature(archive.fileobj) != initial:
            raise ValueError("checkpoint source changed during verification")
        source = RankReader(reader, stored, manifest, chunk_bytes)
        try:
            yield source
            if _signature(archive.fileobj) != initial:
                raise ValueError("checkpoint source changed during conversion")
        finally:
            source.closed = True
