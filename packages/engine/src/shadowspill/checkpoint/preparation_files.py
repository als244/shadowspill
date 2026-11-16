"""Immutable local grants for a trusted CPU preparation process."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO

from shadowspill_client.protocol.requests import ArtifactRef

if TYPE_CHECKING:
    from shadowspill.training.worker.conversion_records import (
        ConversionPrepared,
        ConversionWritten,
    )

    from .conversion import ConvertedState


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


def write_ranks(
    converted: ConvertedState,
    description: ConversionPrepared,
    directory: Path,
    *,
    chunk_bytes: int,
) -> ConversionWritten:
    """Write and confirm the admitted rank set before its owner can publish it."""
    from shadowspill.training.worker.checkpoint_records import RankSaved
    from shadowspill.training.worker.conversion_records import ConversionWritten

    from .exports import write_weights
    from .save import write_checkpoint

    info = description.info
    write = write_checkpoint if info.kind == "checkpoint" else write_weights
    completed = []
    for rank in description.ranks:
        path = directory / f"rank-{rank.rank}.ssp"
        with path.open("xb") as stream:
            manifest = write(
                converted.ranks[rank.rank],
                stream,
                metadata=rank.metadata(info),
                chunk_bytes=chunk_bytes,
                weights=info.weights,
                frozen_state_id=info.frozen_state_id,
            )
            if stream.tell() != rank.size_bytes:
                raise ValueError("converted rank size differs from its admission")
        rank.confirm(info, manifest)
        with path.open("rb") as stream:
            sha256, size = file_digest(stream)
        if size != rank.size_bytes:
            raise ValueError("converted rank size changed before completion")
        completed.append(RankSaved(sha256, rank.confirmation(info)))
    return ConversionWritten(completed)
