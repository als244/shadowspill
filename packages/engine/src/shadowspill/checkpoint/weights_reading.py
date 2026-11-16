"""Read inference exports through the existing verified tensor reader."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import BinaryIO

from shadowspill.artifacts._values import digest

from .archive import read_manifest
from .exports import _describe
from .inspection import MANIFEST_LIMIT, open_checkpoint
from .reading import RankReader, _signature, _verified_reader
from .state import CheckpointState


@contextmanager
def read_weights(
    state: CheckpointState,
    path: str | os.PathLike[str] | BinaryIO,
    *,
    chunk_bytes: int,
    frozen_state_id: str | None = None,
    max_manifest_bytes: int = MANIFEST_LIMIT,
) -> Iterator[RankReader]:
    """Validate a weights-only rank against its source Plans before any reads.

    Logical views, ties, selected master/compute copies and frozen omissions use
    the same description as the exporter. Every stored tensor is checksummed
    before yielding the ordinary bounded random-access reader. No optimizer or
    RNG state is read, initialized or restored. Combining ranks and selecting a
    target layout remain the initialization owner's responsibilities.
    """
    if type(chunk_bytes) is not int or chunk_bytes < 1:
        raise ValueError("weights source chunk_bytes must be a positive integer")
    with open_checkpoint(path, max_manifest_bytes=max_manifest_bytes) as archive:
        initial = _signature(archive.fileobj)
        manifest = read_manifest(archive)
        weights = manifest.get("weights")
        if (
            weights not in ("master", "compute")
            or not isinstance(manifest.get("metadata"), dict)
            or not isinstance(manifest.get("tensors"), list)
        ):
            raise ValueError("invalid weights archive metadata")
        description = _describe(state, manifest["metadata"], weights, frozen_state_id)
        if set(manifest) != {*description.fields, "tensors"} or digest(
            {key: manifest[key] for key in description.fields}
        ) != digest(description.fields):
            raise ValueError("weights archive differs from source Plans or frozen base")
        with _verified_reader(
            archive,
            manifest,
            {chunk.name: chunk for chunk in description.chunks},
            set(),
            chunk_bytes=chunk_bytes,
            initial=initial,
        ) as source:
            yield source
