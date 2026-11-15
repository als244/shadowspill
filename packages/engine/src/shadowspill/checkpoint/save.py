"""Atomic checkpoint archives streamed from borrowed bounded host storage."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from collections.abc import Buffer, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from shadowspill.artifacts._values import encoded
from shadowspill.artifacts.tensors import TensorChunk

from .archive import (
    MANIFEST_LIMIT,
    ContentHash,
    WriteAll,
    WriteSink,
    finish,
    header,
    padding,
)
from .frozen import omitted_parameters, validate_identity
from .identity import state_identity
from .tensors import byte_view, geometry, region_bytes
from .weights import WeightRepresentation, WeightSelection, select_weights

if TYPE_CHECKING:
    from .state import CheckpointExport, CheckpointState


def _entry(chunk: TensorChunk, index: int) -> dict[str, Any]:
    return {
        **geometry(chunk),
        "member": f"tensors/{index}",
        "bytes": region_bytes(chunk),
    }


class _Writer:
    def __init__(self, archive: WriteSink):
        self.archive = archive
        self.entries: list[dict[str, Any]] = []
        self.active = False
        self.names: set[str] = set()
        self.written = 0
        self.digest = ContentHash()
        self.hash_worker: ThreadPoolExecutor | None = None

    def abort(self) -> None:
        if self.hash_worker is not None:
            self.hash_worker.shutdown(wait=True)
            self.hash_worker = None

    def close(self) -> None:
        try:
            self.close_tensor()
        finally:
            self.abort()

    def close_tensor(self) -> None:
        if self.active:
            if self.written != self.entries[-1]["bytes"]:
                raise ValueError("incomplete tensor export")
            self.entries[-1]["blake3"] = self.digest.hexdigest()
            padding(self.archive, self.written)
            self.active = False

    def write(self, chunk: TensorChunk, tensor: torch.Tensor) -> None:
        if chunk.offset == chunk.region_offset:
            self.close_tensor()
            if chunk.name in self.names:
                raise ValueError(f"duplicate tensor export: {chunk.name}")
            self.names.add(chunk.name)
            entry = _entry(chunk, len(self.entries))
            self.entries.append(entry)
            header(self.archive, entry["member"], entry["bytes"])
            self.written, self.active = 0, True
            self.digest.reset(tensor.nbytes)
        if (
            not self.active
            or any(
                self.entries[-1][key] != value for key, value in geometry(chunk).items()
            )
            or (chunk.offset - chunk.region_offset) * tensor.element_size()
            != self.written
        ):
            raise ValueError("tensor export chunks are not contiguous and consistent")
        view = byte_view(tensor)
        if view.nbytes >= 1 << 20:
            if self.hash_worker is None:
                self.hash_worker = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="checkpoint-hash"
                )
            pending = self.hash_worker.submit(self.digest.update, view)
            try:
                self.archive.write(view)
            finally:
                # Export buffers are borrowed only until this callback returns.
                # Complete hashing even when the archive write fails.
                pending.result()
        else:
            self.archive.write(view)
            self.digest.update(view)
        self.written += view.nbytes


def save_checkpoint(
    executor: CheckpointExport,
    path: str | os.PathLike[str],
    *,
    metadata: Mapping[str, Any] | None = None,
    chunk_bytes: int = 64 << 20,
    frozen_state_id: str | None = None,
    weights: WeightRepresentation = "master",
) -> None:
    """Save all persistent tensors without a full-model host snapshot.

    Metadata can include completed updates and caller-owned scheduler/RNG state
    encoded as JSON data. This helper does not discover application state.
    Distributed callers supply a distinct path per logical rank; each archive
    contains only that rank's owned tensor regions and records its identity.
    A checkpoint binds to the training state, the tensors by name, geometry
    and owned region (``state``); the Program that saved it is recorded for
    information only.
    With frozen_state_id, omit unchanged frozen parameters; retain trainable
    tensors, optimizer state, buffers and anything mutated by an admitted Plan.
    ``weights="master"`` saves ordinary master weights once and derives compute
    copies on load. Replicated compute weights with sharded masters are rebuilt
    collectively from those shards, without storing the compute replicas.
    ``"compute"`` instead reconstructs masters from rounded
    compute values. Encoded representations or incompatible owned layouts retain
    both copies, with the reason recorded in ``retained_weight_copies``.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=f".{destination.name}.", delete=False
        ) as stream:
            temporary = stream.name
            write_checkpoint(
                executor,
                stream.file,
                metadata=metadata,
                chunk_bytes=chunk_bytes,
                frozen_state_id=frozen_state_id,
                weights=weights,
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        temporary = None
        directory = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def write_checkpoint(
    executor: CheckpointExport,
    stream: WriteSink,
    *,
    metadata: Mapping[str, Any] | None = None,
    chunk_bytes: int = 64 << 20,
    frozen_state_id: str | None = None,
    weights: WeightRepresentation = "master",
) -> dict[str, Any]:
    """Write the existing uncompressed rank archive to a borrowed blocking sink.

    No seek, full-state snapshot, flush, close or publication is performed. The
    owner must publish only after this returns and its own durability checks pass.
    On failure, the owner discards the partial stream. Return the exact manifest
    written, so a coordinator need not reread tensor bytes to inspect this rank.
    """
    saved_metadata, omitted, selection = _settings(
        executor, metadata, frozen_state_id, weights
    )
    target = WriteAll(stream)
    writer = _Writer(target)

    def write(chunk: TensorChunk, tensor: torch.Tensor) -> None:
        if (
            chunk.name not in omitted
            and chunk.name not in selection.derived
            and chunk.name not in selection.gathered
        ):
            writer.write(chunk, tensor)

    try:
        executor.export_tensors(write, chunk_bytes=chunk_bytes)
        writer.close()
        _validate_export(executor, writer.entries, omitted, selection)
    except BaseException:
        writer.abort()
        raise
    manifest = _manifest(
        executor,
        writer.entries,
        saved_metadata,
        omitted,
        selection,
        frozen_state_id,
        weights,
    )
    finish(target, encoded(manifest))
    return manifest


def _validate_export(
    state: CheckpointState,
    entries: list[dict[str, Any]],
    omitted: set[str],
    selection: WeightSelection,
) -> None:
    """A borrowed exporter cannot silently omit required saved representations."""
    exported = {entry["name"]: entry for entry in entries}
    expected = set()
    for chunk in state.tensor_chunks():
        if (
            chunk.name in omitted
            or chunk.name in selection.derived
            or chunk.name in selection.gathered
            or not region_bytes(chunk)
        ):
            continue
        expected.add(chunk.name)
        entry = exported.get(chunk.name)
        if (
            entry is None
            or entry["bytes"] != region_bytes(chunk)
            or any(entry.get(key) != value for key, value in geometry(chunk).items())
        ):
            raise ValueError(
                f"checkpoint export differs from its description: {chunk.name}"
            )
    if expected != exported.keys():
        raise ValueError("checkpoint export contains undescribed state")


def _settings(
    executor: CheckpointState,
    metadata: Mapping[str, Any] | None,
    frozen_state_id: str | None,
    weights: WeightRepresentation,
) -> tuple[dict[str, Any], set[str], WeightSelection]:
    raw_metadata = encoded(dict(metadata or {}))
    if len(raw_metadata) > MANIFEST_LIMIT:
        raise ValueError("checkpoint metadata exceeds 16 MiB")
    # Freeze JSON before exporting any borrowed tensor views. A sink or another
    # caller thread must not change the manifest after metadata validation.
    saved_metadata = json.loads(raw_metadata)
    del raw_metadata
    validate_identity(frozen_state_id)
    omitted = set() if frozen_state_id is None else omitted_parameters(executor)
    selection = select_weights(executor, weights)
    return saved_metadata, omitted, selection


def _manifest(
    executor: CheckpointState,
    tensors: list[dict[str, Any]],
    saved_metadata: dict[str, Any],
    omitted: set[str],
    selection: WeightSelection,
    frozen_state_id: str | None,
    weights: WeightRepresentation,
) -> dict[str, Any]:
    return {
        "format": "shadowspill.checkpoint",
        "version": 2,
        "byteorder": sys.byteorder,
        "state": state_identity(executor),
        "program": executor.plan.program.digest,
        "logical_rank": executor.logical_rank,
        "additional_programs": {
            name: plan.program.digest for name, plan in executor.plans.items()
        },
        "tensors": tensors,
        "metadata": saved_metadata,
        "frozen_state_id": frozen_state_id,
        "omitted_parameters": sorted(omitted),
        "weights": weights,
        "derived_tensors": selection.derived,
        "gathered_tensors": selection.gathered,
        "retained_weight_copies": selection.retained,
    }


class _Size:
    def __init__(self) -> None:
        self.size_bytes = 0

    def write(self, data: Buffer, /) -> int:
        with memoryview(data) as view:
            self.size_bytes += view.nbytes
            return view.nbytes


def checkpoint_size(
    executor: CheckpointState,
    *,
    metadata: Mapping[str, Any] | None = None,
    frozen_state_id: str | None = None,
    weights: WeightRepresentation = "master",
) -> int:
    """Exact rank archive bytes from descriptors, without exporting any tensor.

    Use the same immutable metadata/selection and quiescent state for the write.
    The caller still owns resource admission and rejects a changed output size.
    This uses the actual TAR framing and manifest encoder, including hash widths,
    owned regions, weight reconstruction and frozen-base omissions.
    """
    saved_metadata, omitted, selection = _settings(
        executor, metadata, frozen_state_id, weights
    )
    tensors: list[dict[str, Any]] = []
    seen = set()
    counter = _Size()
    for chunk in executor.tensor_chunks():
        if chunk.name in seen:
            raise ValueError(f"duplicate tensor export: {chunk.name}")
        seen.add(chunk.name)
        if (
            chunk.name in omitted
            or chunk.name in selection.derived
            or chunk.name in selection.gathered
            or not region_bytes(chunk)
        ):
            continue
        entry = _entry(chunk, len(tensors))
        entry["blake3"] = "0" * 64  # Content changes the digest, never its byte width.
        tensors.append(entry)
        header(counter, entry["member"], entry["bytes"])
        counter.size_bytes += entry["bytes"]
        padding(counter, entry["bytes"])
    manifest = _manifest(
        executor, tensors, saved_metadata, omitted, selection, frozen_state_id, weights
    )
    finish(counter, encoded(manifest))
    return counter.size_bytes
