"""Atomic checkpoint archives streamed from borrowed bounded host storage."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from shadowspill.artifacts._values import encoded
from shadowspill.artifacts.tensors import TensorChunk

from .archive import (
    MANIFEST_LIMIT,
    WriteAll,
    WriteSink,
    finish,
    header,
    padding,
)
from .encoding import ArchiveSize, TensorWriter, tensor_entry
from .frozen import omitted_parameters, validate_identity
from .identity import state_identity
from .tensors import geometry, region_bytes
from .weights import WeightRepresentation, WeightSelection, select_weights

if TYPE_CHECKING:
    from .state import CheckpointExport, CheckpointState


def save_checkpoint(
    executor: CheckpointExport,
    path: str | os.PathLike[str],
    *,
    metadata: Mapping[str, Any] | None = None,
    chunk_bytes: int = 64 << 20,
    frozen_state_id: str | None = None,
    weights: WeightRepresentation = "master",
) -> None:
    """Save a worker's persistent state as one checkpoint archive.

    The state streams through the executor's bounded export, without a host
    copy of the whole model, and the archive replaces ``path`` atomically. A
    checkpoint binds to the training state: every persistent tensor's name,
    geometry and owned region. Any Executor with the same state restores it,
    whatever its Program, kernels, microbatch geometry, budgets or ordering;
    the saving Program is recorded for information only. Each worker of a
    distributed run saves its own regions to its own path. Application
    state, such as a scheduler's or random generators', belongs in
    ``metadata``.

    Parameters
    ----------
    executor : Executor
        An initialized executor. An outstanding run is waited for first.
    path : str or os.PathLike
        The archive to write; its directory is created.
    metadata : mapping of str to object, optional
        JSON data saved with the state, at most 16 MiB encoded, and returned
        by `load_checkpoint`.
    chunk_bytes : int, default 64 << 20
        Largest chunk read from the executor at once, in bytes.
    frozen_state_id : str, optional
        Identity of a frozen base the caller keeps elsewhere. With it, the
        parameters no admitted Plan changes are left out, and loading needs
        the same identity and a loader for them. Trainable parameters,
        optimizer state, buffers and anything a Plan changes are always
        saved.
    weights : str, default "master"
        Which copy of a weight held in two precisions is saved: ``"master"``
        saves the masters, and loading derives the compute copies, collectively
        for replicated compute weights with sharded masters; ``"compute"``
        saves the compute copies, and loading derives masters from their
        rounded values. Encoded representations and incompatible layouts
        keep both copies, with the reason in the archive's
        ``retained_weight_copies``.

    Raises
    ------
    RuntimeError
        If the executor is closed, failed or not initialized.
    ValueError
        If ``metadata`` exceeds 16 MiB, ``frozen_state_id`` is empty, or
        ``weights`` is neither ``"master"`` nor ``"compute"``.
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
    """Write a worker's checkpoint archive to a stream.

    The archive is the one `save_checkpoint` saves, streamed through the
    executor's bounded export without seeking. The stream is not
    flushed or closed: the caller makes the archive durable and publishes
    it, and discards what was written if this raises. Its size is what
    `checkpoint_size` returns for the same arguments.

    Parameters
    ----------
    executor : Executor or checkpoint state
        An initialized executor, or any object that describes and exports
        one worker's state as an executor does: ``plan``, ``plans``,
        ``logical_rank``, ``tensor_chunks`` and ``export_tensors``. An
        executor's outstanding run is waited for first.
    stream : writable binary file
        Receives the archive through ``write``, which returns the number of
        bytes it took; a partial write is continued.
    metadata : mapping of str to object, optional
        JSON data saved with the state, at most 16 MiB encoded, and returned
        by `load_checkpoint`.
    chunk_bytes : int, default 64 << 20
        Largest chunk read from the executor at once, in bytes.
    frozen_state_id : str, optional
        Identity of a frozen base the caller keeps elsewhere, as for
        `save_checkpoint`.
    weights : str, default "master"
        Which copy of a weight held in two precisions is saved,
        ``"master"`` or ``"compute"``, as for `save_checkpoint`.

    Returns
    -------
    dict
        The archive's manifest, as written: each tensor's name, geometry,
        size and BLAKE3 checksum, the ``metadata``, and the state identity
        and settings `load_checkpoint` checks.

    Raises
    ------
    RuntimeError
        If the executor is closed, failed or not initialized.
    ValueError
        If ``metadata`` or the manifest holding it exceeds 16 MiB,
        ``frozen_state_id`` is empty, or ``weights`` is neither
        ``"master"`` nor ``"compute"``.
    OSError
        If ``write`` returns anything but a positive count no larger than
        the data it was given.
    """
    saved_metadata, omitted, selection = _settings(
        executor, metadata, frozen_state_id, weights
    )
    target = WriteAll(stream)
    writer = TensorWriter(target)

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
        "version": 1,
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


def checkpoint_size(
    executor: CheckpointState,
    *,
    metadata: Mapping[str, Any] | None = None,
    frozen_state_id: str | None = None,
    weights: WeightRepresentation = "master",
) -> int:
    """Return the exact size of a worker's checkpoint archive, in bytes.

    It is computed from the tensor descriptions and the manifest, without
    reading any tensor. `write_checkpoint` and `save_checkpoint` with the
    same arguments write exactly this many bytes.

    Parameters
    ----------
    executor : Executor or checkpoint state
        The executor whose state the archive holds, or any object that
        describes one worker's state as an executor does: ``plan``,
        ``plans``, ``logical_rank`` and ``tensor_chunks``.
    metadata : mapping of str to object, optional
        JSON data saved with the state, as for `write_checkpoint`.
    frozen_state_id : str, optional
        Identity of a frozen base the caller keeps elsewhere, as for
        `save_checkpoint`.
    weights : str, default "master"
        Which copy of a weight held in two precisions is saved,
        ``"master"`` or ``"compute"``, as for `save_checkpoint`.

    Returns
    -------
    int
        The archive's size, in bytes.

    Raises
    ------
    ValueError
        If ``metadata`` or the manifest holding it exceeds 16 MiB,
        ``frozen_state_id`` is empty, or ``weights`` is neither
        ``"master"`` nor ``"compute"``.
    """
    saved_metadata, omitted, selection = _settings(
        executor, metadata, frozen_state_id, weights
    )
    tensors: list[dict[str, Any]] = []
    seen = set()
    counter = ArchiveSize()
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
        entry = tensor_entry(chunk, len(tensors))
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
