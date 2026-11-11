"""Atomic checkpoint archives streamed from Executor-owned host storage."""

from __future__ import annotations

import os
import sys
import tempfile
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

import torch

from shadowspill.artifacts._values import encoded
from shadowspill.artifacts.tensors import TensorChunk

from .archive import ContentHash, finish, header, padding
from .frozen import omitted_parameters, validate_identity
from .identity import state_identity
from .tensors import byte_view, geometry, region_bytes
from .weights import WeightRepresentation, select_weights

if TYPE_CHECKING:
    from shadowspill.execution import Executor


class _Writer:
    def __init__(self, archive: IO[bytes]):
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
            member = f"tensors/{len(self.entries)}"
            self.entries.append(
                {
                    **geometry(chunk),
                    "member": member,
                    "bytes": region_bytes(chunk),
                }
            )
            header(self.archive, member, region_bytes(chunk))
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
    executor: Executor,
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
    copies on load. ``"compute"`` instead reconstructs masters from rounded
    compute values. Encoded representations or incompatible owned layouts retain
    both copies, with the reason recorded in ``retained_weight_copies``.
    """
    # Reject non-JSON metadata before streaming model-sized payloads.
    if len(encoded(dict(metadata or {}))) > 16 << 20:
        raise ValueError("checkpoint metadata exceeds 16 MiB")
    validate_identity(frozen_state_id)
    omitted = set() if frozen_state_id is None else omitted_parameters(executor)
    selection = select_weights(executor, weights)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=f".{destination.name}.", delete=False
        ) as stream:
            temporary = stream.name
            writer = _Writer(stream.file)

            def write(chunk: TensorChunk, tensor: torch.Tensor) -> None:
                if chunk.name not in omitted and chunk.name not in selection.derived:
                    writer.write(chunk, tensor)

            try:
                executor.export_tensors(write, chunk_bytes=chunk_bytes)
                writer.close()
            except BaseException:
                writer.abort()
                raise
            manifest = encoded(
                {
                    "format": "shadowspill.checkpoint",
                    "version": 2,
                    "byteorder": sys.byteorder,
                    "state": state_identity(executor),
                    "program": executor.plan.program.digest,
                    "logical_rank": executor.logical_rank,
                    "additional_programs": {
                        name: plan.program.digest
                        for name, plan in executor.plans.items()
                    },
                    "tensors": writer.entries,
                    "metadata": dict(metadata or {}),
                    "frozen_state_id": frozen_state_id,
                    "omitted_parameters": sorted(omitted),
                    "weights": weights,
                    "derived_tensors": selection.derived,
                    "retained_weight_copies": selection.retained,
                }
            )
            finish(stream.file, manifest)
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
