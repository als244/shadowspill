"""Atomic checkpoint archives streamed from Executor-owned host storage."""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, cast

import torch

from shadowspill.artifacts._values import encoded
from shadowspill.execution.tensors import TensorChunk

from .tensors import byte_view, geometry

if TYPE_CHECKING:
    from shadowspill.execution import Executor


class _Writer:
    def __init__(self, archive: zipfile.ZipFile):
        self.archive = archive
        self.entries: list[dict[str, Any]] = []
        self.stream: BinaryIO | None = None
        self.written = 0
        self.digest = hashlib.sha256()

    def abort(self) -> None:
        if self.stream is not None:
            self.stream.close()
            self.stream = None

    def close_tensor(self) -> None:
        if self.stream is not None:
            self.stream.close()
            self.stream = None
            if self.written != self.entries[-1]["bytes"]:
                raise ValueError("incomplete tensor export")
            self.entries[-1]["sha256"] = self.digest.hexdigest()

    def write(self, chunk: TensorChunk, tensor: torch.Tensor) -> None:
        if chunk.offset == 0:
            self.close_tensor()
            if any(entry["name"] == chunk.name for entry in self.entries):
                raise ValueError(f"duplicate tensor export: {chunk.name}")
            member = f"tensors/{len(self.entries)}"
            self.entries.append(
                {
                    **geometry(chunk),
                    "member": member,
                    "bytes": chunk.storage_elements * tensor.element_size(),
                }
            )
            self.stream = cast(
                BinaryIO, self.archive.open(member, "w", force_zip64=True)
            )
            self.written, self.digest = 0, hashlib.sha256()
        if (
            self.stream is None
            or any(
                self.entries[-1][key] != value for key, value in geometry(chunk).items()
            )
            or chunk.offset * tensor.element_size() != self.written
        ):
            raise ValueError("tensor export chunks are not contiguous and consistent")
        view = byte_view(tensor)
        self.stream.write(view)
        self.digest.update(view)
        self.written += view.nbytes


def save_checkpoint(
    executor: Executor,
    path: str | os.PathLike[str],
    *,
    metadata: Mapping[str, Any] | None = None,
    chunk_bytes: int = 64 << 20,
) -> None:
    """Save all persistent tensors without a full-model host snapshot.

    Metadata can include completed updates and caller-owned scheduler/RNG state
    encoded as JSON data. This helper does not discover application state.
    """
    # Reject non-JSON metadata before streaming model-sized payloads.
    if len(encoded(dict(metadata or {}))) > 16 << 20:
        raise ValueError("checkpoint metadata exceeds 16 MiB")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=f".{destination.name}.", delete=False
        ) as stream:
            temporary = stream.name
            with zipfile.ZipFile(
                stream, "w", compression=zipfile.ZIP_STORED
            ) as archive:
                writer = _Writer(archive)
                try:
                    executor.export_tensors(writer.write, chunk_bytes=chunk_bytes)
                    writer.close_tensor()
                except BaseException:
                    writer.abort()
                    raise
                manifest = encoded(
                    {
                        "format": "shadowspill.checkpoint",
                        "version": 1,
                        "byteorder": sys.byteorder,
                        "program": executor.plan.program.digest,
                        "tensors": writer.entries,
                        "metadata": dict(metadata or {}),
                    }
                )
                if len(manifest) > 16 << 20:
                    raise ValueError("checkpoint manifest exceeds 16 MiB")
                archive.writestr("manifest.json", manifest)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
