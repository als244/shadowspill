"""Restore bounded chunks into Executor storage; file parsing stays outside it."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import zipfile
from typing import TYPE_CHECKING, Any, cast

import torch

from shadowspill.execution.tensors import TensorChunk

from .tensors import byte_view, geometry, region_bytes

if TYPE_CHECKING:
    from shadowspill.execution import Executor


def _manifest(archive: zipfile.ZipFile, executor: Executor) -> dict[str, Any]:
    if archive.getinfo("manifest.json").file_size > 16 << 20:
        raise ValueError("checkpoint manifest exceeds 16 MiB")
    manifest = json.loads(archive.read("manifest.json"))
    if (
        manifest.get("format") != "shadowspill.checkpoint"
        or manifest.get("version") != 1
        or manifest.get("byteorder") != sys.byteorder
        or manifest.get("program") != executor.plan.program.digest
        or manifest.get("logical_rank", 0) != executor.logical_rank
        or manifest.get("additional_programs", {})
        != {name: plan.program.digest for name, plan in executor.plans.items()}
        or not isinstance(manifest.get("metadata"), dict)
    ):
        raise ValueError("checkpoint format or Program identity differs")
    return cast(dict[str, Any], manifest)


def checkpoint_metadata(
    executor: Executor, path: str | os.PathLike[str]
) -> dict[str, Any]:
    """Read loop compatibility information before restoring any tensors."""
    with zipfile.ZipFile(path) as archive:
        return dict(_manifest(archive, executor)["metadata"])


class _Reader:
    def __init__(self, archive: zipfile.ZipFile, manifest: dict[str, Any]):
        self.archive = archive
        entries = manifest["tensors"]
        self.entries = {entry["name"]: entry for entry in entries}
        members = [entry["member"] for entry in entries]
        if (
            len(self.entries) != len(entries)
            or sorted(archive.namelist())
            != sorted(
                [
                    "manifest.json",
                    *members,
                ]
            )
            or len(set(members)) != len(members)
        ):
            raise ValueError("invalid checkpoint tensor inventory")
        for entry in entries:
            info = archive.getinfo(entry["member"])
            if (
                info.file_size != entry["bytes"]
                or info.compress_type != zipfile.ZIP_STORED
            ):
                raise ValueError("checkpoint tensor size or encoding differs")
        self.stream: zipfile.ZipExtFile | None = None
        self.name = ""
        self.consumed = 0
        self.digest = hashlib.sha256()

    def fill(self, chunk: TensorChunk, tensor: torch.Tensor) -> None:
        entry = self.entries.get(chunk.name)
        if (
            entry is None
            or any(entry[key] != value for key, value in geometry(chunk).items())
            or entry["bytes"] != region_bytes(chunk)
        ):
            raise ValueError(f"checkpoint tensor geometry differs: {chunk.name}")
        if chunk.offset == chunk.region_offset:
            if self.stream is not None:
                raise ValueError("previous checkpoint tensor was not consumed")
            self.stream = cast(zipfile.ZipExtFile, self.archive.open(entry["member"]))
            self.name, self.consumed = chunk.name, 0
            self.digest = hashlib.sha256()
        if (
            self.stream is None
            or self.name != chunk.name
            or (
                (chunk.offset - chunk.region_offset) * tensor.element_size()
                != self.consumed
            )
        ):
            raise ValueError(
                "checkpoint requests must consume each tensor sequentially"
            )
        view = byte_view(tensor)
        if self.stream.readinto(view) != view.nbytes:
            raise ValueError(f"checkpoint tensor is truncated: {chunk.name}")
        self.digest.update(view)
        self.consumed += view.nbytes
        if self.consumed == entry["bytes"]:
            self.stream.close()
            self.stream = None
            if self.digest.hexdigest() != entry["sha256"]:
                raise ValueError(f"checkpoint tensor checksum differs: {chunk.name}")

    def close(self) -> None:
        if self.stream is not None:
            self.stream.close()
            self.stream = None


def load_checkpoint(
    executor: Executor,
    path: str | os.PathLike[str],
    *,
    chunk_bytes: int = 64 << 20,
) -> dict[str, Any]:
    """Restore a checkpoint for the same Program and return caller metadata.

    Budgets and planning order may differ. Program identity and every requested
    tensor's geometry/bytes must match. A failure while filling tensors leaves
    Executor uninitialized. All persistent tensors, including CPU controls, load.
    """
    with zipfile.ZipFile(path) as archive:
        manifest = _manifest(archive, executor)
        reader = _Reader(archive, manifest)
        try:
            executor.initialize(reader.fill, chunk_bytes=chunk_bytes, restore=True)
        finally:
            reader.close()
        return dict(manifest["metadata"])
