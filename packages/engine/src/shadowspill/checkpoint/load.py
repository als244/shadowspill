"""Restore bounded chunks into Executor storage; file parsing stays outside it."""

from __future__ import annotations

import os
import re
import sys
import tarfile
from collections.abc import Callable, Collection, Mapping
from dataclasses import replace
from io import BufferedReader
from typing import TYPE_CHECKING, Any, BinaryIO, cast

import torch

from shadowspill.artifacts.tensors import TensorChunk
from shadowspill.parallelism import GroupRef

from .archive import ContentHash, read_manifest
from .frozen import validate_restore
from .identity import ARCHIVE_KEYS, describe_difference, inventory, state_identity
from .inspection import MANIFEST_LIMIT, open_checkpoint
from .tensors import byte_view, geometry, region_bytes
from .weights import select_weights

if TYPE_CHECKING:
    from shadowspill.execution import Executor

    from .state import CheckpointState


def _manifest(archive: tarfile.TarFile, executor: CheckpointState) -> dict[str, Any]:
    manifest = read_manifest(archive)
    if (
        manifest.get("format") != "shadowspill.checkpoint"
        or manifest.get("version") != 2
        or manifest.get("byteorder") != sys.byteorder
        or manifest.get("logical_rank", 0) != executor.logical_rank
        or not isinstance(manifest.get("metadata"), dict)
        or not isinstance(manifest.get("tensors"), list)
    ):
        raise ValueError("checkpoint format or logical rank differs")
    current = state_identity(executor)
    if manifest.get("state") != current:
        raise ValueError(
            "checkpoint state identity differs: "
            f"saved {manifest.get('state')}, this executor {current}; "
            + describe_difference(
                manifest["tensors"],
                inventory(executor),
                omitted=(
                    *manifest.get("omitted_parameters", ()),
                    *manifest.get("derived_tensors", {}),
                    *manifest.get("gathered_tensors", {}),
                ),
            )
        )
    selection = select_weights(executor, manifest.get("weights"))
    if (
        manifest.get("derived_tensors") != selection.derived
        or manifest.get("gathered_tensors") != selection.gathered
        or manifest.get("retained_weight_copies") != selection.retained
        or (set(selection.derived) | set(selection.gathered)).intersection(
            entry["name"] for entry in manifest["tensors"]
        )
    ):
        raise ValueError("checkpoint weight reconstruction differs")
    return manifest


def checkpoint_metadata(
    executor: CheckpointState, path: str | os.PathLike[str]
) -> dict[str, Any]:
    """Read loop compatibility information before restoring any tensors."""
    with open_checkpoint(path) as archive:
        return dict(_manifest(archive, executor)["metadata"])


class _Reader:
    def __init__(
        self,
        archive: tarfile.TarFile,
        manifest: dict[str, Any],
        chunks: dict[str, TensorChunk],
        omitted: set[str],
    ):
        self.archive = archive
        self.stream = cast(BufferedReader, archive.fileobj)
        entries = manifest["tensors"]
        self.entries = {entry["name"]: entry for entry in entries}
        members = [entry["member"] for entry in entries]
        if (
            len(self.entries) != len(entries)
            or sorted(archive.getnames())
            != sorted(
                [
                    "manifest.json",
                    "manifest.blake3",
                    *members,
                ]
            )
            or len(set(members)) != len(members)
        ):
            raise ValueError("invalid checkpoint tensor inventory")
        expected = (
            {name for name, chunk in chunks.items() if region_bytes(chunk)}
            - omitted
            - set(manifest["derived_tensors"])
            - set(manifest["gathered_tensors"])
        )
        if missing := expected - self.entries.keys():
            raise ValueError(
                f"checkpoint tensor geometry differs: {min(missing)} (missing tensor)"
            )
        if self.entries.keys() - expected:
            raise ValueError("invalid checkpoint tensor inventory")
        for entry in entries:
            chunk = chunks[entry["name"]]
            expected_geometry = geometry(chunk)
            if (
                entry.keys() != expected_geometry.keys() | ARCHIVE_KEYS
                or any(
                    type(entry.get(key)) is not type(value)
                    or entry[key] != value
                    or (
                        isinstance(value, list)
                        and any(type(item) is not int for item in entry[key])
                    )
                    for key, value in expected_geometry.items()
                )
                or type(entry.get("bytes")) is not int
                or entry["bytes"] != region_bytes(chunk)
            ):
                raise ValueError(f"checkpoint tensor geometry differs: {chunk.name}")
            checksum = entry.get("blake3")
            if (
                not isinstance(checksum, str)
                or re.fullmatch(r"[0-9a-f]{64}", checksum) is None
            ):
                raise ValueError(f"checkpoint tensor checksum is invalid: {chunk.name}")
            info = archive.getmember(entry["member"])
            if info.size != entry["bytes"]:
                raise ValueError("checkpoint tensor size or encoding differs")
        self.active = False
        self.name = ""
        self.consumed = 0
        self.digest = ContentHash()
        self.scratch: torch.Tensor | None = None

    def fill(self, chunk: TensorChunk, tensor: torch.Tensor) -> None:
        entry = self.entries.get(chunk.name)
        if (
            entry is None
            or any(entry[key] != value for key, value in geometry(chunk).items())
            or entry["bytes"] != region_bytes(chunk)
        ):
            raise ValueError(f"checkpoint tensor geometry differs: {chunk.name}")
        if chunk.offset == chunk.region_offset:
            if self.active:
                raise ValueError("previous checkpoint tensor was not consumed")
            self.stream.seek(self.archive.getmember(entry["member"]).offset_data)
            self.active = True
            self.name, self.consumed = chunk.name, 0
            self.digest.reset(tensor.nbytes)
        if (
            not self.active
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
        if view.nbytes > entry["bytes"] - self.consumed:
            raise ValueError(f"checkpoint request exceeds tensor: {chunk.name}")
        # Read directly into the destination; ExFileObject.readinto would first
        # allocate a separate bytes object containing the same payload.
        if self.stream.readinto(view) != view.nbytes:
            raise ValueError(f"checkpoint tensor is truncated: {chunk.name}")
        self.digest.update(view)
        self.consumed += view.nbytes
        if self.consumed == entry["bytes"]:
            self.active = False
            if self.digest.hexdigest() != entry["blake3"]:
                raise ValueError(f"checkpoint tensor checksum differs: {chunk.name}")

    def derive(
        self,
        source: TensorChunk,
        request: TensorChunk,
        tensor: torch.Tensor,
        chunk_bytes: int,
    ) -> None:
        """Read and validate one selected copy, casting into a bounded destination.

        A strict subregion still checksums the entire selected source. Reading
        it again avoids model-sized staging or a restore-order dependency in
        Executor; even the borrowed scratch is bounded by ``chunk_bytes``.
        """
        width = min(
            max(1, chunk_bytes // source.dtype.itemsize),
            source.storage_elements
            if source.region_elements is None
            else source.region_elements,
        )
        if (
            self.scratch is None
            or self.scratch.dtype != source.dtype
            or self.scratch.numel() < width
        ):
            self.scratch = torch.empty(width, dtype=source.dtype, device="cpu")

        def consume(
            begin: int, end: int, destination: torch.Tensor | None = None
        ) -> None:
            for offset in range(begin, end, width):
                count = min(width, end - offset)
                assert self.scratch is not None
                temporary = self.scratch[:count]
                self.fill(replace(source, offset=offset), temporary)
                if destination is not None:
                    destination[offset - begin : offset - begin + count].copy_(
                        temporary
                    )

        if request.offset == request.region_offset:
            consume(source.region_offset, request.region_offset)
        end = request.offset + tensor.numel()
        consume(request.offset, end, tensor)
        region_end = request.region_offset + (
            request.storage_elements
            if request.region_elements is None
            else request.region_elements
        )
        if end == region_end:
            source_end = source.region_offset + (
                source.storage_elements
                if source.region_elements is None
                else source.region_elements
            )
            consume(end, source_end)
            # A completed local cast must not retain its scratch into the later
            # collective reconstruction, which already owns two bounded chunks.
            self.scratch = None


def _initialize(
    executor: Executor,
    reader: _Reader,
    manifest: Mapping[str, Any],
    chunks: dict[str, TensorChunk],
    omitted: set[str],
    *,
    chunk_bytes: int,
    fill_tensor: Callable[[TensorChunk, torch.Tensor], None] | None,
    restore: bool | Collection[str] = True,
) -> None:
    """Apply verified selections through the existing Executor initialization."""
    derived = manifest["derived_tensors"]
    gathered = manifest["gathered_tensors"]

    def fill(chunk: TensorChunk, tensor: torch.Tensor) -> None:
        if chunk.name in omitted:
            assert fill_tensor is not None
            fill_tensor(chunk, tensor)
        elif chunk.name in derived:
            reader.derive(chunks[derived[chunk.name]], chunk, tensor, chunk_bytes)
        elif chunk.name in gathered:
            # Every valid element is filled after the coordinated local
            # reads. Avoid an unnecessary full-replica zeroing pass.
            return
        else:
            reader.fill(chunk, tensor)

    executor.initialize(
        fill,
        chunk_bytes=chunk_bytes,
        restore=restore,
        replicated_copies={
            name: (item["source"], GroupRef(name, item["groups"]))
            for name, item in gathered.items()
        },
    )


def load_checkpoint(
    executor: Executor,
    path: str | os.PathLike[str],
    *,
    chunk_bytes: int = 64 << 20,
    frozen_state_id: str | None = None,
    fill_tensor: Callable[[TensorChunk, torch.Tensor], None] | None = None,
) -> dict[str, Any]:
    """Restore a checkpoint of the same training state and return caller metadata.

    The checkpoint binds to the tensors by name, geometry and owned region,
    not to the Program: a rebuilt Program, other kernels, another microbatch
    geometry or budget restore it, and a different inventory is refused with
    the tensors named. A failure while filling tensors leaves
    Executor uninitialized. All persistent tensors, including CPU controls, load.
    Partial checkpoints require the same frozen_state_id and a bounded loader
    for the omitted base parameters, even when the Executor was initialized.
    Ordinary omitted weight copies are reconstructed by bounded casts from the
    archive's selected representation; encoded representations remain physical.
    Replicated compute weights are reconstructed collectively from restored
    master shards using bounded CPU transfers. All distributed workers call load.
    """
    with open_checkpoint(path) as archive:
        manifest = _manifest(archive, executor)
        omitted = validate_restore(executor, manifest, frozen_state_id)
        if omitted and fill_tensor is None:
            raise ValueError(
                "partial checkpoint requires fill_tensor for the frozen base"
            )
        chunks = {chunk.name: chunk for chunk in executor.tensor_chunks()}
        reader = _Reader(archive, manifest, chunks, omitted)
        _initialize(
            executor,
            reader,
            manifest,
            chunks,
            omitted,
            chunk_bytes=chunk_bytes,
            fill_tensor=fill_tensor,
        )
        return dict(manifest["metadata"])


def validate_checkpoint(
    executor: CheckpointState,
    path: str | os.PathLike[str] | BinaryIO,
    *,
    frozen_state_id: str | None = None,
    max_manifest_bytes: int = MANIFEST_LIMIT,
) -> dict[str, Any]:
    """Return the validated rank manifest, without reading tensor payloads.

    The owner separately verifies immutable archive bytes during bounded delivery.
    Load still verifies each tensor while reading; corruption or transfer failure
    after release invalidates state. This does not authorize unvalidated uploads.
    """
    with open_checkpoint(path, max_manifest_bytes=max_manifest_bytes) as archive:
        manifest = _manifest(archive, executor)
        omitted = validate_restore(executor, manifest, frozen_state_id)
        chunks = {chunk.name: chunk for chunk in executor.tensor_chunks()}
        _Reader(archive, manifest, chunks, omitted)
        return manifest
