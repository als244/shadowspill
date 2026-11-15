"""Shared bounded tensor archive framing, hashing and exact byte counting."""

from __future__ import annotations

from collections.abc import Buffer
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import torch

from shadowspill.artifacts.tensors import TensorChunk

from .archive import ContentHash, WriteSink, header, padding
from .tensors import byte_view, geometry, region_bytes


def tensor_entry(chunk: TensorChunk, index: int) -> dict[str, Any]:
    return {
        **geometry(chunk),
        "member": f"tensors/{index}",
        "bytes": region_bytes(chunk),
    }


class TensorWriter:
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
            entry = tensor_entry(chunk, len(self.entries))
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


class ArchiveSize:
    def __init__(self) -> None:
        self.size_bytes = 0

    def write(self, data: Buffer, /) -> int:
        with memoryview(data) as view:
            self.size_bytes += view.nbytes
            return view.nbytes
