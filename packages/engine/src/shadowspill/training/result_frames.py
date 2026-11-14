"""Assemble declared caller-order tensor outputs in bounded unpublished files."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager
from itertools import groupby
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import BinaryIO

from shadowspill_client.buffers import BufferBudget, BufferLease
from shadowspill_client.protocol.requests import ArtifactRef
from shadowspill_client.protocol.tensors import tensor_bytes

from .inputs.frame_reader import FrameReader
from .inputs.frames import FRAME_CHUNK_BYTES, BufferWriter, InputFrame
from .inputs.packing import PackedAssignment
from .outputs import OutputLayout
from .result_tensors import write_tensor_region
from .worker.output_frame import WorkerOutputFrame


class ResultFrames:
    """One scratch file, serial worker reads and the existing binary encoder.

    Each named tensor becomes a separate frame. Its ``inputs`` leaves are keyed
    by decimal caller Datum index, or ``global`` for a global output. This keeps
    original shapes (including empty tensors) without padding the downloaded
    values. The output name belongs to the result's artifact mapping.

    Construction admits metadata only. The publisher must retain this owner
    before start(), so a failed file deletion remains reachable and charged.
    """

    def __init__(
        self,
        layout: OutputLayout,
        assignment: PackedAssignment,
        *,
        root: Path,
        memory: BufferBudget,
        disk: BufferBudget,
    ) -> None:
        metadata = sum(
            512 + len(dest.name.encode()) + 64 * len(dest.shape)
            for dest in layout.destinations(assignment)
        ) + 4096 * len(layout.outputs.tensors)
        self._metadata = memory.reserve(
            4096 + 8 * metadata + FRAME_CHUNK_BYTES, timeout=0
        )
        self.layout = layout
        self.assignment: PackedAssignment | None = assignment
        self._root, self.memory, self._disk = root, memory, disk
        self._disk_lease: BufferLease | None = None
        self._directory: TemporaryDirectory[str] | None = None
        self._stream: BinaryIO | None = None
        self.frames: dict[str, InputFrame] = {}
        self._regions: dict[
            tuple[str, int | None], tuple[int, tuple[int, ...], str]
        ] = {}
        self._paths: dict[tuple[str, tuple[str, ...]], tuple[int, int]] = {}
        self._size = self._global_size = 0
        self._started = self.ready = self.closed = False
        try:
            for name in layout.outputs.tensors:
                shapes: dict[tuple[str, ...], tuple[int, ...]] = {}
                dtypes: dict[tuple[str, ...], str] = {}
                for dest in layout.destinations(assignment):
                    if dest.name != name:
                        continue
                    path = (
                        "inputs",
                        "global" if dest.datum is None else str(dest.datum),
                    )
                    shapes[path], dtypes[path] = dest.shape, dest.dtype
                    size = tensor_bytes(dest.dtype, dest.shape)
                    self._regions[name, dest.datum] = self._size, dest.shape, dest.dtype
                    self._paths[name, path] = self._size, size
                    self._size += size
                    if dest.datum is None:
                        self._global_size = max(self._global_size, size)
                self.frames[name] = InputFrame(shapes, dtypes)
        except BaseException:
            self.frames.clear()
            self._regions.clear()
            self._paths.clear()
            self._metadata.close()
            raise

    def start(self) -> None:
        if self.closed or self._started:
            raise RuntimeError("result assembly is closed or already started")
        self._started = True
        self._disk_lease = self._disk.reserve(self._size + self._global_size, timeout=0)
        self._directory = TemporaryDirectory(prefix="results-", dir=self._root)
        self._stream = (Path(self._directory.name) / "tensors").open("w+b", buffering=0)
        self._stream.truncate(self._size + self._global_size)

    def _chunks(self, offset: int, size: int) -> Iterator[memoryview]:
        assert self._stream is not None
        while size:
            self._stream.seek(offset)
            block = self._stream.read(min(size, FRAME_CHUNK_BYTES))
            try:
                if not block:
                    raise OSError("result assembly file became truncated")
                yield memoryview(block)
                offset += len(block)
                size -= len(block)
            finally:
                del block

    def assemble(
        self,
        frames: Mapping[int, WorkerOutputFrame],
        open_rank: Callable[[int], AbstractContextManager[BinaryIO]],
    ) -> None:
        if self.closed or self._stream is None or self.ready:
            raise RuntimeError("result assembly is not writable")
        if self.assignment is None:
            raise RuntimeError("result assembly already consumed its input metadata")
        assignment, self.assignment = self.assignment, None
        seen: set[tuple[str, int | None]] = set()
        for (name, _buffer, datum), parts in groupby(
            self.layout.pieces(assignment),
            key=lambda p: (p.destination.name, p.buffer, p.destination.datum),
        ):
            destination, shape, dtype = self._regions[name, datum]
            if datum is not None and (name, datum) in seen:
                raise ValueError("caller Datum output occurs in multiple buffers")
            first = True
            for piece in parts:
                frame = frames[piece.logical_rank]
                index = frame.buffers.index(piece.buffer)
                path = ("inputs", str(index), "values", name)
                if (
                    frame.frame.shapes[path] != piece.source_shape
                    or frame.frame.dtypes[path] != dtype
                ):
                    raise ValueError("worker output differs from its prepared geometry")
                with (
                    open_rank(piece.logical_rank) as stream,
                    FrameReader(stream, frame.frame, self.memory) as reader,
                ):
                    write_tensor_region(
                        self._stream,
                        offset=self._size if datum is None else destination,
                        shape=shape,
                        dtype=dtype,
                        axes=piece.destination_axes,
                        chunks=reader.chunks(path, piece.source_axes),
                        budget=self.memory,
                        reduction="sum"
                        if piece.worker_reduction == "sum" and not first
                        else None,
                    )
                first = False
            if datum is None:
                write_tensor_region(
                    self._stream,
                    offset=destination,
                    shape=shape,
                    dtype=dtype,
                    axes=tuple(slice(None) for _ in shape),
                    chunks=self._chunks(self._size, tensor_bytes(dtype, shape)),
                    budget=self.memory,
                    reduction=self.layout.global_reductions[name]
                    if (name, datum) in seen
                    else None,
                )
            seen.add((name, datum))
        if seen != self._regions.keys():
            raise ValueError("prepared output pieces do not cover caller destinations")
        self.ready = True

    def write(self, name: str, stream: BufferWriter, artifact_id: str) -> ArtifactRef:
        if self.closed or not self.ready:
            raise RuntimeError("caller-order result frames are not ready")
        return self.frames[name].write_chunks(
            stream,
            lambda path: self._chunks(*self._paths[name, path]),
            self.memory,
            artifact_id,
        )

    def close(self) -> None:
        if self.closed:
            return
        # A failed scratch deletion retains output bytes, never the input batch.
        self.assignment = None
        if self._stream is not None:
            self._stream.close()
        if self._directory is not None:
            self._directory.cleanup()
        self.frames.clear()
        self._regions.clear()
        self._paths.clear()
        if self._disk_lease is not None:
            self._disk_lease.close()
        self._metadata.close()
        self.closed = True
