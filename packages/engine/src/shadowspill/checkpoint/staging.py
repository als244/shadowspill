"""Bounded unpublished target storage, including consistency of tied writes."""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import MappingProxyType

import torch

from shadowspill.artifacts._values import nonnegative_int
from shadowspill.artifacts.tensors import TensorChunk

from .archive import WriteAll
from .mapping import LogicalView
from .tensors import byte_view, region_bytes

type TensorKey = tuple[int, str]


def staging_bytes(chunks: Mapping[TensorKey, TensorChunk]) -> int:
    """Exact logical disk demand: payload plus one written flag per element."""
    return sum(
        region_bytes(chunk) * (chunk.dtype.itemsize + 1) // chunk.dtype.itemsize
        for chunk in chunks.values()
    )


class TensorStaging:
    """One private conversion directory; ownership belongs to ``stage_tensors``.

    Written flags live on disk as well as payloads. This keeps arbitrary strided
    or overlapping tied views bounded without retaining a model-sized bitmap or
    interval list in host memory. Unaddressed storage holes remain zero. Logical
    coverage and compatibility are the conversion owner's separate preflight.
    """

    def __init__(
        self,
        directory: Path,
        chunks: Mapping[TensorKey, TensorChunk],
        scratch_bytes: int,
    ):
        self._directory = directory
        self.chunks = MappingProxyType(dict(chunks))
        self._indices = {key: index for index, key in enumerate(chunks)}
        maximum = max((chunk.dtype.itemsize for chunk in chunks.values()), default=1)
        self._width = scratch_bytes // (maximum + 1)
        self._values = torch.empty(
            self._width * maximum, dtype=torch.uint8, device="cpu"
        )
        self._marks = torch.empty(self._width, dtype=torch.uint8, device="cpu")
        self._valid, self._sealed, self._closed = True, False, False
        for key, chunk in chunks.items():
            for suffix, length in (
                ("data", region_bytes(chunk)),
                ("marks", region_bytes(chunk) // chunk.dtype.itemsize),
            ):
                with self._path(key, suffix).open("xb", buffering=0) as stream:
                    stream.truncate(length)

    def _path(self, key: TensorKey, suffix: str) -> Path:
        return self._directory / f"{self._indices[key]}.{suffix}"

    def _check(self) -> None:
        if not self._valid or self._closed:
            raise ValueError("checkpoint target staging is invalid or closed")

    def _all_written(self, marks: torch.Tensor) -> bool:
        expected = self._values[: marks.numel()]
        expected.fill_(1)
        return torch.equal(marks, expected)

    def write(self, key: TensorKey, offset: int, values: torch.Tensor) -> None:
        """Accept a borrowed region; overlapping writes must agree bit-for-bit."""
        self._check()
        if self._sealed:
            raise ValueError("checkpoint target staging is already sealed")
        try:
            self._write(key, offset, values)
        except BaseException:
            self._valid = False
            raise

    def _write(self, key: TensorKey, offset: int, values: torch.Tensor) -> None:
        chunk = self.chunks[key]
        nonnegative_int(offset, "target archive offset")
        incoming = byte_view(values)
        width = chunk.dtype.itemsize
        if values.dtype != chunk.dtype or (
            offset + values.numel()
        ) * width > region_bytes(chunk):
            raise ValueError("checkpoint target write exceeds its owned geometry")
        with (
            self._path(key, "data").open("r+b", buffering=0) as data,
            self._path(key, "marks").open("r+b", buffering=0) as written,
        ):
            for start in range(0, values.numel(), self._width):
                count = min(self._width, values.numel() - start)
                marks = self._marks[:count]
                written.seek(offset + start)
                if written.readinto(byte_view(marks)) != count:
                    raise ValueError("checkpoint target coverage file is truncated")
                portion = incoming[start * width : (start + count) * width]
                if torch.any(marks).item():
                    complete = self._all_written(marks)
                    old = self._values[: count * width]
                    data.seek((offset + start) * width)
                    if data.readinto(byte_view(old)) != old.numel():
                        raise ValueError("checkpoint target payload is truncated")
                    expected = values[start : start + count].view(torch.uint8)
                    if complete:
                        if not torch.equal(old, expected):
                            raise ValueError(
                                "tied checkpoint views write conflicting values"
                            )
                    else:
                        # Mixed coverage is uncommon. Inspect runs without a
                        # tensor-sized Python list or extra boolean array.
                        flags = byte_view(marks)
                        begin = 0
                        while begin < count:
                            flag = flags[begin]
                            if flag not in (0, 1):
                                raise ValueError(
                                    "invalid checkpoint target coverage flag"
                                )
                            stop = begin + 1
                            while stop < count and flags[stop] == flag:
                                stop += 1
                            if flag and not torch.equal(
                                old[begin * width : stop * width],
                                expected[begin * width : stop * width],
                            ):
                                raise ValueError(
                                    "tied checkpoint views write conflicting values"
                                )
                            begin = stop
                data.seek((offset + start) * width)
                WriteAll(data).write(portion)
                marks.fill_(1)
                written.seek(offset + start)
                WriteAll(written).write(byte_view(marks))

    def require_written(self, key: TensorKey, view: LogicalView) -> None:
        """Check every addressed logical value, allowing unreferenced padding."""
        self._check()
        try:
            chunk = self.chunks[key]
            if (view.storage_elements, view.region_offset, view.region_elements) != (
                chunk.storage_elements,
                chunk.region_offset,
                region_bytes(chunk) // chunk.dtype.itemsize,
            ):
                raise ValueError("checkpoint target view has different owned storage")
            with self._path(key, "marks").open("rb", buffering=0) as stream:
                for span in view.spans():
                    for start in range(0, span.elements, self._width):
                        count = min(self._width, span.elements - start)
                        marks = self._marks[:count]
                        stream.seek(span.archive_offset + start)
                        if stream.readinto(
                            byte_view(marks)
                        ) != count or not self._all_written(marks):
                            raise ValueError(
                                "checkpoint target has unwritten logical values"
                            )
        except BaseException:
            self._valid = False
            raise

    def seal(self) -> None:
        """Stop writes after the owner has validated all required logical views."""
        self._check()
        self._sealed = True

    def export(
        self,
        rank: int,
        callback: Callable[[TensorChunk, torch.Tensor], None],
        *,
        chunk_bytes: int,
        names: Collection[str] | None = None,
    ) -> None:
        """Feed the ordinary checkpoint encoder, borrowing the same staging buffer."""
        self._check()
        if not self._sealed:
            raise ValueError("checkpoint target must be sealed before export")
        if type(chunk_bytes) is not int or chunk_bytes < 1:
            raise ValueError("checkpoint export chunk_bytes must be a positive integer")
        for (owner, name), chunk in self.chunks.items():
            if owner != rank or (names is not None and name not in names):
                continue
            width = chunk.dtype.itemsize
            step = min(self._values.numel(), chunk_bytes) // width
            if not step:
                raise ValueError("checkpoint export budget cannot hold one element")
            elements = region_bytes(chunk) // width
            with self._path((owner, name), "data").open("rb", buffering=0) as stream:
                for start in range(0, elements, step):
                    count = min(step, elements - start)
                    values = self._values[: count * width].view(chunk.dtype)
                    if stream.readinto(byte_view(values)) != values.nbytes:
                        raise ValueError("checkpoint target payload is truncated")
                    callback(replace(chunk, offset=chunk.region_offset + start), values)


@contextmanager
def stage_tensors(
    chunks: Mapping[TensorKey, TensorChunk],
    directory: Path,
    *,
    scratch_bytes: int,
    max_disk_bytes: int,
) -> Iterator[TensorStaging]:
    """Admit private scratch before allocation; always remove temporary state.

    Disk demand includes the written flags even when the filesystem can represent
    untouched zeros sparsely. Metadata and final archive files need separate owner
    reservations. This helper never creates or replaces a published artifact.
    """
    chunks = dict(chunks)
    for (rank, name), chunk in chunks.items():
        nonnegative_int(rank, "target logical rank")
        for label, value in (
            ("storage_elements", chunk.storage_elements),
            ("region_offset", chunk.region_offset),
            (
                "region_elements",
                chunk.storage_elements
                if chunk.region_elements is None
                else chunk.region_elements,
            ),
        ):
            nonnegative_int(value, label)
        if (
            chunk.name != name
            or not isinstance(name, str)
            or not name
            or chunk.offset != chunk.region_offset
            or chunk.region_offset + region_bytes(chunk) // chunk.dtype.itemsize
            > chunk.storage_elements
        ):
            raise ValueError("invalid checkpoint target descriptor")
    nonnegative_int(max_disk_bytes, "checkpoint target disk allowance")
    maximum = max((chunk.dtype.itemsize for chunk in chunks.values()), default=1)
    if type(scratch_bytes) is not int or scratch_bytes < maximum + 1:
        raise ValueError("checkpoint target scratch cannot hold one value and flag")
    if staging_bytes(chunks) > max_disk_bytes:
        raise ValueError("checkpoint target staging exceeds its disk allowance")
    with TemporaryDirectory(prefix=".checkpoint-conversion-", dir=directory) as path:
        staging = TensorStaging(Path(path), chunks, scratch_bytes)
        try:
            yield staging
        finally:
            staging._closed = True
            del staging._values, staging._marks
