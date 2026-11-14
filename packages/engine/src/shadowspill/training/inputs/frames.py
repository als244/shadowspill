"""Bounded binary frames for already owned, fixed physical input arrays."""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.payload import MAGIC, MEDIA_TYPE
from shadowspill_client.protocol.payload_sizes import (
    MAX_TREE_DEPTH,
    TREE_NODE_BYTES,
    PayloadSize,
    manifest_json,
)
from shadowspill_client.protocol.requests import ArtifactRef
from shadowspill_client.protocol.tensors import storage_dtype, tensor_bytes

type Path = tuple[str, ...]
FRAME_CHUNK_BYTES = 64 << 10


class BufferWriter(Protocol):
    def write(self, data: memoryview, /) -> int | None: ...


class InputFrame:
    """One physical buffer/rank uses the existing versioned Datum wire format.

    The fixed header and byte counts are resolved on CPU during preparation.
    Encoding borrows engine-owned arrays, so it must finish before their owners
    close. Unlike a caller submission, it needs no second full tensor snapshot.
    """

    def __init__(
        self, shapes: Mapping[Path, tuple[int, ...]], dtypes: Mapping[Path, str]
    ) -> None:
        if shapes.keys() != dtypes.keys():
            raise ValueError("frame shapes and dtypes require exactly the same paths")
        paths = sorted(shapes)
        trees: dict[str, Any] = {"inputs": {}, "loss_inputs": {}}
        nodes: set[Path] = {("inputs",), ("loss_inputs",)}
        sizes, layouts = [], []
        for index, path in enumerate(paths):
            if (
                not 2 <= len(path) <= MAX_TREE_DEPTH + 1
                or path[0] not in trees
                or any(type(key) is not str or not key for key in path)
            ):
                raise ValueError("frame paths require bounded model/loss tensor leaves")
            shape, dtype = tuple(shapes[path]), dtypes[path]
            sizes.append(tensor_bytes(dtype, shape))
            layouts.append({"dtype": dtype, "shape": shape})
            tree = trees[path[0]]
            for depth, key in enumerate(path[1:], 1):
                nodes.add(path[: depth + 1])
                if not isinstance(tree, dict) or (
                    key in tree and depth == len(path) - 1
                ):
                    raise ValueError("frame tensor paths overlap")
                if depth == len(path) - 1:
                    tree[key] = index
                else:
                    tree = tree.setdefault(key, {})
        self.header = manifest_json([trees], layouts)
        tree_bytes = len(nodes) * TREE_NODE_BYTES + sum(
            len(path[-1].encode("utf-8")) for path in nodes if len(path) > 1
        )
        self.size = PayloadSize(
            sum(sizes), len(self.header), tree_bytes, max(sizes, default=0)
        )
        self.shapes = MappingProxyType({path: tuple(shapes[path]) for path in paths})
        self.dtypes = MappingProxyType({path: dtypes[path] for path in paths})
        # Include bounded metadata accounting plus the sole contiguous scratch
        # block used when a worker's routed slice is noncontiguous.
        self.write_bytes = 8 * self.size.header + self.size.tree + FRAME_CHUNK_BYTES

    def write(
        self,
        stream: BufferWriter,
        fields: Mapping[Path, NDArray[Any]],
        budget: BufferBudget,
        artifact_id: str,
    ) -> ArtifactRef:
        if fields.keys() != self.shapes.keys() or any(
            value.shape != self.shapes[path]
            or value.dtype != storage_dtype(self.dtypes[path])
            for path, value in fields.items()
        ):
            raise ValueError("input arrays differ from their prepared frame layout")
        with budget.reserve(self.write_bytes, timeout=0):
            digest = hashlib.sha256()

            def write(data: memoryview) -> None:
                for start in range(0, len(data), FRAME_CHUNK_BYTES):
                    part = data[start : start + FRAME_CHUNK_BYTES]
                    digest.update(part)
                    while part:
                        size = stream.write(part)
                        if type(size) is not int or not 0 < size <= len(part):
                            raise OSError("prepared input frame write made no progress")
                        part = part[size:]

            write(memoryview(MAGIC + struct.pack("<Q", len(self.header))))
            write(memoryview(self.header))
            for path in self.shapes:
                array = fields[path]
                if not array.size:
                    continue
                if array.flags.c_contiguous:
                    write(memoryview(array).cast("B"))
                else:
                    # A full reshape/tobytes would allocate an unadmitted copy.
                    # The contig operand flag forces each strided piece into at
                    # most one declared scratch block, in wire (C) order.
                    with np.nditer(
                        array,
                        flags=["external_loop", "buffered", "zerosize_ok"],
                        # NumPy's current stub misspells "contig" as "config".
                        op_flags=cast(Any, ["readonly", "contig"]),
                        order="C",
                        buffersize=FRAME_CHUNK_BYTES // array.dtype.itemsize,
                    ) as chunks:
                        for chunk in chunks:
                            # A single-operand iterator yields one array.
                            write(memoryview(cast(NDArray[Any], chunk)).cast("B"))
            return ArtifactRef(
                artifact_id, digest.hexdigest(), self.size.wire, MEDIA_TYPE
            )
